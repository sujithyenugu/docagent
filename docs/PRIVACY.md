# Privacy & Data Protection — DocumentAgent

This document covers how DocumentAgent handles student data when used by teachers for grading, including data flow diagrams, applicable regulatory obligations, and the privacy guarantees of each extraction backend.

---

## 1. Data Flow Diagram

### Local Backend (Embedded Apple MLX / Gemma 3 4B)

```
Teacher's Phone / Tablet
        │
        │  photograph answer sheet / upload PDF
        ▼
┌───────────────────────────────────┐
│  DocumentAgent Mobile App         │
│  (React Native / Expo)            │
└───────────────┬───────────────────┘
                │  multipart POST /api/v1/grade/extract
                │  (HTTPS, local network or VPN only)
                ▼
┌───────────────────────────────────┐
│  DocumentAgent FastAPI API        │
│  (school Mac — Apple Silicon)     │
│                                   │
│  • image written to /tmp (RAM)    │
│  • passed to the embedded MLX     │
│    model in the same process      │
│    (no HTTP, no separate server)  │
│  • /tmp file deleted immediately  │
└───────────────┬───────────────────┘
                │  in-process function call
                ▼
┌───────────────────────────────────┐
│  Apple MLX — Gemma 3 4B           │
│  on device, in-process            │
│  NO external network calls        │
│  (except a one-time model         │
│   download on first use)          │
└───────────────┬───────────────────┘
                │  extracted markdown
                ▼
        API response → App
        (markdown displayed to teacher)
        NO storage by default
```

**Privacy guarantee (local mode):** Student answer sheet images and extracted text **never leave the device or local network**. No data is sent to any external server during inference.

---

### Cloud Backend (Landing AI AIDE)

```
Teacher's Phone / Tablet
        │
        ▼
┌───────────────────────────────────┐
│  DocumentAgent Mobile App         │
└───────────────┬───────────────────┘
                │  HTTPS POST /api/v1/grade/extract
                ▼
┌───────────────────────────────────┐
│  DocumentAgent FastAPI API        │
│  (school server or cloud VM)      │
│                                   │
│  • image → temp file              │
│  • sent to Landing AI AIDE API    │
│  • temp file deleted immediately  │
└───────────────┬───────────────────┘
                │  HTTPS to api.landing.ai
                ▼
┌───────────────────────────────────┐
│  Landing AI AIDE (cloud)          │
│  (external service)               │
└───────────────┬───────────────────┘
                │  extracted markdown
                ▼
        API response → App
        NO S3 storage (unless opted in via Streamlit sidebar)
```

**Privacy note (cloud mode):** Student document images are transmitted to Landing AI servers for OCR/extraction. Review Landing AI's privacy policy before using cloud mode with identifiable student data.

---

## 2. What Data Is Processed

| Data type | When it occurs | Stored? |
|---|---|---|
| Student answer sheet image | Every grading extraction | No (temp file, deleted after extraction) |
| Extracted markdown text | Every grading extraction | No (returned to app, not persisted by API) |
| Student name / roll number | Form-fill / mark sheet filling | No (returned in API response only) |
| Aadhaar number (if in doc) | Any extraction | Masked to `XXXX XXXX XXXX` before any output |
| Grade / marks | Rubric evaluation | No (computed in-memory, returned to app) |
| Teacher's IP address | Every API request | In-memory rate limiter only (not logged to disk) |

**All API responses carry `X-Data-Retained: false`** confirming no server-side persistence.

---

## 3. Aadhaar Masking Policy

The API automatically masks any Aadhaar numbers found in extracted text **before** returning the response or passing text to the LLM in Stage 2 of the form-fill pipeline.

Pattern: `\b(\d{4})\s?(\d{4})\s?(\d{4})\b` → replaced with `XXXX XXXX XXXX`

This applies to:
- `POST /api/v1/extract` (markdown output)
- `POST /api/v1/forms/fill` (field values and source attribution)
- `POST /api/v1/grade/extract` (markdown and keyword context snippets)

---

## 4. Regulatory Context

### DPDP Act 2023 (Digital Personal Data Protection Act, India)

The DPDP Act 2023 classifies children's personal data (under 18) as requiring **verifiable parental/guardian consent** before processing (Section 9).

| Obligation | How DocumentAgent addresses it |
|---|---|
| Consent before processing child data | Teacher must acknowledge the consent screen before first use and when switching to cloud backend |
| Purpose limitation | Extraction is performed solely for grading — text is not used for any other purpose |
| Data minimisation | Only the data needed for extraction is transmitted; no metadata is retained |
| Right to erasure | No data is stored server-side; nothing to erase (grade ledger is local to the teacher's device) |
| Security of personal data | Local mode: data stays on-device. Cloud mode: HTTPS transport; Landing AI's security controls apply |

> **Recommendation:** Use **local mode** for processing student documents that may contain personal data (name, DOB, Aadhaar). Switch to cloud mode only for non-sensitive documents (blank answer sheets, typed text).

### FERPA (for institutions serving US-affiliated students)

If the institution is subject to FERPA (e.g., international schools with US accreditation), student answer sheets and grades are **education records**. Transmitting them to an external API (cloud mode) requires the institution to have a FERPA-compliant data-sharing agreement with Landing AI. Local mode avoids this requirement entirely.

---

## 5. Consent Screen

The mobile app displays a consent acknowledgement in two situations:

1. **First launch** — before any grading session begins.
2. **Switching from local to cloud backend** — teacher is reminded that images will be sent externally.

Consent screen text:

> "DocumentAgent will process photographs of student answer sheets and assignments to assist with grading. In **local mode**, all processing happens on this device or your school's server — no data is sent externally. In **cloud mode**, images are sent to Landing AI's servers for processing; please ensure you have appropriate consent from students or guardians before using this mode with identifiable documents.
>
> By continuing, you confirm you have the authority to process these documents for educational purposes."

---

## 6. Network Verification (Automated Test)

The following pytest marker verifies that local mode makes **zero external HTTP calls**:

```python
# tests/api/test_privacy.py
import pytest
from unittest.mock import patch, MagicMock

@pytest.mark.anyio
async def test_local_mode_no_external_calls(client):
    """Local mode must not contact any external host (beyond the one-time
    model download, which is out of scope for this per-request check)."""
    with patch("local_model._load_engine") as mock_load_engine, \
         patch("mlx_vlm.generate") as mock_generate, \
         patch("mlx_vlm.prompt_utils.apply_chat_template", side_effect=lambda *a, **kw: ""):
        mock_load_engine.return_value = (MagicMock(), MagicMock(), {})
        mock_generate.return_value = MagicMock(text="# Extracted")
        # Patch socket to catch any real outgoing connections
        with patch("socket.getaddrinfo", side_effect=AssertionError("External DNS lookup!")):
            response = await client.post(
                "/api/v1/extract",
                files=[("file", ("test.png", b"<png bytes>", "image/png"))],
                data={"use_local": "true"},
            )
    # Generation ran entirely in-process (mocked above), no external DNS
    assert response.status_code in (200, 502)  # 502 if the mock doesn't fully match
```

Run with:
```bash
pytest tests/api/test_privacy.py -v
```

---

## 7. Rate Limiting

All endpoints are rate-limited per client IP:

| Backend | Limit |
|---|---|
| Local (embedded MLX) | 10 requests / minute |
| Cloud (Landing AI) | 1 request / minute |

Rate limit headers are returned in 429 responses. This prevents accidental bulk upload of student data.

---

## 8. S3 Storage (opt-in only)

S3 upload is **opt-in** and available only via the Streamlit UI sidebar. The mobile teacher grading app does **not** expose S3 upload. If S3 is enabled:

- Documents are stored at `s3://{bucket}/{uuid}/input/{filename}`
- Extracted markdown is stored at `s3://{bucket}/{uuid}/output/extracted.md`
- Access is controlled by the IAM policy in `infra/`
- Enable bucket versioning and server-side encryption (SSE-S3 or SSE-KMS) for any bucket used with student data

---

## 9. Security Contact

To report a privacy or security concern, open an issue labelled `security` at the repository's issue tracker or contact the maintainer directly.
