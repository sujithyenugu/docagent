# DocumentAgent — Architecture & Technical Reference

---

## Architecture Diagram

### Phase 1/2 — Cloud & Local (Dev / Demo)

```
User uploads document (PDF / image)
        │
        ▼
┌────────────────────────────────┐
│    Streamlit Web App           │  app.py — entry, nav; views/{cloud,local}.py — pages
│  (dedicated Cloud/Local pages) │  common.py — shared upload/result UI
└──────────────┬─────────────────┘
               │
      ┌────────┴────────────────┐
      │                         │
      ▼                         ▼
 Cloud Path                Local Path
 (Landing AI               local_model.py
  AIDE API)                LocalModelClient
      │                         │
      │                         │  PDF → pymupdf renders each page
      │                         │        as PNG at 96 DPI
      │                         │
      │                         ▼
      │                  mlx_vlm.load() / generate()
      │                  — in-process, same Python program
      │                  (no HTTP, no separate server)
      │                         │
      │                         ▼
      │                  Qwen2.5-VL-3B (VLM variant, MLX/Metal)
      │                  vision model — on device
      │                  no data leaves the machine
      │                         │
      └────────┬────────────────┘
               ▼
      Structured Output
      (Markdown — text, tables, key-value pairs)
               │
               ▼
      Two-column result display
      (document preview │ extracted markdown)
      Optional: download .md or save to S3
```

#### Local path — end-to-end call flow

```
views/local.py: write uploaded bytes → /tmp/tmp_upload<hash>.jpg
    │
    ▼
LocalModelClient.parse(Path("/tmp/..."))
    │
    ├─ PDF?  → _parse_pdf()
    │              fitz.open() → per-page PNG at 96 DPI, written to a temp file
    │              _run_generate() × N pages
    │
    └─ Image? → _parse_image()
                   _run_generate() × 1, directly on the file path
                       │
                       ▼
               _load_engine(repo_id)  — loaded once per process, cached
                   mlx_vlm.load(repo_id) → (model, processor)
                   mlx_vlm.utils.load_config(repo_id) → config
                       │
                       ▼
               apply_chat_template(processor, config, prompt, num_images=1)
               mlx_vlm.generate(
                   model, processor, formatted_prompt,
                   image=[image_path], max_tokens=2048, temperature=0.1,
               )
               ← GenerationResult.text
    │
    ▼
LocalModelResponse(markdown, metadata{page_count, duration_ms})
    │
views/local.py: finally block — temp file deleted
common.py: render_result() — two-column result display
```

First use of a given model repo triggers a one-time `huggingface_hub` download
(cached under `~/.cache/huggingface`); every extraction after that reuses the
already-loaded model in memory — no per-request reload.

### Phase 3 — Event-Driven Pipeline (Production)

```
User / Streamlit App
        │
        │ upload doc (PDF/image)
        ▼
┌───────────────────┐
│   AWS S3 Bucket   │  {uuid}/input/{filename}
│  (input prefix)   │
└────────┬──────────┘
         │ S3 ObjectCreated event
         ▼
┌───────────────────┐
│   AWS Lambda      │  documentagent-ade-processor
│  (handler.py)     │  • fetches Landing AI key from Secrets Manager
│                   │  • calls Landing AI ADE API
│                   │  • writes markdown → {uuid}/output/extracted.md
└────────┬──────────┘
         │ StartIngestionJob
         ▼
┌───────────────────┐
│ Bedrock Knowledge │  data source: S3 */output/*
│ Base + OpenSearch │  embedding: amazon.titan-embed-text-v2:0
│ Serverless        │
└────────┬──────────┘
         │ retrieve()
         ▼
┌───────────────────┐
│  Strands Agent    │  agent.py — interactive CLI
│  (Claude 3.5)     │  query_documents tool → KB retrieval
└───────────────────┘
```

---

## Technical Stack

| Layer | Technology |
|---|---|
| Document Extraction API | Landing AI AIDE (`landingai-ade`) |
| OCR Baseline | Tesseract, PaddleOCR |
| Local Vision Model | Qwen2.5-VL-3B (VLM variant, 4-bit MLX — `mlx-community/Qwen2.5-VL-3B-Instruct-4bit`) |
| Local Inference Runtime | **Apple MLX** (`mlx`, `mlx-vlm`) — embedded, in-process, Apple Silicon only |
| Local Client | `local_model.py` — `LocalModelClient` wraps `mlx_vlm` |
| Demo Interface | Streamlit (`app.py`, `views/`, `common.py`), Jupyter Notebook |
| Cloud Storage | AWS S3 (boto3) — `storage.py` |
| Serverless Compute | AWS Lambda — `lambda/handler.py` |
| Permissions | AWS IAM — `infra/` |
| Secrets | AWS Secrets Manager |
| AOSS REST signing | `requests-aws4auth` (SigV4 for OpenSearch Serverless) |
| Vector Knowledge Base | AWS Bedrock Knowledge Base + OpenSearch Serverless |
| Agent Framework | Strands Agents — `agent.py` |
| Default Agent Model | `amazon.nova-lite-v1:0` (AWS-native, no payment gate) |
| Language | Python 3.10+ |
| Tests | pytest + pytest-cov (`tests/`) |

---

## Benchmark: Cloud vs Local

**Current comparison** — all four models run under the same corrected
scoring harness against this project's own 20-document dataset (invoices,
handwritten ledgers, government forms, ID-style cards), via
[`benchmarks/eval_models.py`](benchmarks/eval_models.py). Full per-category
breakdown in [`benchmarks/model_comparison.md`](benchmarks/model_comparison.md):

| Model | Backend | Field Acc | Fuzzy Acc | Avg Latency | Cost/page |
|---|---|---|---|---|---|
| Qwen3-VL-32B | LM Studio (benchmark-only — not used by the shipped app) | 52.8% | 52.8% | 28.1s | $0.00 |
| **Qwen2.5-VL-3B** | **Local — Apple MLX (current default)** | **49.6%** | 50.0% | **2.7s** | $0.00 |
| Landing AI AIDE (`dpt-2-latest`) | Cloud | 49.1% | 49.1% | 8.5s | $0.015 |
| Gemma-3-4B | Local — Apple MLX (former default) | 44.2% | 45.0% | 3.2s | $0.00 |

All four land within ~9 percentage points of each other. Cloud AIDE no
longer shows the large accuracy edge the historical baseline below
suggested — the current local default is essentially tied with it while
running ~3x faster at no per-page cost.

This required fixing a harness bug first: an initial run scored AIDE at
only 19.1% field accuracy, because Landing AI's ADE API returns tables as
literal HTML (`<tr><td>Key:</td><td>Value</td></tr>`) rather than Markdown
pipe tables — the field-extraction heuristic in `run_benchmark.py` had only
ever been exercised against the local MLX models' Markdown-table output,
and mangled every HTML-table value it touched (e.g. `</td><td
id="0-3">Nisha Kapoor</td></tr>` instead of `Nisha Kapoor`). Fixed by
`_html_tables_to_pipes()`, which normalises HTML table rows into the same
`| Key | Value |` shape the parser already understood. (Also note: this
run used `dpt-2-latest`, not the app's hardcoded default `dpt-2-mini` —
the latter returned `422 Unsupported model` against the API key used here.)

**Historical baseline** (measured via LM Studio, CPU-only local inference,
under an earlier/different scoring pass — kept for reference, not directly
comparable to the corrected table above):

| Metric | Cloud (AIDE) | Local (Gemma 3 4B, LM Studio/CPU) |
|---|---|---|
| Overall field accuracy | 90.0% | 78.9% |
| Avg latency / page | 2.5s | 18.1s (CPU) / ~6s (GPU, estimated) |
| Cost per page | ~$0.015 | $0.00 |
| Hardest category | Ledgers 80.0% | Ledgers 60.0% |

The local runtime is now Apple MLX (Metal-accelerated) instead of LM Studio.
Qwen2.5-VL-3B has one known limitation: an `mlx-vlm` tokenizer bug
(`IndexError` in `detokenizer.add_token`, upstream in the `mlx-vlm` package,
not this project's code) intermittently fails a small fraction of
documents; the harness handles this gracefully (scores that document 0%
rather than crashing the run).

Set `LOCAL_MODEL_REPO=mlx-community/Qwen2.5-VL-7B-Instruct-4bit` to try the
larger 7B variant for potentially better accuracy at the cost of a bigger
download and slower inference — not yet benchmarked on this dataset.

**Decision rule:**

| Use case | Recommendation |
|---|---|
| Default / no strong constraint | **Local (Qwen2.5-VL-3B)** — accuracy is now on par with cloud, free, ~3x faster |
| Aadhaar / PAN / medical / sensitive student PII | **Local (Qwen2.5-VL-3B)** — no data leaves machine |
| Air-gapped school deployment | **Local (Qwen2.5-VL-3B)** — no external dependency |
| Willing to pay per page for the small remaining edge on structured tables | **Cloud (AIDE)** — modest lead on invoices/govt forms, $0.015/page |

The mobile app's **Settings → Extraction Backend** toggle lets a user choose either
path; its current default is Cloud, a holdover from when Cloud had a large accuracy
lead — worth revisiting now that Local is roughly at parity (see table above).

---

## Phase 3 — Strands Agent Model Selection

The Strands Agent only performs **Q&A on already-extracted markdown** — the generative
model never sees raw images. Choose based on stage:

| Stage | Model (`BEDROCK_MODEL_ID`) | Est. cost / 10 queries | Notes |
|---|---|---|---|
| Testing | `amazon.nova-lite-v1:0` ← **default** | ~$0.003 | AWS-native, no Anthropic payment gate |
| Pre-demo | `anthropic.claude-haiku-4-5-20251001-v1:0` | ~$0.05 | Fast, excellent quality |
| Demo | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` | ~$0.17 | Best accuracy for nuanced Q&A |

To switch models at any time — no code changes needed, just update `.env`:
```
BEDROCK_MODEL_ID=anthropic.claude-haiku-4-5-20251001-v1:0
```

---

## Privacy & Security

| Path | Where documents go | Recommendation |
|---|---|---|
| Cloud (Landing AI) | Sent to Landing AI servers during extraction | Suitable for non-sensitive business documents |
| Local (Qwen2.5-VL-3B) | Processed on-device, in-process — no external API calls during inference | Use for Aadhaar, PAN, financial records |

- Landing AI API key stored in Secrets Manager for the Lambda path — never in env vars as plaintext.
- No document storage by default; opt-in S3 toggle in the Streamlit sidebar.
- Local mode: a one-time model download (~2.5 GB) on first use. After that, no network calls during inference — the model runs embedded in the same process, with no separate local server involved.
- Local mode is only available on Apple Silicon macOS (MLX requirement) — including inside the Docker image, where the containerized Linux runtime has no path to Metal, so Local mode is unavailable there and Cloud mode is used instead.

---

## Prerequisites

- Python 3.10+
- [Landing AI API key](https://app.landing.ai/) — required for cloud extraction
- AWS account with access to S3, Lambda, Bedrock, Secrets Manager — required for Phase 3
- Apple Silicon Mac (macOS 14+) — required for local extraction (Apple MLX); the model downloads automatically on first use, no manual setup
