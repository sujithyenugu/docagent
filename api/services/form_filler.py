"""Form-fill service — two-stage extract → structured-fields pipeline.

Stage 1: Run the extraction service on every uploaded document to get markdown.
Stage 2: Use the local vision model in JSON mode to extract structured fields
         defined by the requested form template.

Aadhaar numbers are masked in all output (XXXX XXXX XXXX).

Supported templates
-------------------
Admin / identity (use_case: admin)
  lease_agreement        — residential lease / rent agreement
  rent_receipt           — monthly rent receipt
  dl_renewal             — driving licence renewal application
  affidavit              — general affidavit
  bank_account           — bank account opening form

Grading (use_case: grading)
  cbse_marksheet         — CBSE Class 10/12 mark sheet
  icse_marksheet         — ICSE mark sheet
  school_progress_report — school term progress report
  admit_card_verification — admit card field extraction for verification

Source documents
----------------
Student documents (grading):
  admit_card             — student admit card
  roll_number_sheet      — class roll number sheet
  answer_booklet         — answer booklet page(s)
  cbse_marksheet         — CBSE mark sheet (source)

Identity documents (admin):
  aadhaar     — Aadhaar card (front + back)
  pan         — PAN card
  dl          — Driving Licence
  voter_id    — Voter ID (EPIC)
"""

import json
import logging
import re
import time
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Template definitions
# ---------------------------------------------------------------------------

_TEMPLATES: dict[str, dict] = {
    # ── Admin / identity ──────────────────────────────────────────────────
    "lease_agreement": {
        "required": ["tenant_name", "landlord_name", "property_address", "monthly_rent"],
        "optional": ["start_date", "duration_months", "security_deposit"],
        "use_case": "admin",
    },
    "rent_receipt": {
        "required": ["tenant_name", "landlord_name", "amount", "month"],
        "optional": ["property_address", "receipt_number", "payment_mode"],
        "use_case": "admin",
    },
    "dl_renewal": {
        "required": ["applicant_name", "dl_number", "dob", "address"],
        "optional": ["vehicle_class", "blood_group", "expiry_date"],
        "use_case": "admin",
    },
    "affidavit": {
        "required": ["deponent_name", "statement", "date"],
        "optional": ["deponent_address", "notary_name"],
        "use_case": "admin",
    },
    "bank_account": {
        "required": ["applicant_name", "dob", "address", "pan_number"],
        "optional": ["occupation", "annual_income", "nominee_name"],
        "use_case": "admin",
    },
    # ── Grading ───────────────────────────────────────────────────────────
    "cbse_marksheet": {
        "required": [
            "student_name", "roll_number", "school_name",
            "class_grade", "english_marks", "hindi_marks",
            "mathematics_marks", "science_marks", "social_science_marks",
            "total_marks", "percentage",
        ],
        "optional": ["dob", "exam_year", "grade", "result", "school_code"],
        "use_case": "grading",
    },
    "icse_marksheet": {
        "required": [
            "student_name", "roll_number", "school_name",
            "class_grade", "english_marks", "total_marks", "percentage",
        ],
        "optional": [
            "dob", "exam_year", "grade", "result",
            "second_language_marks", "history_civics_marks",
            "geography_marks", "mathematics_marks", "science_marks",
        ],
        "use_case": "grading",
    },
    "school_progress_report": {
        "required": [
            "student_name", "roll_number", "class_section",
            "term", "academic_year", "total_marks", "percentage", "rank",
        ],
        "optional": [
            "teacher_remarks", "attendance_percentage",
            "english_marks", "hindi_marks", "mathematics_marks",
            "science_marks", "social_science_marks",
        ],
        "use_case": "grading",
    },
    "admit_card_verification": {
        "required": [
            "student_name", "roll_number", "dob",
            "exam_name", "exam_date", "exam_centre",
        ],
        "optional": ["father_name", "mother_name", "school_name", "class_grade"],
        "use_case": "grading",
    },
}

_AADHAAR_RE = re.compile(r"\b(\d{4})\s?(\d{4})\s?(\d{4})\b")


def _mask_aadhaar(text: str) -> str:
    """Replace all Aadhaar numbers with XXXX XXXX XXXX."""
    return _AADHAAR_RE.sub("XXXX XXXX XXXX", text)


def _extract_json_from_text(text: str) -> dict:
    """Parse JSON from LLM response, tolerating markdown code-fence wrappers."""
    # Strip ```json ... ``` or ``` ... ``` wrappers if present
    stripped = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped.strip())
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse failed (%s); raw text: %.200s", exc, text)
        return {}


def _call_llm_json(prompt: str) -> dict:
    """Call the shared local MLX engine with a JSON-extraction prompt.

    Returns an empty dict if the model is unavailable or returns unparseable output.
    """
    from local_model import generate_text

    try:
        raw = generate_text(prompt, max_tokens=1024, temperature=0.0)
        return _extract_json_from_text(raw or "{}")
    except Exception as exc:
        logger.warning("Local model JSON call failed: %s", exc)
        return {}


def _build_extraction_prompt(fields: list[str], markdown: str) -> str:
    field_list = ", ".join(f'"{f}"' for f in fields)
    return (
        f"You are a form-filling assistant for Indian government documents.\n"
        f"Extract the following fields from the provided document text.\n"
        f"Fields to extract: [{field_list}]\n\n"
        f"Rules:\n"
        f"- Return ONLY a valid JSON object with exactly these keys.\n"
        f"- Set the value to null if a field cannot be found.\n"
        f"- Do NOT include any explanation, markdown, or extra text.\n"
        f"- Dates should be in DD/MM/YYYY format.\n"
        f"- Monetary amounts should be numeric strings (e.g. '15000').\n\n"
        f"Document text:\n{markdown}\n\n"
        f"JSON:"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class FormFillResult:
    def __init__(
        self,
        fields: dict[str, Any],
        field_sources: dict[str, str],
        missing_fields: list[str],
    ) -> None:
        self.fields = fields
        self.field_sources = field_sources
        self.missing_fields = missing_fields


def fill(
    docs: list[tuple[bytes, str]],
    template_id: str,
    *,
    use_local: bool = True,
    api_key: str | None = None,
    model: str = "dpt-2-latest",
    answer_key: dict | None = None,
) -> FormFillResult:
    """Run the two-stage form-fill pipeline.

    Args:
        docs:        List of (file_bytes, filename) tuples.
        template_id: One of the keys in ``_TEMPLATES``.
        use_local:   Use the embedded local MLX backend for stage-1 extraction.
        api_key:     Required when ``use_local=False``.
        model:       ADE model variant (cloud only).
        answer_key:  Optional dict ``{field: expected_value}`` for grading templates.
                     When supplied, missing required fields are compared against the key.

    Returns:
        :class:`FormFillResult` with extracted fields, sources, and missing fields.

    Raises:
        ValueError: Unknown template_id or no documents provided.
    """
    if template_id not in _TEMPLATES:
        raise ValueError(
            f"Unknown template '{template_id}'. "
            f"Available: {', '.join(_TEMPLATES)}"
        )
    if not docs:
        raise ValueError("At least one document is required.")

    template = _TEMPLATES[template_id]
    all_fields = template["required"] + template["optional"]

    # ── Stage 1: extract markdown from each document ──────────────────────
    from api.services.extraction import run as extract_run

    markdowns: list[tuple[str, str]] = []  # (markdown, filename)
    for doc_bytes, filename in docs:
        result = extract_run(
            doc_bytes, filename, use_local=use_local, api_key=api_key, model=model
        )
        markdowns.append((_mask_aadhaar(result["markdown"]), filename))
        logger.info("stage-1 extract | file=%s pages=%s", filename, result["page_count"])

    # ── Stage 2: extract structured fields from combined markdown ─────────
    combined_markdown = "\n\n---\n\n".join(
        f"[Document: {fname}]\n{md}" for md, fname in markdowns
    )
    prompt = _build_extraction_prompt(all_fields, combined_markdown)

    start = time.time()
    raw_fields = _call_llm_json(prompt)
    logger.info("stage-2 field extract | duration_ms=%d", int((time.time() - start) * 1000))

    # ── Post-process ──────────────────────────────────────────────────────
    fields: dict[str, Any] = {}
    field_sources: dict[str, str] = {}
    missing_fields: list[str] = []

    for field in all_fields:
        value = raw_fields.get(field)
        fields[field] = value
        if value is not None:
            # Attribute the field to whichever document's markdown contains its value
            source_file = _find_source(str(value), markdowns) or "unknown"
            field_sources[field] = source_file
        elif field in template["required"]:
            missing_fields.append(field)

    return FormFillResult(
        fields=fields,
        field_sources=field_sources,
        missing_fields=missing_fields,
    )


def _find_source(value: str, markdowns: list[tuple[str, str]]) -> str | None:
    """Return the filename of the first document whose markdown contains *value*."""
    if not value or value.lower() == "null":
        return None
    for md, fname in markdowns:
        if value.lower() in md.lower():
            return fname
    return markdowns[0][1] if markdowns else None


def list_templates() -> list[dict]:
    """Return a list of available template descriptors."""
    return [
        {
            "template_id": tid,
            "required_fields": tmpl["required"],
            "optional_fields": tmpl["optional"],
            "use_case": tmpl.get("use_case", "admin"),
        }
        for tid, tmpl in _TEMPLATES.items()
    ]
