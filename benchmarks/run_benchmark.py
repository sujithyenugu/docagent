#!/usr/bin/env python3
"""Benchmark Landing AI AIDE vs local Gemma 3 4B on Indian document categories.

Metrics
-------
- Field Accuracy     — exact-match rate of extracted key-value fields vs ground truth
- Fuzzy Accuracy     — token-level F1 match (handles minor OCR/formatting differences)
- End-to-end latency — wall-clock ms per page including any encoding overhead
- Cost per page      — estimated credits/USD (AIDE: ~$0.015/page; local: $0.00)

Categories tested
-----------------
- invoices      — GST invoices (structured tables, numeric fields)
- ledgers       — Handwritten daily ledgers (semi-structured, noisy)
- govt_forms    — Government application forms (key-value, mixed Hindi/English)
- id_docs       — Aadhaar-format and PAN-format cards

Usage
-----
    # Local backend (embedded Apple MLX — Apple Silicon Mac required)
    python benchmarks/run_benchmark.py \\
        --dataset-dir benchmarks/dataset \\
        --backend local \\
        --output benchmarks/run_local.json

    # Cloud backend (requires Landing AI API key)
    python benchmarks/run_benchmark.py \\
        --dataset-dir benchmarks/dataset \\
        --backend cloud \\
        --api-key YOUR_KEY \\
        --output benchmarks/run_cloud.json

    # LM Studio backend (benchmark-only — compares a model served through LM
    # Studio; not used by the shipped app, which runs local models embedded
    # via Apple MLX instead)
    python benchmarks/run_benchmark.py \\
        --dataset-dir benchmarks/dataset \\
        --backend lmstudio \\
        --model qwen2.5-vl-7b-instruct \\
        --lmstudio-base-url http://localhost:1234/v1 \\
        --output benchmarks/run_lmstudio.json

    # Compare two pre-run JSON result files and write results.md
    python benchmarks/run_benchmark.py \\
        --compare benchmarks/run_local.json benchmarks/run_cloud.json \\
        --write-results-md

    # CI mode — exit 1 if field accuracy < 80% for any category
    python benchmarks/run_benchmark.py \\
        --dataset-dir benchmarks/dataset --backend local --strict
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

# Make the project root importable (for `from api.services... import ...`
# below) regardless of how this script is invoked — direct `python
# benchmarks/run_benchmark.py` only puts this file's own directory on
# sys.path, not the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cost constants
# ---------------------------------------------------------------------------

_CLOUD_COST_USD_PER_PAGE = 0.015   # approximate Landing AI AIDE credit ≈ $0.015
_LOCAL_COST_USD_PER_PAGE = 0.000   # free after one-time model download
_LMSTUDIO_COST_USD_PER_PAGE = 0.000  # free — runs on the user's own machine

_DEFAULT_LMSTUDIO_BASE_URL = "http://localhost:1234/v1"

# Same instruction given to the embedded MLX model (local_model.py's
# _EXTRACT_PROMPT) — keeping the prompt identical is what makes an
# LM-Studio-served model comparable to the local/cloud rows in the same report.
_LMSTUDIO_EXTRACT_PROMPT = (
    "Extract all text, tables, and key-value pairs from this document image. "
    "Return clean Markdown. Preserve table structure. Do not add commentary."
)


def _call_lmstudio(
    image_bytes: bytes,
    filename: str,
    base_url: str,
    model: str,
    timeout: float = 180.0,
) -> dict[str, Any]:
    """Send one document image to a model loaded in LM Studio for extraction.

    Benchmark-only: LM Studio was removed as a runtime dependency of the
    shipped app (replaced by embedded Apple MLX — see ARCHITECTURE.md) but
    stays useful here as *one more model source* to compare against local
    MLX and cloud AIDE on this project's own dataset. Talks to LM Studio's
    OpenAI-compatible REST server directly via `requests` rather than the
    `openai` package, since that package was removed from requirements.txt
    when LM Studio was dropped from the app itself.
    """
    import base64
    import mimetypes

    import requests

    mime = mimetypes.guess_type(filename)[0] or "image/png"
    data_url = f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"

    resp = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _LMSTUDIO_EXTRACT_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "max_tokens": 2048,
            "temperature": 0.1,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"] or ""
    return {"markdown": text, "page_count": 1, "model": model}


# ---------------------------------------------------------------------------
# Field extraction from markdown
# ---------------------------------------------------------------------------

_STRIP = re.compile(r"[^\w\s\.\-\/\+@₹$]")

# Currency-word tokens stripped during normalisation so "Rs. 10489" compares
# equal to "10489" — models often echo the currency label the document uses,
# which shouldn't count as a mismatch against a ground-truth value that
# stores just the number.
_CURRENCY_WORDS = {"rs", "inr", "usd", "aed"}


def _normalise(text: str) -> str:
    """Lower-case, collapse whitespace, strip punctuation/currency words for comparison.

    Apostrophes are deleted outright (not turned into a space) before the
    general punctuation strip: "Father's" must become "fathers", not
    "father s" — the latter splits into two tokens and stops "Father's
    Name" ever matching the expected key "fathers_name" (normalised to
    "fathers name"), leaving the generic key "name" as the only candidate
    and silently mismatching the field.
    """
    text = str(text).lower().replace("'", "").replace("’", "")
    text = _STRIP.sub(" ", text)
    tokens = [t for t in text.split() if t.rstrip(".") not in _CURRENCY_WORDS]
    return " ".join(tokens)


def _token_f1(pred: str, gold: str) -> float:
    """Token-level F1 between predicted and gold strings."""
    pred_tokens = set(_normalise(pred).split())
    gold_tokens = set(_normalise(gold).split())
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = pred_tokens & gold_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _strip_md_emphasis(text: str) -> str:
    """Remove markdown bold/italic asterisks and surrounding whitespace."""
    return re.sub(r"\*+", "", text).strip()


def _best_matching_key(k_norm: str, expected_keys: list[str]) -> str | None:
    """Pick the single expected key that best matches a document-side label.

    Prefers an exact normalised match; falls back to substring containment
    (e.g. document label "Applicant Name" matching expected key "name") but
    always prefers the most specific (longest) candidate. Without this,
    checking "does either string contain the other" independently for every
    expected key lets a generic key like "name" match a *more specific*
    document label like "Father's Name" purely because "name" is a literal
    substring of it — verified in practice to swap `name`/`fathers_name`
    values on every ID-document benchmark run, for every model tested
    (local MLX and LM Studio alike), since it's a parsing bug independent
    of what the model actually outputs.
    """
    best_key: str | None = None
    best_score = -1
    for ek in expected_keys:
        ek_norm = _normalise(ek).replace("_", " ")
        if not ek_norm:
            continue
        if ek_norm == k_norm:
            score = 1000 + len(ek_norm)
        elif ek_norm in k_norm or k_norm in ek_norm:
            score = len(ek_norm)
        else:
            continue
        if score > best_score:
            best_score = score
            best_key = ek
    return best_key


_HTML_TR = re.compile(r"<tr[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_HTML_CELL = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
_HTML_TAG = re.compile(r"<[^>]+>")


def _html_tables_to_pipes(markdown: str) -> str:
    """Rewrite literal HTML <tr>/<td> table rows as "| cell | cell |" rows.

    Landing AI's ADE API renders tables as literal HTML
    (`<tr><td>Applicant Name:</td><td>Nisha Kapoor</td></tr>`), not Markdown
    pipe tables — the local MLX models never do this, so the table-row
    parser below never needed to handle it. Left unconverted, the line falls
    through to the colon-separated pattern, which partitions on the *first*
    colon in the raw HTML and produces a value still full of markup
    (`</td><td id="0-3">Nisha Kapoor</td></tr>`) — technically present but
    never exact-matches the clean ground-truth value. Converting each row
    upfront lets the one table parser serve both output styles.
    """
    if "<tr" not in markdown.lower():
        return markdown

    def _row_to_pipes(m: re.Match) -> str:
        cells = [_HTML_TAG.sub(" ", c).strip() for c in _HTML_CELL.findall(m.group(1))]
        return "| " + " | ".join(cells) + " |"

    return _HTML_TR.sub(_row_to_pipes, markdown)


def _extract_fields_from_markdown(markdown: str, expected_keys: list[str]) -> dict[str, str]:
    """Heuristically extract key-value pairs from extracted markdown text.

    Looks for patterns like:
        Key: Value
        **Key**: Value
        **Key:** Value      (bold wraps the colon too — the local MLX model's
                             actual style; must strip asterisks from the
                             VALUE too, not just the key, or every extracted
                             value gets a stray leading "**")
        | Key | Value |
        <tr><td>Key</td><td>Value</td></tr>  (Landing AI ADE's table style —
                             normalised to "| Key | Value |" up front, see
                             `_html_tables_to_pipes`)

    Returns a dict of lowercased-key → extracted-value-string. Each line
    contributes to at most one expected key (see `_best_matching_key`), and
    a line already resolved via the table-row pattern skips the colon-
    separated pattern — re-running both on "| Key: | Value |" style rows
    (table AND colon syntax on the same line) previously let the colon pass
    re-assign a pipe-contaminated value ("| DEEPAK VERMA |") over the clean
    table-parsed one.
    """
    extracted: dict[str, str] = {}
    lines = _html_tables_to_pipes(markdown).splitlines()

    for line in lines:
        matched_via_table = False

        # Table row: | key | value |
        parts = [_strip_md_emphasis(p) for p in line.split("|") if p.strip()]
        if len(parts) == 2:
            k_norm = _normalise(parts[0])
            best = _best_matching_key(k_norm, expected_keys)
            if best:
                extracted[best] = parts[1]
                matched_via_table = True

        # Colon-separated: Key: Value (asterisks/pipes may wrap either side)
        if not matched_via_table and ":" in line:
            raw_key, _, raw_val = line.partition(":")
            raw_key = _strip_md_emphasis(raw_key).strip("| \t")
            raw_val = _strip_md_emphasis(raw_val).strip("| \t")
            k_norm = _normalise(raw_key)
            best = _best_matching_key(k_norm, expected_keys)
            if best and raw_val:
                extracted[best] = raw_val

    return extracted


# ---------------------------------------------------------------------------
# Per-document evaluation
# ---------------------------------------------------------------------------

def _evaluate_document(
    img_path: Path,
    gt: dict[str, Any],
    backend: str,
    api_key: str,
    model: str,
    local_model_repo: str | None = None,
    lmstudio_base_url: str = _DEFAULT_LMSTUDIO_BASE_URL,
) -> dict[str, Any]:
    """Run extraction on one image and return evaluation metrics."""
    fields: dict[str, str] = gt.get("fields", {})
    expected_keys = list(fields.keys())

    t0 = time.perf_counter()
    resolved_model = model
    try:
        doc_bytes = img_path.read_bytes()
        if backend == "local":
            from api.services.extraction import run_local
            result = run_local(doc_bytes, img_path.name, repo_id=local_model_repo)
        elif backend == "lmstudio":
            result = _call_lmstudio(doc_bytes, img_path.name, lmstudio_base_url, model)
        else:
            from api.services.extraction import run_cloud
            result = run_cloud(doc_bytes, img_path.name, api_key=api_key, model=model)
        markdown = result["markdown"]
        page_count = result.get("page_count", 1)
        resolved_model = result.get("model", resolved_model)
        extraction_ok = True
    except Exception as exc:
        logger.error("Extraction failed for %s: %s", img_path.name, exc)
        markdown = ""
        page_count = 1
        extraction_ok = False

    latency_ms = int((time.perf_counter() - t0) * 1000)
    if backend == "cloud":
        cost_per_page = _CLOUD_COST_USD_PER_PAGE
    elif backend == "lmstudio":
        cost_per_page = _LMSTUDIO_COST_USD_PER_PAGE
    else:
        cost_per_page = _LOCAL_COST_USD_PER_PAGE
    total_cost = cost_per_page * max(page_count, 1)

    extracted = _extract_fields_from_markdown(markdown, expected_keys)

    # Exact-match accuracy
    exact_matches = 0
    fuzzy_scores: list[float] = []
    field_results: dict[str, dict] = {}

    for key, gold_val in fields.items():
        pred_val = extracted.get(key, "")
        exact = _normalise(pred_val) == _normalise(gold_val)
        f1 = _token_f1(pred_val, gold_val)
        exact_matches += int(exact)
        fuzzy_scores.append(f1)
        field_results[key] = {
            "gold": gold_val,
            "pred": pred_val,
            "exact": exact,
            "token_f1": round(f1, 3),
        }

    n = len(fields)
    field_accuracy = exact_matches / n if n else 0.0
    fuzzy_accuracy = sum(fuzzy_scores) / len(fuzzy_scores) if fuzzy_scores else 0.0

    return {
        "file": img_path.name,
        "category": gt.get("category", "unknown"),
        "model": resolved_model,
        "extraction_ok": extraction_ok,
        "latency_ms": latency_ms,
        "cost_usd": round(total_cost, 5),
        "field_accuracy": round(field_accuracy, 3),
        "fuzzy_accuracy": round(fuzzy_accuracy, 3),
        "num_fields": n,
        "exact_matches": exact_matches,
        "field_results": field_results,
    }


# ---------------------------------------------------------------------------
# Full benchmark run
# ---------------------------------------------------------------------------

def run_benchmark(
    dataset_dir: Path,
    backend: str,
    api_key: str = "",
    model: str = "dpt-2-latest",
    local_model_repo: str | None = None,
    lmstudio_base_url: str = _DEFAULT_LMSTUDIO_BASE_URL,
) -> dict[str, Any]:
    """Run benchmark over all images in dataset_dir.

    Expects layout: dataset_dir/{category}/{stem}.png + {stem}.json

    Args:
        local_model_repo: For backend="local", override the MLX model repo id
            for this run only (defaults to LOCAL_MODEL_REPO / the app's
            configured default). Lets one process benchmark multiple local
            models against the same dataset — see eval_models.py.
        lmstudio_base_url: For backend="lmstudio", the base URL of a running
            LM Studio server (OpenAI-compatible REST API). `model` is the
            model name as loaded in LM Studio. Benchmark-only — LM Studio is
            not used by the shipped app; this exists to compare a model
            served through it against the embedded MLX and cloud paths.
    """
    all_results: list[dict] = []
    categories = sorted([d.name for d in dataset_dir.iterdir() if d.is_dir()])

    if not categories:
        logger.error("No category subdirectories found in %s", dataset_dir)
        return {}

    for cat in categories:
        cat_dir = dataset_dir / cat
        images = sorted(cat_dir.glob("*.png"))
        if not images:
            logger.warning("No PNG images in %s", cat_dir)
            continue
        for img_path in images:
            gt_path = img_path.with_suffix(".json")
            if not gt_path.exists():
                logger.warning("No ground truth for %s — skipping", img_path.name)
                continue
            gt = json.loads(gt_path.read_text(encoding="utf-8"))
            logger.info("Evaluating %s/%s ...", cat, img_path.name)
            result = _evaluate_document(
                img_path, gt, backend, api_key, model,
                local_model_repo=local_model_repo,
                lmstudio_base_url=lmstudio_base_url,
            )
            all_results.append(result)
            logger.info(
                "  %-40s exact=%.0f%% fuzzy=%.0f%% latency=%dms cost=$%.4f",
                img_path.name,
                result["field_accuracy"] * 100,
                result["fuzzy_accuracy"] * 100,
                result["latency_ms"],
                result["cost_usd"],
            )

    # Aggregate by category
    by_cat: dict[str, list] = {}
    for r in all_results:
        by_cat.setdefault(r["category"], []).append(r)

    category_summary: dict[str, dict] = {}
    for cat, rows in by_cat.items():
        category_summary[cat] = {
            "count": len(rows),
            "field_accuracy": round(sum(r["field_accuracy"] for r in rows) / len(rows), 3),
            "fuzzy_accuracy": round(sum(r["fuzzy_accuracy"] for r in rows) / len(rows), 3),
            "avg_latency_ms": round(sum(r["latency_ms"] for r in rows) / len(rows)),
            "total_cost_usd": round(sum(r["cost_usd"] for r in rows), 4),
        }

    overall = all_results
    # Prefer the actual resolved model name reported by extraction (accurate
    # even if local_model_repo wasn't passed, or the default model changes).
    resolved_local_model = (
        overall[0]["model"] if overall and backend in ("local", "lmstudio") else local_model_repo
    )
    summary = {
        "backend": backend,
        "model": model if backend == "cloud" else (resolved_local_model or "unknown"),
        "total_documents": len(overall),
        "field_accuracy": round(sum(r["field_accuracy"] for r in overall) / len(overall), 3) if overall else 0,
        "fuzzy_accuracy": round(sum(r["fuzzy_accuracy"] for r in overall) / len(overall), 3) if overall else 0,
        "avg_latency_ms": round(sum(r["latency_ms"] for r in overall) / len(overall)) if overall else 0,
        "total_cost_usd": round(sum(r["cost_usd"] for r in overall), 4),
        "categories": category_summary,
        "documents": all_results,
    }
    return summary


# ---------------------------------------------------------------------------
# Markdown report writer
# ---------------------------------------------------------------------------

def _write_results_md(local_summary: dict, cloud_summary: dict, out_path: Path) -> None:
    """Write a filled benchmarks/results.md comparing local and cloud."""

    def pct(v: float) -> str:
        return f"{v * 100:.1f}%"

    def ms(v: int) -> str:
        if v >= 1000:
            return f"{v / 1000:.1f}s"
        return f"{v}ms"

    lines: list[str] = [
        "# Phase 2 Benchmark Results: Landing AI AIDE vs Gemma 3 4B (Local)",
        "",
        f"> **Last run:** {_today()}",
        f"> **Dataset:** {local_summary.get('total_documents', '?')} documents "
        f"(5 per category × 4 categories)",
        "",
        "---",
        "",
        "## Setup",
        "",
        "| Item | Cloud Path | Local Path |",
        "|---|---|---|",
        f"| Model | Landing AI AIDE (`{cloud_summary.get('model', 'dpt-2-latest')}`) | "
        f"Gemma 3 4B IT (VLM, 4-bit MLX) |",
        "| Runtime | Landing AI API | Apple MLX (embedded, in-process) |",
        "| Hardware | Cloud (Landing AI infra) | Local Apple Silicon Mac |",
        "| Cost | ~$0.015/page | $0.00/page (after model download) |",
        "",
        "---",
        "",
        "## Dataset",
        "",
        "20 Indian document images across 4 categories (5 documents each):",
        "",
        "| Category | Description |",
        "|---|---|",
        "| Invoices | GST invoices with line items, CGST/SGST calculations |",
        "| Handwritten ledgers | Shop daily ledgers with debit/credit entries |",
        "| Government forms | Ration card / certificate applications |",
        "| ID-style documents | Synthetic Aadhaar-format and PAN-format cards |",
        "",
        "Source: `benchmarks/dataset/` — synthetically generated by "
        "`scripts/generate_benchmark_dataset.py` (seed=42). "
        "All PII is synthetic; no real personal data is used.",
        "",
        "---",
        "",
        "## Results",
        "",
        "### Summary Table",
        "",
        "| Category | AIDE Field Acc | Gemma Field Acc | AIDE Fuzzy | Gemma Fuzzy | "
        "AIDE Latency | Gemma Latency | AIDE Cost/doc |",
        "|---|---|---|---|---|---|---|---|",
    ]

    cat_order = ["invoice", "ledger", "govt_form", "id_doc"]
    cat_labels = {
        "invoice": "Invoices",
        "ledger": "Ledgers (handwritten)",
        "govt_form": "Govt Forms",
        "id_doc": "ID Documents",
    }
    for cat in cat_order:
        lc = local_summary.get("categories", {}).get(cat, {})
        cc = cloud_summary.get("categories", {}).get(cat, {})
        if not lc and not cc:
            continue
        label = cat_labels.get(cat, cat)
        lines.append(
            f"| {label} "
            f"| {pct(cc.get('field_accuracy', 0))} "
            f"| {pct(lc.get('field_accuracy', 0))} "
            f"| {pct(cc.get('fuzzy_accuracy', 0))} "
            f"| {pct(lc.get('fuzzy_accuracy', 0))} "
            f"| {ms(cc.get('avg_latency_ms', 0))} "
            f"| {ms(lc.get('avg_latency_ms', 0))} "
            f"| ${cc.get('total_cost_usd', 0) / max(cc.get('count', 1), 1):.4f} |"
        )

    lines += [
        "",
        "### Aggregate",
        "",
        "| Metric | Cloud (AIDE) | Local (Gemma 3 4B) |",
        "|---|---|---|",
        f"| Overall field accuracy | {pct(cloud_summary.get('field_accuracy', 0))} "
        f"| {pct(local_summary.get('field_accuracy', 0))} |",
        f"| Overall fuzzy accuracy | {pct(cloud_summary.get('fuzzy_accuracy', 0))} "
        f"| {pct(local_summary.get('fuzzy_accuracy', 0))} |",
        f"| Avg latency / page | {ms(cloud_summary.get('avg_latency_ms', 0))} "
        f"| {ms(local_summary.get('avg_latency_ms', 0))} |",
        f"| Cost per page | ~$0.015 | $0.00 |",
        f"| Total dataset cost | ${cloud_summary.get('total_cost_usd', 0):.4f} | $0.00 |",
        "",
        "---",
        "",
        "## Recommendation",
        "",
        "Based on benchmark results:",
        "",
        "- **Use cloud (AIDE)** when field accuracy is paramount — especially for "
        "structured tables (invoices, mark sheets), mixed Hindi/English forms, and "
        "accuracy-sensitive workflows like payroll or tax documents.",
        "- **Use local (Gemma 3 4B via embedded Apple MLX)** when documents must not leave "
        "the machine — Aadhaar, PAN, medical records, student exam sheets containing "
        "roll numbers. Accuracy on simple printed documents is within acceptable range; "
        "handwritten content sees a ~8–12 pp accuracy drop vs AIDE.",
        "",
        "See [ARCHITECTURE.md](../ARCHITECTURE.md) for the full decision framework.",
        "",
        "---",
        "",
        "## Reproducing This Benchmark",
        "",
        "```bash",
        "# 1. Generate the dataset (one-time, ~20 seconds)",
        "python scripts/generate_benchmark_dataset.py",
        "",
        "# 2a. Run local backend (embedded Apple MLX — Apple Silicon Mac required)",
        "python benchmarks/run_benchmark.py \\",
        "    --dataset-dir benchmarks/dataset \\",
        "    --backend local \\",
        "    --output benchmarks/run_local.json",
        "",
        "# 2b. Run cloud backend (requires Landing AI API key)",
        "python benchmarks/run_benchmark.py \\",
        "    --dataset-dir benchmarks/dataset \\",
        "    --backend cloud \\",
        "    --api-key YOUR_KEY \\",
        "    --output benchmarks/run_cloud.json",
        "",
        "# 3. Regenerate this results.md",
        "python benchmarks/run_benchmark.py \\",
        "    --compare benchmarks/run_local.json benchmarks/run_cloud.json \\",
        "    --write-results-md",
        "```",
        "",
        "Hardware used for reference run: Apple M2 Pro (local), Landing AI hosted (cloud).",
        "",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", out_path)


def _today() -> str:
    import datetime
    return datetime.date.today().isoformat()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark AIDE vs local Gemma on Indian documents")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset-dir", help="Path to benchmarks/dataset directory")
    group.add_argument("--compare", nargs=2, metavar=("LOCAL_JSON", "CLOUD_JSON"),
                       help="Compare two pre-run JSON result files")

    parser.add_argument("--backend", choices=["local", "cloud", "lmstudio"], default="local",
                        help="Extraction backend (used with --dataset-dir)")
    parser.add_argument("--api-key", default="", help="Landing AI API key (cloud only)")
    parser.add_argument("--model", default="dpt-2-latest",
                        help="ADE model variant (cloud), or the model name as loaded "
                             "in LM Studio (lmstudio)")
    parser.add_argument("--local-model", default=None,
                        help="Override the local MLX model repo id for this run "
                             "(local backend only), e.g. "
                             "mlx-community/Qwen2.5-VL-7B-Instruct-4bit")
    parser.add_argument("--lmstudio-base-url", default=_DEFAULT_LMSTUDIO_BASE_URL,
                        help="LM Studio server base URL (lmstudio backend only, "
                             f"default: {_DEFAULT_LMSTUDIO_BASE_URL}). Benchmark-only — "
                             "not used by the shipped app.")
    parser.add_argument("--output", default="", help="Write JSON results to this path")
    parser.add_argument("--write-results-md", action="store_true",
                        help="Write benchmarks/results.md (used with --compare)")
    parser.add_argument("--strict", action="store_true",
                        help="Exit 1 if field accuracy < 80%% for any category")
    parser.add_argument("--format", choices=["text", "json", "markdown"], default="text",
                        help="Output format for terminal (default: text)")
    args = parser.parse_args()

    if args.compare:
        local_json, cloud_json = args.compare
        local_summary = json.loads(Path(local_json).read_text())
        cloud_summary = json.loads(Path(cloud_json).read_text())
        if args.write_results_md:
            _write_results_md(
                local_summary, cloud_summary,
                Path("benchmarks/results.md"),
            )
            return
        # Just print comparison
        for name, s in [("Local", local_summary), ("Cloud", cloud_summary)]:
            print(f"\n=== {name} ({s.get('model', '?')}) ===")
            print(f"Field accuracy: {s.get('field_accuracy', 0) * 100:.1f}%")
            print(f"Fuzzy accuracy: {s.get('fuzzy_accuracy', 0) * 100:.1f}%")
            print(f"Avg latency:    {s.get('avg_latency_ms', 0)} ms")
        return

    # Run benchmark
    import sys
    summary = run_benchmark(
        dataset_dir=Path(args.dataset_dir),
        backend=args.backend,
        api_key=args.api_key,
        model=args.model,
        local_model_repo=args.local_model,
        lmstudio_base_url=args.lmstudio_base_url,
    )

    if not summary:
        logger.error("No results — check --dataset-dir path and dataset structure")
        sys.exit(1)

    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        logger.info("Results written to %s", args.output)

    if args.format == "json":
        print(json.dumps(summary, indent=2))
    elif args.format == "markdown":
        print("| Category | Count | Field Acc | Fuzzy Acc | Avg Latency | Cost |")
        print("|---|---|---|---|---|---|")
        for cat, row in summary.get("categories", {}).items():
            cost = _CLOUD_COST_USD_PER_PAGE * row["count"] if args.backend == "cloud" else 0
            print(
                f"| {cat} | {row['count']} "
                f"| {row['field_accuracy'] * 100:.1f}% "
                f"| {row['fuzzy_accuracy'] * 100:.1f}% "
                f"| {row['avg_latency_ms']}ms "
                f"| ${cost:.4f} |"
            )
        print(f"\nOverall field accuracy: {summary['field_accuracy'] * 100:.1f}%")
    else:
        print(f"\nBackend:           {summary['backend']}")
        print(f"Documents:         {summary['total_documents']}")
        print(f"Field accuracy:    {summary['field_accuracy'] * 100:.1f}%")
        print(f"Fuzzy accuracy:    {summary['fuzzy_accuracy'] * 100:.1f}%")
        print(f"Avg latency:       {summary['avg_latency_ms']} ms")
        print(f"Total cost:        ${summary['total_cost_usd']:.4f}")
        print("\nBy category:")
        for cat, row in summary.get("categories", {}).items():
            print(
                f"  {cat:<20} field={row['field_accuracy'] * 100:.1f}%  "
                f"fuzzy={row['fuzzy_accuracy'] * 100:.1f}%  "
                f"latency={row['avg_latency_ms']}ms"
            )

    if args.strict:
        threshold = 0.80
        failures = [
            cat for cat, row in summary.get("categories", {}).items()
            if row["field_accuracy"] < threshold
        ]
        if failures:
            logger.error("Field accuracy below 80%% for: %s", ", ".join(failures))
            sys.exit(1)
        logger.info("All categories passed the 80%% field accuracy threshold.")


if __name__ == "__main__":
    main()
