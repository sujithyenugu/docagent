#!/usr/bin/env python3
"""Evaluate the form-fill pipeline on the synthetic ID document dataset.

Metrics
-------
Field Accuracy   — % of extracted field values that exactly match ground truth
                   (case-insensitive, whitespace-normalised)
Field Recall     — % of ground-truth non-null fields that were extracted (non-null)
Missing Rate     — % of required fields not extracted

Acceptance thresholds (per plan)
---------------------------------
Name accuracy   > 95%
ID Number acc.  > 99%  (aadhaar_number, pan_number, dl_number, epic_number, passport_number)
Address acc.    > 80%

Usage
-----
    python benchmarks/eval_form_fill.py \
        --eval-dir benchmarks/datasets/synthetic_ids/eval \
        --backend local

    # Breakdown by variation and document type
    python benchmarks/eval_form_fill.py --eval-dir ... --breakdown

    # CI (strict mode exits 1 on threshold failure)
    python benchmarks/eval_form_fill.py --eval-dir ... --strict
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

# Make the project root importable (for `from api.services... import ...`
# below) regardless of how this script is invoked — direct `python
# benchmarks/eval_form_fill.py` only puts this file's own directory on
# sys.path, not the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_ID_FIELDS = {"aadhaar_number", "pan_number", "dl_number", "epic_number", "passport_number"}
_NAME_FIELDS = {"name_en", "name_hi"}
_ADDRESS_FIELDS = {"address"}

_THRESHOLDS = {
    "name": 0.95,
    "id_number": 0.99,
    "address": 0.80,
}


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _normalise(value: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for comparison."""
    import re
    if not value:
        return ""
    value = value.lower().strip()
    value = re.sub(r"\s+", " ", value)
    # Remove common punctuation differences
    value = re.sub(r"[,\.\-/]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _field_match(predicted: str | None, ground_truth: str | None) -> bool:
    """Return True if predicted matches ground truth after normalisation."""
    if predicted is None and ground_truth is None:
        return True
    if predicted is None or ground_truth is None:
        return False
    return _normalise(str(predicted)) == _normalise(str(ground_truth))


# ---------------------------------------------------------------------------
# Extraction runner
# ---------------------------------------------------------------------------

def _extract_fields(
    image_path: Path,
    doc_type: str,
    backend: str,
    api_key: str,
    model: str,
) -> dict:
    """Run form-fill pipeline and return extracted fields dict."""
    from api.services.form_filler import fill

    img_bytes = image_path.read_bytes()

    # Map doc_type to a relevant template (best match)
    template_map = {
        "aadhaar": "bank_account",  # Aadhaar fields overlap with bank account
        "pan": "bank_account",
        "dl": "dl_renewal",
        "voter_id": "affidavit",
        "passport": "bank_account",
    }
    template_id = template_map.get(doc_type, "bank_account")

    try:
        result = fill(
            [(img_bytes, image_path.name)],
            template_id,
            use_local=(backend == "local"),
            api_key=api_key or None,
            model=model,
        )
        return result.fields
    except Exception as exc:
        logger.error("Form fill failed for %s: %s", image_path.name, exc)
        return {}


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    eval_dir: Path,
    backend: str = "local",
    api_key: str = "",
    model: str = "dpt-2-latest",
) -> dict:
    """Evaluate all images in *eval_dir* against their ground truth JSONs.

    Returns nested summary dict:
        {
            "overall": {field_accuracy, field_recall, missing_rate},
            "by_doc_type": {doc_type: {...}},
            "by_variation": {variation: {...}},
            "by_field_category": {name/id_number/address: field_accuracy},
        }
    """
    images = sorted(eval_dir.glob("*.png"))
    if not images:
        logger.warning("No PNG files in %s", eval_dir)
        return {}

    all_records: list[dict] = []

    for img_path in images:
        gt_path = img_path.with_suffix(".json")
        if not gt_path.exists():
            logger.warning("No GT for %s — skipping", img_path.name)
            continue

        gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
        doc_type = gt_data.get("doc_type", "unknown")
        variation = gt_data.get("variation", "unknown")
        gt_fields = gt_data.get("fields", {})

        predicted_fields = _extract_fields(img_path, doc_type, backend, api_key, model)

        for field, gt_value in gt_fields.items():
            pred_value = predicted_fields.get(field)
            match = _field_match(pred_value, gt_value)
            extracted = pred_value is not None

            # Categorise field
            if field in _NAME_FIELDS:
                category = "name"
            elif field in _ID_FIELDS:
                category = "id_number"
            elif field in _ADDRESS_FIELDS:
                category = "address"
            else:
                category = "other"

            all_records.append({
                "file": img_path.name,
                "doc_type": doc_type,
                "variation": variation,
                "field": field,
                "category": category,
                "gt_value": gt_value,
                "pred_value": pred_value,
                "match": match,
                "extracted": extracted,
            })

        logger.debug("Processed %s: %d fields", img_path.name, len(gt_fields))

    if not all_records:
        return {}

    def _metrics(records: list[dict]) -> dict:
        if not records:
            return {"count": 0, "field_accuracy": 0.0, "field_recall": 0.0, "missing_rate": 0.0}
        accuracy = sum(r["match"] for r in records) / len(records)
        recall = sum(r["extracted"] for r in records) / len(records)
        missing = 1.0 - recall
        return {
            "count": len(records),
            "field_accuracy": accuracy,
            "field_recall": recall,
            "missing_rate": missing,
        }

    # Overall
    summary: dict = {"overall": _metrics(all_records)}

    # By doc type
    by_doc: dict[str, list] = defaultdict(list)
    for r in all_records:
        by_doc[r["doc_type"]].append(r)
    summary["by_doc_type"] = {k: _metrics(v) for k, v in by_doc.items()}

    # By variation
    by_var: dict[str, list] = defaultdict(list)
    for r in all_records:
        by_var[r["variation"]].append(r)
    summary["by_variation"] = {k: _metrics(v) for k, v in by_var.items()}

    # By field category (for threshold checks)
    by_cat: dict[str, list] = defaultdict(list)
    for r in all_records:
        by_cat[r["category"]].append(r)
    summary["by_field_category"] = {k: _metrics(v) for k, v in by_cat.items()}

    return summary


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def _print_table(summary: dict, breakdown: bool = False) -> None:
    def _row(label: str, row: dict) -> str:
        return (
            f"{label:<25} {row['count']:>6} "
            f"{row['field_accuracy']:>10.3f} "
            f"{row['field_recall']:>10.3f} "
            f"{row['missing_rate']:>10.3f}"
        )

    header = f"{'Section':<25} {'Count':>6} {'Accuracy':>10} {'Recall':>10} {'Missing':>10}"
    print(header)
    print("-" * len(header))
    print(_row("overall", summary["overall"]))

    if breakdown:
        print("\n-- By Document Type --")
        for k, v in summary.get("by_doc_type", {}).items():
            print(_row(k, v))
        print("\n-- By Variation --")
        for k, v in summary.get("by_variation", {}).items():
            print(_row(k, v))

    print("\n-- Field Category Thresholds --")
    thresholds = {"name": 0.95, "id_number": 0.99, "address": 0.80}
    for cat, row in summary.get("by_field_category", {}).items():
        thresh = thresholds.get(cat)
        if thresh:
            status = "PASS" if row["field_accuracy"] >= thresh else "FAIL"
            print(f"  {cat:<15} acc={row['field_accuracy']:.3f}  threshold={thresh:.2f}  {status}")


def _print_markdown(summary: dict, breakdown: bool = False) -> None:
    print("## Form Fill Evaluation Results\n")
    print("| Section | Count | Accuracy | Recall | Missing Rate |")
    print("|---|---|---|---|---|")
    print(
        f"| **overall** | {summary['overall']['count']} "
        f"| {summary['overall']['field_accuracy']:.3f} "
        f"| {summary['overall']['field_recall']:.3f} "
        f"| {summary['overall']['missing_rate']:.3f} |"
    )

    if breakdown:
        print("\n### By Document Type\n")
        print("| Doc Type | Count | Accuracy | Recall | Missing Rate |")
        print("|---|---|---|---|---|")
        for k, v in summary.get("by_doc_type", {}).items():
            print(f"| {k} | {v['count']} | {v['field_accuracy']:.3f} | {v['field_recall']:.3f} | {v['missing_rate']:.3f} |")

        print("\n### By Variation\n")
        print("| Variation | Count | Accuracy | Recall | Missing Rate |")
        print("|---|---|---|---|---|")
        for k, v in summary.get("by_variation", {}).items():
            print(f"| {k} | {v['count']} | {v['field_accuracy']:.3f} | {v['field_recall']:.3f} | {v['missing_rate']:.3f} |")

    print("\n### Field Category Thresholds\n")
    print("| Category | Accuracy | Threshold | Status |")
    print("|---|---|---|---|")
    thresholds = {"name": 0.95, "id_number": 0.99, "address": 0.80}
    for cat, row in summary.get("by_field_category", {}).items():
        thresh = thresholds.get(cat)
        if thresh:
            status = "✅ PASS" if row["field_accuracy"] >= thresh else "❌ FAIL"
            print(f"| {cat} | {row['field_accuracy']:.3f} | {thresh:.2f} | {status} |")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate form-fill pipeline on synthetic ID dataset")
    parser.add_argument("--eval-dir", required=True, help="Eval directory with PNG + JSON pairs")
    parser.add_argument("--backend", choices=["local", "cloud"], default="local")
    parser.add_argument("--api-key", default="", help="Landing AI API key (cloud only)")
    parser.add_argument("--model", default="dpt-2-latest")
    parser.add_argument(
        "--format", choices=["text", "markdown", "json"], default="text"
    )
    parser.add_argument("--breakdown", action="store_true", help="Show per-doc-type and per-variation tables")
    parser.add_argument("--strict", action="store_true", help="Exit 1 if any threshold fails (CI)")
    args = parser.parse_args()

    summary = evaluate(
        Path(args.eval_dir),
        backend=args.backend,
        api_key=args.api_key,
        model=args.model,
    )

    if not summary:
        logger.error("No results — check --eval-dir path")
        sys.exit(1)

    if args.format == "json":
        print(json.dumps(summary, indent=2))
    elif args.format == "markdown":
        _print_markdown(summary, breakdown=args.breakdown)
    else:
        _print_table(summary, breakdown=args.breakdown)

    if args.strict:
        failures = []
        for cat, row in summary.get("by_field_category", {}).items():
            thresh = _THRESHOLDS.get(cat)
            if thresh and row["field_accuracy"] < thresh:
                failures.append(f"{cat} ({row['field_accuracy']:.3f} < {thresh})")
        if failures:
            logger.error("Threshold failures: %s", "; ".join(failures))
            sys.exit(1)
        logger.info("All field accuracy thresholds passed.")


if __name__ == "__main__":
    main()
