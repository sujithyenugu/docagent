#!/usr/bin/env python3
"""Evaluate the form-fill pipeline on the synthetic student document dataset.

Metrics
-------
Field Accuracy    — % of extracted fields that exactly match ground truth
Field Recall      — % of ground-truth non-null fields that were extracted (non-null)
Missing Rate      — % of required fields not extracted
Mark Deviation    — Mean Absolute Error between extracted marks and ground truth marks
                    (reported only when marks fields are present)

Field thresholds (per plan)
---------------------------
Student Name   > 95%
Roll Number    > 99%
Marks Awarded  > 90%

Usage
-----
    python benchmarks/eval_teacher_grading_forms.py \
        --eval-dir benchmarks/datasets/synthetic_student_docs/eval \
        --backend local

    python benchmarks/eval_teacher_grading_forms.py \
        --eval-dir ... --breakdown --format markdown

    # CI strict mode
    python benchmarks/eval_teacher_grading_forms.py --eval-dir ... --strict
"""

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

# Make the project root importable (for `from api.services... import ...`
# below) regardless of how this script is invoked — direct `python
# benchmarks/eval_teacher_grading_forms.py` only puts this file's own
# directory on sys.path, not the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_NAME_FIELDS = {"student_name"}
_ID_FIELDS = {"roll_number", "admission_number"}
_MARKS_FIELDS = {
    "english_marks", "hindi_marks", "mathematics_marks",
    "science_marks", "social_science_marks", "total_marks",
}

_THRESHOLDS = {
    "student_name": 0.95,
    "roll_number": 0.99,
    "marks": 0.90,
}


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _normalise(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[,.\-/]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _field_match(predicted: str | None, ground_truth: str | None) -> bool:
    if predicted is None and ground_truth is None:
        return True
    if predicted is None or ground_truth is None:
        return False
    return _normalise(str(predicted)) == _normalise(str(ground_truth))


def _mark_deviation(predicted: str | None, ground_truth: str | None) -> float | None:
    """Return absolute difference between two mark values, or None if unparseable."""
    if predicted is None or ground_truth is None:
        return None
    try:
        return abs(float(str(predicted).strip()) - float(str(ground_truth).strip()))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Doc-type → template mapping
# ---------------------------------------------------------------------------

_TEMPLATE_MAP = {
    "admit_card": "admit_card_verification",
    "roll_number_sheet": "admit_card_verification",
    "answer_booklet": "admit_card_verification",
    "cbse_marksheet": "cbse_marksheet",
    "school_id_card": "admit_card_verification",
}


def _extract_fields(image_path: Path, doc_type: str, backend: str, api_key: str, model: str) -> dict:
    from api.services.form_filler import fill

    img_bytes = image_path.read_bytes()
    template_id = _TEMPLATE_MAP.get(doc_type, "admit_card_verification")

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

            # Field category
            if field in _NAME_FIELDS:
                category = "student_name"
            elif field in _ID_FIELDS:
                category = "roll_number"
            elif field in _MARKS_FIELDS:
                category = "marks"
            else:
                category = "other"

            mark_dev = _mark_deviation(pred_value, gt_value) if category == "marks" else None

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
                "mark_deviation": mark_dev,
            })

        logger.debug("Processed %s | fields=%d", img_path.name, len(gt_fields))

    if not all_records:
        return {}

    def _metrics(records: list[dict]) -> dict:
        if not records:
            return {"count": 0, "field_accuracy": 0.0, "field_recall": 0.0, "missing_rate": 0.0, "mark_mae": None}
        accuracy = sum(r["match"] for r in records) / len(records)
        recall = sum(r["extracted"] for r in records) / len(records)
        devs = [r["mark_deviation"] for r in records if r["mark_deviation"] is not None]
        mae = sum(devs) / len(devs) if devs else None
        return {
            "count": len(records),
            "field_accuracy": accuracy,
            "field_recall": recall,
            "missing_rate": 1.0 - recall,
            "mark_mae": mae,
        }

    summary: dict = {"overall": _metrics(all_records)}

    by_doc: dict[str, list] = defaultdict(list)
    for r in all_records:
        by_doc[r["doc_type"]].append(r)
    summary["by_doc_type"] = {k: _metrics(v) for k, v in by_doc.items()}

    by_var: dict[str, list] = defaultdict(list)
    for r in all_records:
        by_var[r["variation"]].append(r)
    summary["by_variation"] = {k: _metrics(v) for k, v in by_var.items()}

    by_cat: dict[str, list] = defaultdict(list)
    for r in all_records:
        by_cat[r["category"]].append(r)
    summary["by_field_category"] = {k: _metrics(v) for k, v in by_cat.items()}

    return summary


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _print_table(summary: dict, breakdown: bool = False) -> None:
    def _row(label: str, row: dict) -> str:
        mae = f"{row['mark_mae']:.2f}" if row.get("mark_mae") is not None else " n/a"
        return (
            f"{label:<28} {row['count']:>6} "
            f"{row['field_accuracy']:>10.3f} "
            f"{row['field_recall']:>10.3f} "
            f"{row['missing_rate']:>10.3f} "
            f"{mae:>9}"
        )

    header = f"{'Section':<28} {'Count':>6} {'Accuracy':>10} {'Recall':>10} {'Missing':>10} {'MarkMAE':>9}"
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
    for cat, row in summary.get("by_field_category", {}).items():
        thresh = _THRESHOLDS.get(cat)
        if thresh:
            status = "PASS" if row["field_accuracy"] >= thresh else "FAIL"
            mae_str = f"  MarkMAE={row['mark_mae']:.2f}" if row.get("mark_mae") is not None else ""
            print(f"  {cat:<20} acc={row['field_accuracy']:.3f}  thresh={thresh:.2f}  {status}{mae_str}")


def _print_markdown(summary: dict, breakdown: bool = False) -> None:
    print("## Teacher Grading Form-Fill Evaluation\n")
    print("| Section | Count | Accuracy | Recall | Missing | Mark MAE |")
    print("|---|---|---|---|---|---|")

    def _row(label: str, row: dict) -> str:
        mae = f"{row['mark_mae']:.2f}" if row.get("mark_mae") is not None else "n/a"
        return (
            f"| **{label}** | {row['count']} "
            f"| {row['field_accuracy']:.3f} "
            f"| {row['field_recall']:.3f} "
            f"| {row['missing_rate']:.3f} "
            f"| {mae} |"
        )

    print(_row("overall", summary["overall"]))

    if breakdown:
        print("\n### By Document Type\n")
        print("| Doc Type | Count | Accuracy | Recall | Missing | Mark MAE |")
        print("|---|---|---|---|---|---|")
        for k, v in summary.get("by_doc_type", {}).items():
            print(_row(k, v))

        print("\n### By Variation\n")
        print("| Variation | Count | Accuracy | Recall | Missing | Mark MAE |")
        print("|---|---|---|---|---|---|")
        for k, v in summary.get("by_variation", {}).items():
            print(_row(k, v))

    print("\n### Field Category Thresholds\n")
    print("| Category | Accuracy | Threshold | Mark MAE | Status |")
    print("|---|---|---|---|---|")
    for cat, row in summary.get("by_field_category", {}).items():
        thresh = _THRESHOLDS.get(cat)
        if thresh:
            status = "✅ PASS" if row["field_accuracy"] >= thresh else "❌ FAIL"
            mae_str = f"{row['mark_mae']:.2f}" if row.get("mark_mae") is not None else "n/a"
            print(f"| {cat} | {row['field_accuracy']:.3f} | {thresh:.2f} | {mae_str} | {status} |")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate teacher grading form-fill pipeline on synthetic student documents"
    )
    parser.add_argument("--eval-dir", required=True, help="Eval dir with PNG + JSON pairs")
    parser.add_argument("--backend", choices=["local", "cloud"], default="local")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="dpt-2-latest")
    parser.add_argument("--format", choices=["text", "markdown", "json"], default="text")
    parser.add_argument("--breakdown", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 on any threshold failure (CI mode)",
    )
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
