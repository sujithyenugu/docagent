#!/usr/bin/env python3
"""Evaluate document extraction on the student assignment dataset for teacher grading.

Metrics
-------
CER          — Character Error Rate (primary extraction quality)
WER          — Word Error Rate
BLEU-1       — Unigram precision
Keyword Recall — % of answer key terms found in extracted text (grading-critical)
Marks Agreement — Mean Absolute Error between awarded marks and ground truth marks

Acceptance thresholds
---------------------
CER < 15% for Hindi, English, mixed
Keyword Recall >= 80% for all scripts

Usage
-----
    python benchmarks/eval_teacher_grading.py \
        --eval-dir benchmarks/datasets/student_assignments/eval \
        --backend local

    python benchmarks/eval_teacher_grading.py \
        --eval-dir benchmarks/datasets/student_assignments/eval \
        --backend cloud --api-key YOUR_KEY

    # Markdown table output
    python benchmarks/eval_teacher_grading.py --eval-dir ... --format markdown

    # CI mode (exits 1 if any threshold fails)
    python benchmarks/eval_teacher_grading.py --eval-dir ... --strict
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

# Make the project root importable (for `from api.services... import ...`
# below) regardless of how this script is invoked — direct `python
# benchmarks/eval_teacher_grading.py` only puts this file's own directory on
# sys.path, not the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_CER_THRESHOLD = 0.15
_KEYWORD_RECALL_THRESHOLD = 0.80


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _levenshtein(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            old = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = old
    return dp[n]


def cer(hypothesis: str, reference: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return min(_levenshtein(hypothesis, reference) / len(reference), 1.0)


def wer(hypothesis: str, reference: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return min(_levenshtein(ref_words, hyp_words) / len(ref_words), 1.0)


def bleu1(hypothesis: str, reference: str) -> float:
    hyp_words = hypothesis.lower().split()
    ref_set = set(reference.lower().split())
    if not hyp_words:
        return 0.0
    return sum(1 for w in hyp_words if w in ref_set) / len(hyp_words)


def keyword_recall(hypothesis: str, answer_key: dict) -> tuple[float, int, int]:
    """Compute keyword recall over all questions in the answer key.

    Returns: (recall_rate, keywords_found, keywords_total)
    """
    hyp_lower = hypothesis.lower()
    total = 0
    found = 0
    for q_data in answer_key.values():
        for kw in q_data.get("keywords", []):
            total += 1
            if kw.lower() in hyp_lower:
                found += 1
    if total == 0:
        return 1.0, 0, 0
    return found / total, found, total


def marks_agreement(hypothesis: str, answer_key: dict) -> tuple[float | None, float, float]:
    """Compute awarded marks from keyword matching and compare to max marks.

    Returns: (mean_absolute_error, awarded_total, possible_total)
    If no answer key or no marks, returns (None, 0, 0).
    """
    if not answer_key:
        return None, 0.0, 0.0

    hyp_lower = hypothesis.lower()
    awarded = 0.0
    possible = 0.0
    errors = []

    for q_data in answer_key.values():
        q_marks = float(q_data.get("marks", 0))
        keywords = q_data.get("keywords", [])
        if not keywords:
            continue

        found = sum(1 for kw in keywords if kw.lower() in hyp_lower)
        q_awarded = round(found / len(keywords) * q_marks, 2)
        awarded += q_awarded
        possible += q_marks
        errors.append(abs(q_marks - q_awarded))  # vs perfect score as proxy

    mae = sum(errors) / len(errors) if errors else None
    return mae, awarded, possible


# ---------------------------------------------------------------------------
# Extraction runner
# ---------------------------------------------------------------------------

def _extract(image_path: Path, backend: str, api_key: str, model: str) -> str:
    img_bytes = image_path.read_bytes()
    filename = image_path.name

    if backend == "local":
        from api.services.extraction import run_local
        result = run_local(img_bytes, filename)
    else:
        from api.services.extraction import run_cloud
        result = run_cloud(img_bytes, filename, api_key=api_key, model=model)

    return result["markdown"]


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    eval_dir: Path,
    backend: str = "local",
    api_key: str = "",
    model: str = "dpt-2-latest",
) -> dict:
    """Run evaluation over all PNG images in *eval_dir*.

    Each PNG must have a paired JSON with keys:
        text         — ground truth answer text
        script       — "hindi" | "english" | "mixed"
        answer_key   — {q1: {keywords, marks}, ...}

    Returns nested summary dict with per-script and overall metrics.
    """
    pairs = sorted(eval_dir.glob("*.png"))
    if not pairs:
        logger.warning("No PNG files in %s", eval_dir)
        return {}

    by_script: dict[str, list] = defaultdict(list)

    for img_path in pairs:
        gt_path = img_path.with_suffix(".json")
        if not gt_path.exists():
            logger.warning("No ground truth for %s — skipping", img_path.name)
            continue

        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        reference = gt.get("text", "")
        script = gt.get("script", "english")
        answer_key = gt.get("answer_key", {})

        try:
            hypothesis = _extract(img_path, backend, api_key, model)
        except Exception as exc:
            logger.error("Extraction failed for %s: %s", img_path.name, exc)
            hypothesis = ""

        c = cer(hypothesis, reference)
        w = wer(hypothesis, reference)
        b = bleu1(hypothesis, reference)
        kr, kw_found, kw_total = keyword_recall(hypothesis, answer_key)
        mae, awarded, possible = marks_agreement(hypothesis, answer_key)

        record = {
            "file": img_path.name,
            "script": script,
            "cer": c,
            "wer": w,
            "bleu1": b,
            "keyword_recall": kr,
            "kw_found": kw_found,
            "kw_total": kw_total,
            "marks_mae": mae,
            "marks_awarded": awarded,
            "marks_possible": possible,
        }
        by_script[script].append(record)
        logger.info(
            "%-30s script=%-8s CER=%.3f WER=%.3f BLEU1=%.3f KR=%.3f",
            img_path.name, script, c, w, b, kr,
        )

    if not any(by_script.values()):
        return {}

    def _agg(records: list[dict]) -> dict:
        if not records:
            return {}
        n = len(records)
        mae_vals = [r["marks_mae"] for r in records if r["marks_mae"] is not None]
        return {
            "count": n,
            "cer": sum(r["cer"] for r in records) / n,
            "wer": sum(r["wer"] for r in records) / n,
            "bleu1": sum(r["bleu1"] for r in records) / n,
            "keyword_recall": sum(r["keyword_recall"] for r in records) / n,
            "marks_mae": sum(mae_vals) / len(mae_vals) if mae_vals else None,
        }

    all_records = [r for recs in by_script.values() for r in recs]
    summary: dict = {
        "overall": _agg(all_records),
        "by_script": {s: _agg(recs) for s, recs in by_script.items()},
    }
    return summary


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def _print_table(summary: dict) -> None:
    header = (
        f"{'Script':<12} {'N':>5} {'CER':>7} {'WER':>7} "
        f"{'BLEU-1':>7} {'KR':>7} {'MAE':>7} "
        f"{'CER<15%':>8} {'KR>=80%':>8}"
    )
    print(header)
    print("-" * len(header))

    def _row(label: str, row: dict) -> str:
        cer_pass = "PASS" if row["cer"] < _CER_THRESHOLD else "FAIL"
        kr_pass = "PASS" if row["keyword_recall"] >= _KEYWORD_RECALL_THRESHOLD else "FAIL"
        mae_str = f"{row['marks_mae']:.2f}" if row.get("marks_mae") is not None else "  n/a"
        return (
            f"{label:<12} {row['count']:>5} {row['cer']:>7.3f} {row['wer']:>7.3f} "
            f"{row['bleu1']:>7.3f} {row['keyword_recall']:>7.3f} {mae_str:>7} "
            f"{cer_pass:>8} {kr_pass:>8}"
        )

    print(_row("overall", summary["overall"]))
    print()
    for script, row in summary.get("by_script", {}).items():
        print(_row(script, row))


def _print_markdown(summary: dict) -> None:
    print("## Teacher Grading Evaluation Results\n")
    header = "| Script | N | CER | WER | BLEU-1 | Keyword Recall | Marks MAE | CER<15% | KR>=80% |"
    print(header)
    print("|---|---|---|---|---|---|---|---|---|")

    def _row(label: str, row: dict) -> str:
        cer_pass = "✅" if row["cer"] < _CER_THRESHOLD else "❌"
        kr_pass = "✅" if row["keyword_recall"] >= _KEYWORD_RECALL_THRESHOLD else "❌"
        mae_str = f"{row['marks_mae']:.2f}" if row.get("marks_mae") is not None else "n/a"
        return (
            f"| **{label}** | {row['count']} | {row['cer']:.3f} | {row['wer']:.3f} "
            f"| {row['bleu1']:.3f} | {row['keyword_recall']:.3f} | {mae_str} "
            f"| {cer_pass} | {kr_pass} |"
        )

    print(_row("overall", summary["overall"]))
    for script, row in summary.get("by_script", {}).items():
        print(_row(script, row))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate extraction accuracy on student assignments for teacher grading"
    )
    parser.add_argument("--eval-dir", required=True, help="Eval directory with PNG + JSON pairs")
    parser.add_argument("--backend", choices=["local", "cloud"], default="local")
    parser.add_argument("--api-key", default="", help="Landing AI API key (cloud only)")
    parser.add_argument("--model", default="dpt-2-latest")
    parser.add_argument("--format", choices=["text", "markdown", "json"], default="text")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if CER >= 15%% or Keyword Recall < 80%% for any script (CI mode)",
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
        _print_markdown(summary)
    else:
        _print_table(summary)

    if args.strict:
        failures = []
        for script, row in summary.get("by_script", {}).items():
            if row["cer"] >= _CER_THRESHOLD:
                failures.append(f"CER {script} ({row['cer']:.3f} >= {_CER_THRESHOLD})")
            if row["keyword_recall"] < _KEYWORD_RECALL_THRESHOLD:
                failures.append(f"KR {script} ({row['keyword_recall']:.3f} < {_KEYWORD_RECALL_THRESHOLD})")
        if failures:
            logger.error("Threshold failures: %s", "; ".join(failures))
            sys.exit(1)
        logger.info("All thresholds passed.")


if __name__ == "__main__":
    main()
