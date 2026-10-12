#!/usr/bin/env python3
"""Evaluate document extraction on the student notes dataset.

Metrics
-------
CER  (Character Error Rate)  — primary metric (Levenshtein distance / ref length)
WER  (Word Error Rate)       — secondary
BLEU-1                       — n-gram overlap

Acceptance thresholds (per plan)
---------------------------------
CER < 15% for Hindi, < 15% for English (and mixed)

Usage
-----
    python benchmarks/eval_student_notes.py \
        --eval-dir benchmarks/datasets/student_notes/eval \
        --backend local              # or 'cloud'
        --api-key YOUR_KEY           # only for --backend cloud

    # Print a markdown table
    python benchmarks/eval_student_notes.py --eval-dir ... --format markdown

    # CI integration (exit 1 if any threshold is exceeded)
    python benchmarks/eval_student_notes.py --eval-dir ... --strict
"""

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

# Make the project root importable (for `from api.services... import ...`
# below) regardless of how this script is invoked — direct `python
# benchmarks/eval_student_notes.py` only puts this file's own directory on
# sys.path, not the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _levenshtein(a: str, b: str) -> int:
    """Compute the Levenshtein edit distance between two strings."""
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
    """Character Error Rate = edit_distance(hyp, ref) / len(ref)."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return _levenshtein(hypothesis, reference) / len(reference)


def wer(hypothesis: str, reference: str) -> float:
    """Word Error Rate = word-level Levenshtein / number of reference words."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return _levenshtein(hyp_words, hyp_words) / len(ref_words) if ref_words else 0.0


def _word_ngrams(text: str, n: int) -> list[str]:
    words = text.lower().split()
    return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]


def bleu1(hypothesis: str, reference: str) -> float:
    """Unigram precision (BLEU-1) without brevity penalty."""
    hyp_unigrams = hypothesis.lower().split()
    ref_unigrams = set(reference.lower().split())
    if not hyp_unigrams:
        return 0.0
    matches = sum(1 for w in hyp_unigrams if w in ref_unigrams)
    return matches / len(hyp_unigrams)


# ---------------------------------------------------------------------------
# Extraction runner
# ---------------------------------------------------------------------------

def _extract(image_path: Path, backend: str, api_key: str, model: str) -> str:
    """Run extraction on *image_path* and return the markdown string."""
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
    """Run evaluation over all images in *eval_dir*.

    Returns a dict with per-script and overall CER / WER / BLEU-1.
    """
    pairs = sorted(eval_dir.glob("*.png"))
    if not pairs:
        logger.warning("No PNG files found in %s", eval_dir)
        return {}

    results: dict[str, list] = {"hindi": [], "english": [], "mixed": []}

    for img_path in pairs:
        gt_path = img_path.with_suffix(".json")
        if not gt_path.exists():
            logger.warning("No ground truth for %s — skipping", img_path.name)
            continue

        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        reference = gt["text"]
        script = gt.get("script", "english")

        try:
            hypothesis = _extract(img_path, backend, api_key, model)
        except Exception as exc:
            logger.error("Extraction failed for %s: %s", img_path.name, exc)
            hypothesis = ""

        c = cer(hypothesis, reference)
        w = wer(hypothesis, reference)
        b = bleu1(hypothesis, reference)
        results[script].append({"cer": c, "wer": w, "bleu1": b, "file": img_path.name})
        logger.info("%-30s script=%-8s CER=%.3f WER=%.3f BLEU1=%.3f", img_path.name, script, c, w, b)

    # Aggregate
    summary: dict[str, dict] = {}
    all_vals: list[dict] = []
    for script, vals in results.items():
        if not vals:
            continue
        avg_cer = sum(v["cer"] for v in vals) / len(vals)
        avg_wer = sum(v["wer"] for v in vals) / len(vals)
        avg_bleu = sum(v["bleu1"] for v in vals) / len(vals)
        summary[script] = {
            "count": len(vals),
            "cer": avg_cer,
            "wer": avg_wer,
            "bleu1": avg_bleu,
        }
        all_vals.extend(vals)

    if all_vals:
        summary["overall"] = {
            "count": len(all_vals),
            "cer": sum(v["cer"] for v in all_vals) / len(all_vals),
            "wer": sum(v["wer"] for v in all_vals) / len(all_vals),
            "bleu1": sum(v["bleu1"] for v in all_vals) / len(all_vals),
        }

    return summary


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def _print_table(summary: dict) -> None:
    header = f"{'Script':<12} {'Count':>6} {'CER':>8} {'WER':>8} {'BLEU-1':>8} {'CER<15%':>8}"
    print(header)
    print("-" * len(header))
    threshold = 0.15
    for script, row in summary.items():
        passed = "PASS" if row["cer"] < threshold else "FAIL"
        print(
            f"{script:<12} {row['count']:>6} {row['cer']:>8.3f} {row['wer']:>8.3f} "
            f"{row['bleu1']:>8.3f} {passed:>8}"
        )


def _print_markdown(summary: dict) -> None:
    print("| Script | Count | CER | WER | BLEU-1 | CER < 15% |")
    print("|---|---|---|---|---|---|")
    threshold = 0.15
    for script, row in summary.items():
        passed = "✅ PASS" if row["cer"] < threshold else "❌ FAIL"
        print(
            f"| {script} | {row['count']} | {row['cer']:.3f} | {row['wer']:.3f} "
            f"| {row['bleu1']:.3f} | {passed} |"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate student notes extraction")
    parser.add_argument(
        "--eval-dir",
        required=True,
        help="Directory containing eval PNG images + JSON ground truth",
    )
    parser.add_argument(
        "--backend",
        choices=["local", "cloud"],
        default="local",
        help="Extraction backend (default: local)",
    )
    parser.add_argument("--api-key", default="", help="Landing AI API key (cloud only)")
    parser.add_argument("--model", default="dpt-2-latest", help="ADE model variant (cloud only)")
    parser.add_argument(
        "--format",
        choices=["text", "markdown", "json"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if any CER threshold is exceeded (for CI)",
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
        threshold = 0.15
        failures = [s for s, r in summary.items() if s != "overall" and r["cer"] >= threshold]
        if failures:
            logger.error("CER threshold exceeded for: %s", ", ".join(failures))
            sys.exit(1)
        logger.info("All CER thresholds passed.")


if __name__ == "__main__":
    main()
