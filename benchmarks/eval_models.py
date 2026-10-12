#!/usr/bin/env python3
"""Compare multiple extraction models/backends on the same benchmark dataset.

Runs run_benchmark.py's evaluation logic once per model spec and writes a
single side-by-side comparison report (JSON + Markdown) — the direct way to
answer "which model should we use" instead of manually diffing separate
two-way `run_benchmark.py --compare` runs.

Model specs
-----------
Each --model value is "label=kind:value":
    --model gemma4b=local:mlx-community/gemma-3-4b-it-4bit
    --model qwen3b=local:mlx-community/Qwen2.5-VL-3B-Instruct-4bit
    --model qwen7b=local:mlx-community/Qwen2.5-VL-7B-Instruct-4bit
    --model aide-latest=cloud:dpt-2-latest
    --model qwen-lmstudio=lmstudio:qwen2.5-vl-7b-instruct

`kind` is "local" (value = MLX repo id, Apple Silicon Mac required), "cloud"
(value = ADE model variant, requires --api-key — "dpt-2-mini" has returned
"422 Unsupported model" against current API keys; "dpt-2-latest" is known
to work), or "lmstudio" (value = the model name as loaded in a running LM
Studio instance, requires --lmstudio-base-url reachable — default
http://localhost:1234/v1).
LM Studio is **not** used by the shipped app (local mode runs embedded via
Apple MLX instead) — this kind exists purely so a model only available
through LM Studio (e.g. a GGUF-quantized Qwen build) can still be compared
side by side with the MLX and cloud rows on this project's dataset.

Usage
-----
    # Explicit model list
    python benchmarks/eval_models.py \\
        --dataset-dir benchmarks/dataset \\
        --api-key YOUR_LANDING_AI_KEY \\
        --model gemma4b=local:mlx-community/gemma-3-4b-it-4bit \\
        --model qwen7b=local:mlx-community/Qwen2.5-VL-7B-Instruct-4bit \\
        --model aide-latest=cloud:dpt-2-latest \\
        --output benchmarks/model_comparison.json \\
        --write-report benchmarks/model_comparison.md

    # Built-in candidate set (current default + two Qwen2.5-VL sizes + AIDE)
    python benchmarks/eval_models.py --dataset-dir benchmarks/dataset \\
        --api-key YOUR_KEY --defaults

    # CI mode — exit 1 if every local model underperforms cloud by >15pp
    python benchmarks/eval_models.py --dataset-dir benchmarks/dataset \\
        --api-key YOUR_KEY --defaults --strict
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from run_benchmark import _DEFAULT_LMSTUDIO_BASE_URL, run_benchmark

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Candidate models for --defaults. Picked because Qwen2.5-VL is documented to
# score meaningfully higher than Gemma-3 on general OCR benchmarks (OCRBench);
# this script exists to verify that holds on *this* project's actual Indian
# document dataset (handwriting, mixed Hindi/English, GST invoices) rather
# than assume external benchmarks transfer directly.
DEFAULT_MODEL_SPECS = [
    "gemma-4b=local:mlx-community/gemma-3-4b-it-4bit",
    "qwen2.5-vl-3b=local:mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
    "qwen2.5-vl-7b=local:mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
    "aide=cloud:dpt-2-latest",
]


def _parse_model_spec(spec: str) -> tuple[str, str, str]:
    """Parse "label=kind:value" into (label, kind, value)."""
    if "=" not in spec or ":" not in spec.split("=", 1)[1]:
        raise ValueError(
            f"Invalid --model spec '{spec}'. Expected 'label=kind:value', "
            "e.g. 'qwen7b=local:mlx-community/Qwen2.5-VL-7B-Instruct-4bit'."
        )
    label, rest = spec.split("=", 1)
    kind, value = rest.split(":", 1)
    if kind not in ("local", "cloud", "lmstudio"):
        raise ValueError(
            f"Invalid kind '{kind}' in --model spec '{spec}' — must be "
            "'local', 'cloud', or 'lmstudio'."
        )
    return label, kind, value


def run_comparison(
    dataset_dir: Path,
    model_specs: list[str],
    api_key: str = "",
    lmstudio_base_url: str = _DEFAULT_LMSTUDIO_BASE_URL,
) -> dict[str, dict[str, Any]]:
    """Run every model spec against dataset_dir, returning {label: summary}."""
    results: dict[str, dict[str, Any]] = {}
    for spec in model_specs:
        label, kind, value = _parse_model_spec(spec)
        logger.info("=== Evaluating %s (%s: %s) ===", label, kind, value)

        if kind == "cloud" and not api_key:
            logger.warning("Skipping %s — cloud model requires --api-key", label)
            continue

        try:
            summary = run_benchmark(
                dataset_dir=dataset_dir,
                backend=kind,
                api_key=api_key,
                model=value if kind in ("cloud", "lmstudio") else "dpt-2-latest",
                local_model_repo=value if kind == "local" else None,
                lmstudio_base_url=lmstudio_base_url,
            )
        except Exception as exc:
            logger.error("Model %s failed entirely: %s", label, exc)
            continue

        if not summary:
            logger.warning("No results for %s — skipping", label)
            continue

        summary["label"] = label
        summary["spec"] = spec
        results[label] = summary
        logger.info(
            "  %s: field_acc=%.1f%% fuzzy_acc=%.1f%% avg_latency=%dms",
            label,
            summary["field_accuracy"] * 100,
            summary["fuzzy_accuracy"] * 100,
            summary["avg_latency_ms"],
        )
    return results


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _ms(v: int) -> str:
    return f"{v / 1000:.1f}s" if v >= 1000 else f"{v}ms"


def write_report(results: dict[str, dict[str, Any]], out_path: Path) -> None:
    labels = list(results.keys())
    all_categories: list[str] = sorted(
        {cat for s in results.values() for cat in s.get("categories", {})}
    )

    lines: list[str] = [
        "# Model Comparison — Extraction Quality Benchmark",
        "",
        f"Compared {len(labels)} model(s) on the same dataset "
        f"({next(iter(results.values()))['total_documents']} documents) "
        "so accuracy differences reflect the model, not the data.",
        "",
        "## Overall",
        "",
        "| Model | Backend | Field Acc | Fuzzy Acc | Avg Latency | Total Cost |",
        "|---|---|---|---|---|---|",
    ]
    # Rank by field accuracy, best first.
    for label in sorted(labels, key=lambda l: results[l]["field_accuracy"], reverse=True):
        s = results[label]
        lines.append(
            f"| **{label}** | {s['backend']} ({s.get('model', '?')}) "
            f"| {_pct(s['field_accuracy'])} | {_pct(s['fuzzy_accuracy'])} "
            f"| {_ms(s['avg_latency_ms'])} | ${s['total_cost_usd']:.4f} |"
        )

    lines += ["", "## By Category", ""]
    for cat in all_categories:
        lines.append(f"### {cat}")
        lines.append("")
        lines.append("| Model | Field Acc | Fuzzy Acc | Avg Latency |")
        lines.append("|---|---|---|---|")
        for label in sorted(
            labels,
            key=lambda l: results[l].get("categories", {}).get(cat, {}).get("field_accuracy", 0),
            reverse=True,
        ):
            row = results[label].get("categories", {}).get(cat)
            if not row:
                continue
            lines.append(
                f"| {label} | {_pct(row['field_accuracy'])} | {_pct(row['fuzzy_accuracy'])} "
                f"| {_ms(row['avg_latency_ms'])} |"
            )
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", out_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiple models on the same benchmark dataset")
    parser.add_argument("--dataset-dir", required=True, help="Path to benchmarks/dataset directory")
    parser.add_argument("--model", action="append", default=[],
                        help="Model spec 'label=kind:value' (repeatable). See module docstring.")
    parser.add_argument("--defaults", action="store_true",
                        help="Use the built-in candidate set instead of --model")
    parser.add_argument("--api-key", default="", help="Landing AI API key (required for cloud specs)")
    parser.add_argument("--lmstudio-base-url", default=_DEFAULT_LMSTUDIO_BASE_URL,
                        help="LM Studio server base URL (required for lmstudio specs, "
                             f"default: {_DEFAULT_LMSTUDIO_BASE_URL})")
    parser.add_argument("--output", default="benchmarks/model_comparison.json",
                        help="Write combined JSON results here")
    parser.add_argument("--write-report", default="benchmarks/model_comparison.md",
                        help="Write the Markdown comparison report here")
    parser.add_argument("--strict", action="store_true",
                        help="Exit 1 if every local model's field accuracy trails "
                             "the best cloud model by more than 15 percentage points")
    args = parser.parse_args()

    model_specs = DEFAULT_MODEL_SPECS if args.defaults else args.model
    if not model_specs:
        logger.error("No models specified — pass --model (repeatable) or --defaults")
        sys.exit(1)

    results = run_comparison(
        Path(args.dataset_dir), model_specs,
        api_key=args.api_key, lmstudio_base_url=args.lmstudio_base_url,
    )
    if not results:
        logger.error("No successful model runs — nothing to report")
        sys.exit(1)

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False))
        logger.info("Wrote %s", args.output)

    if args.write_report:
        write_report(results, Path(args.write_report))

    print("\n=== Summary (best field accuracy first) ===")
    for label in sorted(results, key=lambda l: results[l]["field_accuracy"], reverse=True):
        s = results[label]
        print(
            f"  {label:<20} field={s['field_accuracy'] * 100:5.1f}%  "
            f"fuzzy={s['fuzzy_accuracy'] * 100:5.1f}%  "
            f"latency={s['avg_latency_ms']:6d}ms  "
            f"({s['backend']}: {s.get('model', '?')})"
        )

    if args.strict:
        cloud_best = max(
            (s["field_accuracy"] for s in results.values() if s["backend"] == "cloud"),
            default=None,
        )
        local_best = max(
            (s["field_accuracy"] for s in results.values() if s["backend"] == "local"),
            default=None,
        )
        if cloud_best is not None and local_best is not None and (cloud_best - local_best) > 0.15:
            logger.error(
                "Best local model trails best cloud model by %.1f pp (> 15pp threshold)",
                (cloud_best - local_best) * 100,
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
