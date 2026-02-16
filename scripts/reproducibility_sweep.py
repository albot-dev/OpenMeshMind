#!/usr/bin/env python3
"""
Run multi-seed generality evaluations and summarize reproducibility metrics.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIN_SWEEP_SEEDS = 3


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_seed(
    seed: int,
    top_k: int,
    long_context_top_k: int,
    include_distributed_reference: bool,
) -> dict[str, float]:
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / f"generality_seed_{seed}.json"
        cmd = [
            sys.executable,
            "scripts/evaluate_generality.py",
            "--seed",
            str(seed),
            "--top-k",
            str(top_k),
            "--long-context-top-k",
            str(long_context_top_k),
            "--json-out",
            str(out),
            "--quiet",
        ]
        if not include_distributed_reference:
            cmd.append("--skip-distributed-reference")

        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"evaluate_generality failed for seed={seed} rc={proc.returncode}\n{proc.stdout}"
            )
        with out.open("r", encoding="utf-8") as f:
            report = json.load(f)

    tasks = report["tasks"]
    run = {
        "seed": seed,
        "overall_score": float(report["aggregate"]["overall_score"]),
        "classification_accuracy": float(tasks["classification"]["metrics"]["accuracy"]),
        "classification_macro_f1": float(tasks["classification"]["metrics"]["macro_f1"]),
        "retrieval_recall_at_1": float(tasks["retrieval"]["metrics"]["recall_at_1"]),
        "retrieval_mrr": float(tasks["retrieval"]["metrics"]["mrr"]),
        "long_context_recall_at_1": float(tasks["long_context_retrieval"]["metrics"]["recall_at_1"]),
        "long_context_mrr": float(tasks["long_context_retrieval"]["metrics"]["mrr"]),
        "instruction_pass_rate": float(tasks["instruction_following"]["metrics"]["pass_rate"]),
        "conversation_pass_rate": float(tasks["conversation_continuity"]["metrics"]["pass_rate"]),
        "tool_pass_rate": float(tasks["tool_use"]["metrics"]["pass_rate"]),
        "multi_step_tool_pass_rate": float(tasks["multi_step_tool_use"]["metrics"]["pass_rate"]),
        "multi_step_tool_chain_pass_rate": float(
            tasks["multi_step_tool_use"]["metrics"]["chain_pass_rate"]
        ),
    }
    distributed = tasks.get("distributed_reference")
    if distributed:
        d = distributed["metrics"]
        run["int8_accuracy_drop"] = float(d["int8_accuracy_drop"])
        run["int8_comm_savings_percent"] = float(d["int8_comm_savings_percent"])
    adapter = tasks.get("adapter_reference")
    if adapter:
        a = adapter["metrics"]
        run["adapter_int8_accuracy_drop"] = float(a["int8_accuracy_drop"])
        run["adapter_int8_comm_savings_percent"] = float(a["int8_comm_savings_percent"])
    return run


def summarize_metric(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.pstdev(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def summarize_runs(runs: list[dict[str, float]]) -> dict[str, object]:
    keys = [
        "overall_score",
        "classification_accuracy",
        "classification_macro_f1",
        "retrieval_recall_at_1",
        "retrieval_mrr",
        "long_context_recall_at_1",
        "long_context_mrr",
        "instruction_pass_rate",
        "conversation_pass_rate",
        "tool_pass_rate",
        "multi_step_tool_pass_rate",
        "multi_step_tool_chain_pass_rate",
    ]
    summary = {key: summarize_metric([float(run[key]) for run in runs]) for key in keys}
    if runs and "int8_accuracy_drop" in runs[0]:
        summary["int8_accuracy_drop"] = summarize_metric(
            [float(run["int8_accuracy_drop"]) for run in runs]
        )
        summary["int8_comm_savings_percent"] = summarize_metric(
            [float(run["int8_comm_savings_percent"]) for run in runs]
        )
    if runs and "adapter_int8_accuracy_drop" in runs[0]:
        summary["adapter_int8_accuracy_drop"] = summarize_metric(
            [float(run["adapter_int8_accuracy_drop"]) for run in runs]
        )
        summary["adapter_int8_comm_savings_percent"] = summarize_metric(
            [float(run["adapter_int8_comm_savings_percent"]) for run in runs]
        )
    return summary


def print_summary(report: dict[str, object]) -> None:
    print("Reproducibility sweep summary\n")
    cfg = report["config"]
    print(
        f"seeds={cfg['seeds']}, top_k={cfg['top_k']}, "
        f"include_distributed_reference={cfg['include_distributed_reference']}"
    )
    summary = report["summary"]
    print(
        "overall score:"
        f" mean={summary['overall_score']['mean']:.4f},"
        f" std={summary['overall_score']['std']:.4f},"
        f" min={summary['overall_score']['min']:.4f}"
    )
    print(
        "instruction/conversation/tool pass:"
        f" instruction_mean={summary['instruction_pass_rate']['mean']:.4f},"
        f" conversation_mean={summary['conversation_pass_rate']['mean']:.4f},"
        f" tool_mean={summary['tool_pass_rate']['mean']:.4f}"
    )
    print(
        "long-context/multi-step:"
        f" long_context_r1_mean={summary['long_context_recall_at_1']['mean']:.4f},"
        f" long_context_mrr_mean={summary['long_context_mrr']['mean']:.4f},"
        f" multi_step_tool_mean={summary['multi_step_tool_pass_rate']['mean']:.4f},"
        f" multi_step_chain_mean={summary['multi_step_tool_chain_pass_rate']['mean']:.4f}"
    )
    if "int8_accuracy_drop" in summary:
        print(
            "distributed int8:"
            f" drop_mean={summary['int8_accuracy_drop']['mean']:.4f},"
            f" comm_savings_mean={summary['int8_comm_savings_percent']['mean']:.2f}%"
        )
    if "adapter_int8_accuracy_drop" in summary:
        print(
            "adapter int8:"
            f" drop_mean={summary['adapter_int8_accuracy_drop']['mean']:.4f},"
            f" comm_savings_mean={summary['adapter_int8_comm_savings_percent']['mean']:.2f}%"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        default="7,17,27",
        help="Comma-separated seeds for repeatability sweep (minimum 3).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Retrieval top-k value passed to evaluate_generality.",
    )
    parser.add_argument(
        "--long-context-top-k",
        type=int,
        default=3,
        help="Long-context retrieval top-k value passed to evaluate_generality.",
    )
    parser.add_argument(
        "--skip-distributed-reference",
        action="store_true",
        help="Skip distributed reference task in each run.",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional output JSON path.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console summary output.",
    )
    args = parser.parse_args()

    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    if len(seeds) < MIN_SWEEP_SEEDS:
        print(f"Need at least {MIN_SWEEP_SEEDS} seeds for a reproducibility sweep.")
        return 1

    runs = [
        run_seed(
            seed=seed,
            top_k=args.top_k,
            long_context_top_k=args.long_context_top_k,
            include_distributed_reference=not args.skip_distributed_reference,
        )
        for seed in seeds
    ]

    report = {
        "schema_version": 1,
        "generated_utc": utc_now_iso(),
        "config": {
            "seeds": seeds,
            "top_k": args.top_k,
            "long_context_top_k": args.long_context_top_k,
            "include_distributed_reference": not args.skip_distributed_reference,
        },
        "runs": runs,
        "summary": summarize_runs(runs=runs),
    }
    if not args.quiet:
        print_summary(report)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
