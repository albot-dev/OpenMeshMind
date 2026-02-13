#!/usr/bin/env python3
"""
Benchmark harness for constrained-hardware latency and memory measurements.

This script executes core experiments in isolated subprocesses and records:
- wall-clock runtime
- peak Python heap memory (tracemalloc)
- peak RSS (when available)
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SELF_PATH = Path(__file__).resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def normalize_ru_maxrss(ru_maxrss: int) -> int:
    # Linux: kilobytes. macOS/BSD: bytes.
    if sys.platform.startswith("linux"):
        return ru_maxrss * 1024
    return ru_maxrss


def run_fedavg_task(mode: str) -> dict[str, object]:
    from experiments import fedavg_cpu_only as fed

    seeds = [7] if mode == "reduced" else [7, 17, 27]
    secure = False if mode == "reduced" else True
    report = fed.run_experiment(
        seeds=seeds,
        dropout_rate=0.0,
        non_iid_severity=fed.DEFAULT_NON_IID_SEVERITY,
        secure_aggregation=secure,
    )
    methods = report["methods"]
    return {
        "task": "fedavg_baseline",
        "config": report["config"],
        "metrics": {
            "centralized_accuracy": methods["centralized"]["accuracy_mean"],
            "fedavg_int8_accuracy": methods["fedavg_int8"]["accuracy_mean"],
            "communication_reduction_percent": report["communication_reduction_percent"],
            "fedavg_runtime_mean_sec": methods["fedavg_int8"]["runtime_mean_sec"],
        },
    }


def run_retrieval_task(mode: str) -> dict[str, object]:
    from experiments import local_retrieval_baseline as retrieval

    report = retrieval.run_retrieval(
        corpus_path=str(ROOT / "data" / "retrieval_corpus.json"),
        queries_path=str(ROOT / "data" / "retrieval_queries.json"),
        top_k=3 if mode == "reduced" else 5,
    )
    return {
        "task": "retrieval_baseline",
        "config": report["config"],
        "metrics": {
            "recall_at_1": report["metrics"]["recall_at_1"],
            "recall_at_k": report["metrics"]["recall_at_k"],
            "mrr": report["metrics"]["mrr"],
            "retrieval_latency_mean_ms": report["metrics"]["latency_mean_ms"],
        },
    }


def run_classification_task(mode: str) -> dict[str, object]:
    from experiments import local_classification_baseline as classification

    report = classification.run_classification(
        seed=7,
        samples_per_label=20 if mode == "reduced" else 60,
        test_fraction=0.2,
        steps=900 if mode == "reduced" else 2200,
        learning_rate=0.18,
        measure_latency=True,
    )
    return {
        "task": "classification_baseline",
        "config": report["config"],
        "metrics": {
            "accuracy": report["metrics"]["accuracy"],
            "macro_f1": report["metrics"]["macro_f1"],
            "latency_mean_ms": report["metrics"]["latency_mean_ms"],
            "train_runtime_sec": report["metrics"]["train_runtime_sec"],
        },
    }


def run_single_task(task: str, mode: str) -> dict[str, object]:
    task_map = {
        "fedavg_baseline": run_fedavg_task,
        "retrieval_baseline": run_retrieval_task,
        "classification_baseline": run_classification_task,
    }
    runner = task_map.get(task)
    if runner is None:
        raise ValueError(f"Unknown task: {task}")

    tracemalloc.start()
    t0 = time.perf_counter()
    payload = runner(mode)
    dt = time.perf_counter() - t0
    _, peak_heap = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_rss = normalize_ru_maxrss(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    return {
        "task": task,
        "mode": mode,
        "runtime_sec": dt,
        "peak_heap_bytes": peak_heap,
        "peak_rss_bytes": peak_rss,
        "payload": payload,
    }


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    idx = int(0.95 * (len(values) - 1))
    return sorted(values)[idx]


def call_internal_task(task: str, mode: str) -> dict[str, object]:
    proc = subprocess.run(
        [sys.executable, str(SELF_PATH), "--internal-task", task, "--mode", mode],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


def run_suite(mode: str, repeats: int) -> dict[str, object]:
    tasks = ["fedavg_baseline", "retrieval_baseline", "classification_baseline"]
    benchmark_entries: list[dict[str, object]] = []
    total_start = time.perf_counter()

    for task in tasks:
        runs = [call_internal_task(task=task, mode=mode) for _ in range(repeats)]
        runtimes = [r["runtime_sec"] for r in runs]
        peaks_heap = [r["peak_heap_bytes"] for r in runs]
        peaks_rss = [r["peak_rss_bytes"] for r in runs]
        payload = runs[0]["payload"]
        benchmark_entries.append(
            {
                "name": task,
                "repeats": repeats,
                "runtime_mean_sec": sum(runtimes) / len(runtimes),
                "runtime_p95_sec": p95(runtimes),
                "peak_heap_mean_bytes": int(sum(peaks_heap) / len(peaks_heap)),
                "peak_heap_max_bytes": max(peaks_heap),
                "peak_rss_mean_bytes": int(sum(peaks_rss) / len(peaks_rss)),
                "peak_rss_max_bytes": max(peaks_rss),
                "payload": payload,
            }
        )

    total_runtime = time.perf_counter() - total_start
    return {
        "schema_version": 1,
        "config": {
            "mode": mode,
            "repeats": repeats,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cwd": str(ROOT),
        },
        "benchmarks": benchmark_entries,
        "summary": {
            "total_runtime_sec": total_runtime,
            "max_peak_rss_bytes": max(b["peak_rss_max_bytes"] for b in benchmark_entries),
            "max_peak_heap_bytes": max(b["peak_heap_max_bytes"] for b in benchmark_entries),
        },
    }


def summarize(report: dict[str, object]) -> None:
    print("Benchmark suite (constrained-hardware view)\n")
    print(
        f"mode={report['config']['mode']}, repeats={report['config']['repeats']}, "
        f"python={report['config']['python_version']}"
    )
    print(
        f"{'Task':<20} {'Runtime mean':<14} {'Peak RSS':<14} {'Peak heap':<14} {'Key metric'}"
    )
    print("-" * 86)
    for entry in report["benchmarks"]:
        payload = entry["payload"]
        if entry["name"] == "fedavg_baseline":
            key_metric = (
                f"int8_acc={payload['metrics']['fedavg_int8_accuracy']:.4f}, "
                f"comm_save={payload['metrics']['communication_reduction_percent']:.2f}%"
            )
        elif entry["name"] == "classification_baseline":
            key_metric = (
                f"acc={payload['metrics']['accuracy']:.4f}, "
                f"f1={payload['metrics']['macro_f1']:.4f}"
            )
        else:
            key_metric = (
                f"r@1={payload['metrics']['recall_at_1']:.4f}, "
                f"mrr={payload['metrics']['mrr']:.4f}"
            )
        print(
            f"{entry['name']:<20} "
            f"{entry['runtime_mean_sec']:<14.3f} "
            f"{entry['peak_rss_max_bytes'] / (1024*1024):<14.2f} "
            f"{entry['peak_heap_max_bytes'] / (1024*1024):<14.2f} "
            f"{key_metric}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["reduced", "full"],
        default="full",
        help="Benchmark mode: reduced for CI, full for local deep runs.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=0,
        help="Override repeat count (default: 1 for reduced, 3 for full).",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional path to write machine-readable JSON report.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress human-readable summary.",
    )
    parser.add_argument(
        "--internal-task",
        default="",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args.internal_task:
        output = run_single_task(task=args.internal_task, mode=args.mode)
        json.dump(output, sys.stdout, sort_keys=True)
        return

    repeats = args.repeats if args.repeats > 0 else (1 if args.mode == "reduced" else 3)
    report = run_suite(mode=args.mode, repeats=repeats)
    if not args.quiet:
        summarize(report)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
