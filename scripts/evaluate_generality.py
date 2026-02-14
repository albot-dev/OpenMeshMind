#!/usr/bin/env python3
"""
Evaluate local generalist MVP readiness on commodity hardware.

Task families:
- classification quality (local CPU baseline)
- retrieval quality (local corpus lookup)
- instruction following (runtime action compliance)
- tool use (runtime calculator correctness)

Optional:
- distributed reference comparison (centralized vs federated int8/sparse)
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import fedavg_classification_utility as utility_fed
from experiments import local_classification_baseline as classification
from experiments import local_retrieval_baseline as retrieval
from scripts.local_generalist_runtime import LocalGeneralistRuntime


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    idx = int(0.95 * (len(values) - 1))
    return sorted(values)[idx]


def normalize_ru_maxrss(ru_maxrss: int) -> int:
    # Linux: kilobytes. macOS/BSD: bytes.
    if sys.platform.startswith("linux"):
        return ru_maxrss * 1024
    return ru_maxrss


def evaluate_classification(seed: int) -> dict[str, object]:
    start = time.perf_counter()
    report = classification.run_classification(
        seed=seed,
        samples_per_label=24,
        test_fraction=0.25,
        steps=900,
        learning_rate=0.18,
        measure_latency=True,
    )
    runtime = time.perf_counter() - start
    metrics = report["metrics"]
    return {
        "runtime_sec": runtime,
        "metrics": {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "latency_mean_ms": metrics["latency_mean_ms"],
            "latency_p95_ms": metrics["latency_p95_ms"],
            "train_runtime_sec": metrics["train_runtime_sec"],
        },
        "counts": report["counts"],
        "score": (metrics["accuracy"] + metrics["macro_f1"]) / 2.0,
    }


def evaluate_retrieval(top_k: int) -> dict[str, object]:
    start = time.perf_counter()
    report = retrieval.run_retrieval(
        corpus_path="data/retrieval_corpus.json",
        queries_path="data/retrieval_queries.json",
        top_k=top_k,
    )
    runtime = time.perf_counter() - start
    metrics = report["metrics"]
    return {
        "runtime_sec": runtime,
        "metrics": {
            "recall_at_1": metrics["recall_at_1"],
            "recall_at_k": metrics["recall_at_k"],
            "mrr": metrics["mrr"],
            "latency_mean_ms": metrics["latency_mean_ms"],
            "latency_p95_ms": metrics["latency_p95_ms"],
        },
        "counts": report["counts"],
        "score": (metrics["recall_at_1"] + metrics["mrr"]) / 2.0,
    }


def evaluate_instruction_following(runtime: LocalGeneralistRuntime) -> dict[str, object]:
    # Ordered prompts to verify multi-step behavior including memory, retrieval, and exact response control.
    cases = [
        {
            "id": "remember_note",
            "prompt": "Remember that my preferred sync time is 14:00 UTC.",
            "contains": ["Remembered", "14:00"],
        },
        {
            "id": "recall_note",
            "prompt": "What did I ask you to remember?",
            "contains": ["14:00", "UTC"],
        },
        {
            "id": "retrieval_lookup",
            "prompt": "Lookup in the corpus what mentions secure aggregation.",
            "contains_any": ["aggregate", "masks", "client updates"],
        },
        {
            "id": "exact_response",
            "prompt": "Respond exactly with: ACK READY",
            "equals": "ACK READY",
        },
    ]

    passed = 0
    latencies: list[float] = []
    details: list[dict[str, object]] = []
    start = time.perf_counter()

    for item in cases:
        result = runtime.respond(item["prompt"])
        answer = str(result["answer"])
        latencies.append(float(result["latency_ms"]))
        ok = True
        if "contains" in item:
            ok = all(token in answer for token in item["contains"])
        if ok and "contains_any" in item:
            ok = any(token in answer.lower() for token in item["contains_any"])
        if ok and "equals" in item:
            ok = answer.strip() == item["equals"]
        if ok:
            passed += 1
        details.append(
            {
                "id": item["id"],
                "ok": ok,
                "intent": result["intent"],
                "answer": answer,
                "latency_ms": result["latency_ms"],
            }
        )

    runtime_sec = time.perf_counter() - start
    pass_rate = passed / len(cases)
    return {
        "runtime_sec": runtime_sec,
        "counts": {"cases": len(cases), "passed": passed},
        "metrics": {
            "pass_rate": pass_rate,
            "latency_mean_ms": statistics.mean(latencies),
            "latency_p95_ms": p95(latencies),
        },
        "score": pass_rate,
        "details": details,
    }


def evaluate_tool_use(runtime: LocalGeneralistRuntime) -> dict[str, object]:
    cases = [
        ("Calculate (12 + 8) * 3", "60"),
        ("Compute 150 / 6", "25"),
        ("Evaluate 19 + 5 - 3", "21"),
        ("What is 7 * 9", "63"),
    ]
    passed = 0
    latencies: list[float] = []
    details: list[dict[str, object]] = []
    start = time.perf_counter()
    for prompt, expected in cases:
        result = runtime.respond(prompt)
        answer = str(result["answer"]).strip()
        ok = answer == expected
        if ok:
            passed += 1
        latencies.append(float(result["latency_ms"]))
        details.append(
            {
                "prompt": prompt,
                "expected": expected,
                "answer": answer,
                "ok": ok,
                "latency_ms": result["latency_ms"],
            }
        )
    runtime_sec = time.perf_counter() - start
    pass_rate = passed / len(cases)
    return {
        "runtime_sec": runtime_sec,
        "counts": {"cases": len(cases), "passed": passed},
        "metrics": {
            "pass_rate": pass_rate,
            "latency_mean_ms": statistics.mean(latencies),
            "latency_p95_ms": p95(latencies),
        },
        "score": pass_rate,
        "details": details,
    }


def evaluate_distributed_reference(seed: int) -> dict[str, object]:
    # Reduced configuration to keep this executable on commodity CPUs.
    start = time.perf_counter()
    report = utility_fed.run_experiment(
        seeds=[seed],
        modes=["fp32", "int8", "sparse"],
        samples_per_label=14,
        test_fraction=0.25,
        n_clients=5,
        rounds=6,
        local_steps=3,
        batch_size=8,
        learning_rate=0.18,
        non_iid_severity=1.2,
        sparse_ratio=0.2,
    )
    runtime_sec = time.perf_counter() - start
    methods = report["methods"]
    centralized = methods["centralized"]
    int8 = methods["fedavg_int8"]
    sparse = methods["fedavg_sparse"]
    comm = report["communication_savings_percent"]
    quality_drop = report["quality_drop_vs_centralized"]

    return {
        "runtime_sec": runtime_sec,
        "metrics": {
            "centralized_accuracy": centralized["accuracy_mean"],
            "int8_accuracy": int8["accuracy_mean"],
            "sparse_accuracy": sparse["accuracy_mean"],
            "int8_accuracy_drop": quality_drop["int8_accuracy_drop"],
            "sparse_accuracy_drop": quality_drop["sparse_accuracy_drop"],
            "int8_comm_savings_percent": comm.get("int8_vs_fp32_percent", 0.0),
            "sparse_comm_savings_percent": comm.get("sparse_vs_fp32_percent", 0.0),
        },
        "score": max(0.0, 1.0 - max(0.0, quality_drop["int8_accuracy_drop"])) * (
            comm.get("int8_vs_fp32_percent", 0.0) / 100.0
        ),
    }


def build_aggregate(tasks: dict[str, dict[str, object]]) -> dict[str, object]:
    task_scores = [float(payload["score"]) for payload in tasks.values()]
    runtime_total = sum(float(payload["runtime_sec"]) for payload in tasks.values())
    weighted = statistics.mean(task_scores) if task_scores else 0.0
    return {
        "task_count": len(tasks),
        "task_scores": {name: payload["score"] for name, payload in tasks.items()},
        "overall_score": weighted,
        "runtime_total_sec": runtime_total,
    }


def summarize(report: dict[str, object]) -> None:
    print("Generality MVP evaluation\n")
    cfg = report["config"]
    print(
        f"seed={cfg['seed']}, top_k={cfg['top_k']}, "
        f"include_distributed_reference={cfg['include_distributed_reference']}"
    )

    cls = report["tasks"]["classification"]["metrics"]
    ret = report["tasks"]["retrieval"]["metrics"]
    ins = report["tasks"]["instruction_following"]["metrics"]
    tool = report["tasks"]["tool_use"]["metrics"]
    print(f"classification: acc={cls['accuracy']:.4f}, f1={cls['macro_f1']:.4f}")
    print(f"retrieval: r@1={ret['recall_at_1']:.4f}, mrr={ret['mrr']:.4f}")
    print(f"instruction: pass_rate={ins['pass_rate']:.4f}")
    print(f"tool_use: pass_rate={tool['pass_rate']:.4f}")
    if "distributed_reference" in report["tasks"]:
        dist = report["tasks"]["distributed_reference"]["metrics"]
        print(
            "distributed: "
            f"int8_drop={dist['int8_accuracy_drop']:+.4f}, "
            f"int8_comm_save={dist['int8_comm_savings_percent']:.2f}%"
        )

    print(
        "aggregate: "
        f"overall_score={report['aggregate']['overall_score']:.4f}, "
        f"runtime_total_sec={report['aggregate']['runtime_total_sec']:.2f}"
    )
    print(
        "resources: "
        f"peak_rss_mb={report['resources']['peak_rss_bytes'] / (1024*1024):.2f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Reproducibility seed for deterministic components.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Retrieval top-k for retrieval task.",
    )
    parser.add_argument(
        "--skip-distributed-reference",
        action="store_true",
        help="Skip centralized vs federated comparison task.",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional output JSON path.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console summary.",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    runtime = LocalGeneralistRuntime(seed=args.seed, top_k=args.top_k)
    tasks: dict[str, dict[str, object]] = {
        "classification": evaluate_classification(seed=args.seed),
        "retrieval": evaluate_retrieval(top_k=args.top_k),
        "instruction_following": evaluate_instruction_following(runtime=runtime),
        "tool_use": evaluate_tool_use(runtime=runtime),
    }
    if not args.skip_distributed_reference:
        tasks["distributed_reference"] = evaluate_distributed_reference(seed=args.seed)

    report = {
        "schema_version": 1,
        "generated_utc": utc_now_iso(),
        "config": {
            "seed": args.seed,
            "top_k": args.top_k,
            "include_distributed_reference": not args.skip_distributed_reference,
        },
        "tasks": tasks,
        "aggregate": build_aggregate(tasks=tasks),
        "resources": {
            "total_wall_clock_sec": time.perf_counter() - started,
            "peak_rss_bytes": normalize_ru_maxrss(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
    }

    if not args.quiet:
        summarize(report)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
