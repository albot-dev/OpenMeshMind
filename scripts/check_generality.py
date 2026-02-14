#!/usr/bin/env python3
"""
Validate generality MVP metrics against baseline readiness thresholds.
"""

from __future__ import annotations

import argparse
import json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_json", help="Path to JSON report from scripts/evaluate_generality.py")
    parser.add_argument(
        "--expected-schema-version",
        type=int,
        default=1,
        help="Expected metrics schema version (default: 1).",
    )
    parser.add_argument("--min-classification-accuracy", type=float, default=0.80)
    parser.add_argument("--min-classification-f1", type=float, default=0.78)
    parser.add_argument("--min-retrieval-recall-at-1", type=float, default=0.60)
    parser.add_argument("--min-retrieval-mrr", type=float, default=0.75)
    parser.add_argument("--min-instruction-pass-rate", type=float, default=0.75)
    parser.add_argument("--min-tool-pass-rate", type=float, default=0.80)
    parser.add_argument("--min-overall-score", type=float, default=0.75)
    parser.add_argument("--max-total-runtime-sec", type=float, default=180.0)
    parser.add_argument("--max-int8-accuracy-drop", type=float, default=0.10)
    parser.add_argument("--min-int8-comm-savings-percent", type=float, default=40.0)
    args = parser.parse_args()

    with open(args.metrics_json, "r", encoding="utf-8") as f:
        report = json.load(f)

    failures: list[str] = []
    if report.get("schema_version") != args.expected_schema_version:
        failures.append(
            f"schema_version={report.get('schema_version')} "
            f"(expected {args.expected_schema_version})"
        )

    tasks = report.get("tasks", {})
    required = ["classification", "retrieval", "instruction_following", "tool_use"]
    for name in required:
        if name not in tasks:
            failures.append(f"missing task section: {name}")

    if failures:
        print("Generality metrics summary unavailable due to missing sections.")
        print("\nValidation failed:")
        for issue in failures:
            print(f"- {issue}")
        return 1

    cls = tasks["classification"]["metrics"]
    ret = tasks["retrieval"]["metrics"]
    ins = tasks["instruction_following"]["metrics"]
    tool = tasks["tool_use"]["metrics"]
    aggregate = report.get("aggregate", {})
    resources = report.get("resources", {})

    if cls["accuracy"] < args.min_classification_accuracy:
        failures.append(
            f"classification accuracy {cls['accuracy']:.4f} < {args.min_classification_accuracy:.4f}"
        )
    if cls["macro_f1"] < args.min_classification_f1:
        failures.append(f"classification macro_f1 {cls['macro_f1']:.4f} < {args.min_classification_f1:.4f}")
    if ret["recall_at_1"] < args.min_retrieval_recall_at_1:
        failures.append(
            f"retrieval recall@1 {ret['recall_at_1']:.4f} < {args.min_retrieval_recall_at_1:.4f}"
        )
    if ret["mrr"] < args.min_retrieval_mrr:
        failures.append(f"retrieval mrr {ret['mrr']:.4f} < {args.min_retrieval_mrr:.4f}")
    if ins["pass_rate"] < args.min_instruction_pass_rate:
        failures.append(
            f"instruction pass_rate {ins['pass_rate']:.4f} < {args.min_instruction_pass_rate:.4f}"
        )
    if tool["pass_rate"] < args.min_tool_pass_rate:
        failures.append(f"tool pass_rate {tool['pass_rate']:.4f} < {args.min_tool_pass_rate:.4f}")

    overall = aggregate.get("overall_score", 0.0)
    if overall < args.min_overall_score:
        failures.append(f"overall score {overall:.4f} < {args.min_overall_score:.4f}")

    runtime_total = resources.get("total_wall_clock_sec", aggregate.get("runtime_total_sec", 0.0))
    if runtime_total > args.max_total_runtime_sec:
        failures.append(f"total runtime {runtime_total:.2f}s > {args.max_total_runtime_sec:.2f}s")

    distributed = tasks.get("distributed_reference")
    if distributed is not None:
        dist = distributed.get("metrics", {})
        int8_drop = float(dist.get("int8_accuracy_drop", 0.0))
        int8_comm = float(dist.get("int8_comm_savings_percent", 0.0))
        if int8_drop > args.max_int8_accuracy_drop:
            failures.append(
                f"distributed int8_accuracy_drop {int8_drop:.4f} > {args.max_int8_accuracy_drop:.4f}"
            )
        if int8_comm < args.min_int8_comm_savings_percent:
            failures.append(
                "distributed int8_comm_savings_percent "
                f"{int8_comm:.2f}% < {args.min_int8_comm_savings_percent:.2f}%"
            )

    print("Generality metrics summary")
    print(f"- schema_version: {report.get('schema_version')}")
    print(f"- classification accuracy: {cls['accuracy']:.4f}")
    print(f"- classification macro_f1: {cls['macro_f1']:.4f}")
    print(f"- retrieval recall@1: {ret['recall_at_1']:.4f}")
    print(f"- retrieval mrr: {ret['mrr']:.4f}")
    print(f"- instruction pass_rate: {ins['pass_rate']:.4f}")
    print(f"- tool pass_rate: {tool['pass_rate']:.4f}")
    print(f"- overall score: {overall:.4f}")
    print(f"- total runtime: {runtime_total:.2f}s")
    if distributed is not None:
        dist = distributed["metrics"]
        print(f"- int8 accuracy drop vs centralized: {dist['int8_accuracy_drop']:+.4f}")
        print(f"- int8 communication savings vs fp32: {dist['int8_comm_savings_percent']:.2f}%")

    if failures:
        print("\nValidation failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nValidation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
