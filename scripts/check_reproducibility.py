#!/usr/bin/env python3
"""
Validate reproducibility sweep metrics.
"""

from __future__ import annotations

import argparse
import json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_json", help="Path to reproducibility sweep JSON report.")
    parser.add_argument(
        "--expected-schema-version",
        type=int,
        default=1,
        help="Expected schema version (default: 1).",
    )
    parser.add_argument("--min-runs", type=int, default=3)
    parser.add_argument("--min-overall-score-mean", type=float, default=0.75)
    parser.add_argument("--max-overall-score-std", type=float, default=0.10)
    parser.add_argument("--min-classification-accuracy-mean", type=float, default=0.80)
    parser.add_argument("--min-retrieval-recall-at-1-mean", type=float, default=0.60)
    parser.add_argument("--min-instruction-pass-rate-mean", type=float, default=0.70)
    parser.add_argument("--min-tool-pass-rate-mean", type=float, default=0.80)
    parser.add_argument("--max-int8-accuracy-drop-mean", type=float, default=0.10)
    parser.add_argument("--min-int8-comm-savings-mean", type=float, default=40.0)
    parser.add_argument("--max-adapter-int8-accuracy-drop-mean", type=float, default=0.25)
    parser.add_argument("--min-adapter-int8-comm-savings-mean", type=float, default=40.0)
    args = parser.parse_args()

    with open(args.metrics_json, "r", encoding="utf-8") as f:
        report = json.load(f)

    failures: list[str] = []
    if report.get("schema_version") != args.expected_schema_version:
        failures.append(
            f"schema_version={report.get('schema_version')} "
            f"(expected {args.expected_schema_version})"
        )

    runs = report.get("runs", [])
    if len(runs) < args.min_runs:
        failures.append(f"run_count={len(runs)} (expected >= {args.min_runs})")

    summary = report.get("summary", {})
    overall = summary.get("overall_score", {})
    cls_acc = summary.get("classification_accuracy", {})
    ret = summary.get("retrieval_recall_at_1", {})
    ins = summary.get("instruction_pass_rate", {})
    tool = summary.get("tool_pass_rate", {})

    overall_mean = float(overall.get("mean", 0.0))
    overall_std = float(overall.get("std", 0.0))
    cls_mean = float(cls_acc.get("mean", 0.0))
    ret_mean = float(ret.get("mean", 0.0))
    ins_mean = float(ins.get("mean", 0.0))
    tool_mean = float(tool.get("mean", 0.0))

    if overall_mean < args.min_overall_score_mean:
        failures.append(
            f"overall_score mean {overall_mean:.4f} < {args.min_overall_score_mean:.4f}"
        )
    if overall_std > args.max_overall_score_std:
        failures.append(f"overall_score std {overall_std:.4f} > {args.max_overall_score_std:.4f}")
    if cls_mean < args.min_classification_accuracy_mean:
        failures.append(
            f"classification_accuracy mean {cls_mean:.4f} < {args.min_classification_accuracy_mean:.4f}"
        )
    if ret_mean < args.min_retrieval_recall_at_1_mean:
        failures.append(
            f"retrieval_recall@1 mean {ret_mean:.4f} < {args.min_retrieval_recall_at_1_mean:.4f}"
        )
    if ins_mean < args.min_instruction_pass_rate_mean:
        failures.append(
            f"instruction_pass_rate mean {ins_mean:.4f} < {args.min_instruction_pass_rate_mean:.4f}"
        )
    if tool_mean < args.min_tool_pass_rate_mean:
        failures.append(f"tool_pass_rate mean {tool_mean:.4f} < {args.min_tool_pass_rate_mean:.4f}")

    if "int8_accuracy_drop" in summary:
        int8_drop_mean = float(summary["int8_accuracy_drop"].get("mean", 0.0))
        int8_savings_mean = float(summary["int8_comm_savings_percent"].get("mean", 0.0))
        if int8_drop_mean > args.max_int8_accuracy_drop_mean:
            failures.append(
                "int8_accuracy_drop mean "
                f"{int8_drop_mean:.4f} > {args.max_int8_accuracy_drop_mean:.4f}"
            )
        if int8_savings_mean < args.min_int8_comm_savings_mean:
            failures.append(
                "int8_comm_savings mean "
                f"{int8_savings_mean:.2f}% < {args.min_int8_comm_savings_mean:.2f}%"
            )
    if "adapter_int8_accuracy_drop" in summary:
        adapter_drop_mean = float(summary["adapter_int8_accuracy_drop"].get("mean", 0.0))
        adapter_savings_mean = float(summary["adapter_int8_comm_savings_percent"].get("mean", 0.0))
        if adapter_drop_mean > args.max_adapter_int8_accuracy_drop_mean:
            failures.append(
                "adapter_int8_accuracy_drop mean "
                f"{adapter_drop_mean:.4f} > {args.max_adapter_int8_accuracy_drop_mean:.4f}"
            )
        if adapter_savings_mean < args.min_adapter_int8_comm_savings_mean:
            failures.append(
                "adapter_int8_comm_savings mean "
                f"{adapter_savings_mean:.2f}% < {args.min_adapter_int8_comm_savings_mean:.2f}%"
            )

    print("Reproducibility metrics summary")
    print(f"- schema_version: {report.get('schema_version')}")
    print(f"- run_count: {len(runs)}")
    print(f"- overall score mean/std: {overall_mean:.4f}/{overall_std:.4f}")
    print(f"- classification accuracy mean: {cls_mean:.4f}")
    print(f"- retrieval recall@1 mean: {ret_mean:.4f}")
    print(f"- instruction pass mean: {ins_mean:.4f}")
    print(f"- tool pass mean: {tool_mean:.4f}")
    if "int8_accuracy_drop" in summary:
        print(f"- int8 drop mean: {summary['int8_accuracy_drop'].get('mean'):.4f}")
        print(f"- int8 comm savings mean: {summary['int8_comm_savings_percent'].get('mean'):.2f}%")
    if "adapter_int8_accuracy_drop" in summary:
        print(f"- adapter int8 drop mean: {summary['adapter_int8_accuracy_drop'].get('mean'):.4f}")
        print(
            "- adapter int8 comm savings mean: "
            f"{summary['adapter_int8_comm_savings_percent'].get('mean'):.2f}%"
        )

    if failures:
        print("\nValidation failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nValidation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
