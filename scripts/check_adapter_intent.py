#!/usr/bin/env python3
"""
Validate federated adapter-style intent experiment metrics.
"""

from __future__ import annotations

import argparse
import json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_json", help="Path to adapter metrics JSON report.")
    parser.add_argument(
        "--expected-schema-version",
        type=int,
        default=1,
        help="Expected schema version (default: 1).",
    )
    parser.add_argument(
        "--min-centralized-accuracy",
        type=float,
        default=0.70,
        help="Minimum centralized adapter accuracy (default: 0.70).",
    )
    parser.add_argument(
        "--max-int8-accuracy-drop",
        type=float,
        default=0.20,
        help="Maximum int8 accuracy drop vs centralized (default: 0.20).",
    )
    parser.add_argument(
        "--min-int8-comm-savings",
        type=float,
        default=35.0,
        help="Minimum int8 communication savings percent vs fp32 (default: 35).",
    )
    args = parser.parse_args()

    with open(args.metrics_json, "r", encoding="utf-8") as f:
        report = json.load(f)

    failures: list[str] = []
    if report.get("schema_version") != args.expected_schema_version:
        failures.append(
            f"schema_version={report.get('schema_version')} "
            f"(expected {args.expected_schema_version})"
        )

    methods = report.get("methods", {})
    central = methods.get("centralized", {})
    int8 = methods.get("fedavg_int8", {})
    quality_drop = report.get("quality_drop_vs_centralized", {})
    comm = report.get("communication_savings_percent", {})

    central_acc = central.get("accuracy_mean", 0.0)
    int8_acc = int8.get("accuracy_mean", 0.0)
    int8_drop = quality_drop.get("int8_accuracy_drop", 0.0)
    int8_savings = comm.get("int8_vs_fp32_percent", 0.0)

    if central_acc < args.min_centralized_accuracy:
        failures.append(
            f"centralized accuracy {central_acc:.4f} < {args.min_centralized_accuracy:.4f}"
        )
    if int8_drop > args.max_int8_accuracy_drop:
        failures.append(
            f"int8 accuracy drop {int8_drop:.4f} > {args.max_int8_accuracy_drop:.4f}"
        )
    if int8_savings < args.min_int8_comm_savings:
        failures.append(
            f"int8 comm savings {int8_savings:.2f}% < {args.min_int8_comm_savings:.2f}%"
        )

    print("Adapter intent metrics summary")
    print(f"- schema_version: {report.get('schema_version')}")
    print(f"- centralized accuracy: {central_acc:.4f}")
    print(f"- fedavg_int8 accuracy: {int8_acc:.4f}")
    print(f"- int8 accuracy drop: {int8_drop:+.4f}")
    print(f"- int8 comm savings vs fp32: {int8_savings:.2f}%")

    if failures:
        print("\nValidation failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nValidation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
