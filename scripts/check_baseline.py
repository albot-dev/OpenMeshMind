#!/usr/bin/env python3
"""
Validate baseline experiment metrics for CI and reproducibility checks.
"""

from __future__ import annotations

import argparse
import json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "metrics_json",
        help="Path to JSON report from experiments/fedavg_cpu_only.py --json-out",
    )
    parser.add_argument(
        "--max-accuracy-drop",
        type=float,
        default=0.03,
        help="Maximum allowed drop vs centralized for fedavg_int8 (default: 0.03).",
    )
    parser.add_argument(
        "--min-int8-accuracy",
        type=float,
        default=0.82,
        help="Minimum absolute accuracy for fedavg_int8 (default: 0.82).",
    )
    parser.add_argument(
        "--min-comm-reduction",
        type=float,
        default=50.0,
        help="Minimum communication reduction percent from int8 (default: 50).",
    )
    args = parser.parse_args()

    with open(args.metrics_json, "r", encoding="utf-8") as f:
        report = json.load(f)

    methods = report["methods"]
    acc_central = methods["centralized"]["accuracy_mean"]
    acc_int8 = methods["fedavg_int8"]["accuracy_mean"]
    comm_reduction = report["communication_reduction_percent"]

    accuracy_drop = acc_central - acc_int8
    failures: list[str] = []

    if accuracy_drop > args.max_accuracy_drop:
        failures.append(
            f"fedavg_int8 accuracy drop {accuracy_drop:.4f} exceeds "
            f"max {args.max_accuracy_drop:.4f}"
        )
    if acc_int8 < args.min_int8_accuracy:
        failures.append(
            f"fedavg_int8 accuracy {acc_int8:.4f} below min {args.min_int8_accuracy:.4f}"
        )
    if comm_reduction < args.min_comm_reduction:
        failures.append(
            f"communication reduction {comm_reduction:.2f}% below min "
            f"{args.min_comm_reduction:.2f}%"
        )

    print("Baseline metrics summary")
    print(f"- centralized accuracy: {acc_central:.4f}")
    print(f"- fedavg_int8 accuracy: {acc_int8:.4f}")
    print(f"- accuracy drop: {accuracy_drop:.4f}")
    print(f"- communication reduction: {comm_reduction:.2f}%")

    if failures:
        print("\nValidation failed:")
        for issue in failures:
            print(f"- {issue}")
        return 1

    print("\nValidation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
