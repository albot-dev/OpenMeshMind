#!/usr/bin/env python3
"""
Validate fairness simulation output from the CPU FedAvg experiment.
"""

from __future__ import annotations

import argparse
import json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_json", help="Path to fairness JSON report.")
    parser.add_argument(
        "--expected-schema-version",
        type=int,
        default=2,
        help="Expected metrics schema version (default: 2).",
    )
    parser.add_argument(
        "--min-int8-jain-improvement",
        type=float,
        default=0.05,
        help=(
            "Minimum required improvement in Jain fairness index for fedavg_int8 "
            "vs fedavg_fp32 (default: 0.05)."
        ),
    )
    parser.add_argument(
        "--min-int8-contributed-clients-gain",
        type=float,
        default=0.5,
        help=(
            "Minimum required gain in contributed_clients_per_round_mean for "
            "fedavg_int8 vs fedavg_fp32 (default: 0.5)."
        ),
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
    fp32 = methods.get("fedavg_fp32", {})
    int8 = methods.get("fedavg_int8", {})
    fp32_fair = fp32.get("fairness")
    int8_fair = int8.get("fairness")

    if not fp32_fair:
        failures.append("fedavg_fp32 fairness block is missing")
    if not int8_fair:
        failures.append("fedavg_int8 fairness block is missing")
    if not fp32.get("fairness_clients"):
        failures.append("fedavg_fp32 fairness_clients block is missing")
    if not int8.get("fairness_clients"):
        failures.append("fedavg_int8 fairness_clients block is missing")

    if failures:
        print("Fairness metrics summary")
        print(f"- schema_version: {report.get('schema_version')}")
        print("\nValidation failed:")
        for issue in failures:
            print(f"- {issue}")
        return 1

    fp32_jain = fp32_fair["contribution_jain_index_mean"]
    int8_jain = int8_fair["contribution_jain_index_mean"]
    fp32_gap = fp32_fair["contribution_rate_gap_mean"]
    int8_gap = int8_fair["contribution_rate_gap_mean"]
    fp32_slowest = fp32_fair["slowest_fastest_contribution_ratio_mean"]
    int8_slowest = int8_fair["slowest_fastest_contribution_ratio_mean"]
    fp32_contrib = fp32_fair["contributed_clients_per_round_mean_mean"]
    int8_contrib = int8_fair["contributed_clients_per_round_mean_mean"]

    if int8_jain - fp32_jain < args.min_int8_jain_improvement:
        failures.append(
            f"Jain improvement too small: fp32={fp32_jain:.4f}, int8={int8_jain:.4f}, "
            f"required delta>={args.min_int8_jain_improvement:.4f}"
        )
    if int8_gap > fp32_gap:
        failures.append(
            f"Contribution gap got worse: fp32={fp32_gap:.4f}, int8={int8_gap:.4f}"
        )
    if int8_slowest < fp32_slowest:
        failures.append(
            "Slowest/fastest contribution ratio got worse: "
            f"fp32={fp32_slowest:.4f}, int8={int8_slowest:.4f}"
        )
    if int8_contrib - fp32_contrib < args.min_int8_contributed_clients_gain:
        failures.append(
            "Contributed clients/round gain too small: "
            f"fp32={fp32_contrib:.4f}, int8={int8_contrib:.4f}, "
            f"required delta>={args.min_int8_contributed_clients_gain:.4f}"
        )

    print("Fairness metrics summary")
    print(f"- schema_version: {report.get('schema_version')}")
    print(f"- fp32 Jain: {fp32_jain:.4f}")
    print(f"- int8 Jain: {int8_jain:.4f}")
    print(f"- fp32 contribution gap: {fp32_gap:.4f}")
    print(f"- int8 contribution gap: {int8_gap:.4f}")
    print(f"- fp32 contributed clients/round: {fp32_contrib:.4f}")
    print(f"- int8 contributed clients/round: {int8_contrib:.4f}")

    if failures:
        print("\nValidation failed:")
        for issue in failures:
            print(f"- {issue}")
        return 1

    print("\nValidation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
