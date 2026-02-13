#!/usr/bin/env python3
"""
Validate fairness stress metrics from fedavg_classification_utility.py.
"""

from __future__ import annotations

import argparse
import json


def validate_single_report(
    report: dict[str, object],
    min_jain_improvement: float,
    min_gap_improvement: float,
    min_contrib_gain: float,
) -> tuple[list[str], str]:
    failures: list[str] = []
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
        failures.append("fedavg_fp32 fairness_clients is missing")
    if not int8.get("fairness_clients"):
        failures.append("fedavg_int8 fairness_clients is missing")
    if failures:
        return failures, "missing fairness blocks"

    jain_gain = int8_fair["contribution_jain_index_mean"] - fp32_fair["contribution_jain_index_mean"]
    gap_improvement = fp32_fair["contribution_rate_gap_mean"] - int8_fair["contribution_rate_gap_mean"]
    contrib_gain = (
        int8_fair["contributed_clients_per_round_mean_mean"]
        - fp32_fair["contributed_clients_per_round_mean_mean"]
    )

    if jain_gain < min_jain_improvement:
        failures.append(
            f"int8 Jain gain too small: {jain_gain:.4f} < {min_jain_improvement:.4f}"
        )
    if gap_improvement < min_gap_improvement:
        failures.append(
            f"int8 gap improvement too small: {gap_improvement:.4f} < {min_gap_improvement:.4f}"
        )
    if contrib_gain < min_contrib_gain:
        failures.append(
            f"int8 contributed-clients gain too small: {contrib_gain:.4f} < {min_contrib_gain:.4f}"
        )

    summary = (
        f"jain_gain={jain_gain:.4f}, "
        f"gap_improvement={gap_improvement:.4f}, "
        f"contrib_gain={contrib_gain:.4f}"
    )
    return failures, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_json", help="Path to fairness JSON report.")
    parser.add_argument(
        "--expected-schema-version",
        type=int,
        default=1,
        help="Expected schema version (default: 1).",
    )
    parser.add_argument(
        "--min-int8-jain-improvement",
        type=float,
        default=0.05,
        help="Minimum int8 Jain fairness gain vs fp32 (default: 0.05).",
    )
    parser.add_argument(
        "--min-int8-gap-improvement",
        type=float,
        default=0.05,
        help="Minimum int8 contribution-gap improvement vs fp32 (default: 0.05).",
    )
    parser.add_argument(
        "--min-int8-contributed-clients-gain",
        type=float,
        default=0.5,
        help="Minimum gain in contributed clients/round for int8 vs fp32 (default: 0.5).",
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

    scenario_reports = report.get("scenarios")
    if scenario_reports is None:
        scenario_reports = [report]

    print("Utility fairness validation summary")
    print(f"- schema_version: {report.get('schema_version')}")
    print(f"- scenarios: {len(scenario_reports)}")

    for idx, scenario in enumerate(scenario_reports):
        scenario_failures, summary = validate_single_report(
            report=scenario,
            min_jain_improvement=args.min_int8_jain_improvement,
            min_gap_improvement=args.min_int8_gap_improvement,
            min_contrib_gain=args.min_int8_contributed_clients_gain,
        )
        print(f"- scenario[{idx}]: {summary}")
        failures.extend([f"scenario[{idx}] {item}" for item in scenario_failures])

    if failures:
        print("\nValidation failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nValidation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
