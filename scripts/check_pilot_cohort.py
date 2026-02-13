#!/usr/bin/env python3
"""
Validate pilot cohort metrics schema and readiness thresholds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return (isinstance(value, int) or isinstance(value, float)) and not isinstance(value, bool)


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def validate_against_schema(
    payload: object,
    schema: dict[str, object],
    path: str,
    failures: list[str],
) -> None:
    expected_type = schema.get("type")
    if expected_type == "object":
        if not isinstance(payload, dict):
            failures.append(f"{path}: expected object")
            return

        required = schema.get("required", [])
        for key in required:
            if key not in payload:
                failures.append(f"{path}.{key}: missing required field")

        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            for key in payload:
                if key not in properties:
                    failures.append(f"{path}.{key}: unexpected field")

        for key, child_schema in properties.items():
            if key in payload:
                validate_against_schema(
                    payload=payload[key],
                    schema=child_schema,
                    path=f"{path}.{key}",
                    failures=failures,
                )
        return

    if expected_type == "array":
        if not isinstance(payload, list):
            failures.append(f"{path}: expected array")
        return

    if expected_type == "string":
        if not isinstance(payload, str):
            failures.append(f"{path}: expected string")
            return
        min_length = schema.get("minLength")
        if min_length is not None and len(payload) < min_length:
            failures.append(f"{path}: length {len(payload)} < minLength {min_length}")

    elif expected_type == "boolean":
        if not isinstance(payload, bool):
            failures.append(f"{path}: expected boolean")
            return

    elif expected_type == "integer":
        if not _is_integer(payload):
            failures.append(f"{path}: expected integer")
            return

    elif expected_type == "number":
        if not _is_number(payload):
            failures.append(f"{path}: expected number")
            return

    const_value = schema.get("const")
    if const_value is not None and payload != const_value:
        failures.append(f"{path}: value {payload!r} != const {const_value!r}")

    minimum = schema.get("minimum")
    if minimum is not None and _is_number(payload) and payload < minimum:
        failures.append(f"{path}: value {payload} < minimum {minimum}")

    maximum = schema.get("maximum")
    if maximum is not None and _is_number(payload) and payload > maximum:
        failures.append(f"{path}: value {payload} > maximum {maximum}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_json", help="Path to pilot cohort metrics JSON report.")
    parser.add_argument(
        "--schema",
        default="schemas/pilot_cohort.schema.v1.json",
        help="Path to pilot cohort schema JSON.",
    )
    parser.add_argument(
        "--expected-schema-version",
        type=int,
        default=1,
        help="Expected schema version (default: 1).",
    )
    parser.add_argument(
        "--min-node-count",
        type=int,
        default=1,
        help="Minimum required node count (default: 1).",
    )
    parser.add_argument(
        "--min-uptime-ratio-mean",
        type=float,
        default=0.90,
        help="Minimum allowed uptime mean ratio (default: 0.90).",
    )
    parser.add_argument(
        "--min-last-cycle-ok-ratio",
        type=float,
        default=0.80,
        help="Minimum allowed last-cycle success ratio (default: 0.80).",
    )
    parser.add_argument(
        "--min-status-collected-ratio",
        type=float,
        default=0.0,
        help="Minimum required status_collected_ratio (default: 0.0).",
    )
    parser.add_argument(
        "--max-open-milestones",
        type=int,
        default=None,
        help="Optional cap for open milestones in cohort summary.",
    )
    parser.add_argument(
        "--max-open-issues",
        type=int,
        default=None,
        help="Optional cap for open issues in cohort summary.",
    )
    args = parser.parse_args()

    report_path = _resolve(args.metrics_json)
    schema_path = _resolve(args.schema)

    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)
    with schema_path.open("r", encoding="utf-8") as f:
        schema = json.load(f)

    failures: list[str] = []
    validate_against_schema(payload=report, schema=schema, path="$", failures=failures)

    if report.get("schema_version") != args.expected_schema_version:
        failures.append(
            f"$.schema_version: {report.get('schema_version')} "
            f"(expected {args.expected_schema_version})"
        )

    cohort = report.get("cohort", {})
    node_count = cohort.get("node_count")
    summary = report.get("summary", {})
    health = summary.get("health", {})
    status = summary.get("status", {})

    uptime_mean = health.get("uptime_ratio_24h_mean", 0.0)
    ok_ratio = health.get("last_cycle_ok_ratio", 0.0)
    status_collected_ratio = status.get("status_collected_ratio", 0.0)
    open_milestones_max = status.get("open_milestones_max")
    open_issues_max = status.get("open_issues_max")

    if _is_integer(node_count) and node_count < args.min_node_count:
        failures.append(f"$.cohort.node_count: {node_count} < {args.min_node_count}")
    if _is_number(uptime_mean) and uptime_mean < args.min_uptime_ratio_mean:
        failures.append(
            f"$.summary.health.uptime_ratio_24h_mean: {uptime_mean} < {args.min_uptime_ratio_mean}"
        )
    if _is_number(ok_ratio) and ok_ratio < args.min_last_cycle_ok_ratio:
        failures.append(
            f"$.summary.health.last_cycle_ok_ratio: {ok_ratio} < {args.min_last_cycle_ok_ratio}"
        )
    if _is_number(status_collected_ratio) and status_collected_ratio < args.min_status_collected_ratio:
        failures.append(
            "$.summary.status.status_collected_ratio: "
            f"{status_collected_ratio} < {args.min_status_collected_ratio}"
        )

    if args.max_open_milestones is not None and _is_integer(open_milestones_max):
        if open_milestones_max > args.max_open_milestones:
            failures.append(
                f"$.summary.status.open_milestones_max: {open_milestones_max} > {args.max_open_milestones}"
            )
    if args.max_open_issues is not None and _is_integer(open_issues_max):
        if open_issues_max > args.max_open_issues:
            failures.append(f"$.summary.status.open_issues_max: {open_issues_max} > {args.max_open_issues}")

    print("Pilot cohort validation summary")
    print(f"- schema_version: {report.get('schema_version')}")
    print(f"- node_count: {node_count}")
    print(f"- uptime_ratio_24h_mean: {uptime_mean}")
    print(f"- last_cycle_ok_ratio: {ok_ratio}")
    print(f"- status_collected_ratio: {status_collected_ratio}")
    print(f"- open_milestones_max: {open_milestones_max}")
    print(f"- open_issues_max: {open_issues_max}")

    if failures:
        print("\nValidation failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nValidation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
