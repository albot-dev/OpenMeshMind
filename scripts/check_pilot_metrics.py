#!/usr/bin/env python3
"""
Validate pilot metrics payloads against schema and optional readiness gates.
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
    parser.add_argument("metrics_json", help="Path to pilot metrics JSON report.")
    parser.add_argument(
        "--schema",
        default="schemas/pilot_metrics.schema.v1.json",
        help="Path to pilot metrics schema JSON.",
    )
    parser.add_argument(
        "--expected-schema-version",
        type=int,
        default=1,
        help="Expected pilot metrics schema version (default: 1).",
    )
    parser.add_argument(
        "--max-open-milestones",
        type=int,
        default=None,
        help="Optional cap for allowed open milestones.",
    )
    parser.add_argument(
        "--max-open-issues",
        type=int,
        default=None,
        help="Optional cap for allowed open issues.",
    )
    parser.add_argument(
        "--require-status-collected",
        action="store_true",
        help="Fail if GitHub status counters were not collected.",
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

    status = report.get("status", {})
    open_milestones = status.get("open_milestones")
    open_issues = status.get("open_issues")
    status_collected = status.get("collected")

    if args.require_status_collected and status_collected is not True:
        failures.append("$.status.collected: expected true when --require-status-collected is used")
    if args.max_open_milestones is not None and _is_integer(open_milestones):
        if open_milestones > args.max_open_milestones:
            failures.append(
                f"$.status.open_milestones: {open_milestones} > {args.max_open_milestones}"
            )
    if args.max_open_issues is not None and _is_integer(open_issues):
        if open_issues > args.max_open_issues:
            failures.append(f"$.status.open_issues: {open_issues} > {args.max_open_issues}")

    print("Pilot metrics validation summary")
    print(f"- schema_version: {report.get('schema_version')}")
    print(f"- node_id: {report.get('node', {}).get('node_id')}")
    print(f"- mode: {report.get('node', {}).get('mode')}")
    print(f"- uptime_ratio_24h: {report.get('health', {}).get('uptime_ratio_24h')}")
    print(f"- open_milestones: {open_milestones}")
    print(f"- open_issues: {open_issues}")
    print(f"- status_collected: {status_collected}")

    if failures:
        print("\nValidation failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nValidation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
