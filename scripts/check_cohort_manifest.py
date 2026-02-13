#!/usr/bin/env python3
"""
Validate pilot cohort manifest and summarize onboarding startup/failure rates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_node(node: dict[str, object], idx: int, failures: list[str]) -> None:
    required = [
        "node_id",
        "region",
        "hardware_tier",
        "cpu_cores",
        "memory_gb",
        "network_tier",
        "onboarding_status",
        "onboarding_checked_utc",
        "metrics_path",
        "failure_reason",
    ]
    for key in required:
        if key not in node:
            failures.append(f"nodes[{idx}].{key}: missing required field")

    status = node.get("onboarding_status")
    if status not in {"pending", "passed", "failed"}:
        failures.append(f"nodes[{idx}].onboarding_status: invalid value {status!r}")

    cpu = node.get("cpu_cores")
    if not _is_integer(cpu) or cpu <= 0:
        failures.append(f"nodes[{idx}].cpu_cores: expected integer > 0")

    mem = node.get("memory_gb")
    if not _is_integer(mem) or mem <= 0:
        failures.append(f"nodes[{idx}].memory_gb: expected integer > 0")

    failure_reason = node.get("failure_reason")
    if status == "failed" and (not isinstance(failure_reason, str) or not failure_reason.strip()):
        failures.append(f"nodes[{idx}].failure_reason: required when onboarding_status=failed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_json", help="Path to cohort manifest JSON.")
    parser.add_argument(
        "--expected-schema-version",
        type=int,
        default=1,
        help="Expected schema version (default: 1).",
    )
    parser.add_argument(
        "--min-nodes",
        type=int,
        default=5,
        help="Minimum required nodes in manifest (default: 5).",
    )
    parser.add_argument(
        "--min-passed",
        type=int,
        default=5,
        help="Minimum required passed onboarding nodes (default: 5).",
    )
    parser.add_argument(
        "--require-metrics-files",
        action="store_true",
        help="Require metrics_path files to exist for passed nodes.",
    )
    parser.add_argument(
        "--summary-json-out",
        default="",
        help="Optional path to write onboarding summary JSON.",
    )
    args = parser.parse_args()

    manifest_path = _resolve(args.manifest_json)
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    failures: list[str] = []

    if manifest.get("schema_version") != args.expected_schema_version:
        failures.append(
            f"schema_version={manifest.get('schema_version')} (expected {args.expected_schema_version})"
        )

    if not isinstance(manifest.get("cohort_id"), str) or len(manifest.get("cohort_id", "")) < 3:
        failures.append("cohort_id: expected non-empty string length >= 3")

    nodes = manifest.get("nodes")
    if not isinstance(nodes, list):
        failures.append("nodes: expected array")
        nodes = []

    node_ids: set[str] = set()
    passed = 0
    failed = 0
    pending = 0

    for idx, node in enumerate(nodes):
        if not isinstance(node, dict):
            failures.append(f"nodes[{idx}]: expected object")
            continue

        validate_node(node=node, idx=idx, failures=failures)

        node_id = str(node.get("node_id", "")).strip()
        if node_id in node_ids:
            failures.append(f"nodes[{idx}].node_id: duplicate {node_id!r}")
        else:
            node_ids.add(node_id)

        status = node.get("onboarding_status")
        if status == "passed":
            passed += 1
            if args.require_metrics_files:
                metrics_path = _resolve(str(node.get("metrics_path", "")))
                if not metrics_path.exists():
                    failures.append(
                        f"nodes[{idx}].metrics_path: missing file {metrics_path} for passed node"
                    )
                else:
                    try:
                        with metrics_path.open("r", encoding="utf-8") as f:
                            report = json.load(f)
                    except json.JSONDecodeError:
                        failures.append(
                            f"nodes[{idx}].metrics_path: invalid JSON file {metrics_path}"
                        )
                    else:
                        report_node = report.get("node", {}).get("node_id")
                        if isinstance(report_node, str) and report_node and report_node != node_id:
                            failures.append(
                                f"nodes[{idx}].metrics_path: node_id mismatch "
                                f"manifest={node_id} report={report_node}"
                            )
        elif status == "failed":
            failed += 1
        else:
            pending += 1

    total = len(nodes)
    startup_rate = 0.0 if total == 0 else passed / total
    failure_rate = 0.0 if total == 0 else failed / total

    if total < args.min_nodes:
        failures.append(f"node_count={total} < min_nodes={args.min_nodes}")
    if passed < args.min_passed:
        failures.append(f"passed_count={passed} < min_passed={args.min_passed}")

    summary = {
        "schema_version": 1,
        "cohort_id": manifest.get("cohort_id", ""),
        "generated_utc": manifest.get("generated_utc", ""),
        "node_count": total,
        "passed_count": passed,
        "failed_count": failed,
        "pending_count": pending,
        "startup_rate": startup_rate,
        "failure_rate": failure_rate,
    }

    print("Cohort onboarding summary")
    print(f"- cohort_id: {summary['cohort_id']}")
    print(f"- node_count: {total}")
    print(f"- passed_count: {passed}")
    print(f"- failed_count: {failed}")
    print(f"- pending_count: {pending}")
    print(f"- startup_rate: {startup_rate:.4f}")
    print(f"- failure_rate: {failure_rate:.4f}")

    if args.summary_json_out:
        out_path = _resolve(args.summary_json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print(f"- summary_json_out: {out_path}")

    if failures:
        print("\nValidation failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nValidation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
