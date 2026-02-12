#!/usr/bin/env python3
"""
Validate benchmark suite output schema and required latency/memory fields.
"""

from __future__ import annotations

import argparse
import json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_json", help="Path to benchmark JSON report.")
    parser.add_argument(
        "--expected-schema-version",
        type=int,
        default=1,
        help="Expected benchmark schema version (default: 1).",
    )
    parser.add_argument(
        "--expected-mode",
        default="reduced",
        help="Expected benchmark mode in config (default: reduced).",
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
    mode = report.get("config", {}).get("mode")
    if mode != args.expected_mode:
        failures.append(f"mode={mode} (expected {args.expected_mode})")

    benchmarks = report.get("benchmarks", [])
    if not benchmarks:
        failures.append("benchmarks list is empty")

    for entry in benchmarks:
        name = entry.get("name", "<unknown>")
        runtime = entry.get("runtime_mean_sec")
        peak_heap = entry.get("peak_heap_max_bytes")
        peak_rss = entry.get("peak_rss_max_bytes")
        if runtime is None or runtime <= 0:
            failures.append(f"{name}: runtime_mean_sec missing/invalid")
        if peak_heap is None or peak_heap <= 0:
            failures.append(f"{name}: peak_heap_max_bytes missing/invalid")
        if peak_rss is None or peak_rss <= 0:
            failures.append(f"{name}: peak_rss_max_bytes missing/invalid")

    print("Benchmark metrics summary")
    print(f"- schema_version: {report.get('schema_version')}")
    print(f"- mode: {mode}")
    print(f"- benchmark_count: {len(benchmarks)}")

    if failures:
        print("\nValidation failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nValidation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
