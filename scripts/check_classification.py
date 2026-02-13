#!/usr/bin/env python3
"""
Validate local classification baseline metrics for CI checks.
"""

from __future__ import annotations

import argparse
import json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "metrics_json",
        help="Path to JSON report from experiments/local_classification_baseline.py --json-out",
    )
    parser.add_argument(
        "--expected-schema-version",
        type=int,
        default=1,
        help="Expected metrics schema version (default: 1).",
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.85,
        help="Minimum acceptable classification accuracy (default: 0.85).",
    )
    parser.add_argument(
        "--min-macro-f1",
        type=float,
        default=0.85,
        help="Minimum acceptable macro-F1 (default: 0.85).",
    )
    parser.add_argument(
        "--max-train-runtime-sec",
        type=float,
        default=2.0,
        help="Maximum acceptable training runtime seconds (default: 2.0).",
    )
    parser.add_argument(
        "--max-latency-mean-ms",
        type=float,
        default=1.0,
        help="Maximum acceptable mean inference latency (default: 1.0 ms).",
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

    metrics = report.get("metrics", {})
    accuracy = metrics.get("accuracy")
    macro_f1 = metrics.get("macro_f1")
    train_runtime = metrics.get("train_runtime_sec")
    latency_mean = metrics.get("latency_mean_ms")

    if accuracy is None or accuracy < args.min_accuracy:
        failures.append(
            f"accuracy={accuracy} below minimum {args.min_accuracy:.4f}"
        )
    if macro_f1 is None or macro_f1 < args.min_macro_f1:
        failures.append(
            f"macro_f1={macro_f1} below minimum {args.min_macro_f1:.4f}"
        )
    if train_runtime is None or train_runtime > args.max_train_runtime_sec:
        failures.append(
            f"train_runtime_sec={train_runtime} above maximum {args.max_train_runtime_sec:.4f}"
        )
    if latency_mean is None or latency_mean > args.max_latency_mean_ms:
        failures.append(
            f"latency_mean_ms={latency_mean} above maximum {args.max_latency_mean_ms:.4f}"
        )

    counts = report.get("counts", {})
    labels = counts.get("labels", [])
    if not isinstance(labels, list) or len(labels) < 2:
        failures.append("counts.labels must include at least two classes")

    print("Classification metrics summary")
    print(f"- schema_version: {report.get('schema_version')}")
    print(f"- accuracy: {accuracy}")
    print(f"- macro_f1: {macro_f1}")
    print(f"- train_runtime_sec: {train_runtime}")
    print(f"- latency_mean_ms: {latency_mean}")
    print(f"- labels: {len(labels) if isinstance(labels, list) else 0}")

    if failures:
        print("\nValidation failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nValidation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
