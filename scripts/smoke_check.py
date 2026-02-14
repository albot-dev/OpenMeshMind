#!/usr/bin/env python3
"""
One-command smoke path for low-end contributor environments.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_step(name: str, cmd: list[str]) -> tuple[bool, float, str]:
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    duration = time.perf_counter() - start
    if proc.returncode != 0:
        return False, duration, proc.stdout
    return True, duration, proc.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-fairness",
        action="store_true",
        help="Include fairness stress checks in the smoke run.",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional path to write smoke summary JSON.",
    )
    args = parser.parse_args()

    steps: list[tuple[str, list[str]]] = [
        (
            "unit_tests",
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"],
        ),
        (
            "baseline_run",
            [
                sys.executable,
                "experiments/fedavg_cpu_only.py",
                "--seeds",
                "7",
                "--quiet",
                "--json-out",
                "baseline_metrics.json",
            ],
        ),
        (
            "baseline_check",
            [
                sys.executable,
                "scripts/check_baseline.py",
                "baseline_metrics.json",
                "--expected-schema-version",
                "2",
            ],
        ),
        (
            "classification_run",
            [
                sys.executable,
                "experiments/local_classification_baseline.py",
                "--samples-per-label",
                "24",
                "--steps",
                "900",
                "--quiet",
                "--json-out",
                "classification_metrics.json",
            ],
        ),
        (
            "classification_check",
            [
                sys.executable,
                "scripts/check_classification.py",
                "classification_metrics.json",
                "--expected-schema-version",
                "1",
            ],
        ),
        (
            "utility_fedavg_run",
            [
                sys.executable,
                "experiments/fedavg_classification_utility.py",
                "--seeds",
                "7",
                "--rounds",
                "8",
                "--local-steps",
                "4",
                "--quiet",
                "--json-out",
                "utility_fedavg_metrics.json",
            ],
        ),
        (
            "adapter_intent_run",
            [
                sys.executable,
                "experiments/fedavg_adapter_intent.py",
                "--seeds",
                "7",
                "--samples-per-intent",
                "12",
                "--rounds",
                "20",
                "--local-steps",
                "10",
                "--batch-size",
                "8",
                "--learning-rate",
                "0.26",
                "--quiet",
                "--json-out",
                "adapter_intent_metrics.json",
            ],
        ),
        (
            "adapter_intent_check",
            [
                sys.executable,
                "scripts/check_adapter_intent.py",
                "adapter_intent_metrics.json",
                "--expected-schema-version",
                "1",
            ],
        ),
        (
            "benchmark_reduced_run",
            [
                sys.executable,
                "scripts/benchmark_suite.py",
                "--mode",
                "reduced",
                "--quiet",
                "--json-out",
                "benchmark_metrics.json",
            ],
        ),
        (
            "benchmark_check",
            [
                sys.executable,
                "scripts/check_benchmarks.py",
                "benchmark_metrics.json",
                "--expected-schema-version",
                "1",
                "--expected-mode",
                "reduced",
            ],
        ),
        (
            "generality_eval_run",
            [
                sys.executable,
                "scripts/evaluate_generality.py",
                "--skip-distributed-reference",
                "--quiet",
                "--json-out",
                "generality_metrics.json",
            ],
        ),
        (
            "generality_eval_check",
            [
                sys.executable,
                "scripts/check_generality.py",
                "generality_metrics.json",
                "--expected-schema-version",
                "1",
            ],
        ),
    ]

    if args.include_fairness:
        steps.extend(
            [
                (
                    "fedavg_fairness_run",
                    [
                        sys.executable,
                        "experiments/fedavg_cpu_only.py",
                        "--simulate-client-capacity",
                        "--seeds",
                        "7",
                        "--quiet",
                        "--json-out",
                        "fairness_metrics.json",
                    ],
                ),
                (
                    "fedavg_fairness_check",
                    [
                        sys.executable,
                        "scripts/check_fairness.py",
                        "fairness_metrics.json",
                        "--expected-schema-version",
                        "2",
                    ],
                ),
                (
                    "utility_fairness_run",
                    [
                        sys.executable,
                        "experiments/fedavg_classification_utility.py",
                        "--simulate-client-capacity",
                        "--dropout-rate",
                        "0.1",
                        "--round-deadline-sweep",
                        "4.0,4.2",
                        "--seeds",
                        "7",
                        "--quiet",
                        "--json-out",
                        "utility_fairness_metrics.json",
                    ],
                ),
                (
                    "utility_fairness_check",
                    [
                        sys.executable,
                        "scripts/check_utility_fairness.py",
                        "utility_fairness_metrics.json",
                        "--expected-schema-version",
                        "1",
                    ],
                ),
                (
                    "generality_eval_full_run",
                    [
                        sys.executable,
                        "scripts/evaluate_generality.py",
                        "--quiet",
                        "--json-out",
                        "generality_metrics.json",
                    ],
                ),
                (
                    "generality_eval_full_check",
                    [
                        sys.executable,
                        "scripts/check_generality.py",
                        "generality_metrics.json",
                        "--expected-schema-version",
                        "1",
                    ],
                ),
                (
                    "reproducibility_sweep_run",
                    [
                        sys.executable,
                        "scripts/reproducibility_sweep.py",
                        "--seeds",
                        "7,17,27",
                        "--json-out",
                        "reproducibility_metrics.json",
                        "--quiet",
                    ],
                ),
                (
                    "reproducibility_sweep_check",
                    [
                        sys.executable,
                        "scripts/check_reproducibility.py",
                        "reproducibility_metrics.json",
                        "--expected-schema-version",
                        "1",
                    ],
                ),
            ]
        )

    summary: dict[str, object] = {
        "schema_version": 1,
        "include_fairness": args.include_fairness,
        "steps": [],
    }
    total_start = time.perf_counter()
    for name, cmd in steps:
        print(f"[smoke] {name}: {' '.join(cmd)}")
        ok, duration, output = run_step(name=name, cmd=cmd)
        summary["steps"].append(
            {
                "name": name,
                "ok": ok,
                "duration_sec": duration,
            }
        )
        print(f"[smoke] {name}: {'ok' if ok else 'failed'} ({duration:.2f}s)")
        if not ok:
            print(output)
            summary["ok"] = False
            summary["total_duration_sec"] = time.perf_counter() - total_start
            if args.json_out:
                with open(args.json_out, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2, sort_keys=True)
            return 1

    summary["ok"] = True
    summary["total_duration_sec"] = time.perf_counter() - total_start
    print(f"[smoke] total: {summary['total_duration_sec']:.2f}s")
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
