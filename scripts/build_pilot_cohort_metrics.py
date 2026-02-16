#!/usr/bin/env python3
"""
Aggregate multiple pilot metrics files into a cohort-level summary.
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_cmd(args: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        args,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.returncode, proc.stdout.strip()


def default_repo() -> str:
    code, out = run_cmd(["git", "config", "--get", "remote.origin.url"])
    if code != 0 or not out:
        return ""
    cleaned = out[:-4] if out.endswith(".git") else out
    if cleaned.startswith("https://github.com/"):
        return cleaned.split("https://github.com/", 1)[1]
    if cleaned.startswith("git@github.com:"):
        return cleaned.split("git@github.com:", 1)[1]
    return ""


def resolve(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def validate_metrics_file(metrics_path: Path, schema_path: str) -> None:
    code, out = run_cmd(
        [
            sys.executable,
            "scripts/check_pilot_metrics.py",
            str(metrics_path),
            "--schema",
            schema_path,
            "--expected-schema-version",
            "1",
        ]
    )
    if code != 0:
        detail = out if out else "validation failed"
        raise ValueError(f"invalid pilot metrics file {metrics_path}: {detail}")


def _num(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, int) or isinstance(value, float):
        return float(value)
    return default


def _int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    return default


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _max_int(values: list[int]) -> int:
    if not values:
        return 0
    return max(values)


def _min_float(values: list[float]) -> float:
    if not values:
        return 0.0
    return min(values)


def _max_float(values: list[float]) -> float:
    if not values:
        return 0.0
    return max(values)


def node_summary(report: dict[str, object], source: str) -> dict[str, object]:
    node = report.get("node", {})
    health = report.get("health", {})
    quality = report.get("quality", {})
    accessibility = report.get("accessibility", {})
    decentralization = report.get("decentralization", {})
    communication = report.get("communication", {})
    status = report.get("status", {})

    return {
        "source": source,
        "node_id": str(node.get("node_id", "unknown")),
        "mode": str(node.get("mode", "")),
        "uptime_ratio_24h": _num(health.get("uptime_ratio_24h")),
        "last_cycle_ok": bool(health.get("last_cycle_ok", False)),
        "classification_accuracy": _num(quality.get("classification_accuracy")),
        "classification_macro_f1": _num(quality.get("classification_macro_f1")),
        "utility_fedavg_int8_accuracy": _num(quality.get("utility_fedavg_int8_accuracy")),
        "utility_fedavg_int8_macro_f1": _num(quality.get("utility_fedavg_int8_macro_f1")),
        "benchmark_total_runtime_sec": _num(accessibility.get("benchmark_total_runtime_sec")),
        "max_peak_rss_bytes": _int(accessibility.get("max_peak_rss_bytes")),
        "max_peak_heap_bytes": _int(accessibility.get("max_peak_heap_bytes")),
        "baseline_int8_jain_index": _num(decentralization.get("baseline_int8_jain_index")),
        "utility_int8_jain_gain": _num(decentralization.get("utility_int8_jain_gain")),
        "baseline_int8_reduction_percent": _num(communication.get("baseline_int8_reduction_percent")),
        "utility_int8_savings_percent": _num(communication.get("utility_int8_savings_percent")),
        "open_milestones": _int(status.get("open_milestones")),
        "open_issues": _int(status.get("open_issues")),
        "status_collected": bool(status.get("collected", False)),
    }


def load_reports(paths: list[Path], schema_path: str) -> tuple[list[dict[str, object]], list[str]]:
    reports: list[dict[str, object]] = []
    loaded_sources: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        validate_metrics_file(metrics_path=path, schema_path=schema_path)
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        reports.append(payload)
        loaded_sources.append(str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path))
    return reports, loaded_sources


def build_payload(
    node_reports: list[dict[str, object]],
    source_paths: list[str],
    repo: str,
    commit: str,
) -> dict[str, object]:
    nodes = [node_summary(report=report, source=source_paths[idx]) for idx, report in enumerate(node_reports)]

    uptimes = [entry["uptime_ratio_24h"] for entry in nodes]
    last_ok_count = sum(1 for entry in nodes if entry["last_cycle_ok"])
    cls_acc = [entry["classification_accuracy"] for entry in nodes]
    cls_f1 = [entry["classification_macro_f1"] for entry in nodes]
    util_acc = [entry["utility_fedavg_int8_accuracy"] for entry in nodes]
    util_f1 = [entry["utility_fedavg_int8_macro_f1"] for entry in nodes]
    runtime = [entry["benchmark_total_runtime_sec"] for entry in nodes]
    peak_rss = [entry["max_peak_rss_bytes"] for entry in nodes]
    peak_heap = [entry["max_peak_heap_bytes"] for entry in nodes]
    baseline_jain = [entry["baseline_int8_jain_index"] for entry in nodes]
    utility_gain = [entry["utility_int8_jain_gain"] for entry in nodes]
    baseline_comm = [entry["baseline_int8_reduction_percent"] for entry in nodes]
    utility_comm = [entry["utility_int8_savings_percent"] for entry in nodes]
    open_milestones = [entry["open_milestones"] for entry in nodes]
    open_issues = [entry["open_issues"] for entry in nodes]
    status_collected_count = sum(1 for entry in nodes if entry["status_collected"])

    node_count = len(nodes)
    last_cycle_ok_ratio = 0.0 if node_count == 0 else last_ok_count / node_count
    status_collected_ratio = 0.0 if node_count == 0 else status_collected_count / node_count

    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": {
            "node_count": node_count,
            "node_ids": [entry["node_id"] for entry in nodes],
            "sources": source_paths,
        },
        "nodes": nodes,
        "summary": {
            "health": {
                "uptime_ratio_24h_mean": _mean(uptimes),
                "uptime_ratio_24h_min": _min_float(uptimes),
                "uptime_ratio_24h_max": _max_float(uptimes),
                "last_cycle_ok_ratio": last_cycle_ok_ratio,
            },
            "quality": {
                "classification_accuracy_mean": _mean(cls_acc),
                "classification_macro_f1_mean": _mean(cls_f1),
                "utility_fedavg_int8_accuracy_mean": _mean(util_acc),
                "utility_fedavg_int8_macro_f1_mean": _mean(util_f1),
            },
            "accessibility": {
                "benchmark_total_runtime_sec_mean": _mean(runtime),
                "max_peak_rss_bytes_max": _max_int(peak_rss),
                "max_peak_heap_bytes_max": _max_int(peak_heap),
            },
            "decentralization": {
                "baseline_int8_jain_index_mean": _mean(baseline_jain),
                "utility_int8_jain_gain_mean": _mean(utility_gain),
                "utility_int8_jain_gain_min": _min_float(utility_gain),
                "utility_int8_jain_gain_max": _max_float(utility_gain),
            },
            "communication": {
                "baseline_int8_reduction_percent_mean": _mean(baseline_comm),
                "utility_int8_savings_percent_mean": _mean(utility_comm),
            },
            "status": {
                "open_milestones_max": _max_int(open_milestones),
                "open_issues_max": _max_int(open_issues),
                "status_collected_ratio": status_collected_ratio,
            },
        },
        "provenance": {
            "repo": repo,
            "commit": commit,
            "source_count": len(source_paths),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        action="append",
        default=[],
        help="Path to pilot metrics JSON (repeatable).",
    )
    parser.add_argument(
        "--metrics-glob",
        default="",
        help="Optional glob for pilot metrics JSON files.",
    )
    parser.add_argument(
        "--repo",
        default="",
        help="Repository owner/name (default: inferred from origin remote).",
    )
    parser.add_argument(
        "--schema",
        default="schemas/pilot_metrics.schema.v1.json",
        help="Path to pilot metrics schema JSON for per-node validation.",
    )
    parser.add_argument(
        "--json-out",
        default="pilot/pilot_cohort_metrics.json",
        help="Output cohort metrics JSON path.",
    )
    args = parser.parse_args()

    source_paths: list[Path] = []
    for item in args.metrics:
        source_paths.append(resolve(item))

    if args.metrics_glob:
        matched = [Path(path) for path in glob.glob(str(resolve(args.metrics_glob)), recursive=True)]
        source_paths.extend(matched)

    if not source_paths:
        source_paths.append(resolve("pilot/pilot_metrics.json"))

    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in source_paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_paths.append(path)

    try:
        reports, loaded_sources = load_reports(paths=unique_paths, schema_path=args.schema)
    except ValueError as exc:
        print(exc)
        return 1
    if not reports:
        print("No pilot metrics files found. Provide --metrics or --metrics-glob.")
        return 1

    repo = args.repo or default_repo()
    if not repo:
        print("Unable to resolve provenance repo. Set --repo or configure origin remote.")
        return 1

    commit_code, commit_out = run_cmd(["git", "rev-parse", "HEAD"])
    commit = commit_out.strip()
    if commit_code != 0 or not commit:
        print("Unable to resolve git commit with 'git rev-parse HEAD'.")
        if commit_out:
            print(commit_out)
        return 1

    payload = build_payload(
        node_reports=reports,
        source_paths=loaded_sources,
        repo=repo,
        commit=commit,
    )

    out_path = resolve(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print(f"Pilot cohort metrics written to: {out_path}")
    print(f"Nodes aggregated: {payload['cohort']['node_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
