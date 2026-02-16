#!/usr/bin/env python3
"""
Build a cross-machine reliability comparison report from captured machine snapshots.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def run_cmd(args: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        args,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.returncode, proc.stdout.strip()


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
    return float(statistics.mean(values))


def _max(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(max(values))


def _min(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(min(values))


def _range(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(max(values) - min(values))


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _snapshot_meta(snapshot_dir: Path) -> dict[str, object]:
    meta_path = snapshot_dir / "snapshot_meta.json"
    if not meta_path.exists():
        return {
            "machine_id": snapshot_dir.name,
            "label": "",
            "generated_utc": "",
        }
    try:
        return _load_json(meta_path)
    except (json.JSONDecodeError, ValueError):
        return {
            "machine_id": snapshot_dir.name,
            "label": "",
            "generated_utc": "",
        }


def _artifact_path(snapshot_dir: Path, relative: str) -> Path:
    return snapshot_dir / relative


def _checker_failure(
    *,
    script: str,
    artifact: Path,
    failures: list[str],
) -> None:
    code, out = run_cmd(
        [
            sys.executable,
            script,
            str(artifact),
            "--expected-schema-version",
            "1",
        ]
    )
    if code != 0:
        detail = out.splitlines()[-1] if out else "validation failed"
        failures.append(f"{script} failed: {detail}")


def evaluate_snapshot(
    *,
    snapshot_dir: Path,
    run_checkers: bool,
    require_mvp_readiness: bool,
) -> dict[str, object]:
    failures: list[str] = []

    generality_path = _artifact_path(snapshot_dir, "generality_metrics.json")
    repro_path = _artifact_path(snapshot_dir, "reproducibility_metrics.json")
    smoke_path = _artifact_path(snapshot_dir, "smoke_summary.json")
    status_path = _artifact_path(snapshot_dir, "main_track_status.json")
    readiness_path = _artifact_path(snapshot_dir, "mvp_readiness.json")

    required_paths = [generality_path, repro_path, smoke_path, status_path]
    missing_required = [str(path) for path in required_paths if not path.exists()]
    if missing_required:
        failures.append(f"missing required artifacts: {', '.join(missing_required)}")

    meta = _snapshot_meta(snapshot_dir)
    machine_id = str(meta.get("machine_id", snapshot_dir.name) or snapshot_dir.name)
    label = str(meta.get("label", "") or "")

    generality: dict[str, object] = {}
    reproducibility: dict[str, object] = {}
    smoke: dict[str, object] = {}
    status: dict[str, object] = {}
    readiness: dict[str, object] = {}

    for path, target_name in [
        (generality_path, "generality"),
        (repro_path, "reproducibility"),
        (smoke_path, "smoke"),
        (status_path, "status"),
    ]:
        if not path.exists():
            continue
        try:
            payload = _load_json(path)
        except (json.JSONDecodeError, ValueError) as exc:
            failures.append(f"invalid JSON {path}: {exc}")
            continue
        if target_name == "generality":
            generality = payload
        elif target_name == "reproducibility":
            reproducibility = payload
        elif target_name == "smoke":
            smoke = payload
        else:
            status = payload

    if run_checkers:
        if generality_path.exists():
            _checker_failure(
                script="scripts/check_generality.py",
                artifact=generality_path,
                failures=failures,
            )
        if repro_path.exists():
            _checker_failure(
                script="scripts/check_reproducibility.py",
                artifact=repro_path,
                failures=failures,
            )

    smoke_ok = smoke.get("ok") is True
    if not smoke_ok:
        failures.append("smoke_summary ok is not true")

    status_summary = status.get("summary", {})
    all_done = False
    completed_goals = 0
    total_goals = 0
    completed_required_sub_goals = 0
    required_sub_goals = 0
    if isinstance(status_summary, dict):
        all_done = status_summary.get("all_done") is True
        completed_goals = _int(status_summary.get("completed_goals"), 0)
        total_goals = _int(status_summary.get("total_goals"), 0)
        completed_required_sub_goals = _int(status_summary.get("completed_required_sub_goals"), 0)
        required_sub_goals = _int(status_summary.get("required_sub_goals"), 0)

    if not all_done:
        failures.append("main_track_status summary.all_done is not true")

    readiness_ok = None
    if readiness_path.exists():
        try:
            readiness = _load_json(readiness_path)
            readiness_ok = readiness.get("ready") is True
            if not readiness_ok:
                failures.append("mvp_readiness ready is not true")
        except (json.JSONDecodeError, ValueError) as exc:
            failures.append(f"invalid JSON {readiness_path}: {exc}")
    elif require_mvp_readiness:
        failures.append("missing required artifact mvp_readiness.json")

    generality_overall = _num(get_nested(generality, "aggregate", "overall_score"))
    generality_runtime = _num(get_nested(generality, "resources", "total_wall_clock_sec"))
    repro_overall_mean = _num(get_nested(reproducibility, "summary", "overall_score", "mean"))
    repro_overall_std = _num(get_nested(reproducibility, "summary", "overall_score", "std"))
    smoke_duration = _num(smoke.get("total_duration_sec"))

    return {
        "snapshot_dir": str(snapshot_dir),
        "machine_id": machine_id,
        "label": label,
        "ok": len(failures) == 0,
        "failures": failures,
        "metrics": {
            "generality_overall_score": generality_overall,
            "generality_runtime_sec": generality_runtime,
            "repro_overall_mean": repro_overall_mean,
            "repro_overall_std": repro_overall_std,
            "smoke_ok": smoke_ok,
            "smoke_total_duration_sec": smoke_duration,
            "status_all_done": all_done,
            "completed_goals": completed_goals,
            "total_goals": total_goals,
            "completed_required_sub_goals": completed_required_sub_goals,
            "required_sub_goals": required_sub_goals,
            "mvp_readiness_ok": readiness_ok,
        },
    }


def get_nested(payload: dict[str, object], *keys: str) -> object:
    value: object = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def resolve_snapshot_dirs(snapshot_dirs: list[str], snapshot_glob: str) -> list[Path]:
    items: list[Path] = []
    for value in snapshot_dirs:
        items.append(resolve(value))
    if snapshot_glob:
        for value in sorted(glob.glob(str(resolve(snapshot_glob)))):
            items.append(Path(value))

    unique: list[Path] = []
    seen: set[Path] = set()
    for item in items:
        resolved = item.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(item)
    return unique


def to_markdown(payload: dict[str, object]) -> str:
    lines: list[str] = []
    lines.append("# Machine Comparison Report")
    lines.append("")
    lines.append(f"- generated_utc: `{payload['generated_utc']}`")
    summary = payload["summary"]
    lines.append(
        f"- snapshots: `{summary['snapshot_count']}` "
        f"(passing `{summary['passing_count']}`, failing `{summary['failing_count']}`)"
    )
    lines.append(f"- ready: `{summary['ready']}`")
    lines.append(
        "- overall ranges: "
        f"generality `{summary['generality_overall_score_range']:.4f}`, "
        f"repro_mean `{summary['repro_overall_mean_range']:.4f}`, "
        f"repro_std_max `{summary['repro_overall_std_max']:.4f}`"
    )
    lines.append("")
    lines.append("| machine_id | label | ok | generality | repro_mean | repro_std | smoke_sec | failures |")
    lines.append("|---|---|---|---:|---:|---:|---:|---|")
    for item in payload["snapshots"]:
        metrics = item["metrics"]
        failures = "; ".join(item["failures"]) if item["failures"] else ""
        lines.append(
            f"| {item['machine_id']} | {item['label']} | {item['ok']} | "
            f"{metrics['generality_overall_score']:.4f} | "
            f"{metrics['repro_overall_mean']:.4f} | "
            f"{metrics['repro_overall_std']:.4f} | "
            f"{metrics['smoke_total_duration_sec']:.2f} | "
            f"{failures} |"
        )

    if payload["failures"]:
        lines.append("")
        lines.append("## Global Failures")
        for failure in payload["failures"]:
            lines.append(f"- {failure}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot-dir",
        action="append",
        default=[],
        help="Snapshot directory path (repeatable).",
    )
    parser.add_argument(
        "--snapshot-glob",
        default="pilot/machine_snapshots/*",
        help="Glob used to discover snapshot directories.",
    )
    parser.add_argument(
        "--min-snapshots",
        type=int,
        default=3,
        help="Minimum snapshot count required for a passing comparison.",
    )
    parser.add_argument(
        "--max-generality-overall-range",
        type=float,
        default=0.10,
        help="Maximum allowed range for generality overall scores across snapshots.",
    )
    parser.add_argument(
        "--max-repro-overall-mean-range",
        type=float,
        default=0.10,
        help="Maximum allowed range for reproducibility overall mean scores.",
    )
    parser.add_argument(
        "--max-repro-overall-std-max",
        type=float,
        default=0.10,
        help="Maximum allowed max reproducibility overall std across snapshots.",
    )
    parser.add_argument(
        "--require-mvp-readiness",
        action="store_true",
        help="Require mvp_readiness.json with ready=true in every snapshot.",
    )
    parser.add_argument(
        "--skip-checkers",
        action="store_true",
        help="Skip invoking scripts/check_generality.py and scripts/check_reproducibility.py per snapshot.",
    )
    parser.add_argument(
        "--json-out",
        default="reports/machine_comparison.json",
        help="Output JSON comparison report path.",
    )
    parser.add_argument(
        "--md-out",
        default="reports/machine_comparison.md",
        help="Output markdown comparison report path.",
    )
    args = parser.parse_args()

    dirs = resolve_snapshot_dirs(snapshot_dirs=args.snapshot_dir, snapshot_glob=args.snapshot_glob)
    dirs = [item for item in dirs if item.exists() and item.is_dir()]
    if not dirs:
        print("No snapshot directories found.")
        return 1

    snapshots = [
        evaluate_snapshot(
            snapshot_dir=item,
            run_checkers=not args.skip_checkers,
            require_mvp_readiness=args.require_mvp_readiness,
        )
        for item in dirs
    ]

    generality_values = [item["metrics"]["generality_overall_score"] for item in snapshots]
    repro_mean_values = [item["metrics"]["repro_overall_mean"] for item in snapshots]
    repro_std_values = [item["metrics"]["repro_overall_std"] for item in snapshots]
    smoke_duration_values = [item["metrics"]["smoke_total_duration_sec"] for item in snapshots]
    passing_count = sum(1 for item in snapshots if item["ok"])

    failures: list[str] = []
    if len(snapshots) < args.min_snapshots:
        failures.append(f"snapshot_count={len(snapshots)} < min_snapshots={args.min_snapshots}")
    if passing_count < len(snapshots):
        failures.append(f"snapshot_failures={len(snapshots) - passing_count}")

    generality_range = _range(generality_values)
    repro_mean_range = _range(repro_mean_values)
    repro_std_max = _max(repro_std_values)
    if generality_range > args.max_generality_overall_range:
        failures.append(
            "generality_overall_score_range="
            f"{generality_range:.4f} > max_generality_overall_range={args.max_generality_overall_range:.4f}"
        )
    if repro_mean_range > args.max_repro_overall_mean_range:
        failures.append(
            "repro_overall_mean_range="
            f"{repro_mean_range:.4f} > max_repro_overall_mean_range={args.max_repro_overall_mean_range:.4f}"
        )
    if repro_std_max > args.max_repro_overall_std_max:
        failures.append(
            "repro_overall_std_max="
            f"{repro_std_max:.4f} > max_repro_overall_std_max={args.max_repro_overall_std_max:.4f}"
        )

    payload = {
        "schema_version": 1,
        "generated_utc": utc_now_iso(),
        "config": {
            "min_snapshots": args.min_snapshots,
            "max_generality_overall_range": args.max_generality_overall_range,
            "max_repro_overall_mean_range": args.max_repro_overall_mean_range,
            "max_repro_overall_std_max": args.max_repro_overall_std_max,
            "require_mvp_readiness": args.require_mvp_readiness,
            "skip_checkers": args.skip_checkers,
        },
        "summary": {
            "snapshot_count": len(snapshots),
            "passing_count": passing_count,
            "failing_count": len(snapshots) - passing_count,
            "ready": len(failures) == 0,
            "generality_overall_score_mean": _mean(generality_values),
            "generality_overall_score_min": _min(generality_values),
            "generality_overall_score_max": _max(generality_values),
            "generality_overall_score_range": generality_range,
            "repro_overall_mean_mean": _mean(repro_mean_values),
            "repro_overall_mean_min": _min(repro_mean_values),
            "repro_overall_mean_max": _max(repro_mean_values),
            "repro_overall_mean_range": repro_mean_range,
            "repro_overall_std_max": repro_std_max,
            "smoke_total_duration_sec_mean": _mean(smoke_duration_values),
            "smoke_total_duration_sec_max": _max(smoke_duration_values),
        },
        "snapshots": snapshots,
        "failures": failures,
    }

    json_out = resolve(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    with json_out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    md_out = resolve(args.md_out)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(to_markdown(payload), encoding="utf-8")

    print("Machine comparison summary")
    print(f"- snapshot_count: {payload['summary']['snapshot_count']}")
    print(f"- passing_count: {payload['summary']['passing_count']}")
    print(f"- failing_count: {payload['summary']['failing_count']}")
    print(f"- ready: {payload['summary']['ready']}")
    print(f"- generality_overall_score_range: {payload['summary']['generality_overall_score_range']:.4f}")
    print(f"- repro_overall_mean_range: {payload['summary']['repro_overall_mean_range']:.4f}")
    print(f"- repro_overall_std_max: {payload['summary']['repro_overall_std_max']:.4f}")
    print(f"- json_out: {json_out}")
    print(f"- md_out: {md_out}")

    if failures:
        print("\nComparison failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nComparison passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
