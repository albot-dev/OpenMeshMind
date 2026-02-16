#!/usr/bin/env python3
"""
Run failure-injection drills for machine snapshot comparison reliability.
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
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
        if item.exists() and item.is_dir():
            unique.append(item)
    return unique


def copy_snapshots(src_dirs: list[Path], scenario_root: Path) -> list[Path]:
    copied: list[Path] = []
    for src in src_dirs:
        dst = scenario_root / src.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        copied.append(dst)
    return copied


def run_comparison(
    *,
    snapshot_dirs: list[Path],
    min_snapshots: int,
    require_mvp_readiness: bool,
    skip_checkers: bool,
    json_out: Path,
    md_out: Path,
) -> tuple[int, str]:
    cmd = [
        sys.executable,
        "scripts/build_machine_comparison_report.py",
        "--snapshot-glob",
        "",
        "--min-snapshots",
        str(min_snapshots),
        "--json-out",
        str(json_out),
        "--md-out",
        str(md_out),
    ]
    if require_mvp_readiness:
        cmd.append("--require-mvp-readiness")
    if skip_checkers:
        cmd.append("--skip-checkers")
    for item in snapshot_dirs:
        cmd.extend(["--snapshot-dir", str(item)])
    return run_cmd(cmd)


def to_markdown(payload: dict[str, object]) -> str:
    lines: list[str] = []
    lines.append("# Machine Reliability Drill")
    lines.append("")
    lines.append(f"- generated_utc: `{payload['generated_utc']}`")
    lines.append(f"- ok: `{payload['ok']}`")
    lines.append("")
    lines.append("| scenario | expected_success | actual_success | ok | exit_code |")
    lines.append("|---|---|---|---|---:|")
    for scenario in payload["scenarios"]:
        lines.append(
            f"| {scenario['name']} | {scenario['expected_success']} | "
            f"{scenario['actual_success']} | {scenario['ok']} | {scenario['exit_code']} |"
        )
    if payload["failures"]:
        lines.append("")
        lines.append("## Failures")
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
        help="Minimum snapshots for baseline comparison.",
    )
    parser.add_argument(
        "--skip-checkers",
        action="store_true",
        help="Pass --skip-checkers through to comparison runs.",
    )
    parser.add_argument(
        "--out-dir",
        default="reports/machine_reliability_drill",
        help="Output directory for scenario artifacts and reports.",
    )
    parser.add_argument(
        "--json-out",
        default="reports/machine_reliability_drill.json",
        help="Output JSON drill report path.",
    )
    parser.add_argument(
        "--md-out",
        default="reports/machine_reliability_drill.md",
        help="Output markdown drill report path.",
    )
    args = parser.parse_args()

    snapshot_dirs = resolve_snapshot_dirs(args.snapshot_dir, args.snapshot_glob)
    if not snapshot_dirs:
        print("No snapshot directories found.")
        return 1
    if len(snapshot_dirs) < args.min_snapshots:
        print(f"Need at least {args.min_snapshots} snapshots for drill baseline.")
        return 1

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios: list[dict[str, object]] = []
    failures: list[str] = []

    def record_result(name: str, expected_success: bool, exit_code: int, output: str) -> None:
        actual_success = exit_code == 0
        ok = actual_success == expected_success
        scenarios.append(
            {
                "name": name,
                "expected_success": expected_success,
                "actual_success": actual_success,
                "ok": ok,
                "exit_code": exit_code,
                "output_tail": output.splitlines()[-10:],
            }
        )
        if not ok:
            failures.append(
                f"scenario={name} expected_success={expected_success} actual_success={actual_success}"
            )

    # Scenario 1: baseline expected success.
    baseline_json = out_dir / "baseline_comparison.json"
    baseline_md = out_dir / "baseline_comparison.md"
    code, out = run_comparison(
        snapshot_dirs=snapshot_dirs,
        min_snapshots=args.min_snapshots,
        require_mvp_readiness=True,
        skip_checkers=args.skip_checkers,
        json_out=baseline_json,
        md_out=baseline_md,
    )
    record_result("baseline", True, code, out)

    # Scenario 2: missing smoke artifact expected failure.
    missing_smoke_root = out_dir / "scenario_missing_smoke"
    copied_missing_smoke = copy_snapshots(snapshot_dirs, missing_smoke_root)
    if copied_missing_smoke:
        target = copied_missing_smoke[0] / "smoke_summary.json"
        if target.exists():
            target.unlink()
    missing_smoke_json = out_dir / "missing_smoke_comparison.json"
    missing_smoke_md = out_dir / "missing_smoke_comparison.md"
    code, out = run_comparison(
        snapshot_dirs=copied_missing_smoke,
        min_snapshots=args.min_snapshots,
        require_mvp_readiness=True,
        skip_checkers=args.skip_checkers,
        json_out=missing_smoke_json,
        md_out=missing_smoke_md,
    )
    record_result("missing_smoke", False, code, out)

    # Scenario 3: readiness=false expected failure.
    readiness_false_root = out_dir / "scenario_readiness_false"
    copied_readiness_false = copy_snapshots(snapshot_dirs, readiness_false_root)
    if copied_readiness_false:
        readiness_path = copied_readiness_false[0] / "mvp_readiness.json"
        if readiness_path.exists():
            with readiness_path.open("r", encoding="utf-8") as f:
                readiness = json.load(f)
            if isinstance(readiness, dict):
                readiness["ready"] = False
                with readiness_path.open("w", encoding="utf-8") as f:
                    json.dump(readiness, f, indent=2, sort_keys=True)
    readiness_false_json = out_dir / "readiness_false_comparison.json"
    readiness_false_md = out_dir / "readiness_false_comparison.md"
    code, out = run_comparison(
        snapshot_dirs=copied_readiness_false,
        min_snapshots=args.min_snapshots,
        require_mvp_readiness=True,
        skip_checkers=args.skip_checkers,
        json_out=readiness_false_json,
        md_out=readiness_false_md,
    )
    record_result("readiness_false", False, code, out)

    payload = {
        "schema_version": 1,
        "generated_utc": utc_now_iso(),
        "ok": len(failures) == 0,
        "config": {
            "min_snapshots": args.min_snapshots,
            "skip_checkers": args.skip_checkers,
            "snapshot_count": len(snapshot_dirs),
        },
        "scenarios": scenarios,
        "failures": failures,
    }

    json_out = resolve(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    with json_out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    md_out = resolve(args.md_out)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(to_markdown(payload), encoding="utf-8")

    print("Machine reliability drill summary")
    print(f"- snapshot_count: {len(snapshot_dirs)}")
    print(f"- scenarios: {len(scenarios)}")
    print(f"- ok: {payload['ok']}")
    print(f"- json_out: {json_out}")
    print(f"- md_out: {md_out}")

    if failures:
        print("\nDrill failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nDrill passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
