#!/usr/bin/env python3
"""
Capture a standardized local machine snapshot for cross-machine reliability comparison.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import socket
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_REQUIRED_ARTIFACTS = [
    "generality_metrics.json",
    "reproducibility_metrics.json",
    "smoke_summary.json",
    "main_track_status.json",
]

DEFAULT_OPTIONAL_ARTIFACTS = [
    "mvp_readiness.json",
    "baseline_metrics.json",
    "classification_metrics.json",
    "adapter_intent_metrics.json",
    "benchmark_metrics.json",
]


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


def safe_machine_id(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value.strip())
    cleaned = cleaned.strip("-_")
    return cleaned or "machine"


def rel_to_root(path: Path) -> str:
    if path.is_relative_to(ROOT):
        return str(path.relative_to(ROOT))
    return str(path)


def copy_artifacts(source_root: Path, dest_root: Path, artifacts: list[str]) -> list[dict[str, str]]:
    copied: list[dict[str, str]] = []
    for rel in artifacts:
        src = source_root / rel
        if not src.exists():
            continue
        out = dest_root / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)
        copied.append({"source": str(src), "path": rel_to_root(out)})
    return copied


def build_bundle(bundle_path: Path, snapshot_dir: Path) -> int:
    added = 0
    with tarfile.open(bundle_path, "w:gz") as tar:
        for path in sorted(snapshot_dir.rglob("*")):
            if not path.is_file():
                continue
            tar.add(path, arcname=str(path.relative_to(snapshot_dir.parent)))
            added += 1
    return added


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--machine-id",
        default="",
        help="Stable machine identifier (default: hostname-based).",
    )
    parser.add_argument(
        "--label",
        default="",
        help="Optional label for this snapshot (for example, laptop-01).",
    )
    parser.add_argument(
        "--source-root",
        default=".",
        help="Directory containing generated metric artifacts (default: repo root).",
    )
    parser.add_argument(
        "--out-dir",
        default="pilot/machine_snapshots",
        help="Output directory for captured snapshots.",
    )
    parser.add_argument(
        "--required-artifact",
        action="append",
        default=[],
        help=(
            "Required artifact relative path under source-root (repeatable). "
            "Defaults to core strict-gate artifacts."
        ),
    )
    parser.add_argument(
        "--optional-artifact",
        action="append",
        default=[],
        help="Optional artifact relative path under source-root (repeatable).",
    )
    parser.add_argument(
        "--bundle-out",
        default="",
        help="Optional .tgz path for snapshot bundle.",
    )
    args = parser.parse_args()

    source_root = resolve(args.source_root)
    if not source_root.exists():
        print(f"Source root not found: {source_root}")
        return 1

    machine_id = safe_machine_id(args.machine_id or socket.gethostname())
    required_artifacts = list(args.required_artifact) if args.required_artifact else list(
        DEFAULT_REQUIRED_ARTIFACTS
    )
    optional_artifacts = list(args.optional_artifact) if args.optional_artifact else list(
        DEFAULT_OPTIONAL_ARTIFACTS
    )

    missing_required = [item for item in required_artifacts if not (source_root / item).exists()]
    if missing_required:
        print("Missing required artifact(s):")
        for item in missing_required:
            print(f"- {item}")
        return 1

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot_dir = resolve(args.out_dir) / f"{stamp}_{machine_id}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    copied_required = copy_artifacts(
        source_root=source_root,
        dest_root=snapshot_dir,
        artifacts=required_artifacts,
    )
    copied_optional = copy_artifacts(
        source_root=source_root,
        dest_root=snapshot_dir,
        artifacts=optional_artifacts,
    )

    commit_code, commit_out = run_cmd(["git", "rev-parse", "HEAD"])
    commit = commit_out if commit_code == 0 else ""
    repo_code, repo_out = run_cmd(["git", "config", "--get", "remote.origin.url"])
    repo = repo_out if repo_code == 0 else ""

    metadata = {
        "schema_version": 1,
        "generated_utc": utc_now_iso(),
        "machine_id": machine_id,
        "label": args.label,
        "source_root": str(source_root),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "hostname": socket.gethostname(),
        "provenance": {
            "repo_origin": repo,
            "commit": commit,
        },
        "required_artifacts": copied_required,
        "optional_artifacts": copied_optional,
    }

    meta_path = snapshot_dir / "snapshot_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    print(f"Snapshot directory: {snapshot_dir}")
    print(f"Required artifacts copied: {len(copied_required)}")
    print(f"Optional artifacts copied: {len(copied_optional)}")
    print(f"Metadata: {meta_path}")

    if args.bundle_out:
        bundle_path = resolve(args.bundle_out)
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        count = build_bundle(bundle_path=bundle_path, snapshot_dir=snapshot_dir)
        print(f"Snapshot bundle: {bundle_path}")
        print(f"Bundle files: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
