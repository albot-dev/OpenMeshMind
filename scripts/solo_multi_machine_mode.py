#!/usr/bin/env python3
"""
Ingest onboarding bundles from multiple machines and update cohort artifacts automatically.
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ImportedNode:
    node_id: str
    bundle_path: str
    metrics_path: Path
    profile: dict[str, object]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def rel_or_abs(path: Path) -> str:
    if path.is_relative_to(ROOT):
        return str(path.relative_to(ROOT))
    return str(path)


def run_cmd(args: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        args,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.returncode, proc.stdout.strip()


def unique_bundle_paths(bundle_args: list[str], bundles_glob: str) -> list[Path]:
    paths: list[Path] = []
    for item in bundle_args:
        paths.append(resolve(item))
    if bundles_glob:
        for item in sorted(glob.glob(str(resolve(bundles_glob)))):
            paths.append(Path(item))

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def select_member(
    members: list[tarfile.TarInfo],
    suffix: str,
    node_id: str | None = None,
) -> tarfile.TarInfo | None:
    candidates = [m for m in members if m.isfile() and m.name.endswith(suffix)]
    if not candidates:
        return None
    if node_id:
        tagged = [m for m in candidates if f"/{node_id}/" in m.name]
        if tagged:
            return sorted(tagged, key=lambda x: x.name)[0]
    return sorted(candidates, key=lambda x: x.name)[0]


def read_member_bytes(tar: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    extracted = tar.extractfile(member)
    if extracted is None:
        raise ValueError(f"unable to extract bundle member: {member.name}")
    return extracted.read()


def read_member_json(tar: tarfile.TarFile, member: tarfile.TarInfo) -> dict[str, object]:
    raw = read_member_bytes(tar=tar, member=member)
    return json.loads(raw.decode("utf-8"))


def import_bundle(bundle_path: Path, nodes_dir: Path) -> ImportedNode:
    with tarfile.open(bundle_path, "r:gz") as tar:
        members = tar.getmembers()
        metrics_member = select_member(members=members, suffix="pilot_metrics.json")
        if metrics_member is None:
            raise ValueError("bundle missing pilot_metrics.json")

        metrics_payload = read_member_json(tar=tar, member=metrics_member)
        node_id = str(metrics_payload.get("node", {}).get("node_id", "")).strip()
        if not node_id:
            raise ValueError("bundle metrics missing node.node_id")

        node_dir = nodes_dir / node_id
        node_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = node_dir / "pilot_metrics.json"
        metrics_path.write_bytes(read_member_bytes(tar=tar, member=metrics_member))

        optional_targets = {
            "node_state.json": "node_state.json",
            "node_runner.log": "node_runner.log",
            "node_profile.json": "node_profile.json",
            "node_config.json": "onboarding_node_config.json",
        }
        for suffix, out_name in optional_targets.items():
            member = select_member(members=members, suffix=suffix, node_id=node_id)
            if member is None:
                continue
            (node_dir / out_name).write_bytes(read_member_bytes(tar=tar, member=member))

        profile_path = node_dir / "node_profile.json"
        if profile_path.exists():
            with profile_path.open("r", encoding="utf-8") as f:
                profile = json.load(f)
        else:
            profile = {}

        return ImportedNode(
            node_id=node_id,
            bundle_path=rel_or_abs(bundle_path.resolve()),
            metrics_path=metrics_path,
            profile=profile,
        )


def load_or_init_manifest(manifest_path: Path, cohort_id: str) -> dict[str, object]:
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        example_path = ROOT / "pilot" / "cohort_manifest.example.json"
        if example_path.exists():
            with example_path.open("r", encoding="utf-8") as f:
                manifest = json.load(f)
        else:
            manifest = {"schema_version": 1, "cohort_id": "", "generated_utc": "", "nodes": []}

    manifest["schema_version"] = 1
    if cohort_id:
        manifest["cohort_id"] = cohort_id
    if not manifest.get("cohort_id"):
        manifest["cohort_id"] = f"pilot-cohort-{datetime.now(timezone.utc).strftime('%Y%m%d')}"
    manifest["generated_utc"] = utc_now_iso()
    if not isinstance(manifest.get("nodes"), list):
        manifest["nodes"] = []
    return manifest


def int_or_default(value: object, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    return default


def str_or_default(value: object, default: str) -> str:
    if isinstance(value, str) and value.strip():
        return value
    return default


def upsert_manifest_node(
    existing: dict[str, object] | None,
    imported: ImportedNode,
    checked_utc: str,
) -> dict[str, object]:
    base = existing.copy() if existing else {}
    profile = imported.profile if isinstance(imported.profile, dict) else {}

    region = str_or_default(profile.get("region"), str_or_default(base.get("region"), "unknown"))
    hardware_tier = str_or_default(
        profile.get("hardware_tier"),
        str_or_default(base.get("hardware_tier"), "unknown"),
    )
    network_tier = str_or_default(
        profile.get("network_tier"),
        str_or_default(base.get("network_tier"), "unknown"),
    )
    cpu_cores = int_or_default(profile.get("cpu_cores"), int_or_default(base.get("cpu_cores"), 1))
    memory_gb = int_or_default(profile.get("memory_gb"), int_or_default(base.get("memory_gb"), 1))

    if cpu_cores <= 0:
        cpu_cores = 1
    if memory_gb <= 0:
        memory_gb = 1

    return {
        "node_id": imported.node_id,
        "region": region,
        "hardware_tier": hardware_tier,
        "cpu_cores": cpu_cores,
        "memory_gb": memory_gb,
        "network_tier": network_tier,
        "onboarding_status": "passed",
        "onboarding_checked_utc": checked_utc,
        "metrics_path": rel_or_abs(imported.metrics_path),
        "failure_reason": "",
        "source_bundle": imported.bundle_path,
    }


def run_or_fail(cmd: list[str], label: str) -> None:
    code, out = run_cmd(cmd)
    if out:
        print(out)
    if code != 0:
        raise RuntimeError(f"{label} failed with exit code {code}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        action="append",
        default=[],
        help="Path to onboarding bundle .tgz (repeatable).",
    )
    parser.add_argument(
        "--bundles-glob",
        default="pilot/submissions/*_onboarding_*.tgz",
        help="Glob for onboarding bundles (default: pilot/submissions/*_onboarding_*.tgz).",
    )
    parser.add_argument(
        "--manifest",
        default="pilot/cohort_manifest.json",
        help="Path to cohort manifest JSON.",
    )
    parser.add_argument(
        "--cohort-id",
        default="",
        help="Optional cohort_id override.",
    )
    parser.add_argument(
        "--nodes-dir",
        default="pilot/nodes",
        help="Directory to place imported node artifacts.",
    )
    parser.add_argument(
        "--summary-json-out",
        default="pilot/cohort_onboarding_summary.json",
        help="Path for onboarding summary JSON output.",
    )
    parser.add_argument(
        "--min-nodes",
        type=int,
        default=1,
        help="Minimum nodes required for cohort manifest validation (default: 1).",
    )
    parser.add_argument(
        "--min-passed",
        type=int,
        default=1,
        help="Minimum passed nodes required for cohort manifest validation (default: 1).",
    )
    parser.add_argument(
        "--require-metrics-files",
        action="store_true",
        help="Require metrics files for passed nodes during manifest validation.",
    )
    parser.add_argument(
        "--cohort-metrics-out",
        default="pilot/pilot_cohort_metrics.json",
        help="Output path for aggregated cohort metrics.",
    )
    parser.add_argument(
        "--pilot-status-out",
        default="reports/pilot_status.md",
        help="Output path for pilot status report markdown.",
    )
    parser.add_argument(
        "--pilot-status-bundle-out",
        default="reports/pilot_artifacts.tgz",
        help="Output path for pilot status artifact bundle.",
    )
    parser.add_argument(
        "--no-pipeline",
        action="store_true",
        help="Skip cohort metrics/status report generation after import.",
    )
    args = parser.parse_args()

    bundle_paths = unique_bundle_paths(bundle_args=args.bundle, bundles_glob=args.bundles_glob)
    if not bundle_paths:
        print("No bundles found. Provide --bundle and/or --bundles-glob.")
        return 1

    missing = [path for path in bundle_paths if not path.exists()]
    if missing:
        print("Missing bundle file(s):")
        for item in missing:
            print(f"- {item}")
        return 1

    nodes_dir = resolve(args.nodes_dir)
    nodes_dir.mkdir(parents=True, exist_ok=True)

    imported_nodes: list[ImportedNode] = []
    for bundle in bundle_paths:
        try:
            imported = import_bundle(bundle_path=bundle, nodes_dir=nodes_dir)
        except (ValueError, json.JSONDecodeError, tarfile.TarError) as exc:
            print(f"Failed to import bundle {bundle}: {exc}")
            return 1
        imported_nodes.append(imported)

    checked_utc = utc_now_iso()
    manifest_path = resolve(args.manifest)
    manifest = load_or_init_manifest(manifest_path=manifest_path, cohort_id=args.cohort_id)

    nodes = manifest.get("nodes", [])
    node_index: dict[str, dict[str, object]] = {}
    for entry in nodes:
        if isinstance(entry, dict):
            node_id = str(entry.get("node_id", "")).strip()
            if node_id:
                node_index[node_id] = entry

    for imported in imported_nodes:
        updated = upsert_manifest_node(
            existing=node_index.get(imported.node_id),
            imported=imported,
            checked_utc=checked_utc,
        )
        node_index[imported.node_id] = updated

    manifest["nodes"] = [node_index[key] for key in sorted(node_index.keys())]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(f"Imported bundles: {len(imported_nodes)}")
    for node in imported_nodes:
        print(f"- {node.node_id} <= {node.bundle_path}")
    print(f"Updated cohort manifest: {manifest_path}")

    check_cmd = [
        sys.executable,
        "scripts/check_cohort_manifest.py",
        str(manifest_path),
        "--min-nodes",
        str(args.min_nodes),
        "--min-passed",
        str(args.min_passed),
        "--summary-json-out",
        str(resolve(args.summary_json_out)),
    ]
    if args.require_metrics_files:
        check_cmd.append("--require-metrics-files")
    try:
        run_or_fail(check_cmd, "cohort manifest validation")
    except RuntimeError as exc:
        print(exc)
        return 1

    if args.no_pipeline:
        return 0

    metrics_glob = str(nodes_dir / "*" / "pilot_metrics.json")
    try:
        run_or_fail(
            [
                sys.executable,
                "scripts/build_pilot_cohort_metrics.py",
                "--metrics-glob",
                metrics_glob,
                "--json-out",
                args.cohort_metrics_out,
            ],
            "build pilot cohort metrics",
        )
        run_or_fail(
            [
                sys.executable,
                "scripts/check_pilot_cohort.py",
                args.cohort_metrics_out,
                "--min-node-count",
                str(args.min_nodes),
            ],
            "pilot cohort validation",
        )
        run_or_fail(
            [
                sys.executable,
                "scripts/generate_pilot_status_report.py",
                "--cohort-metrics",
                args.cohort_metrics_out,
                "--out",
                args.pilot_status_out,
                "--bundle-out",
                args.pilot_status_bundle_out,
            ],
            "pilot status generation",
        )
    except RuntimeError as exc:
        print(exc)
        return 1

    print("Solo multi-machine import pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
