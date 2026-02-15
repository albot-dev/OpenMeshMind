#!/usr/bin/env python3
"""
Validate a provenance manifest and optionally verify checksums on disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def resolve(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_json", help="Path to provenance manifest JSON.")
    parser.add_argument("--expected-schema-version", type=int, default=1)
    parser.add_argument("--min-artifacts", type=int, default=1)
    parser.add_argument(
        "--verify-sha256",
        action="store_true",
        help="Recompute SHA256 for each artifact and enforce exact match.",
    )
    args = parser.parse_args()

    path = resolve(args.manifest_json)
    with path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    failures: list[str] = []

    if manifest.get("schema_version") != args.expected_schema_version:
        failures.append(
            f"schema_version={manifest.get('schema_version')} (expected {args.expected_schema_version})"
        )

    label = manifest.get("label")
    if not isinstance(label, str) or not label.strip():
        failures.append("label: expected non-empty string")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        failures.append("artifacts: expected array")
        artifacts = []

    if len(artifacts) < args.min_artifacts:
        failures.append(f"artifact_count={len(artifacts)} < min_artifacts={args.min_artifacts}")

    for idx, row in enumerate(artifacts):
        if not isinstance(row, dict):
            failures.append(f"artifacts[{idx}]: expected object")
            continue
        for key in ["path", "sha256", "size_bytes"]:
            if key not in row:
                failures.append(f"artifacts[{idx}].{key}: missing")

        rel_path = row.get("path")
        if not isinstance(rel_path, str) or not rel_path:
            failures.append(f"artifacts[{idx}].path: expected non-empty string")
            continue

        artifact_path = resolve(rel_path)
        if not artifact_path.exists():
            failures.append(f"artifacts[{idx}].path: missing file {artifact_path}")
            continue

        size_bytes = row.get("size_bytes")
        if not isinstance(size_bytes, int) or size_bytes < 0:
            failures.append(f"artifacts[{idx}].size_bytes: expected integer >= 0")
        else:
            actual_size = artifact_path.stat().st_size
            if actual_size != size_bytes:
                failures.append(
                    f"artifacts[{idx}].size_bytes mismatch manifest={size_bytes} actual={actual_size}"
                )

        sha = row.get("sha256")
        if not isinstance(sha, str) or len(sha) != 64:
            failures.append(f"artifacts[{idx}].sha256: expected 64-char hex string")
        elif args.verify_sha256:
            actual_sha = sha256_file(artifact_path)
            if actual_sha != sha:
                failures.append(
                    f"artifacts[{idx}].sha256 mismatch manifest={sha} actual={actual_sha}"
                )

    print("Provenance manifest validation summary")
    print(f"- schema_version: {manifest.get('schema_version')}")
    print(f"- label: {manifest.get('label')}")
    print(f"- artifact_count: {len(artifacts)}")

    if failures:
        print("\nValidation failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nValidation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
