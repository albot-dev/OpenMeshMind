#!/usr/bin/env python3
"""
Build a machine-readable provenance manifest with checksums for artifact files.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_paths(artifacts: list[str], globs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in artifacts:
        paths.append(resolve(item))
    for pattern in globs:
        for item in sorted(glob.glob(str(resolve(pattern)))):
            paths.append(Path(item))

    out: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def build_manifest(
    *,
    label: str,
    repo: str,
    artifacts: list[Path],
) -> dict[str, object]:
    _, commit = run_cmd(["git", "rev-parse", "HEAD"])
    _, branch = run_cmd(["git", "branch", "--show-current"])

    rows: list[dict[str, object]] = []
    for path in sorted(artifacts, key=lambda p: rel_or_abs(p)):
        stat = path.stat()
        rows.append(
            {
                "path": rel_or_abs(path),
                "sha256": sha256_file(path),
                "size_bytes": stat.st_size,
                "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            }
        )

    return {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "repo": repo,
        "git": {
            "commit": commit,
            "branch": branch,
        },
        "artifact_count": len(rows),
        "artifacts": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="Logical manifest label.")
    parser.add_argument("--repo", default="", help="Repository owner/name (auto-detected if omitted).")
    parser.add_argument("--artifact", action="append", default=[], help="Artifact file path (repeatable).")
    parser.add_argument("--glob", action="append", default=[], help="Artifact glob pattern (repeatable).")
    parser.add_argument("--out", default="provenance_manifest.json", help="Output manifest JSON path.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when any requested artifact path does not exist.",
    )
    args = parser.parse_args()

    paths = unique_paths(artifacts=args.artifact, globs=args.glob)
    if not paths:
        print("No artifacts selected. Use --artifact and/or --glob.")
        return 1

    missing = [path for path in paths if not path.exists()]
    if missing and args.strict:
        print("Missing artifact files:")
        for item in missing:
            print(f"- {rel_or_abs(item)}")
        return 1

    existing = [path for path in paths if path.exists() and path.is_file()]
    if not existing:
        print("No existing artifact files found.")
        return 1

    repo = args.repo or default_repo()
    manifest = build_manifest(label=args.label, repo=repo, artifacts=existing)

    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(f"Provenance manifest written to: {out_path}")
    print(f"Artifacts captured: {manifest['artifact_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
