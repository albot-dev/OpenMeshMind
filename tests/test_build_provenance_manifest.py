import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import build_provenance_manifest


class BuildProvenanceManifestTests(unittest.TestCase):
    def test_main_writes_manifest_with_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            artifact = tmp / "artifact.txt"
            artifact.write_text("hello provenance", encoding="utf-8")
            out = tmp / "manifest.json"

            argv = [
                "build_provenance_manifest.py",
                "--label",
                "unit-test",
                "--artifact",
                str(artifact),
                "--out",
                str(out),
            ]
            with mock.patch.object(sys, "argv", argv):
                code = build_provenance_manifest.main()

            self.assertEqual(code, 0)
            self.assertTrue(out.exists())

            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["label"], "unit-test")
            self.assertEqual(payload["artifact_count"], 1)

            row = payload["artifacts"][0]
            expected_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
            self.assertEqual(row["sha256"], expected_sha)
            self.assertEqual(row["size_bytes"], artifact.stat().st_size)

    def test_main_strict_missing_artifact_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            missing = tmp / "missing.txt"
            out = tmp / "manifest.json"

            argv = [
                "build_provenance_manifest.py",
                "--label",
                "unit-test",
                "--artifact",
                str(missing),
                "--out",
                str(out),
                "--strict",
            ]
            with mock.patch.object(sys, "argv", argv):
                code = build_provenance_manifest.main()

            self.assertEqual(code, 1)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
