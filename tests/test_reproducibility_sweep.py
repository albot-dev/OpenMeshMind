import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import reproducibility_sweep


class ReproducibilitySweepTests(unittest.TestCase):
    def test_main_writes_expected_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "repro.json"
            argv = [
                "reproducibility_sweep.py",
                "--seeds",
                "7,17",
                "--skip-distributed-reference",
                "--quiet",
                "--json-out",
                str(out),
            ]
            with mock.patch.object(sys, "argv", argv):
                code = reproducibility_sweep.main()
            self.assertEqual(code, 0)
            self.assertTrue(out.exists())

            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(len(payload["runs"]), 2)
            self.assertIn("overall_score", payload["summary"])
            self.assertIn("classification_accuracy", payload["summary"])
            self.assertNotIn("int8_accuracy_drop", payload["summary"])
