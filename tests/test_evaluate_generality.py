import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import evaluate_generality


class EvaluateGeneralityTests(unittest.TestCase):
    def test_main_writes_expected_schema_without_distributed_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "generality.json"
            argv = [
                "evaluate_generality.py",
                "--skip-distributed-reference",
                "--json-out",
                str(out),
                "--quiet",
            ]
            with mock.patch.object(sys, "argv", argv):
                code = evaluate_generality.main()
            self.assertEqual(code, 0)
            self.assertTrue(out.exists())

            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)
            self.assertIn("classification", payload["tasks"])
            self.assertIn("retrieval", payload["tasks"])
            self.assertIn("instruction_following", payload["tasks"])
            self.assertIn("tool_use", payload["tasks"])
            self.assertNotIn("distributed_reference", payload["tasks"])
            self.assertGreaterEqual(payload["aggregate"]["overall_score"], 0.5)

    def test_distributed_reference_metrics_shape(self) -> None:
        report = evaluate_generality.evaluate_distributed_reference(seed=7)
        metrics = report["metrics"]
        self.assertIn("centralized_accuracy", metrics)
        self.assertIn("int8_accuracy", metrics)
        self.assertIn("int8_accuracy_drop", metrics)
        self.assertIn("int8_comm_savings_percent", metrics)
        self.assertGreaterEqual(metrics["int8_comm_savings_percent"], 0.0)
