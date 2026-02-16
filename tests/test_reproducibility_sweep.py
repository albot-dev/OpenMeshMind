import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import reproducibility_sweep


class ReproducibilitySweepTests(unittest.TestCase):
    def _fake_run_seed(
        self,
        seed: int,
        top_k: int,
        long_context_top_k: int,
        include_distributed_reference: bool,
    ) -> dict[str, float]:
        base = float(seed) / 1000.0
        run = {
            "seed": seed,
            "overall_score": 0.82 + base,
            "classification_accuracy": 0.90 + base,
            "classification_macro_f1": 0.89 + base,
            "retrieval_recall_at_1": 0.80 + base,
            "retrieval_mrr": 0.83 + base,
            "long_context_recall_at_1": 0.84 + base,
            "long_context_mrr": 0.86 + base,
            "instruction_pass_rate": 0.78 + base,
            "conversation_pass_rate": 0.81 + base,
            "tool_pass_rate": 0.94 + base,
            "multi_step_tool_pass_rate": 0.88 + base,
            "multi_step_tool_chain_pass_rate": 0.77 + base,
        }
        if include_distributed_reference:
            run["int8_accuracy_drop"] = 0.05
            run["int8_comm_savings_percent"] = 65.0
        return run

    def test_main_writes_expected_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "repro.json"
            argv = [
                "reproducibility_sweep.py",
                "--skip-distributed-reference",
                "--quiet",
                "--json-out",
                str(out),
            ]
            with mock.patch.object(sys, "argv", argv):
                with mock.patch.object(
                    reproducibility_sweep,
                    "run_seed",
                    side_effect=self._fake_run_seed,
                ):
                    code = reproducibility_sweep.main()
            self.assertEqual(code, 0)
            self.assertTrue(out.exists())

            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["config"]["seeds"], [7, 17, 27])
            self.assertEqual(len(payload["runs"]), 3)
            self.assertIn("overall_score", payload["summary"])
            self.assertIn("classification_accuracy", payload["summary"])
            self.assertIn("long_context_recall_at_1", payload["summary"])
            self.assertIn("long_context_mrr", payload["summary"])
            self.assertIn("conversation_pass_rate", payload["summary"])
            self.assertIn("multi_step_tool_pass_rate", payload["summary"])
            self.assertIn("multi_step_tool_chain_pass_rate", payload["summary"])
            self.assertNotIn("int8_accuracy_drop", payload["summary"])

    def test_main_fails_early_when_seed_count_is_below_default_minimum(self) -> None:
        argv = [
            "reproducibility_sweep.py",
            "--seeds",
            "7,17",
            "--quiet",
        ]
        stdout = io.StringIO()
        with mock.patch.object(sys, "argv", argv):
            with mock.patch("sys.stdout", stdout):
                with mock.patch.object(reproducibility_sweep, "run_seed") as mocked_run_seed:
                    code = reproducibility_sweep.main()
        self.assertEqual(code, 1)
        mocked_run_seed.assert_not_called()
        self.assertIn("Need at least 3 seeds", stdout.getvalue())
