import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import build_machine_comparison_report


def _generality_payload(overall: float) -> dict[str, object]:
    return {
        "schema_version": 1,
        "tasks": {
            "classification": {"metrics": {"accuracy": 0.90, "macro_f1": 0.88}},
            "retrieval": {"metrics": {"recall_at_1": 0.82, "mrr": 0.86}},
            "long_context_retrieval": {"metrics": {"recall_at_1": 0.88, "mrr": 0.90}},
            "instruction_following": {"metrics": {"pass_rate": 0.90}},
            "conversation_continuity": {"metrics": {"pass_rate": 0.92}},
            "tool_use": {"metrics": {"pass_rate": 1.00}},
            "multi_step_tool_use": {"metrics": {"pass_rate": 0.89, "chain_pass_rate": 0.80}},
            "distributed_reference": {
                "metrics": {"int8_accuracy_drop": 0.03, "int8_comm_savings_percent": 70.0}
            },
            "adapter_reference": {
                "metrics": {"int8_accuracy_drop": 0.10, "int8_comm_savings_percent": 65.0}
            },
        },
        "aggregate": {"overall_score": overall},
        "resources": {"total_wall_clock_sec": 9.2},
    }


def _repro_payload(mean: float, std: float) -> dict[str, object]:
    return {
        "schema_version": 1,
        "runs": [{"seed": 7}, {"seed": 17}, {"seed": 27}],
        "summary": {
            "overall_score": {"mean": mean, "std": std},
            "classification_accuracy": {"mean": 0.92},
            "retrieval_recall_at_1": {"mean": 0.82},
            "long_context_recall_at_1": {"mean": 0.84},
            "long_context_mrr": {"mean": 0.87},
            "instruction_pass_rate": {"mean": 0.80},
            "conversation_pass_rate": {"mean": 0.84},
            "tool_pass_rate": {"mean": 0.95},
            "multi_step_tool_pass_rate": {"mean": 0.88},
            "multi_step_tool_chain_pass_rate": {"mean": 0.78},
            "int8_accuracy_drop": {"mean": 0.04},
            "int8_comm_savings_percent": {"mean": 72.0},
        },
    }


def _status_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "summary": {
            "total_goals": 4,
            "completed_goals": 4,
            "required_sub_goals": 9,
            "completed_required_sub_goals": 9,
            "all_done": True,
        },
        "goals": [
            {"id": "g1_cpu_federated_foundation", "status": "done"},
            {"id": "g2_generality_reproducibility", "status": "done"},
            {"id": "g3_accessibility_ops", "status": "done"},
            {"id": "g4_decentralization_fairness", "status": "done"},
        ],
    }


class BuildMachineComparisonReportTests(unittest.TestCase):
    def _run_main(self, argv: list[str]) -> tuple[int, str]:
        stdout = io.StringIO()
        with mock.patch.object(sys, "argv", argv):
            with mock.patch("sys.stdout", stdout):
                code = build_machine_comparison_report.main()
        return code, stdout.getvalue()

    def _write_snapshot(
        self,
        root: Path,
        name: str,
        *,
        overall: float,
        repro_mean: float,
        repro_std: float,
        smoke_ok: bool = True,
    ) -> Path:
        snapshot = root / name
        snapshot.mkdir(parents=True, exist_ok=True)
        (snapshot / "snapshot_meta.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "machine_id": name,
                    "label": name,
                }
            ),
            encoding="utf-8",
        )
        (snapshot / "generality_metrics.json").write_text(
            json.dumps(_generality_payload(overall=overall)),
            encoding="utf-8",
        )
        (snapshot / "reproducibility_metrics.json").write_text(
            json.dumps(_repro_payload(mean=repro_mean, std=repro_std)),
            encoding="utf-8",
        )
        (snapshot / "smoke_summary.json").write_text(
            json.dumps({"ok": smoke_ok, "total_duration_sec": 12.0}),
            encoding="utf-8",
        )
        (snapshot / "main_track_status.json").write_text(
            json.dumps(_status_payload()),
            encoding="utf-8",
        )
        (snapshot / "mvp_readiness.json").write_text(
            json.dumps({"schema_version": 1, "ready": True}),
            encoding="utf-8",
        )
        return snapshot

    def test_build_machine_comparison_passes_for_three_green_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            s1 = self._write_snapshot(tmp, "machine-a", overall=0.86, repro_mean=0.85, repro_std=0.03)
            s2 = self._write_snapshot(tmp, "machine-b", overall=0.88, repro_mean=0.87, repro_std=0.04)
            s3 = self._write_snapshot(tmp, "machine-c", overall=0.89, repro_mean=0.88, repro_std=0.02)
            out_json = tmp / "comparison.json"
            out_md = tmp / "comparison.md"

            code, output = self._run_main(
                [
                    "build_machine_comparison_report.py",
                    "--snapshot-dir",
                    str(s1),
                    "--snapshot-dir",
                    str(s2),
                    "--snapshot-dir",
                    str(s3),
                    "--snapshot-glob",
                    "",
                    "--require-mvp-readiness",
                    "--json-out",
                    str(out_json),
                    "--md-out",
                    str(out_md),
                ]
            )
            self.assertEqual(code, 0)
            self.assertIn("Comparison passed.", output)
            self.assertTrue(out_json.exists())
            self.assertTrue(out_md.exists())

            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertTrue(payload["summary"]["ready"])
            self.assertEqual(payload["summary"]["snapshot_count"], 3)
            self.assertEqual(payload["summary"]["failing_count"], 0)

    def test_build_machine_comparison_fails_when_snapshot_is_not_green(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            s1 = self._write_snapshot(tmp, "machine-a", overall=0.86, repro_mean=0.85, repro_std=0.03)
            s2 = self._write_snapshot(
                tmp,
                "machine-b",
                overall=0.88,
                repro_mean=0.87,
                repro_std=0.04,
                smoke_ok=False,
            )
            s3 = self._write_snapshot(tmp, "machine-c", overall=0.89, repro_mean=0.88, repro_std=0.02)
            out_json = tmp / "comparison_fail.json"
            out_md = tmp / "comparison_fail.md"

            code, output = self._run_main(
                [
                    "build_machine_comparison_report.py",
                    "--snapshot-dir",
                    str(s1),
                    "--snapshot-dir",
                    str(s2),
                    "--snapshot-dir",
                    str(s3),
                    "--snapshot-glob",
                    "",
                    "--skip-checkers",
                    "--json-out",
                    str(out_json),
                    "--md-out",
                    str(out_md),
                ]
            )
            self.assertEqual(code, 1)
            self.assertIn("Comparison failed:", output)
            self.assertIn("snapshot_failures=1", output)

    def test_build_machine_comparison_fails_on_range_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            s1 = self._write_snapshot(tmp, "machine-a", overall=0.60, repro_mean=0.60, repro_std=0.03)
            s2 = self._write_snapshot(tmp, "machine-b", overall=0.89, repro_mean=0.87, repro_std=0.04)
            s3 = self._write_snapshot(tmp, "machine-c", overall=0.90, repro_mean=0.88, repro_std=0.02)
            out_json = tmp / "comparison_range_fail.json"
            out_md = tmp / "comparison_range_fail.md"

            code, output = self._run_main(
                [
                    "build_machine_comparison_report.py",
                    "--snapshot-dir",
                    str(s1),
                    "--snapshot-dir",
                    str(s2),
                    "--snapshot-dir",
                    str(s3),
                    "--snapshot-glob",
                    "",
                    "--max-generality-overall-range",
                    "0.05",
                    "--skip-checkers",
                    "--json-out",
                    str(out_json),
                    "--md-out",
                    str(out_md),
                ]
            )
            self.assertEqual(code, 1)
            self.assertIn("Comparison failed:", output)
            self.assertIn("generality_overall_score_range", output)


if __name__ == "__main__":
    unittest.main()
