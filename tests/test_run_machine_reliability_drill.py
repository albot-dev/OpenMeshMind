import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import run_machine_reliability_drill


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
    }


class RunMachineReliabilityDrillTests(unittest.TestCase):
    def _run_main(self, argv: list[str]) -> tuple[int, str]:
        stdout = io.StringIO()
        with mock.patch.object(sys, "argv", argv):
            with mock.patch("sys.stdout", stdout):
                code = run_machine_reliability_drill.main()
        return code, stdout.getvalue()

    def _write_snapshot(self, root: Path, name: str) -> Path:
        snapshot = root / name
        snapshot.mkdir(parents=True, exist_ok=True)
        (snapshot / "snapshot_meta.json").write_text(
            json.dumps({"schema_version": 1, "machine_id": name}),
            encoding="utf-8",
        )
        (snapshot / "generality_metrics.json").write_text(
            json.dumps({"schema_version": 1, "aggregate": {"overall_score": 0.9}}),
            encoding="utf-8",
        )
        (snapshot / "reproducibility_metrics.json").write_text(
            json.dumps({"schema_version": 1, "summary": {"overall_score": {"mean": 0.9, "std": 0.02}}}),
            encoding="utf-8",
        )
        (snapshot / "smoke_summary.json").write_text(
            json.dumps({"ok": True, "total_duration_sec": 10.0}),
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

    def test_drill_passes_for_expected_scenarios(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            s1 = self._write_snapshot(tmp, "machine-a")
            s2 = self._write_snapshot(tmp, "machine-b")
            s3 = self._write_snapshot(tmp, "machine-c")
            out_dir = tmp / "drill_out"
            out_json = tmp / "drill.json"
            out_md = tmp / "drill.md"

            code, output = self._run_main(
                [
                    "run_machine_reliability_drill.py",
                    "--snapshot-dir",
                    str(s1),
                    "--snapshot-dir",
                    str(s2),
                    "--snapshot-dir",
                    str(s3),
                    "--snapshot-glob",
                    "",
                    "--skip-checkers",
                    "--out-dir",
                    str(out_dir),
                    "--json-out",
                    str(out_json),
                    "--md-out",
                    str(out_md),
                ]
            )
            self.assertEqual(code, 0)
            self.assertIn("Drill passed.", output)
            self.assertTrue(out_json.exists())
            self.assertTrue(out_md.exists())

            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertTrue(payload["ok"])
            self.assertEqual(len(payload["scenarios"]), 3)

    def test_drill_fails_when_not_enough_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            s1 = self._write_snapshot(tmp, "machine-a")
            s2 = self._write_snapshot(tmp, "machine-b")

            code, output = self._run_main(
                [
                    "run_machine_reliability_drill.py",
                    "--snapshot-dir",
                    str(s1),
                    "--snapshot-dir",
                    str(s2),
                    "--snapshot-glob",
                    "",
                    "--min-snapshots",
                    "3",
                ]
            )
            self.assertEqual(code, 1)
            self.assertIn("Need at least 3 snapshots", output)


if __name__ == "__main__":
    unittest.main()
