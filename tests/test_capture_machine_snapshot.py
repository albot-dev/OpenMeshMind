import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import capture_machine_snapshot


class CaptureMachineSnapshotTests(unittest.TestCase):
    def _run_main(self, argv: list[str]) -> tuple[int, str]:
        stdout = io.StringIO()
        with mock.patch.object(sys, "argv", argv):
            with mock.patch("sys.stdout", stdout):
                code = capture_machine_snapshot.main()
        return code, stdout.getvalue()

    def test_capture_snapshot_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source = tmp / "source"
            source.mkdir(parents=True, exist_ok=True)
            for name in [
                "generality_metrics.json",
                "reproducibility_metrics.json",
                "smoke_summary.json",
                "main_track_status.json",
            ]:
                (source / name).write_text("{}", encoding="utf-8")
            (source / "mvp_readiness.json").write_text('{"ready": true}', encoding="utf-8")

            out_dir = tmp / "snapshots"
            code, output = self._run_main(
                [
                    "capture_machine_snapshot.py",
                    "--machine-id",
                    "test-machine-01",
                    "--label",
                    "ci-test",
                    "--source-root",
                    str(source),
                    "--out-dir",
                    str(out_dir),
                ]
            )
            self.assertEqual(code, 0)
            self.assertIn("Snapshot directory:", output)

            snapshots = [path for path in out_dir.iterdir() if path.is_dir()]
            self.assertEqual(len(snapshots), 1)
            snapshot = snapshots[0]
            self.assertTrue((snapshot / "generality_metrics.json").exists())
            self.assertTrue((snapshot / "reproducibility_metrics.json").exists())
            self.assertTrue((snapshot / "smoke_summary.json").exists())
            self.assertTrue((snapshot / "main_track_status.json").exists())
            self.assertTrue((snapshot / "snapshot_meta.json").exists())

            payload = json.loads((snapshot / "snapshot_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["machine_id"], "test-machine-01")
            self.assertEqual(payload["label"], "ci-test")

    def test_capture_snapshot_fails_when_required_artifact_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source = tmp / "source"
            source.mkdir(parents=True, exist_ok=True)
            (source / "generality_metrics.json").write_text("{}", encoding="utf-8")
            (source / "reproducibility_metrics.json").write_text("{}", encoding="utf-8")
            (source / "smoke_summary.json").write_text("{}", encoding="utf-8")
            # missing main_track_status.json

            out_dir = tmp / "snapshots"
            code, output = self._run_main(
                [
                    "capture_machine_snapshot.py",
                    "--source-root",
                    str(source),
                    "--out-dir",
                    str(out_dir),
                ]
            )
            self.assertEqual(code, 1)
            self.assertIn("Missing required artifact(s):", output)


if __name__ == "__main__":
    unittest.main()
