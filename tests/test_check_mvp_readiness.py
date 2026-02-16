import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import check_mvp_readiness


class CheckMvpReadinessTests(unittest.TestCase):
    def _write_json(self, directory: Path, name: str, payload: object) -> Path:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _write_text(self, directory: Path, name: str, text: str) -> Path:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def _run_main(self, argv: list[str]) -> tuple[int, str]:
        stdout = io.StringIO()
        with mock.patch.object(sys, "argv", argv):
            with mock.patch("sys.stdout", stdout):
                code = check_mvp_readiness.main()
        return code, stdout.getvalue()

    def _write_required_files(self, root: Path) -> None:
        for artifact in check_mvp_readiness.DEFAULT_REQUIRED_METRIC_ARTIFACTS:
            self._write_json(root, artifact, {"schema_version": 1})
        for doc in check_mvp_readiness.DEFAULT_REQUIRED_DOCS:
            self._write_text(root, doc, "# ready\n")

    def _write_main_track_status(
        self,
        root: Path,
        *,
        goal_2_status: str = "done",
        goal_4_status: str = "pending",
        required_sub_goals: int = 7,
        completed_required_sub_goals: int = 7,
        completed_goals: int = 3,
        total_goals: int = 4,
        all_done: bool = False,
    ) -> None:
        self._write_json(
            root,
            "main_track_status.json",
            {
                "schema_version": 1,
                "summary": {
                    "required_sub_goals": required_sub_goals,
                    "completed_required_sub_goals": completed_required_sub_goals,
                    "completed_goals": completed_goals,
                    "total_goals": total_goals,
                    "all_done": all_done,
                },
                "goals": [
                    {"id": "g1_cpu_federated_foundation", "status": "done"},
                    {"id": "g2_generality_reproducibility", "status": goal_2_status},
                    {"id": "g3_accessibility_ops", "status": "done"},
                    {"id": "g4_decentralization_fairness", "status": goal_4_status},
                ],
            },
        )

    def test_check_mvp_readiness_pass_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_required_files(root)
            self._write_main_track_status(root)

            code, output = self._run_main(
                [
                    "check_mvp_readiness.py",
                    "--repo-root",
                    str(root),
                    "--json-out",
                    "reports/mvp_readiness.json",
                ]
            )
            self.assertEqual(code, 0)
            self.assertIn("Readiness check passed.", output)

            readiness_report = root / "reports/mvp_readiness.json"
            self.assertTrue(readiness_report.exists())
            payload = json.loads(readiness_report.read_text(encoding="utf-8"))
            self.assertTrue(payload["ready"])

    def test_check_mvp_readiness_fails_when_metric_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_required_files(root)
            self._write_main_track_status(root)

            (root / "benchmark_metrics.json").unlink()

            code, output = self._run_main(
                [
                    "check_mvp_readiness.py",
                    "--repo-root",
                    str(root),
                ]
            )
            self.assertEqual(code, 1)
            self.assertIn("Readiness check failed:", output)
            self.assertIn("missing metric artifacts", output)

    def test_check_mvp_readiness_fails_when_required_goal_not_done(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_required_files(root)
            self._write_main_track_status(
                root,
                goal_2_status="pending",
                completed_required_sub_goals=6,
            )

            code, output = self._run_main(
                [
                    "check_mvp_readiness.py",
                    "--repo-root",
                    str(root),
                ]
            )
            self.assertEqual(code, 1)
            self.assertIn("required goal not done: g2_generality_reproducibility", output)

    def test_check_mvp_readiness_threshold_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_required_files(root)
            self._write_main_track_status(root, all_done=False)

            (root / "docs/COMING_GOALS.md").unlink()

            code_ok, output_ok = self._run_main(
                [
                    "check_mvp_readiness.py",
                    "--repo-root",
                    str(root),
                    "--max-missing-docs",
                    "1",
                ]
            )
            self.assertEqual(code_ok, 0)
            self.assertIn("Readiness check passed.", output_ok)

            code_fail, output_fail = self._run_main(
                [
                    "check_mvp_readiness.py",
                    "--repo-root",
                    str(root),
                    "--max-missing-docs",
                    "1",
                    "--require-all-goals-done",
                ]
            )
            self.assertEqual(code_fail, 1)
            self.assertIn("main_track_status.summary.all_done", output_fail)


if __name__ == "__main__":
    unittest.main()
