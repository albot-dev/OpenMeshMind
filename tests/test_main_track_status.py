import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import main_track_status


class MainTrackStatusTests(unittest.TestCase):
    def _write_json(self, directory: Path, name: str, payload: object) -> None:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _create_required_artifacts(self, root: Path) -> None:
        names = [
            "baseline_metrics.json",
            "classification_metrics.json",
            "adapter_intent_metrics.json",
            "generality_metrics.json",
            "reproducibility_metrics.json",
            "benchmark_metrics.json",
        ]
        for name in names:
            self._write_json(root, name, {"schema_version": 1})
        self._write_json(root, "smoke_summary.json", {"ok": True, "total_duration_sec": 12.0})

    def test_collect_status_all_done_without_fairness(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._create_required_artifacts(root)

            def runner(_root: Path, _cmd: list[str]) -> tuple[int, str]:
                return 0, "Validation passed."

            status = main_track_status.collect_status(
                root=root,
                require_fairness=False,
                require_smoke_summary=True,
                runner=runner,
            )
            summary = status["summary"]
            self.assertTrue(summary["all_done"])
            self.assertEqual(summary["completed_goals"], 4)
            self.assertEqual(summary["required_sub_goals"], 7)
            self.assertEqual(summary["completed_required_sub_goals"], 7)

    def test_collect_status_require_fairness_marks_missing_as_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._create_required_artifacts(root)

            def runner(_root: Path, _cmd: list[str]) -> tuple[int, str]:
                return 0, "Validation passed."

            status = main_track_status.collect_status(
                root=root,
                require_fairness=True,
                require_smoke_summary=True,
                runner=runner,
            )
            summary = status["summary"]
            self.assertFalse(summary["all_done"])

            fairness_goal = next(goal for goal in status["goals"] if goal["id"] == "g4_decentralization_fairness")
            self.assertEqual(fairness_goal["status"], "pending")

    def test_collect_status_failed_gate_marks_goal_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._create_required_artifacts(root)

            def runner(_root: Path, cmd: list[str]) -> tuple[int, str]:
                if "scripts/check_generality.py" in cmd:
                    return 1, "Validation failed: overall score low"
                return 0, "Validation passed."

            status = main_track_status.collect_status(
                root=root,
                require_fairness=False,
                require_smoke_summary=True,
                runner=runner,
            )
            summary = status["summary"]
            self.assertFalse(summary["all_done"])

            generality_goal = next(goal for goal in status["goals"] if goal["id"] == "g2_generality_reproducibility")
            self.assertEqual(generality_goal["status"], "blocked")

    def test_main_fail_on_incomplete_returns_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stdout = io.StringIO()
            with mock.patch.object(
                sys,
                "argv",
                [
                    "main_track_status.py",
                    "--repo-root",
                    str(root),
                    "--fail-on-incomplete",
                    "--json-out",
                    "main_track_status.json",
                    "--md-out",
                    "reports/main_track_status.md",
                ],
            ):
                with mock.patch("sys.stdout", stdout):
                    code = main_track_status.main()

            self.assertEqual(code, 1)
            self.assertTrue((root / "main_track_status.json").exists())
            self.assertTrue((root / "reports/main_track_status.md").exists())


if __name__ == "__main__":
    unittest.main()
