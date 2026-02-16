import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import build_pilot_cohort_metrics


def _valid_metrics(node_id: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "timestamp_utc": "2026-02-16T10:00:00+00:00",
        "node": {
            "node_id": node_id,
            "mode": "reduced",
            "python_version": "3.10.10",
            "platform": "darwin",
        },
        "health": {
            "last_cycle_ok": True,
            "cycle_duration_sec": 10.0,
            "step_count": 8,
            "uptime_ratio_24h": 0.99,
        },
        "quality": {
            "classification_accuracy": 0.90,
            "classification_macro_f1": 0.89,
            "utility_fedavg_int8_accuracy": 0.88,
            "utility_fedavg_int8_macro_f1": 0.87,
        },
        "accessibility": {
            "benchmark_total_runtime_sec": 9.5,
            "max_peak_rss_bytes": 100_000_000,
            "max_peak_heap_bytes": 80_000_000,
        },
        "decentralization": {
            "baseline_int8_jain_index": 0.91,
            "utility_int8_jain_gain": 0.08,
        },
        "communication": {
            "baseline_int8_reduction_percent": 70.0,
            "utility_int8_savings_percent": 65.0,
        },
        "status": {
            "collected": True,
            "open_milestones": 0,
            "open_issues": 0,
        },
        "provenance": {
            "repo": "albot-dev/OpenMeshMind",
            "commit": "abcdef123456",
            "decision_log": "DECISION_LOG.md",
            "provenance_template": "PROVENANCE_TEMPLATE.md",
        },
    }


class BuildPilotCohortMetricsTests(unittest.TestCase):
    def _run_main(self, argv: list[str]) -> tuple[int, str]:
        stdout = io.StringIO()
        with mock.patch.object(sys, "argv", argv):
            with mock.patch("sys.stdout", stdout):
                code = build_pilot_cohort_metrics.main()
        return code, stdout.getvalue()

    def test_main_builds_payload_for_valid_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            metrics_a = tmp / "node_a_metrics.json"
            metrics_b = tmp / "node_b_metrics.json"
            out = tmp / "pilot_cohort_metrics.json"
            metrics_a.write_text(json.dumps(_valid_metrics("node-a01")), encoding="utf-8")
            metrics_b.write_text(json.dumps(_valid_metrics("node-b02")), encoding="utf-8")

            code, output = self._run_main(
                [
                    "build_pilot_cohort_metrics.py",
                    "--metrics",
                    str(metrics_a),
                    "--metrics",
                    str(metrics_b),
                    "--repo",
                    "albot-dev/OpenMeshMind",
                    "--json-out",
                    str(out),
                ]
            )
            self.assertEqual(code, 0)
            self.assertIn("Pilot cohort metrics written to:", output)
            self.assertTrue(out.exists())

            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["cohort"]["node_count"], 2)
            self.assertEqual(payload["provenance"]["repo"], "albot-dev/OpenMeshMind")
            self.assertTrue(payload["provenance"]["commit"])

    def test_main_fails_for_invalid_metrics_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bad = tmp / "bad_metrics.json"
            out = tmp / "pilot_cohort_metrics.json"
            bad.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "node": {"node_id": "node-bad"},
                    }
                ),
                encoding="utf-8",
            )

            code, output = self._run_main(
                [
                    "build_pilot_cohort_metrics.py",
                    "--metrics",
                    str(bad),
                    "--repo",
                    "albot-dev/OpenMeshMind",
                    "--json-out",
                    str(out),
                ]
            )
            self.assertEqual(code, 1)
            self.assertIn("invalid pilot metrics file", output)
            self.assertFalse(out.exists())

    def test_main_fails_when_repo_cannot_be_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            metrics = tmp / "metrics.json"
            out = tmp / "pilot_cohort_metrics.json"
            metrics.write_text(json.dumps(_valid_metrics("node-a01")), encoding="utf-8")

            with mock.patch.object(build_pilot_cohort_metrics, "default_repo", return_value=""):
                code, output = self._run_main(
                    [
                        "build_pilot_cohort_metrics.py",
                        "--metrics",
                        str(metrics),
                        "--json-out",
                        str(out),
                    ]
                )
            self.assertEqual(code, 1)
            self.assertIn("Unable to resolve provenance repo", output)


if __name__ == "__main__":
    unittest.main()
