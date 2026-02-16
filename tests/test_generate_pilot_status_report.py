import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import generate_pilot_status_report


class GeneratePilotStatusReportTests(unittest.TestCase):
    def _run_main(self, argv: list[str]) -> tuple[int, str]:
        stdout = io.StringIO()
        with mock.patch.object(sys, "argv", argv):
            with mock.patch("sys.stdout", stdout):
                code = generate_pilot_status_report.main()
        return code, stdout.getvalue()

    def _write_cohort_metrics(self, path: Path) -> None:
        payload = {
            "schema_version": 1,
            "cohort": {"node_count": 2},
            "summary": {
                "health": {
                    "uptime_ratio_24h_mean": 0.99,
                    "last_cycle_ok_ratio": 1.0,
                },
                "quality": {
                    "classification_accuracy_mean": 0.92,
                    "classification_macro_f1_mean": 0.91,
                },
                "decentralization": {
                    "baseline_int8_jain_index_mean": 0.90,
                    "utility_int8_jain_gain_mean": 0.08,
                },
                "communication": {
                    "baseline_int8_reduction_percent_mean": 70.0,
                    "utility_int8_savings_percent_mean": 65.0,
                },
                "accessibility": {
                    "benchmark_total_runtime_sec_mean": 8.5,
                    "max_peak_rss_bytes_max": 120000000,
                },
                "status": {
                    "open_milestones_max": 0,
                    "open_issues_max": 0,
                },
            },
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_main_passes_with_bundle_when_artifacts_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            cohort = tmp / "cohort.json"
            template = tmp / "template.md"
            out = tmp / "pilot_status.md"
            bundle = tmp / "pilot_artifacts.tgz"
            provenance = tmp / "pilot_status_provenance.json"
            self._write_cohort_metrics(cohort)
            template.write_text(
                "# Pilot Status\n"
                "date={{report_date_utc}}\n"
                "status={{overall_status}}\n"
                "nodes={{active_nodes}}\n",
                encoding="utf-8",
            )
            provenance.write_text('{"schema_version":1}', encoding="utf-8")

            with mock.patch.object(generate_pilot_status_report, "run_cmd", return_value=(0, "ok")):
                code, output = self._run_main(
                    [
                        "generate_pilot_status_report.py",
                        "--pilot-metrics",
                        str(tmp / "missing_pilot.json"),
                        "--cohort-metrics",
                        str(cohort),
                        "--template",
                        str(template),
                        "--out",
                        str(out),
                        "--bundle-out",
                        str(bundle),
                        "--provenance-out",
                        str(provenance),
                    ]
                )

            self.assertEqual(code, 0)
            self.assertIn("Pilot status report written to:", output)
            self.assertIn("Pilot artifact bundle written to:", output)
            self.assertTrue(out.exists())
            self.assertTrue(bundle.exists())

    def test_main_fails_when_template_tokens_remain_unresolved(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            cohort = tmp / "cohort.json"
            template = tmp / "template.md"
            out = tmp / "pilot_status.md"
            self._write_cohort_metrics(cohort)
            template.write_text(
                "# Pilot Status\n"
                "date={{report_date_utc}}\n"
                "unknown={{unknown_token}}\n",
                encoding="utf-8",
            )

            code, output = self._run_main(
                [
                    "generate_pilot_status_report.py",
                    "--cohort-metrics",
                    str(cohort),
                    "--template",
                    str(template),
                    "--out",
                    str(out),
                ]
            )

            self.assertEqual(code, 1)
            self.assertIn("Unresolved template token(s)", output)

    def test_main_fails_when_provenance_generation_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            cohort = tmp / "cohort.json"
            template = tmp / "template.md"
            out = tmp / "pilot_status.md"
            self._write_cohort_metrics(cohort)
            template.write_text("status={{overall_status}}\n", encoding="utf-8")

            with mock.patch.object(
                generate_pilot_status_report,
                "run_cmd",
                return_value=(1, "provenance build failed"),
            ):
                code, output = self._run_main(
                    [
                        "generate_pilot_status_report.py",
                        "--cohort-metrics",
                        str(cohort),
                        "--template",
                        str(template),
                        "--out",
                        str(out),
                    ]
                )

            self.assertEqual(code, 1)
            self.assertIn("Failed to generate pilot status provenance manifest.", output)

    def test_main_fails_when_bundle_inputs_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            cohort = tmp / "cohort.json"
            template = tmp / "template.md"
            out = tmp / "pilot_status.md"
            bundle = tmp / "pilot_artifacts.tgz"
            provenance = tmp / "pilot_status_provenance.json"
            self._write_cohort_metrics(cohort)
            template.write_text("status={{overall_status}}\n", encoding="utf-8")

            with mock.patch.object(generate_pilot_status_report, "run_cmd", return_value=(0, "ok")):
                code, output = self._run_main(
                    [
                        "generate_pilot_status_report.py",
                        "--cohort-metrics",
                        str(cohort),
                        "--template",
                        str(template),
                        "--out",
                        str(out),
                        "--bundle-out",
                        str(bundle),
                        "--provenance-out",
                        str(provenance),
                    ]
                )

            self.assertEqual(code, 1)
            self.assertIn("missing expected artifact(s) for bundle", output)


if __name__ == "__main__":
    unittest.main()
