import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import solo_multi_machine_mode


class SoloMultiMachineModeTests(unittest.TestCase):
    def _run_main(self, argv: list[str]) -> tuple[int, str]:
        stdout = io.StringIO()
        with mock.patch.object(sys, "argv", argv):
            with mock.patch("sys.stdout", stdout):
                code = solo_multi_machine_mode.main()
        return code, stdout.getvalue()

    def test_import_bundle_updates_manifest_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bundle = tmp / "node_a_onboarding.tgz"
            manifest = tmp / "pilot" / "cohort_manifest.json"
            nodes_dir = tmp / "pilot" / "nodes"
            summary = tmp / "pilot" / "cohort_onboarding_summary.json"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "cohort_id": "pilot-cohort-test",
                        "generated_utc": "2026-02-14T00:00:00+00:00",
                        "nodes": [],
                    }
                ),
                encoding="utf-8",
            )

            metrics_payload = {
                "schema_version": 1,
                "node": {
                    "node_id": "node-a01",
                    "mode": "reduced",
                    "python_version": "3.10.10",
                    "platform": "darwin",
                },
                "health": {
                    "last_cycle_ok": True,
                    "cycle_duration_sec": 8.0,
                    "step_count": 8,
                    "uptime_ratio_24h": 1.0,
                },
                "quality": {
                    "classification_accuracy": 0.9,
                    "classification_macro_f1": 0.9,
                    "utility_fedavg_int8_accuracy": 0.9,
                    "utility_fedavg_int8_macro_f1": 0.9,
                },
                "accessibility": {
                    "benchmark_total_runtime_sec": 5.0,
                    "max_peak_rss_bytes": 100,
                    "max_peak_heap_bytes": 100,
                },
                "decentralization": {
                    "baseline_int8_jain_index": 0.9,
                    "utility_int8_jain_gain": 0.1,
                },
                "communication": {
                    "baseline_int8_reduction_percent": 60.0,
                    "utility_int8_savings_percent": 50.0,
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
                "timestamp_utc": "2026-02-14T00:00:00+00:00",
            }
            profile_payload = {
                "schema_version": 1,
                "node_id": "node-a01",
                "region": "us-east",
                "hardware_tier": "mid",
                "network_tier": "home-broadband",
                "cpu_cores": 8,
                "memory_gb": 16,
            }

            bundle.parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(bundle, "w:gz") as tar:
                for name, payload in {
                    "pilot/nodes/node-a01/pilot_metrics.json": metrics_payload,
                    "pilot/nodes/node-a01/node_profile.json": profile_payload,
                }.items():
                    raw = json.dumps(payload).encode("utf-8")
                    info = tarfile.TarInfo(name=name)
                    info.size = len(raw)
                    tar.addfile(info, io.BytesIO(raw))

            code, out = self._run_main(
                [
                    "solo_multi_machine_mode.py",
                    "--bundle",
                    str(bundle),
                    "--bundles-glob",
                    "",
                    "--manifest",
                    str(manifest),
                    "--nodes-dir",
                    str(nodes_dir),
                    "--summary-json-out",
                    str(summary),
                    "--min-nodes",
                    "1",
                    "--min-passed",
                    "1",
                    "--require-metrics-files",
                    "--no-pipeline",
                ]
            )
            self.assertEqual(code, 0)
            self.assertIn("Updated cohort manifest:", out)

            with manifest.open("r", encoding="utf-8") as f:
                manifest_payload = json.load(f)
            self.assertEqual(manifest_payload["schema_version"], 1)
            self.assertEqual(len(manifest_payload["nodes"]), 1)
            self.assertEqual(manifest_payload["nodes"][0]["node_id"], "node-a01")
            self.assertEqual(manifest_payload["nodes"][0]["onboarding_status"], "passed")

            imported_metrics = nodes_dir / "node-a01" / "pilot_metrics.json"
            self.assertTrue(imported_metrics.exists())
            self.assertTrue(summary.exists())

    def test_import_bundle_fails_for_invalid_metrics_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bundle = tmp / "node_bad_onboarding.tgz"
            manifest = tmp / "pilot" / "cohort_manifest.json"
            nodes_dir = tmp / "pilot" / "nodes"
            summary = tmp / "pilot" / "cohort_onboarding_summary.json"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "cohort_id": "pilot-cohort-test",
                        "generated_utc": "2026-02-14T00:00:00+00:00",
                        "nodes": [],
                    }
                ),
                encoding="utf-8",
            )

            invalid_metrics_payload = {
                "schema_version": 1,
                "node": {"node_id": "node-bad01", "mode": "reduced"},
            }
            profile_payload = {
                "schema_version": 1,
                "node_id": "node-bad01",
                "region": "us-east",
                "hardware_tier": "mid",
                "network_tier": "home-broadband",
                "cpu_cores": 8,
                "memory_gb": 16,
            }

            with tarfile.open(bundle, "w:gz") as tar:
                for name, payload in {
                    "pilot/nodes/node-bad01/pilot_metrics.json": invalid_metrics_payload,
                    "pilot/nodes/node-bad01/node_profile.json": profile_payload,
                }.items():
                    raw = json.dumps(payload).encode("utf-8")
                    info = tarfile.TarInfo(name=name)
                    info.size = len(raw)
                    tar.addfile(info, io.BytesIO(raw))

            code, out = self._run_main(
                [
                    "solo_multi_machine_mode.py",
                    "--bundle",
                    str(bundle),
                    "--bundles-glob",
                    "",
                    "--manifest",
                    str(manifest),
                    "--nodes-dir",
                    str(nodes_dir),
                    "--summary-json-out",
                    str(summary),
                    "--min-nodes",
                    "1",
                    "--min-passed",
                    "1",
                    "--require-metrics-files",
                    "--no-pipeline",
                ]
            )
            self.assertEqual(code, 1)
            self.assertIn("Failed to import bundle", out)
            self.assertIn("invalid pilot_metrics.json", out)


if __name__ == "__main__":
    unittest.main()
