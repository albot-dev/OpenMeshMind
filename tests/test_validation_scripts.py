import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import check_baseline
from scripts import check_benchmarks
from scripts import check_classification
from scripts import check_cohort_manifest
from scripts import check_fairness
from scripts import check_generality
from scripts import check_pilot_cohort
from scripts import check_pilot_metrics
from scripts import check_utility_fairness


class ValidationScriptTests(unittest.TestCase):
    def _write_json(self, directory: Path, name: str, payload: object) -> Path:
        path = directory / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _run_main(self, fn, argv: list[str]) -> tuple[int, str]:
        stdout = io.StringIO()
        with mock.patch.object(sys, "argv", argv):
            with mock.patch("sys.stdout", stdout):
                code = fn()
        return code, stdout.getvalue()

    def test_check_baseline_pass_and_fail_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            valid = self._write_json(
                tmp,
                "baseline_ok.json",
                {
                    "schema_version": 2,
                    "methods": {
                        "centralized": {"accuracy_mean": 0.90},
                        "fedavg_int8": {"accuracy_mean": 0.88},
                    },
                    "communication_reduction_percent": 60.0,
                },
            )
            invalid = self._write_json(
                tmp,
                "baseline_bad.json",
                {
                    "schema_version": 2,
                    "methods": {
                        "centralized": {"accuracy_mean": 0.90},
                        "fedavg_int8": {"accuracy_mean": 0.74},
                    },
                    "communication_reduction_percent": 30.0,
                },
            )

            code_ok, out_ok = self._run_main(
                check_baseline.main,
                ["check_baseline.py", str(valid)],
            )
            self.assertEqual(code_ok, 0)
            self.assertIn("Validation passed.", out_ok)

            code_bad, out_bad = self._run_main(
                check_baseline.main,
                ["check_baseline.py", str(invalid)],
            )
            self.assertEqual(code_bad, 1)
            self.assertIn("Validation failed:", out_bad)

    def test_check_cohort_manifest_pass_and_fail_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            metrics = self._write_json(
                tmp,
                "node_a_metrics.json",
                {
                    "schema_version": 1,
                    "node": {"node_id": "node-a01"},
                    "health": {"last_cycle_ok": True},
                },
            )
            valid = self._write_json(
                tmp,
                "cohort_ok.json",
                {
                    "schema_version": 1,
                    "cohort_id": "pilot-cohort-test",
                    "generated_utc": "2026-02-13T22:00:00+00:00",
                    "nodes": [
                        {
                            "node_id": "node-a01",
                            "region": "us-east",
                            "hardware_tier": "mid",
                            "cpu_cores": 8,
                            "memory_gb": 16,
                            "network_tier": "home-broadband",
                            "onboarding_status": "passed",
                            "onboarding_checked_utc": "2026-02-13T22:00:00+00:00",
                            "metrics_path": str(metrics),
                            "failure_reason": "",
                        }
                    ],
                },
            )
            invalid = self._write_json(
                tmp,
                "cohort_bad.json",
                {
                    "schema_version": 1,
                    "cohort_id": "pilot-cohort-test",
                    "generated_utc": "2026-02-13T22:00:00+00:00",
                    "nodes": [
                        {
                            "node_id": "node-b02",
                            "region": "us-west",
                            "hardware_tier": "low",
                            "cpu_cores": 4,
                            "memory_gb": 8,
                            "network_tier": "home-broadband",
                            "onboarding_status": "failed",
                            "onboarding_checked_utc": "2026-02-13T22:00:00+00:00",
                            "metrics_path": "missing.json",
                            "failure_reason": "",
                        }
                    ],
                },
            )

            code_ok, out_ok = self._run_main(
                check_cohort_manifest.main,
                [
                    "check_cohort_manifest.py",
                    str(valid),
                    "--min-nodes",
                    "1",
                    "--min-passed",
                    "1",
                    "--require-metrics-files",
                ],
            )
            self.assertEqual(code_ok, 0)
            self.assertIn("Validation passed.", out_ok)

            code_bad, out_bad = self._run_main(
                check_cohort_manifest.main,
                [
                    "check_cohort_manifest.py",
                    str(invalid),
                    "--min-nodes",
                    "1",
                    "--min-passed",
                    "1",
                ],
            )
            self.assertEqual(code_bad, 1)
            self.assertIn("Validation failed:", out_bad)

    def test_check_benchmarks_pass_and_fail_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            valid = self._write_json(
                tmp,
                "bench_ok.json",
                {
                    "schema_version": 1,
                    "config": {"mode": "reduced"},
                    "benchmarks": [
                        {
                            "name": "fedavg_baseline",
                            "runtime_mean_sec": 1.2,
                            "peak_heap_max_bytes": 1024,
                            "peak_rss_max_bytes": 4096,
                        }
                    ],
                },
            )
            invalid = self._write_json(
                tmp,
                "bench_bad.json",
                {
                    "schema_version": 1,
                    "config": {"mode": "full"},
                    "benchmarks": [
                        {
                            "name": "fedavg_baseline",
                            "runtime_mean_sec": 0.0,
                            "peak_heap_max_bytes": 0,
                            "peak_rss_max_bytes": 0,
                        }
                    ],
                },
            )

            code_ok, out_ok = self._run_main(
                check_benchmarks.main,
                ["check_benchmarks.py", str(valid)],
            )
            self.assertEqual(code_ok, 0)
            self.assertIn("Validation passed.", out_ok)

            code_bad, out_bad = self._run_main(
                check_benchmarks.main,
                ["check_benchmarks.py", str(invalid)],
            )
            self.assertEqual(code_bad, 1)
            self.assertIn("Validation failed:", out_bad)

    def test_check_generality_pass_and_fail_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            valid = self._write_json(
                tmp,
                "generality_ok.json",
                {
                    "schema_version": 1,
                    "tasks": {
                        "classification": {
                            "metrics": {
                                "accuracy": 0.90,
                                "macro_f1": 0.88,
                            }
                        },
                        "retrieval": {
                            "metrics": {
                                "recall_at_1": 0.82,
                                "mrr": 0.86,
                            }
                        },
                        "instruction_following": {
                            "metrics": {
                                "pass_rate": 0.90,
                            }
                        },
                        "tool_use": {
                            "metrics": {
                                "pass_rate": 1.00,
                            }
                        },
                        "distributed_reference": {
                            "metrics": {
                                "int8_accuracy_drop": 0.03,
                                "int8_comm_savings_percent": 70.0,
                            }
                        },
                    },
                    "aggregate": {"overall_score": 0.86},
                    "resources": {"total_wall_clock_sec": 9.2},
                },
            )
            invalid = self._write_json(
                tmp,
                "generality_bad.json",
                {
                    "schema_version": 1,
                    "tasks": {
                        "classification": {
                            "metrics": {
                                "accuracy": 0.70,
                                "macro_f1": 0.65,
                            }
                        },
                        "retrieval": {
                            "metrics": {
                                "recall_at_1": 0.40,
                                "mrr": 0.55,
                            }
                        },
                        "instruction_following": {
                            "metrics": {
                                "pass_rate": 0.50,
                            }
                        },
                        "tool_use": {
                            "metrics": {
                                "pass_rate": 0.40,
                            }
                        },
                        "distributed_reference": {
                            "metrics": {
                                "int8_accuracy_drop": 0.20,
                                "int8_comm_savings_percent": 20.0,
                            }
                        },
                    },
                    "aggregate": {"overall_score": 0.42},
                    "resources": {"total_wall_clock_sec": 420.0},
                },
            )

            code_ok, out_ok = self._run_main(
                check_generality.main,
                ["check_generality.py", str(valid)],
            )
            self.assertEqual(code_ok, 0)
            self.assertIn("Validation passed.", out_ok)

            code_bad, out_bad = self._run_main(
                check_generality.main,
                ["check_generality.py", str(invalid)],
            )
            self.assertEqual(code_bad, 1)
            self.assertIn("Validation failed:", out_bad)

    def test_check_fairness_pass_and_fail_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            valid = self._write_json(
                tmp,
                "fairness_ok.json",
                {
                    "schema_version": 2,
                    "methods": {
                        "fedavg_fp32": {
                            "fairness": {
                                "contribution_jain_index_mean": 0.70,
                                "contribution_rate_gap_mean": 0.95,
                                "slowest_fastest_contribution_ratio_mean": 0.00,
                                "contributed_clients_per_round_mean_mean": 5.2,
                            },
                            "fairness_clients": [{"client_index": 0}],
                        },
                        "fedavg_int8": {
                            "fairness": {
                                "contribution_jain_index_mean": 0.91,
                                "contribution_rate_gap_mean": 0.40,
                                "slowest_fastest_contribution_ratio_mean": 0.45,
                                "contributed_clients_per_round_mean_mean": 7.0,
                            },
                            "fairness_clients": [{"client_index": 0}],
                        },
                    },
                },
            )
            invalid = self._write_json(
                tmp,
                "fairness_bad.json",
                {
                    "schema_version": 2,
                    "methods": {
                        "fedavg_fp32": {
                            "fairness": {
                                "contribution_jain_index_mean": 0.92,
                                "contribution_rate_gap_mean": 0.20,
                                "slowest_fastest_contribution_ratio_mean": 0.55,
                                "contributed_clients_per_round_mean_mean": 7.2,
                            },
                            "fairness_clients": [{"client_index": 0}],
                        },
                        "fedavg_int8": {
                            "fairness": {
                                "contribution_jain_index_mean": 0.88,
                                "contribution_rate_gap_mean": 0.30,
                                "slowest_fastest_contribution_ratio_mean": 0.40,
                                "contributed_clients_per_round_mean_mean": 7.1,
                            },
                            "fairness_clients": [{"client_index": 0}],
                        },
                    },
                },
            )

            code_ok, out_ok = self._run_main(
                check_fairness.main,
                ["check_fairness.py", str(valid)],
            )
            self.assertEqual(code_ok, 0)
            self.assertIn("Validation passed.", out_ok)

            code_bad, out_bad = self._run_main(
                check_fairness.main,
                ["check_fairness.py", str(invalid)],
            )
            self.assertEqual(code_bad, 1)
            self.assertIn("Validation failed:", out_bad)

    def test_check_classification_pass_and_fail_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            valid = self._write_json(
                tmp,
                "classification_ok.json",
                {
                    "schema_version": 1,
                    "counts": {"labels": ["a", "b", "c"]},
                    "metrics": {
                        "accuracy": 0.93,
                        "macro_f1": 0.91,
                        "train_runtime_sec": 0.5,
                        "latency_mean_ms": 0.02,
                    },
                },
            )
            invalid = self._write_json(
                tmp,
                "classification_bad.json",
                {
                    "schema_version": 1,
                    "counts": {"labels": ["a"]},
                    "metrics": {
                        "accuracy": 0.70,
                        "macro_f1": 0.60,
                        "train_runtime_sec": 3.2,
                        "latency_mean_ms": 2.5,
                    },
                },
            )

            code_ok, out_ok = self._run_main(
                check_classification.main,
                ["check_classification.py", str(valid)],
            )
            self.assertEqual(code_ok, 0)
            self.assertIn("Validation passed.", out_ok)

            code_bad, out_bad = self._run_main(
                check_classification.main,
                ["check_classification.py", str(invalid)],
            )
            self.assertEqual(code_bad, 1)
            self.assertIn("Validation failed:", out_bad)

    def test_check_utility_fairness_pass_and_fail_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            valid = self._write_json(
                tmp,
                "utility_fairness_ok.json",
                {
                    "schema_version": 1,
                    "methods": {
                        "fedavg_fp32": {
                            "fairness": {
                                "contribution_jain_index_mean": 0.70,
                                "contribution_rate_gap_mean": 0.90,
                                "contributed_clients_per_round_mean_mean": 4.5,
                            },
                            "fairness_clients": [{"client_index": 0}],
                        },
                        "fedavg_int8": {
                            "fairness": {
                                "contribution_jain_index_mean": 0.90,
                                "contribution_rate_gap_mean": 0.50,
                                "contributed_clients_per_round_mean_mean": 5.5,
                            },
                            "fairness_clients": [{"client_index": 0}],
                        },
                    },
                },
            )
            invalid = self._write_json(
                tmp,
                "utility_fairness_bad.json",
                {
                    "schema_version": 1,
                    "methods": {
                        "fedavg_fp32": {
                            "fairness": {
                                "contribution_jain_index_mean": 0.90,
                                "contribution_rate_gap_mean": 0.30,
                                "contributed_clients_per_round_mean_mean": 6.0,
                            },
                            "fairness_clients": [{"client_index": 0}],
                        },
                        "fedavg_int8": {
                            "fairness": {
                                "contribution_jain_index_mean": 0.88,
                                "contribution_rate_gap_mean": 0.32,
                                "contributed_clients_per_round_mean_mean": 6.1,
                            },
                            "fairness_clients": [{"client_index": 0}],
                        },
                    },
                },
            )

            code_ok, out_ok = self._run_main(
                check_utility_fairness.main,
                ["check_utility_fairness.py", str(valid)],
            )
            self.assertEqual(code_ok, 0)
            self.assertIn("Validation passed.", out_ok)

            code_bad, out_bad = self._run_main(
                check_utility_fairness.main,
                ["check_utility_fairness.py", str(invalid)],
            )
            self.assertEqual(code_bad, 1)
            self.assertIn("Validation failed:", out_bad)

    def test_check_pilot_metrics_pass_and_fail_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            valid = self._write_json(
                tmp,
                "pilot_ok.json",
                {
                    "schema_version": 1,
                    "timestamp_utc": "2026-02-13T22:00:00+00:00",
                    "node": {
                        "node_id": "node-local-1",
                        "mode": "reduced",
                        "python_version": "3.12.0",
                        "platform": "test-platform",
                    },
                    "health": {
                        "last_cycle_ok": True,
                        "cycle_duration_sec": 12.5,
                        "step_count": 1,
                        "uptime_ratio_24h": 0.98,
                    },
                    "quality": {
                        "classification_accuracy": 0.92,
                        "classification_macro_f1": 0.91,
                        "utility_fedavg_int8_accuracy": 0.90,
                        "utility_fedavg_int8_macro_f1": 0.89,
                    },
                    "accessibility": {
                        "benchmark_total_runtime_sec": 8.2,
                        "max_peak_rss_bytes": 1200000,
                        "max_peak_heap_bytes": 900000,
                    },
                    "decentralization": {
                        "baseline_int8_jain_index": 0.88,
                        "utility_int8_jain_gain": 0.07,
                    },
                    "communication": {
                        "baseline_int8_reduction_percent": 60.0,
                        "utility_int8_savings_percent": 55.0,
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
                },
            )
            invalid = self._write_json(
                tmp,
                "pilot_bad.json",
                {
                    "schema_version": 1,
                    "timestamp_utc": "2026-02-13T22:00:00+00:00",
                    "node": {
                        "node_id": "node-local-1",
                        "mode": "reduced",
                        "python_version": "3.12.0",
                        "platform": "test-platform",
                    },
                    "health": {
                        "last_cycle_ok": False,
                        "cycle_duration_sec": 12.5,
                        "step_count": 1,
                        "uptime_ratio_24h": 1.20,
                    },
                    "quality": {
                        "classification_accuracy": 0.92,
                        "classification_macro_f1": 0.91,
                        "utility_fedavg_int8_accuracy": 0.90,
                        "utility_fedavg_int8_macro_f1": 0.89,
                    },
                    "accessibility": {
                        "benchmark_total_runtime_sec": 8.2,
                        "max_peak_rss_bytes": 1200000,
                        "max_peak_heap_bytes": 900000,
                    },
                    "decentralization": {
                        "baseline_int8_jain_index": 0.88,
                        "utility_int8_jain_gain": 0.07,
                    },
                    "communication": {
                        "baseline_int8_reduction_percent": 60.0,
                        "utility_int8_savings_percent": 55.0,
                    },
                    "status": {
                        "collected": False,
                        "open_milestones": 0,
                        "open_issues": 3,
                    },
                    "provenance": {
                        "repo": "albot-dev/OpenMeshMind",
                        "commit": "abcdef123456",
                        "decision_log": "DECISION_LOG.md",
                        "provenance_template": "PROVENANCE_TEMPLATE.md",
                    },
                },
            )

            code_ok, out_ok = self._run_main(
                check_pilot_metrics.main,
                ["check_pilot_metrics.py", str(valid)],
            )
            self.assertEqual(code_ok, 0)
            self.assertIn("Validation passed.", out_ok)

            code_bad, out_bad = self._run_main(
                check_pilot_metrics.main,
                [
                    "check_pilot_metrics.py",
                    str(invalid),
                    "--require-status-collected",
                    "--max-open-issues",
                    "0",
                ],
            )
            self.assertEqual(code_bad, 1)
            self.assertIn("Validation failed:", out_bad)

    def test_check_pilot_cohort_pass_and_fail_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            valid = self._write_json(
                tmp,
                "pilot_cohort_ok.json",
                {
                    "schema_version": 1,
                    "timestamp_utc": "2026-02-13T22:00:00+00:00",
                    "cohort": {
                        "node_count": 3,
                        "node_ids": ["node-a", "node-b", "node-c"],
                        "sources": ["pilot/a.json", "pilot/b.json", "pilot/c.json"],
                    },
                    "nodes": [],
                    "summary": {
                        "health": {
                            "uptime_ratio_24h_mean": 0.97,
                            "uptime_ratio_24h_min": 0.93,
                            "uptime_ratio_24h_max": 1.0,
                            "last_cycle_ok_ratio": 1.0,
                        },
                        "quality": {
                            "classification_accuracy_mean": 0.91,
                            "classification_macro_f1_mean": 0.90,
                            "utility_fedavg_int8_accuracy_mean": 0.89,
                            "utility_fedavg_int8_macro_f1_mean": 0.88,
                        },
                        "accessibility": {
                            "benchmark_total_runtime_sec_mean": 8.0,
                            "max_peak_rss_bytes_max": 1500000,
                            "max_peak_heap_bytes_max": 900000,
                        },
                        "decentralization": {
                            "baseline_int8_jain_index_mean": 0.88,
                            "utility_int8_jain_gain_mean": 0.07,
                            "utility_int8_jain_gain_min": 0.05,
                            "utility_int8_jain_gain_max": 0.08,
                        },
                        "communication": {
                            "baseline_int8_reduction_percent_mean": 60.0,
                            "utility_int8_savings_percent_mean": 55.0,
                        },
                        "status": {
                            "open_milestones_max": 1,
                            "open_issues_max": 3,
                            "status_collected_ratio": 1.0,
                        },
                    },
                    "provenance": {
                        "repo": "albot-dev/OpenMeshMind",
                        "commit": "abcdef123456",
                        "source_count": 3,
                    },
                },
            )
            invalid = self._write_json(
                tmp,
                "pilot_cohort_bad.json",
                {
                    "schema_version": 1,
                    "timestamp_utc": "2026-02-13T22:00:00+00:00",
                    "cohort": {
                        "node_count": 1,
                        "node_ids": ["node-a"],
                        "sources": ["pilot/a.json"],
                    },
                    "nodes": [],
                    "summary": {
                        "health": {
                            "uptime_ratio_24h_mean": 0.60,
                            "uptime_ratio_24h_min": 0.60,
                            "uptime_ratio_24h_max": 0.60,
                            "last_cycle_ok_ratio": 0.0,
                        },
                        "quality": {
                            "classification_accuracy_mean": 0.80,
                            "classification_macro_f1_mean": 0.80,
                            "utility_fedavg_int8_accuracy_mean": 0.79,
                            "utility_fedavg_int8_macro_f1_mean": 0.78,
                        },
                        "accessibility": {
                            "benchmark_total_runtime_sec_mean": 8.0,
                            "max_peak_rss_bytes_max": 1500000,
                            "max_peak_heap_bytes_max": 900000,
                        },
                        "decentralization": {
                            "baseline_int8_jain_index_mean": 0.80,
                            "utility_int8_jain_gain_mean": 0.01,
                            "utility_int8_jain_gain_min": 0.01,
                            "utility_int8_jain_gain_max": 0.01,
                        },
                        "communication": {
                            "baseline_int8_reduction_percent_mean": 50.0,
                            "utility_int8_savings_percent_mean": 45.0,
                        },
                        "status": {
                            "open_milestones_max": 1,
                            "open_issues_max": 8,
                            "status_collected_ratio": 0.0,
                        },
                    },
                    "provenance": {
                        "repo": "albot-dev/OpenMeshMind",
                        "commit": "abcdef123456",
                        "source_count": 1,
                    },
                },
            )

            code_ok, out_ok = self._run_main(
                check_pilot_cohort.main,
                [
                    "check_pilot_cohort.py",
                    str(valid),
                    "--min-node-count",
                    "3",
                    "--min-status-collected-ratio",
                    "1.0",
                ],
            )
            self.assertEqual(code_ok, 0)
            self.assertIn("Validation passed.", out_ok)

            code_bad, out_bad = self._run_main(
                check_pilot_cohort.main,
                [
                    "check_pilot_cohort.py",
                    str(invalid),
                    "--min-node-count",
                    "3",
                    "--min-status-collected-ratio",
                    "1.0",
                    "--max-open-issues",
                    "3",
                ],
            )
            self.assertEqual(code_bad, 1)
            self.assertIn("Validation failed:", out_bad)
