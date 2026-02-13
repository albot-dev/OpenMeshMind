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
from scripts import check_fairness
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
