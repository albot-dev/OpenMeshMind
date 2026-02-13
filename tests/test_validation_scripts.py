import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import check_baseline
from scripts import check_benchmarks


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

