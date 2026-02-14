import unittest

from experiments import fedavg_adapter_intent as adapter


class FedAvgAdapterIntentTests(unittest.TestCase):
    def test_run_experiment_schema_and_metrics(self) -> None:
        report = adapter.run_experiment(
            seeds=[7],
            modes=["fp32", "int8", "sparse"],
            samples_per_intent=10,
            n_clients=4,
            rounds=4,
            local_steps=3,
            batch_size=8,
            learning_rate=0.14,
            sparse_ratio=0.25,
            int8_clip_percentile=0.98,
            rank=3,
            non_iid_severity=1.0,
        )

        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["runs"], 1)
        self.assertIn("centralized", report["methods"])
        self.assertIn("fedavg_fp32", report["methods"])
        self.assertIn("fedavg_int8", report["methods"])
        self.assertIn("fedavg_sparse", report["methods"])
        self.assertIn("int8_vs_fp32_percent", report["communication_savings_percent"])
        self.assertGreater(report["communication_savings_percent"]["int8_vs_fp32_percent"], 0.0)

    def test_modes_must_include_fp32(self) -> None:
        with self.assertRaisesRegex(ValueError, "include fp32"):
            adapter.run_experiment(
                seeds=[7],
                modes=["int8"],
                samples_per_intent=10,
                n_clients=4,
                rounds=3,
                local_steps=2,
                batch_size=8,
                learning_rate=0.14,
                sparse_ratio=0.2,
                int8_clip_percentile=0.98,
                rank=2,
                non_iid_severity=1.0,
            )
