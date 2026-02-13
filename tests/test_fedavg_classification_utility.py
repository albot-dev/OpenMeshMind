import unittest

from experiments import fedavg_classification_utility as utility


class FedAvgClassificationUtilityTests(unittest.TestCase):
    def test_sparse_topk_keeps_largest_values(self) -> None:
        vec = [0.02, -0.9, 0.5, 0.01, -0.3]
        sparse, sent_bytes = utility.sparse_topk(vec, keep_ratio=0.4)

        self.assertEqual(sent_bytes, 2 * 8 + 4)
        self.assertEqual(sparse[1], -0.9)
        self.assertEqual(sparse[2], 0.5)
        self.assertEqual(sparse[0], 0.0)
        self.assertEqual(sparse[3], 0.0)
        self.assertEqual(sparse[4], 0.0)

    def test_run_experiment_reports_modes_and_savings(self) -> None:
        report = utility.run_experiment(
            seeds=[7],
            modes=["fp32", "int8", "sparse"],
            samples_per_label=12,
            test_fraction=0.25,
            n_clients=4,
            rounds=3,
            local_steps=2,
            batch_size=6,
            learning_rate=0.2,
            non_iid_severity=1.0,
            sparse_ratio=0.25,
        )

        self.assertEqual(report["schema_version"], 1)
        self.assertIn("centralized", report["methods"])
        self.assertIn("fedavg_fp32", report["methods"])
        self.assertIn("fedavg_int8", report["methods"])
        self.assertIn("fedavg_sparse", report["methods"])

        int8_saving = report["communication_savings_percent"]["int8_vs_fp32_percent"]
        sparse_saving = report["communication_savings_percent"]["sparse_vs_fp32_percent"]
        self.assertGreater(int8_saving, 0.0)
        self.assertGreater(sparse_saving, 0.0)

        acc = report["methods"]["fedavg_fp32"]["accuracy_mean"]
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

    def test_invalid_modes_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown modes"):
            utility.run_experiment(
                seeds=[7],
                modes=["fp32", "invalid"],
                samples_per_label=10,
                test_fraction=0.25,
                n_clients=4,
                rounds=2,
                local_steps=2,
                batch_size=4,
                learning_rate=0.2,
                non_iid_severity=1.0,
                sparse_ratio=0.25,
            )

        with self.assertRaisesRegex(ValueError, "include fp32"):
            utility.run_experiment(
                seeds=[7],
                modes=["int8", "sparse"],
                samples_per_label=10,
                test_fraction=0.25,
                n_clients=4,
                rounds=2,
                local_steps=2,
                batch_size=4,
                learning_rate=0.2,
                non_iid_severity=1.0,
                sparse_ratio=0.25,
            )
