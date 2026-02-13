import random
import unittest

from experiments import fedavg_cpu_only as fed


class FedAvgCoreTests(unittest.TestCase):
    def test_quantize_int8_roundtrip_error_bound(self) -> None:
        vec = [-3.5, -1.25, -0.1, 0.0, 0.1, 1.7, 3.14]
        qvec, scale = fed.quantize_int8(vec)
        restored = fed.dequantize_int8(qvec, scale)

        self.assertGreater(scale, 0.0)
        self.assertTrue(all(-127 <= value <= 127 for value in qvec))
        for original, decoded in zip(vec, restored):
            self.assertLessEqual(abs(original - decoded), scale + 1e-12)

    def test_secure_mask_updates_preserves_aggregate(self) -> None:
        random.seed(7)
        updates = [
            [0.2, -0.1, 0.05],
            [0.0, 0.4, -0.3],
            [-0.2, 0.3, 0.10],
        ]
        masked, overhead, pair_count = fed.secure_mask_updates(
            [row[:] for row in updates],
            mask_bound=0.01,
        )

        aggregate_raw = [sum(values) for values in zip(*updates)]
        aggregate_masked = [sum(values) for values in zip(*masked)]
        for raw, masked_total in zip(aggregate_raw, aggregate_masked):
            self.assertAlmostEqual(raw, masked_total, places=10)

        self.assertEqual(pair_count, 3)
        self.assertEqual(overhead, pair_count * 32)

    def test_non_iid_partition_preserves_examples(self) -> None:
        data = fed.generate_dataset(seed=11, n_samples=120, n_features=4)
        train_data, _ = fed.train_test_split(data)
        clients = fed.non_iid_partition(train_data, n_clients=5, severity=1.2)
        flattened = [sample for shard in clients for sample in shard]

        self.assertEqual(len(flattened), len(train_data))
        self.assertCountEqual(flattened, train_data)

    def test_dropout_one_has_no_participation(self) -> None:
        rounds = 3
        result = fed.run_once(
            seed=13,
            dropout_rate=1.0,
            non_iid_severity=fed.DEFAULT_NON_IID_SEVERITY,
            secure_aggregation=False,
            n_features=6,
            n_clients=4,
            rounds=rounds,
            local_steps=2,
            batch_size=16,
            lr=0.1,
        )

        for method in ("fedavg_fp32", "fedavg_int8"):
            self.assertEqual(result[method].zero_client_rounds, rounds)
            self.assertEqual(result[method].participation_rate, 0.0)
            self.assertEqual(result[method].uplink_bytes, 0)

    def test_capacity_simulation_emits_fairness_metrics(self) -> None:
        result = fed.run_once(
            seed=5,
            dropout_rate=0.0,
            non_iid_severity=1.2,
            secure_aggregation=False,
            client_capacities=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
            round_deadline=2.0,
            capacity_jitter=0.0,
            n_features=4,
            n_clients=8,
            rounds=2,
            local_steps=1,
            batch_size=8,
            lr=0.1,
        )

        fp32 = result["fedavg_fp32"]
        int8 = result["fedavg_int8"]
        self.assertIsNotNone(fp32.fairness_metrics)
        self.assertIsNotNone(fp32.fairness_clients)
        self.assertEqual(len(fp32.fairness_clients or []), 8)
        self.assertGreater(fp32.fairness_metrics["contribution_rate_gap"], 0.0)
        self.assertLess(
            fp32.fairness_metrics["contribution_jain_index"],
            int8.fairness_metrics["contribution_jain_index"],
        )
