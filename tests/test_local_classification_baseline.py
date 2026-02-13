import unittest

from experiments import local_classification_baseline as baseline


class LocalClassificationBaselineTests(unittest.TestCase):
    def test_generate_dataset_is_balanced(self) -> None:
        rows = baseline.generate_dataset(seed=5, samples_per_label=12)
        self.assertEqual(len(rows), 12 * len(baseline.LABEL_TEMPLATES))

        counts: dict[str, int] = {}
        for _, label in rows:
            counts[label] = counts.get(label, 0) + 1

        for label in baseline.LABEL_TEMPLATES:
            self.assertEqual(counts[label], 12)

    def test_stratified_split_keeps_all_labels(self) -> None:
        rows = baseline.generate_dataset(seed=7, samples_per_label=10)
        train, test = baseline.stratified_split(rows=rows, seed=7, test_fraction=0.2)
        train_labels = {label for _, label in train}
        test_labels = {label for _, label in test}

        self.assertEqual(train_labels, set(baseline.LABEL_TEMPLATES))
        self.assertEqual(test_labels, set(baseline.LABEL_TEMPLATES))

    def test_run_classification_has_expected_schema(self) -> None:
        report = baseline.run_classification(
            seed=11,
            samples_per_label=20,
            test_fraction=0.25,
            steps=800,
            learning_rate=0.2,
            measure_latency=False,
        )

        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(len(report["counts"]["labels"]), 4)
        self.assertGreater(report["counts"]["vocabulary_size"], 10)
        self.assertGreaterEqual(report["metrics"]["accuracy"], 0.6)
        self.assertGreaterEqual(report["metrics"]["macro_f1"], 0.6)
        self.assertEqual(report["metrics"]["latency_mean_ms"], 0.0)
        self.assertEqual(report["metrics"]["latency_p95_ms"], 0.0)

    def test_reproducible_quality_metrics_by_seed(self) -> None:
        cfg = {
            "seed": 17,
            "samples_per_label": 16,
            "test_fraction": 0.25,
            "steps": 700,
            "learning_rate": 0.22,
            "measure_latency": False,
        }
        first = baseline.run_classification(**cfg)
        second = baseline.run_classification(**cfg)

        self.assertEqual(first["metrics"]["accuracy"], second["metrics"]["accuracy"])
        self.assertEqual(first["metrics"]["macro_f1"], second["metrics"]["macro_f1"])
        self.assertEqual(
            first["metrics"]["per_label_f1"],
            second["metrics"]["per_label_f1"],
        )

    def test_invalid_parameters_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "samples_per_label"):
            baseline.generate_dataset(seed=1, samples_per_label=2)

        rows = baseline.generate_dataset(seed=2, samples_per_label=8)
        with self.assertRaisesRegex(ValueError, "test_fraction"):
            baseline.stratified_split(rows=rows, seed=2, test_fraction=1.0)

        train, test = baseline.stratified_split(rows=rows, seed=2, test_fraction=0.2)
        vocab, _, feats = baseline.build_tfidf_features(train_rows=train, all_rows=train + test)
        train_x = feats[: len(train)]
        train_y = [0 for _ in train]
        with self.assertRaisesRegex(ValueError, "steps"):
            baseline.train_softmax_classifier(
                train_x=train_x,
                train_y=train_y,
                n_classes=1,
                n_features=len(vocab),
                steps=0,
                learning_rate=0.1,
                seed=2,
            )
