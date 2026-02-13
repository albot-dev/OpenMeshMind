import json
import tempfile
import unittest
from pathlib import Path

from experiments import local_retrieval_baseline as retrieval


class RetrievalBaselineTests(unittest.TestCase):
    def _write_json(self, path: Path, payload: object) -> None:
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_run_retrieval_reports_expected_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            corpus_path = tmp / "corpus.json"
            queries_path = tmp / "queries.json"

            corpus = [
                {"id": "d_cpu", "text": "federated learning on cpu edge devices"},
                {"id": "d_fin", "text": "retirement planning with index fund savings"},
                {"id": "d_search", "text": "local search ranking with tfidf vectors"},
            ]
            queries = [
                {"query": "cpu edge federated learning", "relevant_id": "d_cpu"},
                {"query": "index fund retirement savings", "relevant_id": "d_fin"},
                {"query": "tfidf search ranking vectors", "relevant_id": "d_search"},
            ]
            self._write_json(corpus_path, corpus)
            self._write_json(queries_path, queries)

            report = retrieval.run_retrieval(
                corpus_path=str(corpus_path),
                queries_path=str(queries_path),
                top_k=2,
            )

            self.assertEqual(report["schema_version"], 1)
            self.assertEqual(report["counts"]["documents"], 3)
            self.assertEqual(report["counts"]["queries"], 3)
            self.assertEqual(report["metrics"]["recall_at_1"], 1.0)
            self.assertEqual(report["metrics"]["recall_at_k"], 1.0)
            self.assertEqual(report["metrics"]["mrr"], 1.0)
            self.assertGreater(report["metrics"]["latency_mean_ms"], 0.0)

    def test_duplicate_document_id_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            corpus_path = tmp / "corpus.json"
            queries_path = tmp / "queries.json"
            self._write_json(
                corpus_path,
                [
                    {"id": "dup", "text": "one"},
                    {"id": "dup", "text": "two"},
                ],
            )
            self._write_json(
                queries_path,
                [{"query": "one", "relevant_id": "dup"}],
            )

            with self.assertRaisesRegex(ValueError, "Duplicate document id"):
                retrieval.run_retrieval(
                    corpus_path=str(corpus_path),
                    queries_path=str(queries_path),
                    top_k=2,
                )

    def test_unknown_relevant_id_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            corpus_path = tmp / "corpus.json"
            queries_path = tmp / "queries.json"
            self._write_json(corpus_path, [{"id": "a", "text": "alpha beta"}])
            self._write_json(
                queries_path,
                [{"query": "alpha", "relevant_id": "missing"}],
            )

            with self.assertRaisesRegex(ValueError, "Unknown relevant_id"):
                retrieval.run_retrieval(
                    corpus_path=str(corpus_path),
                    queries_path=str(queries_path),
                    top_k=1,
                )
