#!/usr/bin/env python3
"""
Dependency-light local retrieval baseline with reproducible metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from collections import Counter


TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def tf_vector(tokens: list[str]) -> dict[str, float]:
    if not tokens:
        return {}
    counts = Counter(tokens)
    total = len(tokens)
    return {term: count / total for term, count in counts.items()}


def compute_idf(doc_tokens: list[list[str]]) -> dict[str, float]:
    n_docs = len(doc_tokens)
    df: Counter[str] = Counter()
    for tokens in doc_tokens:
        for term in set(tokens):
            df[term] += 1
    return {
        term: math.log((n_docs + 1) / (count + 1)) + 1.0
        for term, count in df.items()
    }


def tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    tf = tf_vector(tokens)
    return {term: tf_val * idf.get(term, 0.0) for term, tf_val in tf.items()}


def l2_norm(vec: dict[str, float]) -> float:
    return math.sqrt(sum(value * value for value in vec.values()))


def cosine(a: dict[str, float], b: dict[str, float], b_norm: float) -> float:
    if not a or not b or b_norm == 0.0:
        return 0.0
    dot = 0.0
    for term, value in a.items():
        dot += value * b.get(term, 0.0)
    a_norm = l2_norm(a)
    if a_norm == 0.0:
        return 0.0
    return dot / (a_norm * b_norm)


def load_json(path: str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list.")
    return data


def run_retrieval(
    corpus_path: str,
    queries_path: str,
    top_k: int,
) -> dict[str, object]:
    corpus = load_json(corpus_path)
    queries = load_json(queries_path)

    if not corpus:
        raise ValueError("Corpus is empty.")
    if not queries:
        raise ValueError("Queries are empty.")

    docs = []
    doc_ids: set[str] = set()
    for row in corpus:
        doc_id = row["id"]
        text = row["text"]
        if doc_id in doc_ids:
            raise ValueError(f"Duplicate document id: {doc_id}")
        doc_ids.add(doc_id)
        docs.append((doc_id, tokenize(text)))

    idf = compute_idf([tokens for _, tokens in docs])
    doc_vecs = []
    for doc_id, tokens in docs:
        vec = tfidf_vector(tokens, idf)
        doc_vecs.append((doc_id, vec, l2_norm(vec)))

    total_queries = len(queries)
    hit_at_1 = 0
    hit_at_k = 0
    reciprocal_ranks: list[float] = []
    latencies_ms: list[float] = []

    for row in queries:
        query = row["query"]
        relevant_id = row["relevant_id"]
        if relevant_id not in doc_ids:
            raise ValueError(f"Unknown relevant_id in query set: {relevant_id}")

        t0 = time.perf_counter()
        q_vec = tfidf_vector(tokenize(query), idf)
        scored = []
        for doc_id, d_vec, d_norm in doc_vecs:
            scored.append((doc_id, cosine(q_vec, d_vec, d_norm)))
        scored.sort(key=lambda x: x[1], reverse=True)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt_ms)

        ranked_ids = [doc_id for doc_id, _ in scored]
        rank = ranked_ids.index(relevant_id) + 1 if relevant_id in ranked_ids else None
        if rank == 1:
            hit_at_1 += 1
        if rank is not None and rank <= top_k:
            hit_at_k += 1
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

    return {
        "schema_version": 1,
        "config": {
            "corpus_path": corpus_path,
            "queries_path": queries_path,
            "top_k": top_k,
            "tokenizer": "lower_alnum_regex",
            "scoring": "tfidf_cosine",
        },
        "counts": {
            "documents": len(corpus),
            "queries": total_queries,
            "vocabulary_size": len(idf),
        },
        "metrics": {
            "recall_at_1": hit_at_1 / total_queries,
            "recall_at_k": hit_at_k / total_queries,
            "mrr": statistics.mean(reciprocal_ranks),
            "latency_mean_ms": statistics.mean(latencies_ms),
            "latency_p95_ms": (
                sorted(latencies_ms)[int(0.95 * (len(latencies_ms) - 1))]
                if len(latencies_ms) > 1
                else latencies_ms[0]
            ),
        },
    }


def summarize(report: dict[str, object]) -> None:
    metrics = report["metrics"]
    counts = report["counts"]
    config = report["config"]
    print("Local retrieval baseline (TF-IDF + cosine)\n")
    print(
        f"docs={counts['documents']}, queries={counts['queries']}, "
        f"vocab={counts['vocabulary_size']}, top_k={config['top_k']}"
    )
    print(f"Recall@1: {metrics['recall_at_1']:.4f}")
    print(f"Recall@{config['top_k']}: {metrics['recall_at_k']:.4f}")
    print(f"MRR: {metrics['mrr']:.4f}")
    print(f"Mean latency/query: {metrics['latency_mean_ms']:.3f} ms")
    print(f"P95 latency/query: {metrics['latency_p95_ms']:.3f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        default="data/retrieval_corpus.json",
        help="Path to corpus JSON list with fields: id, text",
    )
    parser.add_argument(
        "--queries",
        default="data/retrieval_queries.json",
        help="Path to query JSON list with fields: query, relevant_id",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top-k cutoff for recall metric (default: 3).",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional path to write machine-readable JSON metrics.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress human-readable summary.",
    )
    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be a positive integer.")

    report = run_retrieval(
        corpus_path=args.corpus,
        queries_path=args.queries,
        top_k=args.top_k,
    )
    if not args.quiet:
        summarize(report)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
