# Local Retrieval Baseline

Dependency-light retrieval baseline using TF-IDF and cosine similarity.

## Command

```bash
python3 experiments/local_retrieval_baseline.py --json-out retrieval_metrics.json
```

## Result snapshot

- `Recall@1`: `0.9167`
- `Recall@3`: `1.0000`
- `MRR`: `0.9583`
- `Mean latency/query`: `0.030 ms`
- `P95 latency/query`: `0.027 ms`
