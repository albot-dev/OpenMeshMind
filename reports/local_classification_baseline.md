# Local Classification Baseline Snapshot

Command used:

```bash
python3 experiments/local_classification_baseline.py --json-out classification_metrics.json
```

Summary output:

- labels: `4`
- train samples: `192`
- test samples: `48`
- vocabulary size: `127`
- accuracy: `1.0000`
- macro-F1: `1.0000`
- train runtime: `0.0178s`
- mean latency per sample: `0.0024ms`
- p95 latency per sample: `0.0028ms`

Config:

- seed: `7`
- samples per label: `60`
- test fraction: `0.20`
- steps: `2200`
- learning rate: `0.18`

Notes:

- This is a CPU-only synthetic utility baseline intended for reproducible local experimentation.
- It is intentionally dependency-light and suitable for constrained development environments.
