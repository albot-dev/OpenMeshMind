# Benchmark Suite Snapshot

Constrained-hardware benchmark snapshot from reduced mode.

## Command

```bash
python3 scripts/benchmark_suite.py --mode reduced --json-out benchmark_metrics.json
```

## Sample output

- `fedavg_baseline`
  - runtime mean: `~2.75s`
  - peak RSS: `~26.27 MiB`
  - peak Python heap: `~4.28 MiB`
- `retrieval_baseline`
  - runtime mean: `~0.02s`
  - peak RSS: `~14.56 MiB`
  - peak Python heap: `~0.83 MiB`
