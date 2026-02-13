# Low-End Contributor Smoke Path

One-command smoke run:

```bash
python3 scripts/smoke_check.py --include-fairness --json-out smoke_summary.json
```

What it runs:

- unit tests
- CPU baseline run + validator
- local classification run + validator
- federated utility classification run
- reduced benchmark + validator
- fairness checks for baseline and utility stress scenarios

Observed envelope on current maintainer machine:

- total smoke duration: `~4.8s`
- largest single step: reduced benchmark `~2.3s`
- all steps passed in one command

Reduced benchmark resource envelope (from `benchmark_metrics.json`):

- `fedavg_baseline`
  - runtime mean: `~1.96s`
  - peak RSS: `~26.6 MiB`
  - peak heap: `~3.47 MiB`
- `retrieval_baseline`
  - runtime mean: `~0.01s`
  - peak RSS: `~14.7 MiB`
  - peak heap: `~0.45 MiB`
- `classification_baseline`
  - runtime mean: `~0.03s`
  - peak RSS: `~15.3 MiB`
  - peak heap: `~0.62 MiB`

Troubleshooting:

- If a validator fails, rerun the exact command printed by `scripts/smoke_check.py` for that step.
- Keep runs deterministic by leaving default seeds in place.
- No network access is required for smoke checks once repo dependencies are available locally.
