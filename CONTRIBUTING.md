# Contributing to OpenMeshMind

## Scope

Contributions should improve at least one of:

- accessibility on commodity hardware
- decentralization and resilience
- transparency and reproducibility

## Development principles

- Prefer CPU-first baselines before GPU-only approaches.
- Keep experiments reproducible and lightweight.
- Favor clear metrics over anecdotal claims.
- Avoid adding heavy dependencies unless the gain is concrete and measured.

## Local validation

Run these before opening a pull request:

```bash
python3 -m unittest discover -s tests -p "test_*.py"
python3 experiments/fedavg_cpu_only.py --json-out baseline_metrics.json
python3 scripts/check_baseline.py baseline_metrics.json
python3 experiments/fedavg_cpu_only.py --simulate-client-capacity --quiet --json-out fairness_metrics.json
python3 scripts/check_fairness.py fairness_metrics.json
python3 experiments/local_classification_baseline.py --json-out classification_metrics.json
python3 scripts/check_classification.py classification_metrics.json
```

## Pull request expectations

- State the problem and expected user impact.
- Include before/after metrics for behavior changes.
- Keep changes focused; split large efforts into incremental PRs.

## Code style

- Use clear names and minimal comments.
- Keep scripts dependency-light when possible.
- Update `README.md` or `ROADMAP.md` when behavior or direction changes.
