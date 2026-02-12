# OpenMeshMind

Practical work toward broadly accessible AI systems that do not depend on specialized hardware or centralized control.

## Current baseline

This repo includes a reproducible, CPU-only federated learning experiment in:

- `experiments/fedavg_cpu_only.py`

It compares centralized training, FedAvg, and quantized FedAvg (int8 updates), and reports:

- accuracy
- runtime
- communication cost

## Why this matters

The project direction is:

- run useful AI on commodity devices (CPU-first)
- use communication-efficient decentralization (federated + quantized updates)
- build transparent governance and open participation

## Run the experiment

```bash
python3 experiments/fedavg_cpu_only.py
```

Generate machine-readable metrics:

```bash
python3 experiments/fedavg_cpu_only.py --json-out baseline_metrics.json
```

Validate thresholds:

```bash
python3 scripts/check_baseline.py baseline_metrics.json
```

Run a drop-out resilience scenario (with automatic comparison vs no drop-out):

```bash
python3 experiments/fedavg_cpu_only.py --dropout-rate 0.35 --json-out dropout_35_metrics.json
```

## Next milestones

See `ROADMAP.md` for phased execution.

## Work tracking

- Phase 1 milestone: <https://github.com/albot-dev/OpenMeshMind/milestone/1>
- Starter issues:
  - <https://github.com/albot-dev/OpenMeshMind/issues/1>
  - <https://github.com/albot-dev/OpenMeshMind/issues/2>
  - <https://github.com/albot-dev/OpenMeshMind/issues/3>
  - <https://github.com/albot-dev/OpenMeshMind/issues/4>
  - <https://github.com/albot-dev/OpenMeshMind/issues/5>
