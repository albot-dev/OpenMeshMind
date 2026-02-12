# ai4all

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

## Next milestones

See `ROADMAP.md` for phased execution.
