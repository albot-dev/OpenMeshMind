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

Run a non-IID severity robustness sweep:

```bash
python3 experiments/fedavg_cpu_only.py --non-iid-sweep 0.2,1.4,3.0 --json-out non_iid_sweep_metrics.json
```

A sample sweep summary is available in `reports/non_iid_severity_sweep.md`.

Run with mock secure aggregation (masked update aggregation + overhead metrics):

```bash
python3 experiments/fedavg_cpu_only.py --secure-aggregation --json-out secure_metrics.json
```

Run the local retrieval baseline (dependency-light TF-IDF):

```bash
python3 experiments/local_retrieval_baseline.py --json-out retrieval_metrics.json
```

Retrieval dataset files:

- `data/retrieval_corpus.json`
- `data/retrieval_queries.json`

Example retrieval summary: `reports/local_retrieval_baseline.md`.

## Policy docs

- `BASELINE_POLICY.md`
- `GOVERNANCE.md`
- `CHANGELOG.md`
- `PROJECTS.md`

## Next milestones

See `ROADMAP.md` for phased execution.

## Work tracking

- Project board: <https://github.com/users/albot-dev/projects/1>
- Active milestone (Phase 2): <https://github.com/albot-dev/OpenMeshMind/milestone/2>
- Current issues:
  - <https://github.com/albot-dev/OpenMeshMind/issues/6>
  - <https://github.com/albot-dev/OpenMeshMind/issues/7>
  - <https://github.com/albot-dev/OpenMeshMind/issues/8>
- Completed Phase 1 milestone: <https://github.com/albot-dev/OpenMeshMind/milestone/1>
- Completed Phase 1 issues:
  - <https://github.com/albot-dev/OpenMeshMind/issues/1>
  - <https://github.com/albot-dev/OpenMeshMind/issues/2>
  - <https://github.com/albot-dev/OpenMeshMind/issues/3>
  - <https://github.com/albot-dev/OpenMeshMind/issues/4>
  - <https://github.com/albot-dev/OpenMeshMind/issues/5>
