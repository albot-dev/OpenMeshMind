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

Run heterogeneous capacity fairness simulation:

```bash
python3 experiments/fedavg_cpu_only.py --simulate-client-capacity --json-out fairness_metrics.json
python3 scripts/check_fairness.py fairness_metrics.json
```

A sample fairness summary is available in `reports/fairness_capacity_simulation.md`.

Run the local classification utility baseline (CPU-only, dependency-light):

```bash
python3 experiments/local_classification_baseline.py --json-out classification_metrics.json
python3 scripts/check_classification.py classification_metrics.json
```

Sample classification summary: `reports/local_classification_baseline.md`.

Run federated utility classification (fp32/int8/sparse):

```bash
python3 experiments/fedavg_classification_utility.py --json-out utility_fedavg_metrics.json
```

Sample federated utility summary: `reports/fedavg_classification_utility.md`.

Run utility fairness stress checks (capacity + churn):

```bash
python3 experiments/fedavg_classification_utility.py \
  --simulate-client-capacity \
  --dropout-rate 0.1 \
  --round-deadline-sweep 4.0,4.2 \
  --json-out utility_fairness_metrics.json
python3 scripts/check_utility_fairness.py utility_fairness_metrics.json
```

Utility fairness stress summary: `reports/fedavg_classification_fairness.md`.

Run the local retrieval baseline (dependency-light TF-IDF):

```bash
python3 experiments/local_retrieval_baseline.py --json-out retrieval_metrics.json
```

Retrieval dataset files:

- `data/retrieval_corpus.json`
- `data/retrieval_queries.json`

Example retrieval summary: `reports/local_retrieval_baseline.md`.

Run the benchmark suite (latency + memory fields):

```bash
python3 scripts/benchmark_suite.py --mode full --json-out benchmark_metrics.json
```

Run reduced benchmark mode (CI-friendly):

```bash
python3 -m unittest discover -s tests -p "test_*.py"
python3 scripts/benchmark_suite.py --mode reduced --json-out benchmark_metrics.json
python3 scripts/check_benchmarks.py benchmark_metrics.json --expected-mode reduced
```

Low-end contributor smoke path (one command):

```bash
python3 scripts/smoke_check.py --include-fairness --json-out smoke_summary.json
```

Runtime/memory expectations and troubleshooting: `reports/low_end_smoke_path.md`.

## Policy docs

- `BASELINE_POLICY.md`
- `GOVERNANCE.md`
- `DECISION_LOG.md`
- `PROVENANCE_TEMPLATE.md`
- `CHANGELOG.md`
- `PROJECTS.md`

## Next milestones

See `ROADMAP.md` for phased execution.

## Work tracking

- Project board: <https://github.com/users/albot-dev/projects/1>
- Active milestone (Phase 3): <https://github.com/albot-dev/OpenMeshMind/milestone/3>
- Current issues:
  - <https://github.com/albot-dev/OpenMeshMind/issues/13>
  - <https://github.com/albot-dev/OpenMeshMind/issues/14>
- Completed Phase 1 milestone: <https://github.com/albot-dev/OpenMeshMind/milestone/1>
- Completed issues:
  - <https://github.com/albot-dev/OpenMeshMind/issues/1>
  - <https://github.com/albot-dev/OpenMeshMind/issues/2>
  - <https://github.com/albot-dev/OpenMeshMind/issues/3>
  - <https://github.com/albot-dev/OpenMeshMind/issues/4>
  - <https://github.com/albot-dev/OpenMeshMind/issues/5>
  - <https://github.com/albot-dev/OpenMeshMind/issues/6>
  - <https://github.com/albot-dev/OpenMeshMind/issues/7>
  - <https://github.com/albot-dev/OpenMeshMind/issues/8>
  - <https://github.com/albot-dev/OpenMeshMind/issues/9>
  - <https://github.com/albot-dev/OpenMeshMind/issues/10>
  - <https://github.com/albot-dev/OpenMeshMind/issues/11>
  - <https://github.com/albot-dev/OpenMeshMind/issues/12>
