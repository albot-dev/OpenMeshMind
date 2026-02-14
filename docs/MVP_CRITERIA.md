# Generalist MVP Criteria

## Purpose

Define a concrete, testable MVP gate for "working general AI on commodity hardware" in this repository.

This MVP is intentionally practical:

- CPU-first
- reproducible from clean checkout
- measurable against explicit thresholds
- open to decentralized extension over time

## MVP scope

A build is considered MVP-ready when it can pass the following task families locally:

1. Local classification quality
2. Local retrieval quality
3. Instruction following (action compliance)
4. Tool use (calculator correctness)
5. Centralized vs federated distributed reference comparison
6. Adapter-style federated proxy check (low-rank intent adapter)

## Required thresholds

These are the default gates implemented in `scripts/check_generality.py`.

| Area | Metric | Threshold |
|---|---|---|
| Classification | accuracy | `>= 0.80` |
| Classification | macro F1 | `>= 0.78` |
| Retrieval | recall@1 | `>= 0.60` |
| Retrieval | MRR | `>= 0.75` |
| Instruction following | pass rate | `>= 0.75` |
| Tool use | pass rate | `>= 0.80` |
| Aggregate | overall score | `>= 0.75` |
| Runtime envelope | total eval wall clock | `<= 180s` |
| Distributed reference | int8 accuracy drop vs centralized | `<= 0.10` |
| Distributed reference | int8 communication savings vs fp32 | `>= 40%` |
| Adapter reference | int8 accuracy drop vs centralized | `<= 0.15` |
| Adapter reference | int8 communication savings vs fp32 | `>= 40%` |

## How to run

Quick gate:

```bash
python3 scripts/evaluate_generality.py --skip-distributed-reference --json-out generality_metrics.json
python3 scripts/check_generality.py generality_metrics.json
```

Full gate with distributed reference:

```bash
python3 scripts/evaluate_generality.py --json-out generality_metrics.json
python3 scripts/check_generality.py generality_metrics.json
python3 experiments/fedavg_adapter_intent.py --json-out adapter_intent_metrics.json
python3 scripts/check_adapter_intent.py adapter_intent_metrics.json
python3 scripts/reproducibility_sweep.py --seeds 7,17,27 --json-out reproducibility_metrics.json
python3 scripts/check_reproducibility.py reproducibility_metrics.json
```

Full smoke path (includes generality checks):

```bash
python3 scripts/smoke_check.py --include-fairness --json-out smoke_summary.json
```

## Current operating mode

- Single-operator multi-machine execution is acceptable for current progress.
- Volunteer data intake stays open via `pilot/VOLUNTEER_DATA_SUBMISSION.md`.
- External volunteer participation is optional for passing this MVP gate.

## Exit conditions for next phase

Before moving beyond MVP:

1. Pass the full gate across at least three independent machine runs.
2. Keep all thresholds passing for two consecutive weekly reports.
3. Record the decision to advance in `DECISION_LOG.md`.
