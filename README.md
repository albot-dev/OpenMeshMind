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

Run federated adapter-style intent training (low-rank adapter proxy):

```bash
python3 experiments/fedavg_adapter_intent.py \
  --rounds 20 \
  --local-steps 10 \
  --batch-size 8 \
  --learning-rate 0.26 \
  --int8-clip-percentile 0.98 \
  --json-out adapter_intent_metrics.json
python3 scripts/check_adapter_intent.py adapter_intent_metrics.json
```

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
- `data/retrieval_long_context_corpus.json`
- `data/retrieval_long_context_queries.json`

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

Run local generalist runtime (intent routing + retrieval + tool use + memory):

```bash
python3 scripts/local_generalist_runtime.py --interactive
```

Run generalist MVP evaluation gate:

```bash
python3 scripts/evaluate_generality.py --json-out generality_metrics.json
python3 scripts/check_generality.py generality_metrics.json
python3 scripts/reproducibility_sweep.py --seeds 7,17,27 --json-out reproducibility_metrics.json
python3 scripts/check_reproducibility.py reproducibility_metrics.json
```

The generality and reproducibility gates include long-context retrieval checks, multi-turn conversation continuity, multi-step tool chains, plus distributed FedAvg and adapter-reference communication/quality checks.

MVP criteria and thresholds: `docs/MVP_CRITERIA.md`.
MVP user-value plan: `docs/MVP_USER_VALUE_PLAN.md`.
MVP Definition of Done checklist: `docs/MVP_DEFINITION_OF_DONE.md`.
MVP granular task breakdown: `docs/MVP_TASK_BREAKDOWN.md`.

Track current coming goals and sub-goals:

```bash
python3 scripts/main_track_status.py \
  --json-out main_track_status.json \
  --md-out reports/main_track_status.md
```

Strict full-gate status check (fairness required):

```bash
python3 scripts/main_track_status.py \
  --require-fairness \
  --require-smoke-summary \
  --fail-on-incomplete \
  --json-out main_track_status.json \
  --md-out reports/main_track_status.md
```

Machine-verifiable MVP readiness check:

```bash
python3 scripts/check_mvp_readiness.py \
  --require-fairness \
  --require-all-goals-done \
  --json-out mvp_readiness.json
```

Current goals document: `docs/COMING_GOALS.md`.
Execution sequencing: `docs/NEXT_STEPS.md`.

Release process and tagging workflow: `RELEASE.md`.

Weekly public status generation (report + artifact bundle):

```bash
python3 scripts/generate_weekly_report.py \
  --out reports/weekly_status.md \
  --bundle-out reports/weekly_artifacts.tgz
```

This also refreshes `reports/weekly_provenance_manifest.json`.

Run pilot node cycle and health checks (single-operator fast path):

```bash
cp pilot/node_config.example.json pilot/node_config.json
bash scripts/volunteer_node_setup.sh --node-id volunteer-node-001
python3 scripts/solo_multi_machine_mode.py --help
python3 scripts/pilot_node_runner.py --config pilot/node_config.json --once
python3 scripts/pilot_node_runner.py --config pilot/node_config.json --health
python3 scripts/check_pilot_metrics.py pilot/pilot_metrics.json --require-status-collected
python3 scripts/check_cohort_manifest.py \
  pilot/cohort_manifest.json \
  --min-nodes 1 \
  --min-passed 0 \
  --require-metrics-files \
  --summary-json-out pilot/cohort_onboarding_summary.json
# Optional public-cohort readiness constraints:
# --min-distinct-regions 3 \
# --min-distinct-hardware-tiers 2 \
# --min-distinct-network-tiers 2 \
# --max-unknown-region-ratio 0.2
python3 scripts/build_pilot_cohort_metrics.py --metrics pilot/pilot_metrics.json --json-out pilot/pilot_cohort_metrics.json
python3 scripts/check_pilot_cohort.py pilot/pilot_cohort_metrics.json --min-node-count 1
python3 scripts/generate_pilot_status_report.py \
  --pilot-metrics pilot/pilot_metrics.json \
  --cohort-metrics pilot/pilot_cohort_metrics.json \
  --out reports/pilot_status.md \
  --bundle-out reports/pilot_artifacts.tgz
python3 scripts/generate_pilot_14_day_report.py \
  --manifest pilot/cohort_manifest.json \
  --onboarding-summary pilot/cohort_onboarding_summary.json \
  --daily-cohort-glob 'pilot/runs/day*/pilot_cohort_metrics.json' \
  --incident-log pilot/incident_log.json \
  --out reports/pilot_14_day_report.md \
  --bundle-out reports/pilot_14_day_artifacts.tgz

# Solo operator import mode (collect bundles from multiple personal machines)
python3 scripts/solo_multi_machine_mode.py \
  --bundles-glob 'pilot/submissions/*_onboarding_*.tgz' \
  --min-nodes 1 \
  --min-passed 0 \
  --require-metrics-files
```

The solo import pipeline now validates each imported `pilot_metrics.json` against `schemas/pilot_metrics.schema.v1.json` and validates the cohort manifest against `pilot/cohort_manifest.schema.v1.json` before continuing.

Provenance manifests are generated automatically:

- `pilot/pilot_status_provenance.json`
- `pilot/pilot_14_day_provenance.json`

Current pilot mode:

- We are currently running a single-operator multi-machine phase to move quickly.
- External volunteer data is optional right now and not a blocker for ongoing development.
- Public volunteer onboarding remains open at any time.

Volunteer data PRs are open:

- Volunteers can run `scripts/volunteer_node_setup.sh` and submit their generated data through PR whenever they want.
- Maintainers can ingest submitted bundles with `scripts/solo_multi_machine_mode.py`.
- Full submission instructions: `pilot/VOLUNTEER_DATA_SUBMISSION.md`.

Pilot operations references:

- `PILOT_NODE.md`
- `PILOT_OPERATIONS.md`
- `PILOT_GOVERNANCE.md`
- `PILOT_DECISION_INTAKE_TEMPLATE.md`
- `pilot/COHORT_ONBOARDING_CHECKLIST.md`
- `pilot/PILOT_14_DAY_RUNBOOK.md`
- `pilot/cohort_manifest.schema.v1.json`
- `pilot/cohort_manifest.example.json`
- `pilot/VOLUNTEER_DATA_SUBMISSION.md`
- `scripts/volunteer_node_setup.sh`
- `scripts/solo_multi_machine_mode.py`
- `reports/PILOT_14_DAY_REPORT_TEMPLATE.md`
- `schemas/pilot_metrics.schema.v1.json`
- `schemas/pilot_cohort.schema.v1.json`
- `reports/PILOT_STATUS_TEMPLATE.md`

## Policy docs

- `LICENSE`
- `RELEASE.md`
- `VERSIONING_POLICY.md`
- `BASELINE_POLICY.md`
- `GOVERNANCE.md`
- `DECISION_LOG.md`
- `PROVENANCE_TEMPLATE.md`
- `CHANGELOG.md`
- `PROJECTS.md`
- `docs/MVP_CRITERIA.md`
- `docs/COMING_GOALS.md`
- `docs/NEXT_STEPS.md`
- `PILOT_NODE.md`
- `PILOT_OPERATIONS.md`
- `PILOT_GOVERNANCE.md`
- `PILOT_DECISION_INTAKE_TEMPLATE.md`

## Next milestones

See `ROADMAP.md` for phased execution, `docs/COMING_GOALS.md` for current goal/sub-goal completion tracking, and `docs/NEXT_STEPS.md` for the upcoming execution order.

## Work tracking

- Project board: <https://github.com/users/albot-dev/projects/1>
- Open milestones: <https://github.com/albot-dev/OpenMeshMind/milestones?state=open>
- Closed milestones: <https://github.com/albot-dev/OpenMeshMind/milestones?state=closed>
- Open issues: <https://github.com/albot-dev/OpenMeshMind/issues?q=is%3Aissue%20state%3Aopen>
- Closed issues: <https://github.com/albot-dev/OpenMeshMind/issues?q=is%3Aissue%20state%3Aclosed>
- Completed Phase 1 milestone: <https://github.com/albot-dev/OpenMeshMind/milestone/1>
- Completed Phase 2 milestone: <https://github.com/albot-dev/OpenMeshMind/milestone/2>
- Completed Phase 3 milestone: <https://github.com/albot-dev/OpenMeshMind/milestone/3>
- Completed Phase 4 milestone: <https://github.com/albot-dev/OpenMeshMind/milestone/4>
- Completed Phase 6 milestone: <https://github.com/albot-dev/OpenMeshMind/milestone/6>
- Completed Phase 7 milestone: <https://github.com/albot-dev/OpenMeshMind/milestone/7>
- Completed Phase 8 milestone: <https://github.com/albot-dev/OpenMeshMind/milestone/8>
