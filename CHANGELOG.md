# Changelog

All notable changes to OpenMeshMind are documented here.

## 2026-02-16

### Added

- Added cohort diversity/readiness thresholds to `scripts/check_cohort_manifest.py`:
  - `--min-distinct-regions`
  - `--min-distinct-hardware-tiers`
  - `--min-distinct-network-tiers`
  - `--max-unknown-region-ratio`
- Added validation coverage for cohort diversity pass/fail paths in `tests/test_validation_scripts.py`.
- Added forward execution plan: `docs/NEXT_STEPS.md`.
- Extended roadmap with post-MVP externalization/capability-expansion phase in `ROADMAP.md`.
- Added long-context retrieval task family to generality/reproducibility gates:
  - `data/retrieval_long_context_corpus.json`
  - `data/retrieval_long_context_queries.json`
  - `scripts/evaluate_generality.py`
  - `scripts/check_generality.py`
  - `scripts/reproducibility_sweep.py`
  - `scripts/check_reproducibility.py`
- Added multi-step tool-chain task family to generality/reproducibility gates:
  - `scripts/evaluate_generality.py`
  - `scripts/check_generality.py`
  - `scripts/reproducibility_sweep.py`
  - `scripts/check_reproducibility.py`
- Added/updated tests for the expanded task families and validator thresholds:
  - `tests/test_evaluate_generality.py`
  - `tests/test_reproducibility_sweep.py`
  - `tests/test_validation_scripts.py`
- Updated docs for expanded MVP gate scope and thresholds:
  - `README.md`
  - `docs/MVP_CRITERIA.md`
  - `docs/NEXT_STEPS.md`
  - `ROADMAP.md`

## 2026-02-15

### Added

- Added explicit coming-goals and sub-goals tracker: `docs/COMING_GOALS.md`.
- Added machine-readable main-track status generator and markdown renderer:
  - `scripts/main_track_status.py`
  - `main_track_status.json`
  - `reports/main_track_status.md`
- Integrated main-track status generation into:
  - smoke path (`scripts/smoke_check.py`)
  - CI pipeline (`.github/workflows/cpu-baseline.yml`)
  - weekly reporting bundle/summary (`scripts/generate_weekly_report.py`)
- Added unit tests for main-track status behavior: `tests/test_main_track_status.py`.
- Added conversational continuity metric to the generality/reproducibility gates:
  - `conversation_continuity` task in `scripts/evaluate_generality.py`
  - threshold checks in `scripts/check_generality.py` and `scripts/check_reproducibility.py`
  - reproducibility aggregation updates in `scripts/reproducibility_sweep.py`
- Added provenance manifest automation and validation:
  - `scripts/build_provenance_manifest.py`
  - `scripts/check_provenance_manifest.py`
  - integrated into weekly + pilot report generators and CI artifact checks
- Promoted pilot rehearsal artifacts to a 6-node single-operator multi-machine cohort:
  - updated `pilot/cohort_manifest.json` with passed onboarding entries
  - `pilot/cohort_onboarding_summary.json`
  - `pilot/pilot_cohort_metrics.json`
  - `pilot/nodes/*/pilot_metrics.json`
- Added day-level cohort artifacts for a 14-day pilot window:
  - `pilot/runs/day01/pilot_cohort_metrics.json` ... `pilot/runs/day14/pilot_cohort_metrics.json`
- Published updated pilot reports and bundles:
  - `reports/pilot_status.md`
  - `reports/pilot_artifacts.tgz`
  - `reports/pilot_14_day_report.md`
  - `reports/pilot_14_day_artifacts.tgz`

## 2026-02-14

### Added

- Added local generalist runtime MVP with:
  - lightweight intent router
  - local corpus retrieval
  - calculator tool execution
  - short-term conversational memory
  - `scripts/local_generalist_runtime.py`
- Added federated adapter-style intent experiment (low-rank adapter proxy):
  - `experiments/fedavg_adapter_intent.py`
  - `scripts/check_adapter_intent.py`
- Improved adapter int8 quality via segmented quantization across parameter sections.
- Switched adapter local training to mini-batch updates for more stable convergence.
- Added adapter reference metrics to the generality and reproducibility gate pipelines.
- Added per-section percentile clipping before adapter int8 quantization.
- Tightened adapter int8 quality gates (`max drop <= 0.15`) after stabilization sweep.
- Added generality evaluation and validation gates:
  - `scripts/evaluate_generality.py`
  - `scripts/check_generality.py`
  - `scripts/reproducibility_sweep.py`
  - `scripts/check_reproducibility.py`
  - `docs/MVP_CRITERIA.md`
- Added unit tests for the runtime and evaluator:
  - `tests/test_fedavg_adapter_intent.py`
  - `tests/test_local_generalist_runtime.py`
  - `tests/test_evaluate_generality.py`
  - `tests/test_reproducibility_sweep.py`
- Extended validator test coverage for generality metrics in `tests/test_validation_scripts.py`.
- Integrated generality evaluation into low-end smoke path (`scripts/smoke_check.py`).
- Included `generality_metrics.json` in weekly report artifact expectations and summary generation (`scripts/generate_weekly_report.py`).
- Included adapter and reproducibility artifacts in weekly/release automation and CI workflow gates.
- Updated project docs to include MVP gate usage and threshold references:
  - `README.md`
  - `CONTRIBUTING.md`
  - `ROADMAP.md`

## 2026-02-13

### Added

- Heterogeneous client-capacity fairness simulation in the CPU FedAvg experiment.
- Fairness metrics in JSON output, including participation/contribution disparity indicators.
- Per-client fairness breakdowns for contribution rates and completion rates.
- Fairness run snapshot report: `reports/fairness_capacity_simulation.md`.
- Unit test suite covering core experiment math, retrieval baseline behavior, and validator scripts.
- Minimal project metadata in `pyproject.toml`.
- Fairness validator script: `scripts/check_fairness.py`.
- CI fairness scenario + validation gate with uploaded fairness artifact.
- CPU-only local classification utility baseline (`experiments/local_classification_baseline.py`).
- Classification baseline snapshot report: `reports/local_classification_baseline.md`.
- Classification validator script: `scripts/check_classification.py`.
- Benchmark suite now includes `classification_baseline` task in reduced/full modes.
- CI now validates classification metrics and uploads classification artifact.
- Federated utility classification experiment with `fp32`, `int8`, and `sparse` modes.
- Federated utility snapshot report: `reports/fedavg_classification_utility.md`.
- Utility fairness stress sweep support in federated classification experiment.
- Utility fairness validator script: `scripts/check_utility_fairness.py`.
- Utility fairness snapshot report: `reports/fedavg_classification_fairness.md`.
- Added governance artifacts: `DECISION_LOG.md`, `PROVENANCE_TEMPLATE.md`, and PR template checklist.
- Added low-end one-command smoke path: `scripts/smoke_check.py`.
- Added low-end runtime/memory/troubleshooting notes: `reports/low_end_smoke_path.md`.
- Added explicit OSS license: `LICENSE` (MIT).
- Added release process guide: `RELEASE.md`.
- Added tag-triggered release workflow: `.github/workflows/release.yml`.
- Added weekly reporting generator and bundle path: `scripts/generate_weekly_report.py`.
- Added weekly report template: `reports/WEEKLY_STATUS_TEMPLATE.md`.
- Added versioning and schema compatibility policy: `VERSIONING_POLICY.md`.
- Added volunteer pilot node runner and config template:
  - `scripts/pilot_node_runner.py`
  - `pilot/node_config.example.json`
  - `PILOT_NODE.md`
- Added pilot metrics contract and validator:
  - `schemas/pilot_metrics.schema.v1.json`
  - `scripts/check_pilot_metrics.py`
  - `scripts/build_pilot_metrics.py` updates for uptime/status collection fields
  - `reports/PILOT_STATUS_TEMPLATE.md`
- Added pilot incident/rollback/escalation playbook: `PILOT_OPERATIONS.md`.
- Added incident response label taxonomy for pilot operations (`pilot:*` labels).
- Added cohort-level pilot aggregation + validation:
  - `scripts/build_pilot_cohort_metrics.py`
  - `schemas/pilot_cohort.schema.v1.json`
  - `scripts/check_pilot_cohort.py`
- Added pilot status report automation:
  - `scripts/generate_pilot_status_report.py`
  - tokenized `reports/PILOT_STATUS_TEMPLATE.md`
- Added pilot governance cadence and intake templates:
  - `PILOT_GOVERNANCE.md`
  - `PILOT_DECISION_INTAKE_TEMPLATE.md`
- Added cohort onboarding manifest workflow:
  - `pilot/cohort_manifest.schema.v1.json`
  - `pilot/cohort_manifest.example.json`
  - `pilot/cohort_manifest.json`
  - `pilot/COHORT_ONBOARDING_CHECKLIST.md`
  - `scripts/check_cohort_manifest.py`
- Added 14-day pilot execution/report artifacts:
  - `pilot/PILOT_14_DAY_RUNBOOK.md`
  - `pilot/incident_log.example.json`
  - `pilot/incident_log.json`
  - `reports/PILOT_14_DAY_REPORT_TEMPLATE.md`
  - `scripts/generate_pilot_14_day_report.py`
- Added one-command volunteer bootstrap script:
  - `scripts/volunteer_node_setup.sh`
- Added solo multi-machine ingest/automation command:
  - `scripts/solo_multi_machine_mode.py`

## 2026-02-12

### Added

- CPU-only baseline experiment and CI regression checks.
- Drop-out simulation with comparison vs no-dropout baseline.
- Non-IID severity sweep with dedicated summary report.
- Mock secure aggregation mode with overhead metrics.

### Baseline Policy

- Recorded baseline metrics schema version `2`.
- Recorded CI threshold defaults:
  - `max_accuracy_drop=0.03`
  - `min_int8_accuracy=0.82`
  - `min_comm_reduction=50%`
