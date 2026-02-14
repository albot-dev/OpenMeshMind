# Changelog

All notable changes to OpenMeshMind are documented here.

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
- Added adapter reference metrics to the generality and reproducibility gate pipelines.
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
