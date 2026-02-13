# Changelog

All notable changes to OpenMeshMind are documented here.

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
