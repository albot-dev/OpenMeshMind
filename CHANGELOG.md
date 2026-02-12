# Changelog

All notable changes to OpenMeshMind are documented here.

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
