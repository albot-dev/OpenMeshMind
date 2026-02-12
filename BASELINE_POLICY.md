# Baseline Metric Policy

This project uses machine-readable baseline metrics to prevent silent regressions in accessibility and quality.

## Schema versioning

- Current metrics schema: `2`
- Schema version is stored in each experiment JSON output at `schema_version`.
- CI validation (`scripts/check_baseline.py`) fails if schema version is not the expected value.

## Baseline thresholds

Default thresholds used by CI:

- `max_accuracy_drop`: `0.03` (int8 FedAvg vs centralized)
- `min_int8_accuracy`: `0.82`
- `min_comm_reduction`: `50%`

## Threshold change process

Any threshold change must include:

1. A GitHub issue describing why the change is needed.
2. Reproducible before/after experiment output.
3. A pull request that updates:
   - CI thresholds in `.github/workflows/cpu-baseline.yml`
   - validation defaults in `scripts/check_baseline.py` (if changed)
   - this policy file (if semantics change)
4. A `CHANGELOG.md` entry describing:
   - old threshold
   - new threshold
   - reason and expected impact

## Compatibility

- Minor metric additions can keep the same schema version.
- Any breaking key rename/removal requires bumping `schema_version`.
