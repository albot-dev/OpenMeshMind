# Coming Goals and Sub-Goals

This document defines the current execution goals for the main track toward broadly accessible, commodity-hardware general AI.

The goals are intentionally practical and machine-checkable.
Completion is tracked by `scripts/main_track_status.py`.

## Goal 1: CPU-First Federated Foundation

Sub-goals:

- Baseline federated gate passes (`baseline_metrics.json` -> `scripts/check_baseline.py`).
- Local classification gate passes (`classification_metrics.json` -> `scripts/check_classification.py`).
- Adapter intent proxy gate passes (`adapter_intent_metrics.json` -> `scripts/check_adapter_intent.py`).

Done when all required sub-goals pass.

## Goal 2: Generality and Reproducibility

Sub-goals:

- Generality gate passes (`generality_metrics.json` -> `scripts/check_generality.py`).
- Reproducibility sweep gate passes (`reproducibility_metrics.json` -> `scripts/check_reproducibility.py`).

Done when all required sub-goals pass.

## Goal 3: Accessibility and Operational Reliability

Sub-goals:

- Reduced benchmark gate passes (`benchmark_metrics.json` -> `scripts/check_benchmarks.py --expected-mode reduced`).
- Smoke summary reports success (`smoke_summary.json` has `ok=true`).

Use `--require-smoke-summary` to make the smoke sub-goal mandatory.

## Goal 4: Decentralization Fairness Resilience

Sub-goals:

- Baseline fairness gate passes (`fairness_metrics.json` -> `scripts/check_fairness.py`).
- Utility fairness gate passes (`utility_fairness_metrics.json` -> `scripts/check_utility_fairness.py`).

Default mode keeps this goal optional during solo quick runs.
Use `--require-fairness` when running full gates or CI.

## Tracking Commands

Generate status artifacts:

```bash
python3 scripts/main_track_status.py \
  --json-out main_track_status.json \
  --md-out reports/main_track_status.md
```

Full strict status gate (used in CI):

```bash
python3 scripts/main_track_status.py \
  --require-fairness \
  --fail-on-incomplete \
  --json-out main_track_status.json \
  --md-out reports/main_track_status.md
```

Smoke-bound strict gate (requires smoke summary too):

```bash
python3 scripts/main_track_status.py \
  --require-fairness \
  --require-smoke-summary \
  --fail-on-incomplete \
  --json-out main_track_status.json \
  --md-out reports/main_track_status.md
```

## Notes

- Volunteer onboarding remains open, but external volunteer data is not required for MVP gate completion.
- Volunteers can still submit data PRs at any time via `pilot/VOLUNTEER_DATA_SUBMISSION.md`.
