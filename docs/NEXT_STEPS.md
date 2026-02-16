# Next Steps Plan

This plan continues from the completed MVP, pilot execution, and provenance milestones.

## Current State (2026-02-16)

- Main-track goals complete: `4/4` (`main_track_status.json`)
- Open issues: `0`
- Open milestones: `0`
- Weekly and pilot provenance manifests are generated and validated.

## Phase N1: External Cohort Readiness Hardening (Can do now)

Goal: tighten validation before wider external volunteer intake.

Sub-goals:

- Add cohort diversity/readiness checks to manifest validation.
- Keep onboarding and runbook docs aligned with those checks.
- Keep solo multi-machine flow green under stricter checks.

Exit criteria:

- `scripts/check_cohort_manifest.py` supports diversity constraints.
- Tests cover pass/fail diversity scenarios.
- Docs show strict-check command examples.

## Phase N2: External Cohort Intake (Needs external participants)

Goal: run a true multi-operator cohort (not single-operator simulation).

Sub-goals:

- Reach at least 5 externally submitted onboarding bundles.
- Pass strict manifest checks including diversity constraints.
- Publish refreshed pilot status + 14-day report from external data.

Exit criteria:

- `pilot/cohort_manifest.json` contains >=5 externally sourced passed nodes.
- `reports/pilot_status.md` and `reports/pilot_14_day_report.md` regenerated from external cohort artifacts.

## Phase N3: Capability Expansion on Commodity Hardware (Can do now)

Goal: increase task breadth while keeping CPU-first reproducibility.

Sub-goals:

- Add harder long-context retrieval and multi-step tool tasks.
- Extend gate thresholds and reproducibility summaries.
- Keep smoke path and CI runtime envelope stable.

Exit criteria:

- New task families are included in `scripts/evaluate_generality.py` and validated by checkers.
- Tests and smoke path remain fully passing.

## Immediate execution order

1. Complete N1 hardening (in progress now).
2. Start N3 task expansion after N1 merges green.
3. Trigger N2 when external submissions are available.
