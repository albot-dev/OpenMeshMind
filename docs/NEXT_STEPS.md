# Next Steps Plan

This plan continues from the completed MVP, pilot execution, and provenance milestones.

## Current State (2026-02-16)

- Main-track goals complete: `4/4` (`main_track_status.json`)
- Weekly and pilot provenance manifests are generated and validated.
- External cohort intake remains open, but currently depends on outside participants.

## Execution References

- MVP user-value plan: `docs/MVP_USER_VALUE_PLAN.md`
- MVP Definition of Done checklist: `docs/MVP_DEFINITION_OF_DONE.md`
- MVP granular task backlog: `docs/MVP_TASK_BREAKDOWN.md`
- MVP threshold reference: `docs/MVP_CRITERIA.md`

## Phase N1: External Cohort Readiness Hardening (Completed)

Goal: tighten validation before wider external volunteer intake.

Sub-goals:

- Add cohort diversity/readiness checks to manifest validation.
- Keep onboarding and runbook docs aligned with those checks.
- Keep solo multi-machine flow green under stricter checks.

Exit criteria:

- `scripts/check_cohort_manifest.py` supports diversity constraints.
- Tests cover pass/fail diversity scenarios.
- Docs show strict-check command examples.

Status:

- Completed.

## Phase N2: External Cohort Intake (Needs external participants)

Goal: run a true multi-operator cohort (not single-operator simulation).

Sub-goals:

- Reach at least 5 externally submitted onboarding bundles.
- Pass strict manifest checks including diversity constraints.
- Publish refreshed pilot status + 14-day report from external data.

Exit criteria:

- `pilot/cohort_manifest.json` contains >=5 externally sourced passed nodes.
- `reports/pilot_status.md` and `reports/pilot_14_day_report.md` regenerated from external cohort artifacts.

## Phase N3: Capability Expansion on Commodity Hardware (Completed)

Goal: increase task breadth while keeping CPU-first reproducibility.

Sub-goals:

- Add harder long-context retrieval and multi-step tool tasks.
- Extend gate thresholds and reproducibility summaries.
- Keep smoke path and CI runtime envelope stable.

Exit criteria:

- New task families are included in `scripts/evaluate_generality.py` and validated by checkers.
- Tests and smoke path remain fully passing.

Status:

- Completed with long-context retrieval and multi-step tool-use tasks.

## Phase N4: Solo Multi-Machine Reliability Hardening (Can do now)

Goal: increase confidence in single-operator, multi-machine reproducibility while external cohort intake is pending.

Sub-goals:

- Run repeated multi-seed sweeps on multiple personal machines and compare variance.
- Add failure-injection checks (simulated slow/late nodes, partial artifact loss, resume flow).
- Publish a machine-comparison report with pass/fail gate evidence.

Exit criteria:

- At least 3 machine snapshots in a shared schema pass current generality + reproducibility gates.
- A repeatable replay script reproduces the comparison report from raw artifacts.

Execution commands:

- `python3 scripts/capture_machine_snapshot.py --machine-id my-machine-01 --out-dir pilot/machine_snapshots`
- `python3 scripts/build_machine_comparison_report.py --snapshot-glob 'pilot/machine_snapshots/*' --min-snapshots 3 --require-mvp-readiness --json-out reports/machine_comparison.json --md-out reports/machine_comparison.md`
- `python3 scripts/run_machine_reliability_drill.py --snapshot-glob 'pilot/machine_snapshots/*' --min-snapshots 3 --json-out reports/machine_reliability_drill.json --md-out reports/machine_reliability_drill.md`

Progress:

- Snapshot capture and comparison automation scripts are implemented and tested.
- Remaining work is collecting real snapshots from at least 3 personal machines and publishing the resulting comparison artifacts.

## Immediate execution order

1. Execute N4 reliability hardening and publish machine-comparison artifacts, aligned to phase targets in `docs/MVP_USER_VALUE_PLAN.md`.
2. Keep N2 intake open for external contributors as data arrives, and validate refreshed artifacts with `docs/MVP_DEFINITION_OF_DONE.md`.
3. Regenerate pilot reports when N2 reaches the required external cohort size, then re-run the strict DoD checklist before release decisions.
