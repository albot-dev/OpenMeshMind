# MVP User Value Plan

## Purpose

Define who gets value from the MVP, what outcomes matter, and which phased milestones keep this repository useful on commodity hardware.

## User personas

| Persona | Core job to be done | MVP user value |
|---|---|---|
| Solo builder on a CPU-only laptop | Validate end-to-end capability quickly | One-command smoke run plus clear pass/fail artifacts |
| Volunteer operator | Contribute useful pilot artifacts without custom infra | Standard setup and validation scripts with predictable outputs |
| Maintainer/release owner | Make go/no-go decisions with low ambiguity | Strict machine-readable status plus reproducibility/fairness evidence |

## User-value outcomes and measurable targets

| Outcome ID | User-value outcome | Target | Verification artifacts |
|---|---|---|---|
| U1 | Fast contributor confidence loop | `smoke_summary.json` reports `ok=true` with fairness enabled | `smoke_summary.json` |
| U2 | Broad local general capability | `scripts/check_generality.py` passes with default thresholds | `generality_metrics.json` |
| U3 | Repeatable, not one-off, capability | `scripts/check_reproducibility.py` passes with `run_count >= 3` | `reproducibility_metrics.json` |
| U4 | Efficient decentralized training quality | Baseline + adapter checks pass with communication savings preserved | `baseline_metrics.json`, `adapter_intent_metrics.json` |
| U5 | Fairness resilience under client heterogeneity | Both fairness validators pass | `fairness_metrics.json`, `utility_fairness_metrics.json` |
| U6 | Single readiness signal for maintainers | Strict status shows `4/4` goals complete and `9/9` required sub-goals complete | `main_track_status.json`, `reports/main_track_status.md` |

## Phased milestones

### Phase M1: Contributor confidence path

Scope:

- Keep baseline, classification, adapter, and reduced benchmark checks green.
- Keep smoke execution simple for low-end/CPU contributors.

Exit targets:

- U1 complete.
- U4 complete.

### Phase M2: Generality and reproducibility stability

Scope:

- Keep full generality task-family gate passing.
- Keep multi-seed reproducibility sweep passing.

Exit targets:

- U2 complete.
- U3 complete.

### Phase M3: Fairness and strict release readiness

Scope:

- Keep fairness metrics passing under heterogeneous and churn-stress scenarios.
- Enforce strict main-track status for release decisions.

Exit targets:

- U5 complete.
- U6 complete.

### Phase M4: External cohort scale-up

Scope:

- Apply the same MVP gates to external cohort updates.
- Regenerate pilot status artifacts from external submissions.

Exit targets:

- External intake target in `docs/NEXT_STEPS.md` Phase N2 is met.
- DoD checklist in `docs/MVP_DEFINITION_OF_DONE.md` remains fully passing after artifact refresh.

## Execution rule

Use `docs/MVP_DEFINITION_OF_DONE.md` as the required verification checklist for MVP-ready merges and release decisions.
