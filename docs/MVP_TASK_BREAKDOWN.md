# MVP Task Breakdown

This backlog decomposes MVP delivery into small, verifiable tasks.

Each task has a Definition of Done (DoD) command so completion can be re-checked at any time.

## Foundation Tasks (M1)

| Task ID | Task | Output | Verification command |
|---|---|---|---|
| T1.1 | Keep baseline gate passing | `baseline_metrics.json` | `python3 scripts/check_baseline.py baseline_metrics.json --expected-schema-version 2` |
| T1.2 | Keep classification gate passing | `classification_metrics.json` | `python3 scripts/check_classification.py classification_metrics.json --expected-schema-version 1` |
| T1.3 | Keep adapter gate passing | `adapter_intent_metrics.json` | `python3 scripts/check_adapter_intent.py adapter_intent_metrics.json --expected-schema-version 1` |
| T1.4 | Keep reduced benchmark envelope passing | `benchmark_metrics.json` | `python3 scripts/check_benchmarks.py benchmark_metrics.json --expected-schema-version 1 --expected-mode reduced` |

## Generality Tasks (M2)

| Task ID | Task | Output | Verification command |
|---|---|---|---|
| T2.1 | Maintain classification/retrieval/instruction/tool generality tasks | `generality_metrics.json` | `python3 scripts/check_generality.py generality_metrics.json --expected-schema-version 1` |
| T2.2 | Maintain long-context retrieval and multi-step tool-chain tasks | `generality_metrics.json` | `python3 scripts/check_generality.py generality_metrics.json --expected-schema-version 1` |
| T2.3 | Keep reproducibility sweep stable over 3+ seeds | `reproducibility_metrics.json` | `python3 scripts/check_reproducibility.py reproducibility_metrics.json --expected-schema-version 1` |

## Readiness Tasks (M3)

| Task ID | Task | Output | Verification command |
|---|---|---|---|
| T3.1 | Keep fairness gates passing in strict mode | `fairness_metrics.json`, `utility_fairness_metrics.json` | `python3 scripts/check_fairness.py fairness_metrics.json --expected-schema-version 2 && python3 scripts/check_utility_fairness.py utility_fairness_metrics.json --expected-schema-version 1` |
| T3.2 | Keep strict goal roll-up green | `main_track_status.json`, `reports/main_track_status.md` | `python3 scripts/main_track_status.py --require-fairness --require-smoke-summary --fail-on-incomplete --json-out main_track_status.json --md-out reports/main_track_status.md` |
| T3.3 | Keep machine-verifiable readiness check green | `mvp_readiness.json` | `python3 scripts/check_mvp_readiness.py --require-fairness --require-all-goals-done --json-out mvp_readiness.json` |
| T3.4 | Keep full strict smoke path green | `smoke_summary.json` | `python3 scripts/smoke_check.py --include-fairness --json-out smoke_summary.json` |

## Externalization Tasks (M4)

| Task ID | Task | Output | Verification command |
|---|---|---|---|
| T4.1 | Validate cohort intake artifacts under strict manifest checks | `pilot/cohort_onboarding_summary.json` | `python3 scripts/check_cohort_manifest.py pilot/cohort_manifest.json --min-nodes 5 --min-passed 5 --require-metrics-files --min-distinct-regions 3 --min-distinct-hardware-tiers 2 --min-distinct-network-tiers 2 --max-unknown-region-ratio 0.5` |
| T4.2 | Regenerate pilot status from refreshed cohort artifacts | `reports/pilot_status.md` | `python3 scripts/generate_pilot_status_report.py --pilot-metrics pilot/pilot_metrics.json --cohort-metrics pilot/pilot_cohort_metrics.json --out reports/pilot_status.md --bundle-out reports/pilot_artifacts.tgz` |
| T4.3 | Regenerate 14-day pilot report from refreshed cohort artifacts | `reports/pilot_14_day_report.md` | `python3 scripts/generate_pilot_14_day_report.py --manifest pilot/cohort_manifest.json --onboarding-summary pilot/cohort_onboarding_summary.json --daily-cohort-glob 'pilot/runs/day*/pilot_cohort_metrics.json' --incident-log pilot/incident_log.json --out reports/pilot_14_day_report.md --bundle-out reports/pilot_14_day_artifacts.tgz` |

## Current execution order

1. Run T3.4 (strict smoke) to refresh all gate artifacts.
2. Run T3.2 and T3.3 to confirm strict readiness.
3. For external submissions, run T4.1 then refresh reports via T4.2 and T4.3.
