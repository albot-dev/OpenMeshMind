# Contributing to OpenMeshMind

## Scope

Contributions should improve at least one of:

- accessibility on commodity hardware
- decentralization and resilience
- transparency and reproducibility

## Development principles

- Prefer CPU-first baselines before GPU-only approaches.
- Keep experiments reproducible and lightweight.
- Favor clear metrics over anecdotal claims.
- Avoid adding heavy dependencies unless the gain is concrete and measured.

## Local validation

Run these before opening a pull request:

```bash
python3 -m unittest discover -s tests -p "test_*.py"
python3 experiments/fedavg_cpu_only.py --json-out baseline_metrics.json
python3 scripts/check_baseline.py baseline_metrics.json
python3 experiments/fedavg_cpu_only.py --simulate-client-capacity --quiet --json-out fairness_metrics.json
python3 scripts/check_fairness.py fairness_metrics.json
python3 experiments/local_classification_baseline.py --json-out classification_metrics.json
python3 scripts/check_classification.py classification_metrics.json
python3 experiments/fedavg_classification_utility.py --json-out utility_fedavg_metrics.json
python3 experiments/fedavg_classification_utility.py --simulate-client-capacity --dropout-rate 0.1 --round-deadline-sweep 4.0,4.2 --quiet --json-out utility_fairness_metrics.json
python3 scripts/check_utility_fairness.py utility_fairness_metrics.json
```

## Low-end quickstart

Run the one-command smoke path:

```bash
python3 scripts/smoke_check.py --include-fairness --json-out smoke_summary.json
```

If a step fails, rerun the exact command shown in smoke output.
For expected runtime/memory envelopes and troubleshooting notes, see `reports/low_end_smoke_path.md`.

## Pilot node validation

Before pilot-related changes are merged, run:

```bash
cp pilot/node_config.example.json pilot/node_config.json
python3 scripts/pilot_node_runner.py --config pilot/node_config.json --once
python3 scripts/pilot_node_runner.py --config pilot/node_config.json --health
python3 scripts/check_pilot_metrics.py pilot/pilot_metrics.json --require-status-collected
python3 scripts/check_cohort_manifest.py pilot/cohort_manifest.json --min-nodes 1 --min-passed 0 --require-metrics-files --summary-json-out pilot/cohort_onboarding_summary.json
python3 scripts/solo_multi_machine_mode.py --bundles-glob 'pilot/submissions/*_onboarding_*.tgz' --min-nodes 1 --min-passed 0 --require-metrics-files
python3 scripts/build_pilot_cohort_metrics.py --metrics pilot/pilot_metrics.json --json-out pilot/pilot_cohort_metrics.json
python3 scripts/check_pilot_cohort.py pilot/pilot_cohort_metrics.json --min-node-count 1
python3 scripts/generate_pilot_status_report.py --out reports/pilot_status.md --bundle-out reports/pilot_artifacts.tgz
python3 scripts/generate_pilot_14_day_report.py --manifest pilot/cohort_manifest.json --onboarding-summary pilot/cohort_onboarding_summary.json --daily-cohort-glob 'pilot/runs/day*/pilot_cohort_metrics.json' --incident-log pilot/incident_log.json --out reports/pilot_14_day_report.md --bundle-out reports/pilot_14_day_artifacts.tgz
```

For the full public cohort gate, use `--min-passed 5`.

Pilot runbook and escalation details:

- `PILOT_NODE.md`
- `PILOT_OPERATIONS.md`
- `PILOT_GOVERNANCE.md`
- `PILOT_DECISION_INTAKE_TEMPLATE.md`
- `pilot/COHORT_ONBOARDING_CHECKLIST.md`
- `pilot/PILOT_14_DAY_RUNBOOK.md`
- `pilot/VOLUNTEER_DATA_SUBMISSION.md`
- `reports/PILOT_STATUS_TEMPLATE.md`

## Weekly reporting

Generate the public weekly status file and artifact bundle:

```bash
python3 scripts/generate_weekly_report.py --out reports/weekly_status.md --bundle-out reports/weekly_artifacts.tgz
```

## Pull request expectations

- State the problem and expected user impact.
- Include before/after metrics for behavior changes.
- Keep changes focused; split large efforts into incremental PRs.
- Use `.github/pull_request_template.md` checklist before requesting review.
- For major experiment/policy changes, update:
  - `DECISION_LOG.md`
  - `PROVENANCE_TEMPLATE.md` (filled in context for the change)
- For release work, follow `RELEASE.md`.

## Code style

- Use clear names and minimal comments.
- Keep scripts dependency-light when possible.
- Update `README.md` or `ROADMAP.md` when behavior or direction changes.
