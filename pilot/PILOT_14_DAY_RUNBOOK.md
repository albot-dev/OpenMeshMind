# 14-Day Pilot Runbook

This runbook defines daily execution for the first 14-day pilot window.

## Before day 1

- Finalize `pilot/cohort_manifest.json` from `pilot/cohort_manifest.example.json`.
- If running solo across personal machines, first import all onboarding bundles:

```bash
python3 scripts/solo_multi_machine_mode.py \
  --bundles-glob 'pilot/submissions/*_onboarding_*.tgz' \
  --min-nodes 1 \
  --min-passed 0 \
  --require-metrics-files \
  --no-pipeline
```

- Validate cohort readiness:

```bash
python3 scripts/check_cohort_manifest.py \
  pilot/cohort_manifest.json \
  --min-nodes 1 \
  --min-passed 0 \
  --require-metrics-files \
  --summary-json-out pilot/cohort_onboarding_summary.json
```

For the full public volunteer cohort gate, use `--min-nodes 5 --min-passed 5`.

- Ensure incident labels and escalation process are active (`PILOT_OPERATIONS.md`).

## Daily cycle (days 1-14)

1. Collect node artifacts from each active node into `pilot/runs/dayXX/nodes/<node_id>/pilot_metrics.json`.
2. Build day-level cohort summary:

```bash
python3 scripts/build_pilot_cohort_metrics.py \
  --metrics-glob pilot/runs/dayXX/nodes/*/pilot_metrics.json \
  --json-out pilot/runs/dayXX/pilot_cohort_metrics.json
```

3. Validate day-level cohort health:

```bash
python3 scripts/check_pilot_cohort.py \
  pilot/runs/dayXX/pilot_cohort_metrics.json \
  --min-node-count 1 \
  --min-uptime-ratio-mean 0.90 \
  --min-last-cycle-ok-ratio 0.80
```

For full public cohort readiness checks, use `--min-node-count 5`.

4. Generate/update daily pilot status draft:

```bash
python3 scripts/generate_pilot_status_report.py \
  --cohort-metrics pilot/runs/dayXX/pilot_cohort_metrics.json \
  --out reports/pilot_status_dayXX.md
```

5. Log incidents in `pilot/incident_log.json` (or open new incident issues if needed).

## Day checkpoints

- Day 1: confirm baseline cohort stability and initial incident response latency.
- Day 3: review fairness spread and node churn trends.
- Day 7: mid-point governance review and threshold adjustment decision if required.
- Day 10: confirm publication-readiness for final report artifacts.
- Day 14: lock data window and generate final 14-day report.

## End-of-window report generation

```bash
python3 scripts/generate_pilot_14_day_report.py \
  --manifest pilot/cohort_manifest.json \
  --onboarding-summary pilot/cohort_onboarding_summary.json \
  --daily-cohort-glob 'pilot/runs/day*/pilot_cohort_metrics.json' \
  --incident-log pilot/incident_log.json \
  --out reports/pilot_14_day_report.md \
  --bundle-out reports/pilot_14_day_artifacts.tgz
```

## Required final outputs

- `reports/pilot_14_day_report.md`
- `reports/pilot_14_day_artifacts.tgz`
- Follow-up issues for each major unresolved risk.
