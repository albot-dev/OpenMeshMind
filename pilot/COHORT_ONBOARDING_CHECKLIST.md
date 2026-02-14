# Pilot Cohort Onboarding Checklist

Use this checklist for each volunteer node included in `pilot/cohort_manifest.json`.

## Pre-checks

- Node assigned anonymized `node_id`.
- Hardware/network metadata recorded in manifest.
- Operator reviewed `PILOT_NODE.md` and `PILOT_OPERATIONS.md`.
- `.env` contains `github_token` with repo read access.

## Execution checks

1. Create node config from template:

```bash
cp pilot/node_config.example.json pilot/node_config.json
```

2. Run one onboarding cycle:

```bash
python3 scripts/pilot_node_runner.py --config pilot/node_config.json --once
```

3. Validate node artifacts:

```bash
python3 scripts/check_pilot_metrics.py pilot/pilot_metrics.json --require-status-collected
```

4. Record onboarding status in `pilot/cohort_manifest.json`:

- `onboarding_status`: `passed` or `failed`
- `onboarding_checked_utc`
- `failure_reason` (required if failed)

## Cohort-level gate

After updating all nodes:

```bash
python3 scripts/check_cohort_manifest.py \
  pilot/cohort_manifest.json \
  --min-nodes 5 \
  --min-passed 5 \
  --require-metrics-files \
  --summary-json-out pilot/cohort_onboarding_summary.json
```

## Reporting handoff

- Include onboarding startup/failure rates in pilot status report.
- Save per-node metrics artifacts under `pilot/nodes/<node_id>/`.
- Link onboarding summary in issue `#21`.
- For solo multi-machine collection, ingest all local bundles and auto-update manifest:
```bash
python3 scripts/solo_multi_machine_mode.py \
  --bundles-glob 'pilot/submissions/*_onboarding_*.tgz' \
  --min-nodes 5 \
  --min-passed 5 \
  --require-metrics-files
```
