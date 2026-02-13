# Pilot 14-Day Report

- Report date (UTC): {{report_date_utc}}
- Pilot window: {{window_label}}
- Cohort ID: {{cohort_id}}
- Commit/tag: {{commit_or_tag}}
- Artifact bundle: {{artifact_bundle_path}}

## Executive summary

- Overall status: `{{overall_status}}`
- Key outcomes: {{key_outcomes}}
- Main risks: {{main_risks}}

## Onboarding outcomes

- Node count in manifest: {{manifest_node_count}}
- Passed onboarding: {{onboarding_passed_count}}
- Failed onboarding: {{onboarding_failed_count}}
- Startup rate: {{onboarding_startup_rate}}
- Failure rate: {{onboarding_failure_rate}}

## 14-day cohort metrics

| Metric | Value |
| --- | --- |
| Day count with valid cohort artifact | {{daily_count}} |
| Mean uptime ratio | {{uptime_mean}} |
| Minimum uptime ratio | {{uptime_min}} |
| Mean last-cycle-ok ratio | {{last_cycle_ok_mean}} |
| Mean classification accuracy | {{classification_accuracy_mean}} |
| Mean classification macro F1 | {{classification_macro_f1_mean}} |
| Mean utility int8 Jain gain | {{utility_jain_gain_mean}} |
| Mean utility int8 savings % | {{utility_comm_savings_mean}} |

## Incident summary

- Total incidents: {{incident_total}}
- SEV-1 incidents: {{incident_sev1_count}}
- SEV-2 incidents: {{incident_sev2_count}}
- SEV-3 incidents: {{incident_sev3_count}}
- Open incidents at close: {{incident_open_count}}

## Follow-ups

- Follow-up issue list: {{follow_up_issues}}
- Ownership and target dates: {{follow_up_owners}}

## Provenance

- Cohort manifest: `pilot/cohort_manifest.json`
- Onboarding summary: `pilot/cohort_onboarding_summary.json`
- Daily cohort artifacts glob: `pilot/runs/day*/pilot_cohort_metrics.json`
- Incident log: `pilot/incident_log.json`
