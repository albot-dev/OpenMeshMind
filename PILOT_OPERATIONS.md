# Pilot Operations Playbook

This playbook defines incident handling, rollback, and escalation for Phase 5 pilot operation.

## Roles and ownership

- Incident commander: maintainer on duty (default `@albot-dev`)
- Node operator: volunteer running the affected node
- Communications owner: maintainer posting status updates and closure summary

## Severity and response SLAs

- `SEV-1` (pilot unavailable or severe data integrity risk)
  - Acknowledge: within 15 minutes
  - Initial mitigation: within 60 minutes
  - Public status update: within 60 minutes
- `SEV-2` (degraded service, partial node failures, recurring validation failures)
  - Acknowledge: within 4 hours
  - Mitigation plan: within 24 hours
- `SEV-3` (non-blocking defects, documentation/process gaps)
  - Acknowledge: within 2 business days
  - Scheduled fix: next weekly planning cycle

## Incident trigger examples

- `scripts/pilot_node_runner.py --health` returns non-zero.
- `scripts/check_pilot_metrics.py` fails schema or readiness checks.
- Uptime ratio drops below 0.90 over active window.
- Multiple volunteer nodes fail the same smoke step.

## Incident response flow

1. Create a GitHub issue using labels: `pilot:incident` and severity label (`pilot:sev1`, `pilot:sev2`, or `pilot:sev3`).
2. Assign an incident commander and owner for remediation.
3. Capture current state artifacts (`pilot/node_state.json`, `pilot/pilot_metrics.json`, `pilot/node_runner.log`).
4. Apply mitigation or rollback (see rollback runbook below).
5. Post updates every SLA interval until resolved.
6. Close with root cause and follow-up actions, then add `pilot:postmortem` label.

## Rollback runbook

1. Stop the node runner process.
2. Checkout last known good release tag or commit.
3. Re-run smoke and pilot metric checks:

```bash
python3 scripts/smoke_check.py --include-fairness --json-out smoke_summary.json
python3 scripts/build_pilot_metrics.py --last-cycle-ok --step-count 1 --uptime-ratio-24h 1.0 --json-out pilot/pilot_metrics.json
python3 scripts/check_pilot_metrics.py pilot/pilot_metrics.json --require-status-collected
```

4. Restart node loop:

```bash
python3 scripts/pilot_node_runner.py --config pilot/node_config.json
```

5. Confirm health and update incident issue with rollback timestamp and restored status.

## Escalation path

1. `@albot-dev` (primary maintainer)
2. active contributors listed in the weekly pilot status report
3. broad community call for triage support if SEV-1 remains unresolved after 60 minutes

## Required pilot labels

- `pilot:incident`
- `pilot:sev1`
- `pilot:sev2`
- `pilot:sev3`
- `pilot:rollback`
- `pilot:postmortem`

## Post-incident closure checklist

- Incident issue includes timeline, impact, root cause, and mitigation.
- Follow-up issue(s) created for preventive actions.
- Weekly status report includes incident summary and current risk status.
