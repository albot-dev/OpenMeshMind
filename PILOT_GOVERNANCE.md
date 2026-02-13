# Pilot Governance Cadence

This document defines the recurring governance process for active pilot operation.

## Weekly cadence

- Schedule: every Monday (UTC)
- Duration: 30 minutes
- Required attendees:
  - pilot incident commander on duty
  - at least one active node operator representative
  - maintainer owner for follow-up assignment

## Weekly agenda

1. Review latest pilot status report (`reports/pilot_status.md`).
2. Review cohort onboarding summary (`pilot/cohort_onboarding_summary.json`).
3. Review progress against 14-day runbook (`pilot/PILOT_14_DAY_RUNBOOK.md`).
4. Review all open `pilot:*` incident issues.
5. Review cohort metrics trend deltas (uptime, fairness, quality, communication).
6. Confirm decision intake items from `PILOT_DECISION_INTAKE_TEMPLATE.md`.
7. Record accepted decisions in `DECISION_LOG.md` with linked issue IDs.

## Ownership expectations

- Pilot chair (default `@albot-dev`) runs the cadence and assigns owners.
- Every accepted action must have:
  - a linked GitHub issue
  - an explicit owner
  - a target date
- Any operational policy change must be captured in both:
  - `DECISION_LOG.md`
  - `PILOT_OPERATIONS.md` or `PILOT_NODE.md` (as applicable)

## Decision traceability rules

- No pilot-impacting decision is considered complete without a public issue reference.
- Decision log entries should include:
  - date
  - decision summary
  - alternatives considered
  - expected impact on accessibility/decentralization

## Escalation tie-in

- Active incidents use the process in `PILOT_OPERATIONS.md`.
- Governance cadence must verify postmortem completion for all closed `SEV-1` and `SEV-2` incidents.
