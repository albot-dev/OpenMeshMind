# Governance

OpenMeshMind is designed to remain useful to the public and not become dependent on a single vendor, model host, or hardware tier.

## Core commitments

- Public reproducibility for baseline results.
- Transparent decision records (issues/PRs/discussions).
- Commodity-hardware viability as a first-class requirement.
- Incremental decentralization of training and inference workflows.

## Decision process

- Major changes should be proposed as GitHub issues first.
- Technical decisions should include tradeoffs and measurable acceptance criteria.
- When in doubt, choose the option that improves accessibility and auditability.
- Record major choices in `DECISION_LOG.md`.
- Attach experiment provenance using `PROVENANCE_TEMPLATE.md` for major baseline changes.

## Release policy

- No release claims without reproducible scripts and metrics.
- Regressions in baseline accessibility metrics must be justified and documented.
- Tagged release process is documented in `RELEASE.md`.

## Baseline threshold governance

- Baseline metric schema/version policy is defined in `BASELINE_POLICY.md`.
- Threshold changes require:
  - a linked GitHub issue with rationale
  - reproducible before/after metrics
  - corresponding updates to CI checks
  - a `CHANGELOG.md` note with old/new threshold values

## Maintainer responsibility

- Keep contribution pathways open and documented.
- Prevent opaque dependencies from becoming project bottlenecks.
- Prioritize public-interest use cases over benchmark-only optimization.

## Pilot operations governance

- Volunteer node operations and minimum requirements are documented in `PILOT_NODE.md`.
- Incident, rollback, severity handling, and escalation rules are documented in `PILOT_OPERATIONS.md`.
- Weekly pilot governance cadence and ownership are documented in `PILOT_GOVERNANCE.md`.
- Pilot decision intake template is documented in `PILOT_DECISION_INTAKE_TEMPLATE.md`.
- Pilot metrics contract and validator are tracked in:
  - `schemas/pilot_metrics.schema.v1.json`
  - `scripts/check_pilot_metrics.py`
  - `schemas/pilot_cohort.schema.v1.json`
  - `scripts/check_pilot_cohort.py`
- Public pilot reporting should use `reports/PILOT_STATUS_TEMPLATE.md`.
- Accepted pilot decisions must be linked to:
  - `DECISION_LOG.md`
  - public GitHub issue references
