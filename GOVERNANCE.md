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
