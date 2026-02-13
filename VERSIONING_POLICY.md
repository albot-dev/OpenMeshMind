# Versioning and Schema Compatibility Policy

## Release versioning

OpenMeshMind uses semantic tag format: `vMAJOR.MINOR.PATCH`.

- `PATCH`: bug fixes, documentation updates, performance/reproducibility improvements that do not break existing usage contracts.
- `MINOR`: additive capabilities (new experiments/metrics/checks) that preserve existing schema and CLI compatibility.
- `MAJOR`: breaking changes in CLI behavior, artifact schema, or compatibility guarantees.

## Artifact schema policy

Machine-readable metrics must include `schema_version`.

Current baseline schema versions:

- `baseline_metrics.json`: `2`
- `benchmark_metrics.json`: `1`
- `classification_metrics.json`: `1`
- `fairness_metrics.json`: `2`
- `utility_fedavg_metrics.json`: `1`
- `utility_fairness_metrics.json`: `1`
- `smoke_summary.json`: `1`

## When to bump schema version

Schema version must bump when any of the following occur:

- key rename/removal
- type change of a required field
- unit/semantic change that makes old interpretation wrong
- nesting changes that break existing parsers

Schema version can remain unchanged for:

- additive optional fields
- extra report sections not required by validators
- additional artifacts that do not alter existing artifact contracts

## Validator expectations

- Validators should default to the current expected schema version.
- Backward compatibility support must be explicit in validator flags and documented in changelog/release notes.

## Release-note minimum content

Every tagged release must include:

- changed scripts/experiments list
- schema version changes (old -> new) and migration notes if breaking
- updated thresholds or governance policy deltas
- reproducibility command(s) used for release validation
