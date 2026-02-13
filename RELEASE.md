# Release Process

This process is required for tagged releases (`vX.Y.Z`).

## Pre-release checklist

1. Update `CHANGELOG.md` with release notes.
2. Update `DECISION_LOG.md` for major architecture/policy decisions included in the release.
3. Add provenance details for major experiment changes using `PROVENANCE_TEMPLATE.md`.
4. Confirm issue/milestone status in `README.md` work tracking.
5. Run one-command smoke validation:

```bash
python3 scripts/smoke_check.py --include-fairness --json-out smoke_summary.json
```

## Versioning guidance

- Use `vMAJOR.MINOR.PATCH` tags.
- `PATCH`: non-breaking fixes and reproducibility improvements.
- `MINOR`: new additive experiments/metrics/workflows without breaking schema contracts.
- `MAJOR`: breaking schema or compatibility changes; requires explicit migration notes.
- Detailed schema/version compatibility rules are defined in `VERSIONING_POLICY.md`.

## Tag and publish

1. Create a release commit if needed.
2. Tag the release:

```bash
git tag v0.1.0
git push origin v0.1.0
```

3. GitHub Actions `release.yml` will:
   - run smoke checks
   - collect release artifacts and checksums
   - create a GitHub Release entry attached to the tag

## Required release artifacts

- `baseline_metrics.json`
- `classification_metrics.json`
- `benchmark_metrics.json`
- `fairness_metrics.json`
- `utility_fedavg_metrics.json`
- `utility_fairness_metrics.json`
- `smoke_summary.json`
- `SHA256SUMS.txt`

## Release notes minimum content

- Link to `CHANGELOG.md` section for the release.
- Any schema version change with migration notes.
- Any threshold/governance change with rationale.
- Validation commands used before tagging.
