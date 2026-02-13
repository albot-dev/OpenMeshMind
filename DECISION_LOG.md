# Decision Log

This log records major architectural or policy decisions with explicit tradeoffs.

## Template

Use this structure for each entry:

```markdown
## YYYY-MM-DD: <decision title>

- Context:
- Decision:
- Alternatives considered:
- Accessibility impact (commodity hardware):
- Decentralization impact:
- Reproducibility impact:
- Links (issue/PR/report):
```

---

## 2026-02-13: Add utility fairness stress gate in CI

- Context:
  - Utility federation added compressed update modes (`fp32`, `int8`, `sparse`).
  - Needed automated checks to detect fairness regressions under heterogeneous client capacity + churn.
- Decision:
  - Add `scripts/check_utility_fairness.py` and run it in CI against a stress sweep (`round_deadline=4.0,4.2`).
- Alternatives considered:
  - Manual fairness review only (rejected: too easy to regress silently).
  - Single-scenario check only (rejected: weaker robustness signal).
- Accessibility impact (commodity hardware):
  - Keeps CI checks CPU-only and lightweight.
- Decentralization impact:
  - Enforces explicit guardrails against exclusion of weaker participants.
- Reproducibility impact:
  - Metrics are machine-readable, versioned, and validated in CI.
- Links (issue/PR/report):
  - Issue: `#12`
  - Report: `reports/fedavg_classification_fairness.md`
