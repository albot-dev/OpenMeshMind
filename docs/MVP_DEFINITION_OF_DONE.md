# MVP Definition of Done

## Pass rule

MVP is done only when every command below exits with code `0` and each expected output string appears.

## Verification checklist

| DoD ID | Criterion | Verification command | Expected output strings |
|---|---|---|---|
| DOD-01 | Full smoke path succeeds (including fairness) | `python3 scripts/smoke_check.py --include-fairness --json-out smoke_summary.json` | `[smoke] unit_tests: ok`, `[smoke] main_track_status: ok`, `[smoke] total:` |
| DOD-02 | Baseline federated value gate passes | `python3 scripts/check_baseline.py baseline_metrics.json --expected-schema-version 2` | `Baseline metrics summary`, `Validation passed.` |
| DOD-03 | Local classification quality gate passes | `python3 scripts/check_classification.py classification_metrics.json --expected-schema-version 1` | `Classification metrics summary`, `Validation passed.` |
| DOD-04 | Adapter communication-quality gate passes | `python3 scripts/check_adapter_intent.py adapter_intent_metrics.json --expected-schema-version 1` | `Adapter intent metrics summary`, `Validation passed.` |
| DOD-05 | Reduced benchmark envelope gate passes | `python3 scripts/check_benchmarks.py benchmark_metrics.json --expected-schema-version 1 --expected-mode reduced` | `Benchmark metrics summary`, `Validation passed.` |
| DOD-06 | Generality gate passes | `python3 scripts/check_generality.py generality_metrics.json --expected-schema-version 1` | `Generality metrics summary`, `Validation passed.` |
| DOD-07 | Reproducibility gate passes | `python3 scripts/check_reproducibility.py reproducibility_metrics.json --expected-schema-version 1` | `Reproducibility metrics summary`, `Validation passed.` |
| DOD-08 | Fairness resilience gates pass | `python3 scripts/check_fairness.py fairness_metrics.json --expected-schema-version 2` | `Fairness metrics summary`, `Validation passed.` |
| DOD-09 | Utility fairness stress gate passes | `python3 scripts/check_utility_fairness.py utility_fairness_metrics.json --expected-schema-version 1` | `Utility fairness validation summary`, `Validation passed.` |
| DOD-10 | Strict main-track status is fully complete | `python3 scripts/main_track_status.py --require-fairness --require-smoke-summary --fail-on-incomplete --json-out main_track_status.json --md-out reports/main_track_status.md` | `Main track goals: 4/4 complete, required sub-goals 9/9` |
| DOD-11 | Smoke summary confirms fairness-enabled success | `python3 -c "import json; s=json.load(open('smoke_summary.json')); print(f\"ok={s.get('ok')} include_fairness={s.get('include_fairness')}\")"` | `ok=True include_fairness=True` |
| DOD-12 | Status JSON confirms strict completion counts | `python3 -c "import json; s=json.load(open('main_track_status.json'))['summary']; print(f\"goals={s['completed_goals']}/{s['total_goals']} required={s['completed_required_sub_goals']}/{s['required_sub_goals']} all_done={s['all_done']}\")"` | `goals=4/4 required=9/9 all_done=True` |
| DOD-13 | Machine-verifiable readiness check passes | `python3 scripts/check_mvp_readiness.py --require-fairness --require-all-goals-done --json-out mvp_readiness.json` | `MVP readiness summary`, `Readiness check passed.` |

## Required artifacts after DoD run

- `baseline_metrics.json`
- `classification_metrics.json`
- `adapter_intent_metrics.json`
- `benchmark_metrics.json`
- `generality_metrics.json`
- `reproducibility_metrics.json`
- `fairness_metrics.json`
- `utility_fairness_metrics.json`
- `smoke_summary.json`
- `main_track_status.json`
- `reports/main_track_status.md`
- `mvp_readiness.json`
