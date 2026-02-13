# Pilot Run Artifacts Layout

Store day-level cohort artifacts for the 14-day pilot here.

Recommended structure:

- `pilot/runs/day01/nodes/<node_id>/pilot_metrics.json`
- `pilot/runs/day01/pilot_cohort_metrics.json`
- `pilot/runs/day02/...`
- `pilot/runs/day14/...`

Use `scripts/build_pilot_cohort_metrics.py` to generate each day-level cohort summary,
then `scripts/generate_pilot_14_day_report.py` at the end of the window.
