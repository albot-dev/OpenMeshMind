# Volunteer Pilot Node Guide

This document defines the minimum setup, startup flow, health checks, and troubleshooting for a volunteer pilot node.

## Minimum node requirements

- CPU: 4 physical cores (x86_64 or arm64)
- Memory: 8 GB RAM minimum, 16 GB recommended
- Disk: 10 GB free disk space
- Network: stable outbound internet, >= 10 Mbps down / 5 Mbps up
- Runtime: Python 3.12+

## Required setup

1. Configure GitHub access token in `.env`:

```bash
github_token=<your-token>
```

2. Create node config from template:

```bash
cp pilot/node_config.example.json pilot/node_config.json
```

3. Run one warm-up cycle:

```bash
python3 scripts/pilot_node_runner.py --config pilot/node_config.json --once
```

A successful cycle produces:

- `pilot/node_state.json`
- `pilot/pilot_metrics.json`
- `pilot/node_runner.log`

## Start node loop

Run continuously with configured interval:

```bash
python3 scripts/pilot_node_runner.py --config pilot/node_config.json
```

For a fixed number of cycles:

```bash
python3 scripts/pilot_node_runner.py --config pilot/node_config.json --max-cycles 4
```

## Health checks

Node state health check:

```bash
python3 scripts/pilot_node_runner.py --config pilot/node_config.json --health --min-uptime-ratio 0.90
```

Pilot metrics schema validation:

```bash
python3 scripts/check_pilot_metrics.py pilot/pilot_metrics.json --require-status-collected
```

Pilot launch readiness gate (no open pilot prep work):

```bash
python3 scripts/check_pilot_metrics.py \
  pilot/pilot_metrics.json \
  --require-status-collected \
  --max-open-milestones 0 \
  --max-open-issues 0
```

## Expected resource envelope per cycle

- End-to-end cycle runtime: typically under 90 seconds on baseline commodity hardware
- Peak resident memory should remain within benchmark envelope published in `benchmark_metrics.json`
- Disk growth is mostly logs and JSON artifacts; rotate `pilot/node_runner.log` if needed

## Troubleshooting

- Config missing:
  - Symptom: `Config not found: .../pilot/node_config.json`
  - Fix: copy `pilot/node_config.example.json` to `pilot/node_config.json`
- GitHub status not collected:
  - Symptom: `status_collected: False` in pilot metrics
  - Fix: ensure `.env` has `github_token`, token has repo read permissions, rerun node cycle
- Smoke step failure:
  - Symptom: `last_cycle_ok: false` in node health
  - Fix: inspect `pilot/node_runner.log`, rerun failing command from log output
- Metrics build failure:
  - Symptom: `build_pilot_metrics rc=1` in `last_error`
  - Fix: regenerate baseline artifacts with `python3 scripts/smoke_check.py --include-fairness --json-out smoke_summary.json`

## Related references

- `PILOT_OPERATIONS.md`
- `schemas/pilot_metrics.schema.v1.json`
- `reports/PILOT_STATUS_TEMPLATE.md`
