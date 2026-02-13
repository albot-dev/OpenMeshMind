# Fairness Snapshot: Heterogeneous Client Capacities

Command used:

```bash
python3 experiments/fedavg_cpu_only.py \
  --simulate-client-capacity \
  --json-out fairness_metrics.json
```

Run config:

- seeds: `7,17,27`
- default capacity profile: `[1.0, 0.95, 0.85, 0.75, 0.6, 0.5, 0.4, 0.35]`
- `round_deadline=4.2`
- `capacity_jitter=0.1`

## Key observations

### FedAvg fp32

- fairness gap (`contribution_rate_gap_mean`): `1.0000`
- Jain fairness index (`contribution_jain_index_mean`): `0.7346`
- capacity->contribution correlation: `0.8326`
- contributed clients/round (`mean`): `5.6571`
- per-client contribution rates (fastest -> slowest): `[1.00, 1.00, 1.00, 1.00, 1.00, 0.66, 0.00, 0.00]`

### FedAvg int8

- fairness gap (`contribution_rate_gap_mean`): `0.3524`
- Jain fairness index (`contribution_jain_index_mean`): `0.9847`
- capacity->contribution correlation: `0.5268`
- contributed clients/round (`mean`): `7.6476`
- per-client contribution rates (fastest -> slowest): `[1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.65]`

## Tradeoff summary

- Heterogeneous capacities create clear contribution disparity under stricter round deadlines.
- Int8 update transport improves fairness in this setup by allowing more lower-capacity clients to finish rounds.
- Communication savings remain high (`59.09%`) while preserving strong accuracy in this run.

## Recommended default for fairness stress tests

- Use `--simulate-client-capacity` with:
  - `--round-deadline 4.2`
  - `--capacity-jitter 0.1`

This keeps the scenario realistic (slowest clients are pressured) while still producing measurable, non-trivial fairness improvements from communication-efficient updates.
