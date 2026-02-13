# Federated Utility Fairness Stress Snapshot

Command used:

```bash
python3 experiments/fedavg_classification_utility.py \
  --simulate-client-capacity \
  --dropout-rate 0.1 \
  --round-deadline-sweep 4.0,4.2 \
  --json-out utility_fairness_metrics.json
python3 scripts/check_utility_fairness.py utility_fairness_metrics.json
```

Sweep setup:

- seeds: `7,17,27`
- modes: `fp32,int8,sparse`
- default capacities (`n_clients=6`): `[1.0, 0.9, 0.75, 0.6, 0.45, 0.35]`
- dropout rate: `0.10`
- capacity jitter: `0.10`
- round deadlines: `4.0`, `4.2`

Results:

## Deadline 4.0

- fp32 contribution gap: `0.950`
- int8 contribution gap: `0.650`
- int8 Jain index: `0.977`
- int8 communication savings vs fp32: `66.78%`
- int8 Jain gain vs fp32: `+0.2616`
- int8 contributed-clients gain/round vs fp32: `+1.1500`

## Deadline 4.2

- fp32 contribution gap: `0.950`
- int8 contribution gap: `0.367`
- int8 Jain index: `0.980`
- int8 communication savings vs fp32: `66.11%`
- int8 Jain gain vs fp32: `+0.2633`
- int8 contributed-clients gain/round vs fp32: `+1.2833`

Recommended defaults for utility fairness checks:

- `--simulate-client-capacity`
- `--dropout-rate 0.1`
- `--round-deadline 4.2`
- `--capacity-jitter 0.1`

Rationale:

- Both stress scenarios show clear int8 fairness gains over fp32.
- `round_deadline=4.2` gives the strongest contribution-gap improvement while retaining large communication savings.
