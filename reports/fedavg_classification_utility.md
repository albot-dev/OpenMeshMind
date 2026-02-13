# Federated Utility Classification Snapshot

Command used:

```bash
python3 experiments/fedavg_classification_utility.py --json-out utility_fedavg_metrics.json
```

Config:

- seeds: `7,17,27`
- modes: `fp32,int8,sparse`
- samples per label: `40`
- test fraction: `0.20`
- clients: `6`
- rounds: `20`
- local steps: `8`
- batch size: `8`
- learning rate: `0.2`
- sparse ratio: `0.2`

Summary:

- centralized accuracy / macro-F1: `1.0000 / 1.0000`
- fedavg fp32 accuracy / macro-F1: `1.0000 / 1.0000`
- fedavg int8 accuracy / macro-F1: `1.0000 / 1.0000`
- fedavg sparse accuracy / macro-F1: `1.0000 / 1.0000`

Communication (mean uplink bytes):

- fp32: `246400`
- int8: `62080` (`74.81%` savings vs fp32)
- sparse: `98720` (`59.94%` savings vs fp32)

Quality deltas vs centralized:

- fp32 accuracy drop: `0.0000`
- int8 accuracy drop: `0.0000`
- sparse accuracy drop: `0.0000`

Notes:

- This utility experiment extends the project with federated CPU-first task execution.
- Int8 and sparse transport both reduce communication with no quality loss in this baseline setup.
