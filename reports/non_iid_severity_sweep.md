# Non-IID Severity Sweep

This report captures a baseline robustness sweep for client heterogeneity using the CPU-only federated experiment.

## Command

```bash
python3 experiments/fedavg_cpu_only.py --non-iid-sweep 0.2,1.4,3.0 --json-out non_iid_sweep_metrics.json
```

## Environment

- Machine: 2 CPU cores, low-memory VM
- Seeds: `7,17,27`
- Dropout rate: `0.0`

## Summary (mean across seeds)

| Non-IID severity | FedAvg fp32 accuracy | FedAvg int8 accuracy | Int8 communication reduction |
| --- | ---: | ---: | ---: |
| `0.20` | `0.8667` | `0.8686` | `69.74%` |
| `1.40` | `0.8699` | `0.8686` | `69.74%` |
| `3.00` | `0.8667` | `0.8667` | `69.74%` |

## Takeaway

Across these severity presets, int8 FedAvg stayed close to fp32 FedAvg in accuracy while preserving a large communication reduction.
