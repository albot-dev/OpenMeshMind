#!/usr/bin/env python3
"""
CPU-only federated learning experiment with no third-party dependencies.

This script compares:
1) Centralized logistic regression training
2) Federated Averaging (FedAvg) with full-precision updates
3) FedAvg with int8-quantized client updates

It is designed to run on very small machines (2 cores / <2 GB RAM).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from dataclasses import dataclass


@dataclass
class RunResult:
    accuracy: float
    runtime_sec: float
    uplink_bytes: int
    participation_rate: float | None = None
    zero_client_rounds: int = 0


DEFAULT_N_FEATURES = 18
DEFAULT_N_CLIENTS = 8
DEFAULT_ROUNDS = 35
DEFAULT_LOCAL_STEPS = 14
DEFAULT_BATCH_SIZE = 20
DEFAULT_LR = 0.11
DEFAULT_NON_IID_SEVERITY = 1.4


def sigmoid(z: float) -> float:
    # Numerically stable sigmoid.
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def predict_prob(w: list[float], b: float, x: list[float]) -> float:
    return sigmoid(dot(w, x) + b)


def accuracy(w: list[float], b: float, data: list[tuple[list[float], int]]) -> float:
    hits = 0
    for x, y in data:
        pred = 1 if predict_prob(w, b, x) >= 0.5 else 0
        hits += pred == y
    return hits / len(data)


def quantize_int8(vec: list[float]) -> tuple[list[int], float]:
    max_abs = max((abs(v) for v in vec), default=0.0)
    if max_abs == 0.0:
        return [0 for _ in vec], 1.0
    scale = max_abs / 127.0
    q = []
    for v in vec:
        qv = int(round(v / scale))
        qv = max(-127, min(127, qv))
        q.append(qv)
    return q, scale


def dequantize_int8(qvec: list[int], scale: float) -> list[float]:
    return [q * scale for q in qvec]


def generate_dataset(
    seed: int,
    n_samples: int = 2600,
    n_features: int = 18,
) -> list[tuple[list[float], int]]:
    random.seed(seed)
    true_w = [random.gauss(0.0, 1.0) for _ in range(n_features)]
    true_b = random.gauss(0.0, 0.5)
    data: list[tuple[list[float], int]] = []

    for _ in range(n_samples):
        x = [random.gauss(0.0, 1.0) for _ in range(n_features)]
        logit = dot(true_w, x) + true_b + 0.35 * random.gauss(0.0, 1.0)
        p = sigmoid(logit)
        y = 1 if random.random() < p else 0
        data.append((x, y))

    random.shuffle(data)
    return data


def train_test_split(
    data: list[tuple[list[float], int]],
    test_fraction: float = 0.2,
) -> tuple[list[tuple[list[float], int]], list[tuple[list[float], int]]]:
    split = int(len(data) * (1.0 - test_fraction))
    return data[:split], data[split:]


def non_iid_partition(
    train_data: list[tuple[list[float], int]],
    n_clients: int = 8,
    severity: float = 1.4,
) -> list[list[tuple[list[float], int]]]:
    # Sort by a class-biased score to control client heterogeneity severity.
    scored = sorted(train_data, key=lambda s: (s[0][0] + severity * s[1], s[1]))
    chunks: list[list[tuple[list[float], int]]] = [[] for _ in range(n_clients)]
    chunk_size = len(scored) // n_clients

    for i in range(n_clients):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < n_clients - 1 else len(scored)
        chunks[i].extend(scored[start:end])
        random.shuffle(chunks[i])
    return chunks


def sgd_steps(
    w: list[float],
    b: float,
    data: list[tuple[list[float], int]],
    steps: int,
    batch_size: int,
    lr: float,
) -> tuple[list[float], float]:
    n_features = len(w)
    n = len(data)

    for _ in range(steps):
        batch = [data[random.randrange(n)] for _ in range(batch_size)]
        grad_w = [0.0] * n_features
        grad_b = 0.0

        for x, y in batch:
            p = predict_prob(w, b, x)
            err = p - y
            grad_b += err
            for j, xj in enumerate(x):
                grad_w[j] += err * xj

        inv = 1.0 / batch_size
        for j in range(n_features):
            w[j] -= lr * (grad_w[j] * inv)
        b -= lr * (grad_b * inv)

    return w, b


def run_centralized(
    train_data: list[tuple[list[float], int]],
    test_data: list[tuple[list[float], int]],
    n_features: int,
    total_steps: int,
    batch_size: int,
    lr: float,
) -> RunResult:
    w = [0.0] * n_features
    b = 0.0

    t0 = time.perf_counter()
    w, b = sgd_steps(w, b, train_data, total_steps, batch_size, lr)
    dt = time.perf_counter() - t0

    return RunResult(accuracy=accuracy(w, b, test_data), runtime_sec=dt, uplink_bytes=0)


def run_fedavg(
    clients: list[list[tuple[list[float], int]]],
    test_data: list[tuple[list[float], int]],
    n_features: int,
    rounds: int,
    local_steps: int,
    batch_size: int,
    lr: float,
    quantized: bool,
    dropout_rate: float = 0.0,
) -> RunResult:
    server_w = [0.0] * n_features
    server_b = 0.0
    total_bytes = 0
    selected_clients = 0
    zero_client_rounds = 0

    t0 = time.perf_counter()
    for _ in range(rounds):
        agg_delta_w = [0.0] * n_features
        agg_delta_b = 0.0

        if dropout_rate <= 0.0:
            active_client_indices = list(range(len(clients)))
        else:
            active_client_indices = [
                idx for idx in range(len(clients)) if random.random() >= dropout_rate
            ]
        selected_clients += len(active_client_indices)

        if not active_client_indices:
            zero_client_rounds += 1
            continue

        active_examples = sum(len(clients[idx]) for idx in active_client_indices)

        for idx in active_client_indices:
            local_data = clients[idx]
            weight = len(local_data) / active_examples
            local_w = list(server_w)
            local_b = server_b
            local_w, local_b = sgd_steps(local_w, local_b, local_data, local_steps, batch_size, lr)

            delta_vec = [lw - sw for lw, sw in zip(local_w, server_w)]
            delta_vec.append(local_b - server_b)

            if quantized:
                qvec, scale = quantize_int8(delta_vec)
                restored = dequantize_int8(qvec, scale)
                total_bytes += len(qvec) + 4  # int8 vector + float32 scale
            else:
                restored = delta_vec
                total_bytes += len(delta_vec) * 4  # float32 equivalent

            for j in range(n_features):
                agg_delta_w[j] += weight * restored[j]
            agg_delta_b += weight * restored[-1]

        for j in range(n_features):
            server_w[j] += agg_delta_w[j]
        server_b += agg_delta_b

    dt = time.perf_counter() - t0
    total_client_slots = rounds * len(clients)
    participation_rate = selected_clients / total_client_slots if total_client_slots else None
    return RunResult(
        accuracy=accuracy(server_w, server_b, test_data),
        runtime_sec=dt,
        uplink_bytes=total_bytes,
        participation_rate=participation_rate,
        zero_client_rounds=zero_client_rounds,
    )


def fmt_bytes(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KiB"
    return f"{num_bytes / (1024 * 1024):.2f} MiB"


def run_once(
    seed: int,
    dropout_rate: float = 0.0,
    non_iid_severity: float = DEFAULT_NON_IID_SEVERITY,
    n_features: int = DEFAULT_N_FEATURES,
    n_clients: int = DEFAULT_N_CLIENTS,
    rounds: int = DEFAULT_ROUNDS,
    local_steps: int = DEFAULT_LOCAL_STEPS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
) -> dict[str, RunResult]:
    data = generate_dataset(seed=seed, n_features=n_features)
    train_data, test_data = train_test_split(data)
    clients = non_iid_partition(train_data, n_clients=n_clients, severity=non_iid_severity)

    total_steps_central = rounds * local_steps * n_clients

    return {
        "centralized": run_centralized(
            train_data=train_data,
            test_data=test_data,
            n_features=n_features,
            total_steps=total_steps_central,
            batch_size=batch_size,
            lr=lr,
        ),
        "fedavg_fp32": run_fedavg(
            clients=clients,
            test_data=test_data,
            n_features=n_features,
            rounds=rounds,
            local_steps=local_steps,
            batch_size=batch_size,
            lr=lr,
            quantized=False,
            dropout_rate=dropout_rate,
        ),
        "fedavg_int8": run_fedavg(
            clients=clients,
            test_data=test_data,
            n_features=n_features,
            rounds=rounds,
            local_steps=local_steps,
            batch_size=batch_size,
            lr=lr,
            quantized=True,
            dropout_rate=dropout_rate,
        ),
    }


def aggregate(
    results: list[dict[str, RunResult]],
    config: dict[str, object],
) -> dict[str, object]:
    methods: dict[str, dict[str, float | int]] = {}
    names = ["centralized", "fedavg_fp32", "fedavg_int8"]

    for name in names:
        accs = [r[name].accuracy for r in results]
        times = [r[name].runtime_sec for r in results]
        bytes_up = [r[name].uplink_bytes for r in results]

        methods[name] = {
            "accuracy_mean": statistics.mean(accs),
            "accuracy_std": statistics.pstdev(accs),
            "runtime_mean_sec": statistics.mean(times),
            "runtime_std_sec": statistics.pstdev(times),
            "uplink_mean_bytes": int(statistics.mean(bytes_up)),
        }
        participation = [
            r[name].participation_rate
            for r in results
            if r[name].participation_rate is not None
        ]
        if participation:
            methods[name]["participation_rate_mean"] = statistics.mean(participation)
            methods[name]["participation_rate_std"] = statistics.pstdev(participation)
            methods[name]["zero_client_rounds_mean"] = statistics.mean(
                [r[name].zero_client_rounds for r in results]
            )

    fp = methods["fedavg_fp32"]["uplink_mean_bytes"]
    q8 = methods["fedavg_int8"]["uplink_mean_bytes"]
    savings = (1.0 - q8 / fp) * 100.0 if fp else 0.0

    return {
        "schema_version": 2,
        "config": config,
        "runs": len(results),
        "methods": methods,
        "communication_reduction_percent": savings,
    }


def compare_to_no_dropout(
    report: dict[str, object],
    no_dropout_report: dict[str, object],
) -> dict[str, object]:
    output: dict[str, object] = {}
    for method in ("fedavg_fp32", "fedavg_int8"):
        current = report["methods"][method]
        baseline = no_dropout_report["methods"][method]
        uplink_current = current["uplink_mean_bytes"]
        uplink_baseline = baseline["uplink_mean_bytes"]
        uplink_change = (
            ((uplink_current / uplink_baseline) - 1.0) * 100.0
            if uplink_baseline
            else 0.0
        )
        output[method] = {
            "accuracy_delta": current["accuracy_mean"] - baseline["accuracy_mean"],
            "runtime_delta_sec": current["runtime_mean_sec"] - baseline["runtime_mean_sec"],
            "uplink_change_percent": uplink_change,
        }
    return output


def run_experiment(
    seeds: list[int],
    dropout_rate: float,
    non_iid_severity: float,
) -> dict[str, object]:
    all_results = [
        run_once(seed, dropout_rate=dropout_rate, non_iid_severity=non_iid_severity)
        for seed in seeds
    ]
    report_config: dict[str, object] = {
        "seeds": seeds,
        "dropout_rate": dropout_rate,
        "non_iid_severity": non_iid_severity,
        "n_features": DEFAULT_N_FEATURES,
        "n_clients": DEFAULT_N_CLIENTS,
        "rounds": DEFAULT_ROUNDS,
        "local_steps": DEFAULT_LOCAL_STEPS,
        "batch_size": DEFAULT_BATCH_SIZE,
        "learning_rate": DEFAULT_LR,
    }
    report = aggregate(all_results, config=report_config)

    if dropout_rate > 0.0:
        no_dropout_results = [
            run_once(seed, dropout_rate=0.0, non_iid_severity=non_iid_severity)
            for seed in seeds
        ]
        no_dropout_config = dict(report_config)
        no_dropout_config["dropout_rate"] = 0.0
        no_dropout_report = aggregate(no_dropout_results, config=no_dropout_config)
        report["comparison_vs_no_dropout"] = compare_to_no_dropout(
            report=report,
            no_dropout_report=no_dropout_report,
        )

    return report


def summarize_sweep(sweep_report: dict[str, object]) -> None:
    print("Non-IID severity robustness sweep\n")
    print(
        f"dropout_rate={sweep_report['dropout_rate']:.2f}, "
        f"scenarios={len(sweep_report['scenarios'])}, "
        f"seeds_per_scenario={sweep_report['runs_per_scenario']}"
    )
    print(
        f"{'Severity':<10} {'FedAvg fp32 acc':<16} {'FedAvg int8 acc':<16} "
        f"{'Int8 comm save':<16}"
    )
    print("-" * 72)

    for scenario in sweep_report["scenarios"]:
        config = scenario["config"]
        methods = scenario["methods"]
        comm_saving = f"{scenario['communication_reduction_percent']:.2f}%"
        print(
            f"{config['non_iid_severity']:<10.2f} "
            f"{methods['fedavg_fp32']['accuracy_mean']:<16.4f} "
            f"{methods['fedavg_int8']['accuracy_mean']:<16.4f} "
            f"{comm_saving:<16}"
        )

def summarize(report: dict[str, object]) -> None:
    methods = report["methods"]
    config = report["config"]
    names = ["centralized", "fedavg_fp32", "fedavg_int8"]
    print("CPU-only experiment: centralized vs federated logistic regression")
    print(f"Metrics over {report['runs']} seeds (mean +/- stdev)\n")
    print(
        "Config:"
        f" dropout_rate={config['dropout_rate']:.2f},"
        f" non_iid_severity={config['non_iid_severity']:.2f},"
        f" clients={config['n_clients']},"
        f" rounds={config['rounds']},"
        f" local_steps={config['local_steps']}"
    )
    print(f"{'Method':<16} {'Acc':<16} {'Runtime':<16} {'Uplink/client net'}")
    print("-" * 72)

    for name in names:
        row = methods[name]
        acc_mean = row["accuracy_mean"]
        acc_std = row["accuracy_std"]
        time_mean = row["runtime_mean_sec"]
        time_std = row["runtime_std_sec"]
        bytes_mean = row["uplink_mean_bytes"]

        print(
            f"{name:<16} "
            f"{acc_mean:.4f} +/- {acc_std:.4f}   "
            f"{time_mean:.2f}s +/- {time_std:.2f}s   "
            f"{fmt_bytes(bytes_mean)}"
        )
        if "participation_rate_mean" in row:
            print(
                " " * 16
                + f"participation={row['participation_rate_mean']:.2%}, "
                + f"zero-rounds={row['zero_client_rounds_mean']:.2f}"
            )

    savings = report["communication_reduction_percent"]
    print("\nCommunication reduction from int8 client updates: " f"{savings:.1f}%")
    if "comparison_vs_no_dropout" in report:
        print("\nChanges vs no-dropout baseline:")
        for method, delta in report["comparison_vs_no_dropout"].items():
            print(
                f"- {method}: "
                f"accuracy_delta={delta['accuracy_delta']:+.4f}, "
                f"runtime_delta={delta['runtime_delta_sec']:+.3f}s, "
                f"uplink_change={delta['uplink_change_percent']:+.2f}%"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        default="7,17,27",
        help="Comma-separated list of integer seeds (default: 7,17,27).",
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.0,
        help=(
            "Per-client probability of dropping out in each federated round "
            "(0.0 <= rate < 1.0)."
        ),
    )
    parser.add_argument(
        "--non-iid-severity",
        type=float,
        default=DEFAULT_NON_IID_SEVERITY,
        help=(
            "Class-bias strength used when partitioning clients into non-IID shards "
            "(higher values increase heterogeneity)."
        ),
    )
    parser.add_argument(
        "--non-iid-sweep",
        default="",
        help=(
            "Comma-separated severities to run in one command. "
            "When provided, overrides --non-iid-severity and returns multi-scenario JSON."
        ),
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional path to write machine-readable JSON metrics.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the human-readable table output.",
    )
    args = parser.parse_args()

    if not (0.0 <= args.dropout_rate < 1.0):
        parser.error("--dropout-rate must be in [0.0, 1.0).")
    if args.non_iid_severity < 0.0:
        parser.error("--non-iid-severity must be >= 0.0.")

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if args.non_iid_sweep.strip():
        severities = [float(s.strip()) for s in args.non_iid_sweep.split(",") if s.strip()]
        if any(sev < 0.0 for sev in severities):
            parser.error("--non-iid-sweep values must be >= 0.0.")
        scenarios = [
            run_experiment(
                seeds=seeds,
                dropout_rate=args.dropout_rate,
                non_iid_severity=severity,
            )
            for severity in severities
        ]
        output: dict[str, object] = {
            "schema_version": 2,
            "sweep_type": "non_iid_severity",
            "dropout_rate": args.dropout_rate,
            "runs_per_scenario": len(seeds),
            "scenarios": scenarios,
        }

        if not args.quiet:
            summarize_sweep(output)
        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, sort_keys=True)
        return

    report = run_experiment(
        seeds=seeds,
        dropout_rate=args.dropout_rate,
        non_iid_severity=args.non_iid_severity,
    )

    if not args.quiet:
        summarize(report)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
