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
    secure_overhead_bytes: int = 0
    secure_mask_pair_count: int = 0
    fairness_metrics: dict[str, float] | None = None
    fairness_clients: list[dict[str, float | int]] | None = None


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


def add_in_place(dst: list[float], src: list[float]) -> None:
    for idx, value in enumerate(src):
        dst[idx] += value


def l2_norm(vec: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vec))


def jain_fairness(values: list[float]) -> float:
    if not values:
        return 0.0
    denom = len(values) * sum(value * value for value in values)
    if denom == 0.0:
        return 0.0
    return (sum(values) ** 2) / denom


def pearson_correlation(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(ys)
    cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    if x_var == 0.0 or y_var == 0.0:
        return 0.0
    return cov / math.sqrt(x_var * y_var)


def secure_mask_updates(
    updates: list[list[float]],
    mask_bound: float = 0.01,
) -> tuple[list[list[float]], int, int]:
    """
    Simulate pairwise masking with seed exchange.

    Returns:
    - masked updates for server aggregation
    - estimated overhead bytes for mask seed exchange
    - number of mask pairs created
    """
    if len(updates) < 2:
        return updates, 0, 0

    dim = len(updates[0])
    masks = [[0.0] * dim for _ in updates]
    mask_pairs = 0
    for i in range(len(updates)):
        for j in range(i + 1, len(updates)):
            mask_pairs += 1
            for k in range(dim):
                val = random.uniform(-mask_bound, mask_bound)
                masks[i][k] += val
                masks[j][k] -= val

    masked_updates = []
    for idx, update in enumerate(updates):
        masked = [value + masks[idx][k] for k, value in enumerate(update)]
        masked_updates.append(masked)

    # Simulate exchange of PRG seeds per pair (16 bytes each direction).
    overhead_bytes = mask_pairs * 32
    return masked_updates, overhead_bytes, mask_pairs


def secure_aggregate(masked_updates: list[list[float]]) -> list[float]:
    dim = len(masked_updates[0])
    aggregate = [0.0] * dim
    for update in masked_updates:
        add_in_place(aggregate, update)
    return aggregate


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
    secure_aggregation: bool = False,
    client_capacities: list[float] | None = None,
    round_deadline: float = 4.2,
    capacity_jitter: float = 0.1,
) -> RunResult:
    if client_capacities is not None and len(client_capacities) != len(clients):
        raise ValueError("client_capacities length must match number of clients.")
    if capacity_jitter < 0.0 or capacity_jitter >= 1.0:
        raise ValueError("capacity_jitter must be in [0.0, 1.0).")
    if round_deadline <= 0.0:
        raise ValueError("round_deadline must be > 0.")

    server_w = [0.0] * n_features
    server_b = 0.0
    total_bytes = 0
    selected_clients = 0
    contributed_clients = 0
    zero_client_rounds = 0
    secure_overhead_bytes = 0
    secure_mask_pair_count = 0
    per_client_selected_rounds = [0 for _ in clients]
    per_client_contributed_rounds = [0 for _ in clients]
    per_client_uplink_bytes = [0 for _ in clients]
    per_client_update_l2_sum = [0.0 for _ in clients]

    t0 = time.perf_counter()
    for _ in range(rounds):
        if dropout_rate <= 0.0:
            active_client_indices = list(range(len(clients)))
        else:
            active_client_indices = [
                idx for idx in range(len(clients)) if random.random() >= dropout_rate
            ]
        selected_clients += len(active_client_indices)
        for idx in active_client_indices:
            per_client_selected_rounds[idx] += 1

        if not active_client_indices:
            zero_client_rounds += 1
            continue

        contributing_client_indices = []
        if client_capacities is None:
            contributing_client_indices = active_client_indices
        else:
            for idx in active_client_indices:
                base_capacity = max(client_capacities[idx], 1e-6)
                jitter = 1.0 + random.uniform(-capacity_jitter, capacity_jitter)
                effective_capacity = max(0.05, base_capacity * jitter)
                normalized_compute = (local_steps / DEFAULT_LOCAL_STEPS) / effective_capacity
                network_factor = (0.45 if quantized else 1.0) / effective_capacity
                if normalized_compute + network_factor <= round_deadline:
                    contributing_client_indices.append(idx)

        if not contributing_client_indices:
            zero_client_rounds += 1
            continue

        contributed_clients += len(contributing_client_indices)
        active_examples = sum(len(clients[idx]) for idx in contributing_client_indices)
        weighted_updates: list[list[float]] = []

        for idx in contributing_client_indices:
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
                sent_bytes = len(qvec) + 4  # int8 vector + float32 scale
            else:
                restored = delta_vec
                sent_bytes = len(delta_vec) * 4  # float32 equivalent
            total_bytes += sent_bytes
            per_client_uplink_bytes[idx] += sent_bytes
            per_client_contributed_rounds[idx] += 1
            per_client_update_l2_sum[idx] += l2_norm(restored)

            weighted_updates.append([weight * val for val in restored])

        if secure_aggregation:
            masked_updates, overhead, mask_pairs = secure_mask_updates(weighted_updates)
            secure_overhead_bytes += overhead
            secure_mask_pair_count += mask_pairs
            aggregate_update = secure_aggregate(masked_updates)
        else:
            aggregate_update = [0.0] * (n_features + 1)
            for update in weighted_updates:
                add_in_place(aggregate_update, update)

        for j in range(n_features):
            server_w[j] += aggregate_update[j]
        server_b += aggregate_update[-1]

    dt = time.perf_counter() - t0
    total_client_slots = rounds * len(clients)
    participation_rate = selected_clients / total_client_slots if total_client_slots else None
    fairness_metrics: dict[str, float] | None = None
    fairness_clients: list[dict[str, float | int]] | None = None

    if client_capacities is not None:
        total_uplink = sum(per_client_uplink_bytes)
        fairness_clients = []
        participation_rates: list[float] = []
        contribution_rates: list[float] = []
        completion_rates: list[float] = []
        for idx, capacity in enumerate(client_capacities):
            selected_rounds = per_client_selected_rounds[idx]
            contributed_rounds = per_client_contributed_rounds[idx]
            client_participation = selected_rounds / rounds if rounds else 0.0
            client_contribution = contributed_rounds / rounds if rounds else 0.0
            completion_rate = (
                contributed_rounds / selected_rounds if selected_rounds else 0.0
            )
            participation_rates.append(client_participation)
            contribution_rates.append(client_contribution)
            completion_rates.append(completion_rate)
            fairness_clients.append(
                {
                    "client_index": idx,
                    "capacity": capacity,
                    "selected_rounds": selected_rounds,
                    "contributed_rounds": contributed_rounds,
                    "participation_rate": client_participation,
                    "contribution_rate": client_contribution,
                    "completion_rate": completion_rate,
                    "uplink_bytes": per_client_uplink_bytes[idx],
                    "uplink_share": (
                        per_client_uplink_bytes[idx] / total_uplink if total_uplink else 0.0
                    ),
                    "update_l2_mean": (
                        per_client_update_l2_sum[idx] / contributed_rounds
                        if contributed_rounds
                        else 0.0
                    ),
                }
            )

        slowest_index = min(
            range(len(client_capacities)),
            key=lambda idx: client_capacities[idx],
        )
        fastest_index = max(
            range(len(client_capacities)),
            key=lambda idx: client_capacities[idx],
        )
        slowest_rate = contribution_rates[slowest_index]
        fastest_rate = contribution_rates[fastest_index]
        fairness_metrics = {
            "participation_rate_gap": max(participation_rates) - min(participation_rates),
            "contribution_rate_gap": max(contribution_rates) - min(contribution_rates),
            "completion_rate_gap": max(completion_rates) - min(completion_rates),
            "contribution_jain_index": jain_fairness(contribution_rates),
            "capacity_contribution_correlation": pearson_correlation(
                client_capacities,
                contribution_rates,
            ),
            "slowest_fastest_contribution_ratio": (
                slowest_rate / fastest_rate if fastest_rate > 0.0 else 0.0
            ),
            "contributed_clients_per_round_mean": (
                contributed_clients / rounds if rounds else 0.0
            ),
        }

    return RunResult(
        accuracy=accuracy(server_w, server_b, test_data),
        runtime_sec=dt,
        uplink_bytes=total_bytes,
        participation_rate=participation_rate,
        zero_client_rounds=zero_client_rounds,
        secure_overhead_bytes=secure_overhead_bytes,
        secure_mask_pair_count=secure_mask_pair_count,
        fairness_metrics=fairness_metrics,
        fairness_clients=fairness_clients,
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
    secure_aggregation: bool = False,
    client_capacities: list[float] | None = None,
    round_deadline: float = 4.2,
    capacity_jitter: float = 0.1,
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
            secure_aggregation=secure_aggregation,
            client_capacities=client_capacities,
            round_deadline=round_deadline,
            capacity_jitter=capacity_jitter,
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
            secure_aggregation=secure_aggregation,
            client_capacities=client_capacities,
            round_deadline=round_deadline,
            capacity_jitter=capacity_jitter,
        ),
    }


def aggregate(
    results: list[dict[str, RunResult]],
    config: dict[str, object],
) -> dict[str, object]:
    methods: dict[str, dict[str, object]] = {}
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
        secure_overheads = [r[name].secure_overhead_bytes for r in results]
        if any(secure_overheads):
            methods[name]["secure_overhead_mean_bytes"] = int(
                statistics.mean(secure_overheads)
            )
            methods[name]["secure_mask_pairs_mean"] = statistics.mean(
                [r[name].secure_mask_pair_count for r in results]
            )
        fairness_runs = [r[name].fairness_metrics for r in results if r[name].fairness_metrics]
        if fairness_runs:
            fairness_summary: dict[str, float] = {}
            for metric in fairness_runs[0]:
                values = [run[metric] for run in fairness_runs]
                fairness_summary[f"{metric}_mean"] = statistics.mean(values)
                fairness_summary[f"{metric}_std"] = statistics.pstdev(values)
            methods[name]["fairness"] = fairness_summary

        fairness_clients_runs = [r[name].fairness_clients for r in results if r[name].fairness_clients]
        if fairness_clients_runs:
            n_clients = len(fairness_clients_runs[0])
            clients_aggregate: list[dict[str, float | int]] = []
            for idx in range(n_clients):
                per_seed = [run[idx] for run in fairness_clients_runs]
                clients_aggregate.append(
                    {
                        "client_index": idx,
                        "capacity": per_seed[0]["capacity"],
                        "selected_rounds_mean": statistics.mean(
                            [row["selected_rounds"] for row in per_seed]
                        ),
                        "contributed_rounds_mean": statistics.mean(
                            [row["contributed_rounds"] for row in per_seed]
                        ),
                        "participation_rate_mean": statistics.mean(
                            [row["participation_rate"] for row in per_seed]
                        ),
                        "contribution_rate_mean": statistics.mean(
                            [row["contribution_rate"] for row in per_seed]
                        ),
                        "completion_rate_mean": statistics.mean(
                            [row["completion_rate"] for row in per_seed]
                        ),
                        "uplink_share_mean": statistics.mean(
                            [row["uplink_share"] for row in per_seed]
                        ),
                        "update_l2_mean": statistics.mean(
                            [row["update_l2_mean"] for row in per_seed]
                        ),
                    }
                )
            methods[name]["fairness_clients"] = clients_aggregate

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
    secure_aggregation: bool,
    client_capacities: list[float] | None = None,
    round_deadline: float = 4.2,
    capacity_jitter: float = 0.1,
) -> dict[str, object]:
    all_results = [
        run_once(
            seed,
            dropout_rate=dropout_rate,
            non_iid_severity=non_iid_severity,
            secure_aggregation=secure_aggregation,
            client_capacities=client_capacities,
            round_deadline=round_deadline,
            capacity_jitter=capacity_jitter,
        )
        for seed in seeds
    ]
    report_config: dict[str, object] = {
        "seeds": seeds,
        "dropout_rate": dropout_rate,
        "non_iid_severity": non_iid_severity,
        "secure_aggregation": secure_aggregation,
        "n_features": DEFAULT_N_FEATURES,
        "n_clients": DEFAULT_N_CLIENTS,
        "rounds": DEFAULT_ROUNDS,
        "local_steps": DEFAULT_LOCAL_STEPS,
        "batch_size": DEFAULT_BATCH_SIZE,
        "learning_rate": DEFAULT_LR,
    }
    if client_capacities is not None:
        report_config["client_capacities"] = client_capacities
        report_config["round_deadline"] = round_deadline
        report_config["capacity_jitter"] = capacity_jitter
    report = aggregate(all_results, config=report_config)

    if dropout_rate > 0.0:
        no_dropout_results = [
            run_once(
                seed,
                dropout_rate=0.0,
                non_iid_severity=non_iid_severity,
                secure_aggregation=secure_aggregation,
                client_capacities=client_capacities,
                round_deadline=round_deadline,
                capacity_jitter=capacity_jitter,
            )
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
        f"secure_aggregation={sweep_report['secure_aggregation']}, "
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
        f" secure_aggregation={config['secure_aggregation']},"
        f" clients={config['n_clients']},"
        f" rounds={config['rounds']},"
        f" local_steps={config['local_steps']}"
    )
    if "client_capacities" in config:
        print(
            "Capacity simulation:"
            f" deadline={config['round_deadline']:.2f},"
            f" jitter={config['capacity_jitter']:.2f},"
            f" capacities={config['client_capacities']}"
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
        if "secure_overhead_mean_bytes" in row:
            print(
                " " * 16
                + f"secure-overhead={fmt_bytes(row['secure_overhead_mean_bytes'])}, "
                + f"mask-pairs={row['secure_mask_pairs_mean']:.2f}"
            )
        if "fairness" in row:
            fairness = row["fairness"]
            print(
                " " * 16
                + f"fairness_gap={fairness['contribution_rate_gap_mean']:.2%}, "
                + f"jain={fairness['contribution_jain_index_mean']:.3f}, "
                + f"cap-corr={fairness['capacity_contribution_correlation_mean']:.3f}"
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
        "--secure-aggregation",
        action="store_true",
        help=(
            "Enable a mock secure aggregation mode where the server aggregates "
            "masked weighted updates."
        ),
    )
    parser.add_argument(
        "--simulate-client-capacity",
        action="store_true",
        help=(
            "Enable heterogeneous client capacity simulation and fairness metrics "
            "using the default capacity profile."
        ),
    )
    parser.add_argument(
        "--client-capacities",
        default="",
        help=(
            "Optional comma-separated client capacities (positive floats). "
            f"Expected length: {DEFAULT_N_CLIENTS}."
        ),
    )
    parser.add_argument(
        "--round-deadline",
        type=float,
        default=4.2,
        help=(
            "Normalized deadline used in capacity simulation. "
            "Lower values increase exclusion of slower clients."
        ),
    )
    parser.add_argument(
        "--capacity-jitter",
        type=float,
        default=0.1,
        help="Relative per-round capacity jitter in [0.0, 1.0).",
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
    if args.round_deadline <= 0.0:
        parser.error("--round-deadline must be > 0.0.")
    if not (0.0 <= args.capacity_jitter < 1.0):
        parser.error("--capacity-jitter must be in [0.0, 1.0).")

    client_capacities: list[float] | None = None
    if args.client_capacities.strip():
        client_capacities = [float(s.strip()) for s in args.client_capacities.split(",") if s.strip()]
    elif args.simulate_client_capacity:
        client_capacities = [1.0, 0.95, 0.85, 0.75, 0.6, 0.5, 0.4, 0.35]

    if client_capacities is not None:
        if len(client_capacities) != DEFAULT_N_CLIENTS:
            parser.error(
                f"--client-capacities must include exactly {DEFAULT_N_CLIENTS} values."
            )
        if any(value <= 0.0 for value in client_capacities):
            parser.error("--client-capacities values must be > 0.0.")

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
                secure_aggregation=args.secure_aggregation,
                client_capacities=client_capacities,
                round_deadline=args.round_deadline,
                capacity_jitter=args.capacity_jitter,
            )
            for severity in severities
        ]
        output: dict[str, object] = {
            "schema_version": 2,
            "sweep_type": "non_iid_severity",
            "dropout_rate": args.dropout_rate,
            "secure_aggregation": args.secure_aggregation,
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
        secure_aggregation=args.secure_aggregation,
        client_capacities=client_capacities,
        round_deadline=args.round_deadline,
        capacity_jitter=args.capacity_jitter,
    )

    if not args.quiet:
        summarize(report)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
