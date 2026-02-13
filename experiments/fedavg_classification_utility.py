#!/usr/bin/env python3
"""
CPU-only federated utility classification experiment.

Compares:
- centralized softmax SGD reference
- FedAvg with fp32 client updates
- FedAvg with int8-quantized client updates
- FedAvg with sparse top-k client updates
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import fedavg_cpu_only as fed
from experiments import local_classification_baseline as cls


@dataclass
class RunResult:
    accuracy: float
    macro_f1: float
    runtime_sec: float
    uplink_bytes: int


def softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    total = sum(exps)
    return [value / total for value in exps]


def flatten_params(weights: list[list[float]], bias: list[float]) -> list[float]:
    vec: list[float] = []
    for row in weights:
        vec.extend(row)
    vec.extend(bias)
    return vec


def unflatten_params(
    vec: list[float],
    n_classes: int,
    n_features: int,
) -> tuple[list[list[float]], list[float]]:
    expected = n_classes * n_features + n_classes
    if len(vec) != expected:
        raise ValueError(f"Invalid parameter vector length {len(vec)} (expected {expected})")
    weights: list[list[float]] = []
    offset = 0
    for _ in range(n_classes):
        weights.append(vec[offset : offset + n_features])
        offset += n_features
    bias = vec[offset : offset + n_classes]
    return weights, bias


def init_model(n_classes: int, n_features: int) -> tuple[list[list[float]], list[float]]:
    return [[0.0] * n_features for _ in range(n_classes)], [0.0] * n_classes


def logits_for_sample(
    weights: list[list[float]],
    bias: list[float],
    features: dict[int, float],
) -> list[float]:
    logits: list[float] = []
    for c, row in enumerate(weights):
        score = bias[c]
        for j, value in features.items():
            score += row[j] * value
        logits.append(score)
    return logits


def local_sgd_steps(
    weights: list[list[float]],
    bias: list[float],
    data_x: list[dict[int, float]],
    data_y: list[int],
    steps: int,
    batch_size: int,
    learning_rate: float,
    rng: random.Random,
) -> tuple[list[list[float]], list[float]]:
    n_classes = len(weights)
    n_features = len(weights[0]) if weights else 0
    if steps <= 0:
        raise ValueError("steps must be > 0.")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be > 0.")

    n = len(data_x)
    if n == 0:
        raise ValueError("local client data is empty.")

    for _ in range(steps):
        grad_w = [[0.0] * n_features for _ in range(n_classes)]
        grad_b = [0.0] * n_classes

        for _ in range(batch_size):
            idx = rng.randrange(n)
            x = data_x[idx]
            y = data_y[idx]
            probs = softmax(logits_for_sample(weights=weights, bias=bias, features=x))
            for c in range(n_classes):
                err = probs[c] - (1.0 if c == y else 0.0)
                grad_b[c] += err
                for j, value in x.items():
                    grad_w[c][j] += err * value

        inv_batch = 1.0 / batch_size
        for c in range(n_classes):
            bias[c] -= learning_rate * grad_b[c] * inv_batch
            row = weights[c]
            for j in range(n_features):
                row[j] -= learning_rate * grad_w[c][j] * inv_batch

    return weights, bias


def predict_classes(
    weights: list[list[float]],
    bias: list[float],
    data_x: list[dict[int, float]],
) -> list[int]:
    pred: list[int] = []
    for x in data_x:
        logits = logits_for_sample(weights=weights, bias=bias, features=x)
        best = max(range(len(logits)), key=lambda idx: logits[idx])
        pred.append(best)
    return pred


def evaluate(
    weights: list[list[float]],
    bias: list[float],
    test_x: list[dict[int, float]],
    test_y: list[int],
    n_classes: int,
) -> tuple[float, float]:
    pred = predict_classes(weights=weights, bias=bias, data_x=test_x)
    hits = sum(1 for t, p in zip(test_y, pred) if t == p)
    accuracy = hits / len(test_y)
    macro_f1, _ = cls.compute_macro_f1(truth=test_y, pred=pred, n_classes=n_classes)
    return accuracy, macro_f1


def partition_clients(
    train_x: list[dict[int, float]],
    train_y: list[int],
    n_clients: int,
    severity: float,
    seed: int,
) -> list[tuple[list[dict[int, float]], list[int]]]:
    if n_clients <= 1:
        raise ValueError("n_clients must be >= 2.")
    if severity < 0.0:
        raise ValueError("severity must be >= 0.")

    combined = list(zip(train_x, train_y))
    # Higher severity increases label clustering to simulate non-IID data.
    scored = sorted(combined, key=lambda row: (severity * row[1], len(row[0]), sum(row[0].values())))
    chunk_size = len(scored) // n_clients
    rng = random.Random(seed + 2026)

    clients: list[tuple[list[dict[int, float]], list[int]]] = []
    for idx in range(n_clients):
        start = idx * chunk_size
        end = (idx + 1) * chunk_size if idx < n_clients - 1 else len(scored)
        chunk = scored[start:end]
        rng.shuffle(chunk)
        xs = [x for x, _ in chunk]
        ys = [y for _, y in chunk]
        clients.append((xs, ys))
    return clients


def sparse_topk(
    vec: list[float],
    keep_ratio: float,
) -> tuple[list[float], int]:
    if not (0.0 < keep_ratio <= 1.0):
        raise ValueError("keep_ratio must be in (0.0, 1.0].")
    k = max(1, int(len(vec) * keep_ratio))
    if k >= len(vec):
        return list(vec), len(vec) * 4

    top_indices = sorted(range(len(vec)), key=lambda idx: abs(vec[idx]), reverse=True)[:k]
    dense = [0.0] * len(vec)
    for idx in top_indices:
        dense[idx] = vec[idx]

    # int32 index + float32 value per entry, plus a tiny header.
    bytes_sent = k * 8 + 4
    return dense, bytes_sent


def compress_delta(
    delta_vec: list[float],
    mode: str,
    sparse_ratio: float,
) -> tuple[list[float], int]:
    if mode == "fp32":
        return delta_vec, len(delta_vec) * 4
    if mode == "int8":
        qvec, scale = fed.quantize_int8(delta_vec)
        restored = fed.dequantize_int8(qvec, scale)
        return restored, len(qvec) + 4
    if mode == "sparse":
        return sparse_topk(delta_vec, keep_ratio=sparse_ratio)
    raise ValueError(f"Unknown mode: {mode}")


def run_centralized(
    train_x: list[dict[int, float]],
    train_y: list[int],
    test_x: list[dict[int, float]],
    test_y: list[int],
    n_classes: int,
    n_features: int,
    total_steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> RunResult:
    weights, bias = init_model(n_classes=n_classes, n_features=n_features)
    rng = random.Random(seed + 300)
    t0 = time.perf_counter()
    weights, bias = local_sgd_steps(
        weights=weights,
        bias=bias,
        data_x=train_x,
        data_y=train_y,
        steps=total_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        rng=rng,
    )
    runtime = time.perf_counter() - t0
    accuracy, macro_f1 = evaluate(
        weights=weights,
        bias=bias,
        test_x=test_x,
        test_y=test_y,
        n_classes=n_classes,
    )
    return RunResult(accuracy=accuracy, macro_f1=macro_f1, runtime_sec=runtime, uplink_bytes=0)


def run_fedavg_mode(
    mode: str,
    clients: list[tuple[list[dict[int, float]], list[int]]],
    test_x: list[dict[int, float]],
    test_y: list[int],
    n_classes: int,
    n_features: int,
    rounds: int,
    local_steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    sparse_ratio: float,
) -> RunResult:
    weights, bias = init_model(n_classes=n_classes, n_features=n_features)
    total_bytes = 0
    rng = random.Random(seed + 700)
    t0 = time.perf_counter()

    for round_idx in range(rounds):
        client_sizes = [len(x) for x, _ in clients]
        active_examples = sum(client_sizes)
        if active_examples == 0:
            raise ValueError("No active examples across clients.")

        aggregate = [0.0] * (n_classes * n_features + n_classes)
        base_vec = flatten_params(weights, bias)
        for client_idx, (client_x, client_y) in enumerate(clients):
            local_weights = [row[:] for row in weights]
            local_bias = list(bias)
            local_rng = random.Random(seed * 1000 + round_idx * 31 + client_idx * 17 + 9)
            local_weights, local_bias = local_sgd_steps(
                weights=local_weights,
                bias=local_bias,
                data_x=client_x,
                data_y=client_y,
                steps=local_steps,
                batch_size=batch_size,
                learning_rate=learning_rate,
                rng=local_rng,
            )
            local_vec = flatten_params(local_weights, local_bias)
            delta = [lv - sv for lv, sv in zip(local_vec, base_vec)]
            restored, sent_bytes = compress_delta(
                delta_vec=delta,
                mode=mode,
                sparse_ratio=sparse_ratio,
            )
            total_bytes += sent_bytes
            weight = len(client_x) / active_examples
            for idx, value in enumerate(restored):
                aggregate[idx] += weight * value

        updated_vec = [sv + dv for sv, dv in zip(base_vec, aggregate)]
        weights, bias = unflatten_params(
            vec=updated_vec,
            n_classes=n_classes,
            n_features=n_features,
        )

    runtime = time.perf_counter() - t0
    accuracy, macro_f1 = evaluate(
        weights=weights,
        bias=bias,
        test_x=test_x,
        test_y=test_y,
        n_classes=n_classes,
    )
    return RunResult(
        accuracy=accuracy,
        macro_f1=macro_f1,
        runtime_sec=runtime,
        uplink_bytes=total_bytes,
    )


def run_once(
    seed: int,
    modes: list[str],
    samples_per_label: int,
    test_fraction: float,
    n_clients: int,
    rounds: int,
    local_steps: int,
    batch_size: int,
    learning_rate: float,
    non_iid_severity: float,
    sparse_ratio: float,
) -> dict[str, RunResult]:
    rows = cls.generate_dataset(seed=seed, samples_per_label=samples_per_label)
    train_rows, test_rows = cls.stratified_split(rows=rows, seed=seed, test_fraction=test_fraction)
    labels = sorted({label for _, label in rows})
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    all_rows = train_rows + test_rows
    vocab, _, features = cls.build_tfidf_features(train_rows=train_rows, all_rows=all_rows)
    train_x = features[: len(train_rows)]
    test_x = features[len(train_rows) :]
    train_y = [label_to_idx[label] for _, label in train_rows]
    test_y = [label_to_idx[label] for _, label in test_rows]

    n_classes = len(labels)
    n_features = len(vocab)
    clients = partition_clients(
        train_x=train_x,
        train_y=train_y,
        n_clients=n_clients,
        severity=non_iid_severity,
        seed=seed,
    )
    total_steps = rounds * local_steps * n_clients

    output: dict[str, RunResult] = {
        "centralized": run_centralized(
            train_x=train_x,
            train_y=train_y,
            test_x=test_x,
            test_y=test_y,
            n_classes=n_classes,
            n_features=n_features,
            total_steps=total_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
        )
    }
    for mode in modes:
        output[f"fedavg_{mode}"] = run_fedavg_mode(
            mode=mode,
            clients=clients,
            test_x=test_x,
            test_y=test_y,
            n_classes=n_classes,
            n_features=n_features,
            rounds=rounds,
            local_steps=local_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            sparse_ratio=sparse_ratio,
        )
    return output


def aggregate(
    results: list[dict[str, RunResult]],
    config: dict[str, object],
    modes: list[str],
) -> dict[str, object]:
    method_names = ["centralized"] + [f"fedavg_{mode}" for mode in modes]
    methods: dict[str, dict[str, float | int]] = {}

    for name in method_names:
        acc = [r[name].accuracy for r in results]
        f1 = [r[name].macro_f1 for r in results]
        runtime = [r[name].runtime_sec for r in results]
        uplink = [r[name].uplink_bytes for r in results]
        methods[name] = {
            "accuracy_mean": statistics.mean(acc),
            "accuracy_std": statistics.pstdev(acc),
            "macro_f1_mean": statistics.mean(f1),
            "macro_f1_std": statistics.pstdev(f1),
            "runtime_mean_sec": statistics.mean(runtime),
            "runtime_std_sec": statistics.pstdev(runtime),
            "uplink_mean_bytes": int(statistics.mean(uplink)),
        }

    fp32_bytes = methods.get("fedavg_fp32", {}).get("uplink_mean_bytes", 0)
    comm_savings: dict[str, float] = {}
    for mode in modes:
        if mode == "fp32":
            continue
        mode_name = f"fedavg_{mode}"
        mode_bytes = methods[mode_name]["uplink_mean_bytes"]
        if fp32_bytes:
            comm_savings[f"{mode}_vs_fp32_percent"] = (1.0 - mode_bytes / fp32_bytes) * 100.0
        else:
            comm_savings[f"{mode}_vs_fp32_percent"] = 0.0

    central_acc = methods["centralized"]["accuracy_mean"]
    central_f1 = methods["centralized"]["macro_f1_mean"]
    quality_drop: dict[str, float] = {}
    for mode in modes:
        mode_name = f"fedavg_{mode}"
        quality_drop[f"{mode}_accuracy_drop"] = central_acc - methods[mode_name]["accuracy_mean"]
        quality_drop[f"{mode}_macro_f1_drop"] = central_f1 - methods[mode_name]["macro_f1_mean"]

    return {
        "schema_version": 1,
        "config": config,
        "runs": len(results),
        "methods": methods,
        "communication_savings_percent": comm_savings,
        "quality_drop_vs_centralized": quality_drop,
    }


def run_experiment(
    seeds: list[int],
    modes: list[str],
    samples_per_label: int,
    test_fraction: float,
    n_clients: int,
    rounds: int,
    local_steps: int,
    batch_size: int,
    learning_rate: float,
    non_iid_severity: float,
    sparse_ratio: float,
) -> dict[str, object]:
    valid_modes = {"fp32", "int8", "sparse"}
    if not modes:
        raise ValueError("At least one mode is required.")
    unknown = [mode for mode in modes if mode not in valid_modes]
    if unknown:
        raise ValueError(f"Unknown modes: {unknown}")
    if "fp32" not in modes:
        raise ValueError("modes must include fp32 as baseline.")

    all_results = [
        run_once(
            seed=seed,
            modes=modes,
            samples_per_label=samples_per_label,
            test_fraction=test_fraction,
            n_clients=n_clients,
            rounds=rounds,
            local_steps=local_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            non_iid_severity=non_iid_severity,
            sparse_ratio=sparse_ratio,
        )
        for seed in seeds
    ]
    config = {
        "seeds": seeds,
        "modes": modes,
        "samples_per_label": samples_per_label,
        "test_fraction": test_fraction,
        "n_clients": n_clients,
        "rounds": rounds,
        "local_steps": local_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "non_iid_severity": non_iid_severity,
        "sparse_ratio": sparse_ratio,
    }
    return aggregate(results=all_results, config=config, modes=modes)


def summarize(report: dict[str, object]) -> None:
    print("Federated utility classification (CPU-only)\n")
    config = report["config"]
    print(
        f"runs={report['runs']}, modes={config['modes']}, "
        f"clients={config['n_clients']}, rounds={config['rounds']}, "
        f"local_steps={config['local_steps']}"
    )
    print(f"{'Method':<16} {'Acc':<16} {'Macro-F1':<16} {'Runtime':<14} {'Uplink'}")
    print("-" * 88)
    for method, metrics in report["methods"].items():
        print(
            f"{method:<16} "
            f"{metrics['accuracy_mean']:.4f} +/- {metrics['accuracy_std']:.4f}   "
            f"{metrics['macro_f1_mean']:.4f} +/- {metrics['macro_f1_std']:.4f}   "
            f"{metrics['runtime_mean_sec']:.3f}s   "
            f"{int(metrics['uplink_mean_bytes'])} B"
        )

    if report["communication_savings_percent"]:
        print("\nCommunication savings vs fp32:")
        for key, value in report["communication_savings_percent"].items():
            print(f"- {key}: {value:.2f}%")
    print("\nQuality drop vs centralized:")
    for key, value in report["quality_drop_vs_centralized"].items():
        print(f"- {key}: {value:+.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        default="7,17,27",
        help="Comma-separated integer seeds (default: 7,17,27).",
    )
    parser.add_argument(
        "--modes",
        default="fp32,int8,sparse",
        help="Comma-separated modes from {fp32,int8,sparse} (default: all).",
    )
    parser.add_argument(
        "--samples-per-label",
        type=int,
        default=40,
        help="Generated samples per label (default: 40).",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Test split fraction (default: 0.2).",
    )
    parser.add_argument(
        "--n-clients",
        type=int,
        default=6,
        help="Number of federated clients (default: 6).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=20,
        help="Federated rounds (default: 20).",
    )
    parser.add_argument(
        "--local-steps",
        type=int,
        default=8,
        help="Local steps per client per round (default: 8).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Local batch size (default: 8).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.2,
        help="Learning rate (default: 0.2).",
    )
    parser.add_argument(
        "--non-iid-severity",
        type=float,
        default=1.0,
        help="Client partition non-IID severity (default: 1.0).",
    )
    parser.add_argument(
        "--sparse-ratio",
        type=float,
        default=0.2,
        help="Top-k sparse ratio for sparse mode in (0,1] (default: 0.2).",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional path to write machine-readable JSON report.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress human-readable summary.",
    )
    args = parser.parse_args()

    if not (0.0 < args.test_fraction < 1.0):
        parser.error("--test-fraction must be in (0,1).")
    if args.n_clients < 2:
        parser.error("--n-clients must be >= 2.")
    if args.rounds <= 0:
        parser.error("--rounds must be > 0.")
    if args.local_steps <= 0:
        parser.error("--local-steps must be > 0.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0.")
    if args.learning_rate <= 0.0:
        parser.error("--learning-rate must be > 0.")
    if args.non_iid_severity < 0.0:
        parser.error("--non-iid-severity must be >= 0.")
    if not (0.0 < args.sparse_ratio <= 1.0):
        parser.error("--sparse-ratio must be in (0,1].")

    seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
    modes = [part.strip() for part in args.modes.split(",") if part.strip()]
    report = run_experiment(
        seeds=seeds,
        modes=modes,
        samples_per_label=args.samples_per_label,
        test_fraction=args.test_fraction,
        n_clients=args.n_clients,
        rounds=args.rounds,
        local_steps=args.local_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        non_iid_severity=args.non_iid_severity,
        sparse_ratio=args.sparse_ratio,
    )
    if not args.quiet:
        summarize(report)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
