#!/usr/bin/env python3
"""
CPU-only federated adapter-style intent experiment.

This is a lightweight proxy for adapter training:
- freeze a base linear classifier (random initialization)
- train only a low-rank adapter (A @ B) + bias
- compare centralized reference vs FedAvg (fp32/int8/sparse updates)
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
from scripts import local_generalist_runtime as runtime


@dataclass
class RunResult:
    accuracy: float
    macro_f1: float
    runtime_sec: float
    uplink_bytes: int


def softmax(logits: list[float]) -> list[float]:
    peak = max(logits)
    exps = [math.exp(v - peak) for v in logits]
    total = sum(exps)
    return [v / total for v in exps]


def init_base_weights(n_classes: int, n_features: int, seed: int) -> list[list[float]]:
    _ = seed
    return [[0.0 for _ in range(n_features)] for _ in range(n_classes)]


def init_adapter(
    n_classes: int,
    n_features: int,
    rank: int,
    seed: int,
) -> tuple[list[list[float]], list[list[float]], list[float]]:
    if rank <= 0:
        raise ValueError("rank must be > 0.")
    rng = random.Random(seed + 193)
    a = [[rng.uniform(-0.02, 0.02) for _ in range(rank)] for _ in range(n_classes)]
    b = [[0.0 for _ in range(n_features)] for _ in range(rank)]
    bias = [0.0 for _ in range(n_classes)]
    return a, b, bias


def flatten_params(a: list[list[float]], b: list[list[float]], bias: list[float]) -> list[float]:
    vec: list[float] = []
    for row in a:
        vec.extend(row)
    for row in b:
        vec.extend(row)
    vec.extend(bias)
    return vec


def unflatten_params(
    vec: list[float],
    n_classes: int,
    n_features: int,
    rank: int,
) -> tuple[list[list[float]], list[list[float]], list[float]]:
    expected = n_classes * rank + rank * n_features + n_classes
    if len(vec) != expected:
        raise ValueError(f"Invalid param vector length={len(vec)} expected={expected}")
    offset = 0
    a: list[list[float]] = []
    for _ in range(n_classes):
        a.append(vec[offset : offset + rank])
        offset += rank
    b: list[list[float]] = []
    for _ in range(rank):
        b.append(vec[offset : offset + n_features])
        offset += n_features
    bias = vec[offset : offset + n_classes]
    return a, b, bias


def sparse_topk(vec: list[float], keep_ratio: float) -> tuple[list[float], int]:
    if not (0.0 < keep_ratio <= 1.0):
        raise ValueError("keep_ratio must be in (0.0, 1.0].")
    k = max(1, int(len(vec) * keep_ratio))
    if k >= len(vec):
        return list(vec), len(vec) * 4
    top_indices = sorted(range(len(vec)), key=lambda idx: abs(vec[idx]), reverse=True)[:k]
    dense = [0.0] * len(vec)
    for idx in top_indices:
        dense[idx] = vec[idx]
    return dense, k * 8 + 4


def compress_delta(delta: list[float], mode: str, sparse_ratio: float) -> tuple[list[float], int]:
    if mode == "fp32":
        return delta, len(delta) * 4
    if mode == "int8":
        q, scale = fed.quantize_int8(delta)
        restored = fed.dequantize_int8(q, scale)
        return restored, len(q) + 4
    if mode == "sparse":
        return sparse_topk(delta, keep_ratio=sparse_ratio)
    raise ValueError(f"Unknown mode: {mode}")


def logits_for_sample(
    x: dict[int, float],
    base_w: list[list[float]],
    a: list[list[float]],
    b: list[list[float]],
    bias: list[float],
) -> list[float]:
    rank = len(b)
    bx = [0.0] * rank
    for r in range(rank):
        accum = 0.0
        brow = b[r]
        for j, value in x.items():
            accum += brow[j] * value
        bx[r] = accum

    logits: list[float] = []
    for c, base_row in enumerate(base_w):
        score = bias[c]
        for j, value in x.items():
            score += base_row[j] * value
        for r in range(rank):
            score += a[c][r] * bx[r]
        logits.append(score)
    return logits


def local_sgd_steps(
    base_w: list[list[float]],
    a: list[list[float]],
    b: list[list[float]],
    bias: list[float],
    data_x: list[dict[int, float]],
    data_y: list[int],
    steps: int,
    learning_rate: float,
    seed: int,
) -> tuple[list[list[float]], list[list[float]], list[float]]:
    if steps <= 0:
        raise ValueError("steps must be > 0.")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be > 0.")
    if not data_x:
        raise ValueError("local data_x is empty.")

    n_classes = len(base_w)
    rank = len(b)
    rng = random.Random(seed)

    for _ in range(steps):
        idx = rng.randrange(len(data_x))
        x = data_x[idx]
        y = data_y[idx]

        logits = logits_for_sample(x=x, base_w=base_w, a=a, b=b, bias=bias)
        probs = softmax(logits)
        errs = [probs[c] - (1.0 if c == y else 0.0) for c in range(n_classes)]

        # Precompute B * x.
        bx = [0.0] * rank
        for r in range(rank):
            accum = 0.0
            brow = b[r]
            for j, value in x.items():
                accum += brow[j] * value
            bx[r] = accum

        # Gradients for A and bias.
        for c in range(n_classes):
            err = errs[c]
            bias[c] -= learning_rate * err
            for r in range(rank):
                a[c][r] -= learning_rate * err * bx[r]

        # Gradients for B use current A and errs.
        coeffs = [0.0] * rank
        for r in range(rank):
            coeffs[r] = sum(errs[c] * a[c][r] for c in range(n_classes))
        for r in range(rank):
            brow = b[r]
            coeff = coeffs[r]
            for j, value in x.items():
                brow[j] -= learning_rate * coeff * value

    return a, b, bias


def predict(
    data_x: list[dict[int, float]],
    base_w: list[list[float]],
    a: list[list[float]],
    b: list[list[float]],
    bias: list[float],
) -> list[int]:
    pred: list[int] = []
    for x in data_x:
        logits = logits_for_sample(x=x, base_w=base_w, a=a, b=b, bias=bias)
        pred.append(max(range(len(logits)), key=lambda idx: logits[idx]))
    return pred


def evaluate(
    data_x: list[dict[int, float]],
    data_y: list[int],
    base_w: list[list[float]],
    a: list[list[float]],
    b: list[list[float]],
    bias: list[float],
    n_classes: int,
) -> tuple[float, float]:
    pred = predict(data_x=data_x, base_w=base_w, a=a, b=b, bias=bias)
    hits = sum(1 for t, p in zip(data_y, pred) if t == p)
    acc = hits / len(data_y)
    macro_f1, _ = cls.compute_macro_f1(truth=data_y, pred=pred, n_classes=n_classes)
    return acc, macro_f1


def partition_clients(
    train_x: list[dict[int, float]],
    train_y: list[int],
    n_clients: int,
    non_iid_severity: float,
    seed: int,
) -> list[tuple[list[dict[int, float]], list[int]]]:
    if n_clients <= 1:
        raise ValueError("n_clients must be >= 2.")
    if non_iid_severity < 0.0:
        raise ValueError("non_iid_severity must be >= 0.")

    combined = list(zip(train_x, train_y))
    if non_iid_severity == 0.0:
        rng = random.Random(seed + 51)
        rng.shuffle(combined)
    else:
        combined = sorted(combined, key=lambda row: (row[1] * non_iid_severity, len(row[0]), sum(row[0].values())))

    chunk = len(combined) // n_clients
    clients: list[tuple[list[dict[int, float]], list[int]]] = []
    for idx in range(n_clients):
        start = idx * chunk
        end = (idx + 1) * chunk if idx < n_clients - 1 else len(combined)
        slice_rows = combined[start:end]
        xs = [x for x, _ in slice_rows]
        ys = [y for _, y in slice_rows]
        clients.append((xs, ys))
    return clients


def build_intent_dataset(
    seed: int,
    samples_per_intent: int,
) -> tuple[list[dict[int, float]], list[int], list[dict[int, float]], list[int], int]:
    rows = runtime.generate_intent_dataset(seed=seed, samples_per_intent=samples_per_intent)
    train_rows, test_rows = cls.stratified_split(rows=rows, seed=seed, test_fraction=0.25)
    all_rows = train_rows + test_rows
    vocab, features = runtime.vectorize(all_rows)
    labels = sorted({label for _, label in all_rows})
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    train_x = features[: len(train_rows)]
    train_y = [label_to_idx[label] for _, label in train_rows]
    test_x = features[len(train_rows) :]
    test_y = [label_to_idx[label] for _, label in test_rows]
    return train_x, train_y, test_x, test_y, len(vocab)


def run_centralized(
    train_x: list[dict[int, float]],
    train_y: list[int],
    test_x: list[dict[int, float]],
    test_y: list[int],
    base_w: list[list[float]],
    rank: int,
    steps: int,
    learning_rate: float,
    seed: int,
) -> RunResult:
    n_classes = len(base_w)
    n_features = len(base_w[0]) if base_w else 0
    a, b, bias = init_adapter(
        n_classes=n_classes,
        n_features=n_features,
        rank=rank,
        seed=seed,
    )
    t0 = time.perf_counter()
    a, b, bias = local_sgd_steps(
        base_w=base_w,
        a=a,
        b=b,
        bias=bias,
        data_x=train_x,
        data_y=train_y,
        steps=steps,
        learning_rate=learning_rate,
        seed=seed + 1001,
    )
    runtime_sec = time.perf_counter() - t0
    acc, macro_f1 = evaluate(
        data_x=test_x,
        data_y=test_y,
        base_w=base_w,
        a=a,
        b=b,
        bias=bias,
        n_classes=n_classes,
    )
    return RunResult(accuracy=acc, macro_f1=macro_f1, runtime_sec=runtime_sec, uplink_bytes=0)


def run_fedavg(
    mode: str,
    clients: list[tuple[list[dict[int, float]], list[int]]],
    test_x: list[dict[int, float]],
    test_y: list[int],
    base_w: list[list[float]],
    rank: int,
    rounds: int,
    local_steps: int,
    learning_rate: float,
    sparse_ratio: float,
    seed: int,
) -> RunResult:
    n_classes = len(base_w)
    n_features = len(base_w[0]) if base_w else 0
    global_a, global_b, global_bias = init_adapter(
        n_classes=n_classes,
        n_features=n_features,
        rank=rank,
        seed=seed,
    )
    total_uplink = 0
    t0 = time.perf_counter()

    for round_idx in range(rounds):
        base_vec = flatten_params(global_a, global_b, global_bias)
        aggregate = [0.0] * len(base_vec)
        total_examples = sum(len(xs) for xs, _ in clients)
        if total_examples <= 0:
            raise ValueError("No client examples available.")

        for client_idx, (client_x, client_y) in enumerate(clients):
            local_a = [row[:] for row in global_a]
            local_b = [row[:] for row in global_b]
            local_bias = list(global_bias)
            local_a, local_b, local_bias = local_sgd_steps(
                base_w=base_w,
                a=local_a,
                b=local_b,
                bias=local_bias,
                data_x=client_x,
                data_y=client_y,
                steps=local_steps,
                learning_rate=learning_rate,
                seed=seed * 100 + round_idx * 17 + client_idx * 7 + 5,
            )

            local_vec = flatten_params(local_a, local_b, local_bias)
            delta = [lv - gv for lv, gv in zip(local_vec, base_vec)]
            restored, bytes_sent = compress_delta(delta=delta, mode=mode, sparse_ratio=sparse_ratio)
            total_uplink += bytes_sent

            weight = len(client_x) / total_examples
            for idx, value in enumerate(restored):
                aggregate[idx] += weight * value

        merged = [gv + dv for gv, dv in zip(base_vec, aggregate)]
        global_a, global_b, global_bias = unflatten_params(
            vec=merged,
            n_classes=n_classes,
            n_features=n_features,
            rank=rank,
        )

    runtime_sec = time.perf_counter() - t0
    acc, macro_f1 = evaluate(
        data_x=test_x,
        data_y=test_y,
        base_w=base_w,
        a=global_a,
        b=global_b,
        bias=global_bias,
        n_classes=n_classes,
    )
    return RunResult(
        accuracy=acc,
        macro_f1=macro_f1,
        runtime_sec=runtime_sec,
        uplink_bytes=total_uplink,
    )


def aggregate(
    results: list[dict[str, RunResult]],
    config: dict[str, object],
    modes: list[str],
) -> dict[str, object]:
    methods: dict[str, dict[str, object]] = {}
    all_names = ["centralized"] + [f"fedavg_{mode}" for mode in modes]
    for name in all_names:
        runs = [row[name] for row in results]
        methods[name] = {
            "accuracy_mean": statistics.mean(item.accuracy for item in runs),
            "accuracy_std": statistics.pstdev(item.accuracy for item in runs),
            "macro_f1_mean": statistics.mean(item.macro_f1 for item in runs),
            "macro_f1_std": statistics.pstdev(item.macro_f1 for item in runs),
            "runtime_mean_sec": statistics.mean(item.runtime_sec for item in runs),
            "runtime_std_sec": statistics.pstdev(item.runtime_sec for item in runs),
            "uplink_mean_bytes": statistics.mean(item.uplink_bytes for item in runs),
            "uplink_std_bytes": statistics.pstdev(item.uplink_bytes for item in runs),
        }

    fp32_bytes = methods.get("fedavg_fp32", {}).get("uplink_mean_bytes", 0.0)
    comm_savings: dict[str, float] = {}
    if fp32_bytes > 0.0:
        for mode in modes:
            mode_name = f"fedavg_{mode}"
            mode_bytes = methods[mode_name]["uplink_mean_bytes"]
            comm_savings[f"{mode}_vs_fp32_percent"] = 100.0 * (1.0 - mode_bytes / fp32_bytes)

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


def run_once(
    seed: int,
    modes: list[str],
    samples_per_intent: int,
    n_clients: int,
    rounds: int,
    local_steps: int,
    learning_rate: float,
    sparse_ratio: float,
    rank: int,
    non_iid_severity: float,
) -> dict[str, RunResult]:
    train_x, train_y, test_x, test_y, n_features = build_intent_dataset(
        seed=seed,
        samples_per_intent=samples_per_intent,
    )
    n_classes = len(set(train_y + test_y))
    base_w = init_base_weights(n_classes=n_classes, n_features=n_features, seed=seed)
    clients = partition_clients(
        train_x=train_x,
        train_y=train_y,
        n_clients=n_clients,
        non_iid_severity=non_iid_severity,
        seed=seed,
    )

    result: dict[str, RunResult] = {}
    result["centralized"] = run_centralized(
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        test_y=test_y,
        base_w=base_w,
        rank=rank,
        steps=rounds * local_steps * max(1, n_clients // 2),
        learning_rate=learning_rate,
        seed=seed,
    )
    for mode in modes:
        result[f"fedavg_{mode}"] = run_fedavg(
            mode=mode,
            clients=clients,
            test_x=test_x,
            test_y=test_y,
            base_w=base_w,
            rank=rank,
            rounds=rounds,
            local_steps=local_steps,
            learning_rate=learning_rate,
            sparse_ratio=sparse_ratio,
            seed=seed,
        )
    return result


def run_experiment(
    seeds: list[int],
    modes: list[str],
    samples_per_intent: int,
    n_clients: int,
    rounds: int,
    local_steps: int,
    learning_rate: float,
    sparse_ratio: float,
    rank: int,
    non_iid_severity: float,
) -> dict[str, object]:
    valid_modes = {"fp32", "int8", "sparse"}
    if not modes:
        raise ValueError("At least one mode is required.")
    unknown = [mode for mode in modes if mode not in valid_modes]
    if unknown:
        raise ValueError(f"Unknown modes: {unknown}")
    if "fp32" not in modes:
        raise ValueError("modes must include fp32 as baseline.")

    results = [
        run_once(
            seed=seed,
            modes=modes,
            samples_per_intent=samples_per_intent,
            n_clients=n_clients,
            rounds=rounds,
            local_steps=local_steps,
            learning_rate=learning_rate,
            sparse_ratio=sparse_ratio,
            rank=rank,
            non_iid_severity=non_iid_severity,
        )
        for seed in seeds
    ]
    config = {
        "seeds": seeds,
        "modes": modes,
        "samples_per_intent": samples_per_intent,
        "n_clients": n_clients,
        "rounds": rounds,
        "local_steps": local_steps,
        "learning_rate": learning_rate,
        "sparse_ratio": sparse_ratio,
        "rank": rank,
        "non_iid_severity": non_iid_severity,
    }
    return aggregate(results=results, config=config, modes=modes)


def summarize(report: dict[str, object]) -> None:
    print("Federated adapter-style intent experiment\n")
    cfg = report["config"]
    print(
        f"runs={report['runs']}, modes={cfg['modes']}, rank={cfg['rank']}, "
        f"clients={cfg['n_clients']}, rounds={cfg['rounds']}"
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
    parser.add_argument("--seeds", default="7,17,27", help="Comma-separated seeds.")
    parser.add_argument(
        "--modes",
        default="fp32,int8,sparse",
        help="Comma-separated federated modes (must include fp32).",
    )
    parser.add_argument(
        "--samples-per-intent",
        type=int,
        default=28,
        help="Synthetic samples per intent label.",
    )
    parser.add_argument("--n-clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-steps", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.16)
    parser.add_argument("--sparse-ratio", type=float, default=0.2)
    parser.add_argument("--rank", type=int, default=4, help="Adapter rank.")
    parser.add_argument("--non-iid-severity", type=float, default=1.2)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json-out", default="", help="Optional JSON output path.")
    args = parser.parse_args()

    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    report = run_experiment(
        seeds=seeds,
        modes=modes,
        samples_per_intent=args.samples_per_intent,
        n_clients=args.n_clients,
        rounds=args.rounds,
        local_steps=args.local_steps,
        learning_rate=args.learning_rate,
        sparse_ratio=args.sparse_ratio,
        rank=args.rank,
        non_iid_severity=args.non_iid_severity,
    )
    if not args.quiet:
        summarize(report)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
