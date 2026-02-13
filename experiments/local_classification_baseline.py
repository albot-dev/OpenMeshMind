#!/usr/bin/env python3
"""
Dependency-light local text classification baseline for commodity CPUs.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import time
from collections import Counter


TOKEN_RE = re.compile(r"[a-z0-9]+")

LABEL_TEMPLATES: dict[str, list[str]] = {
    "budget_planning": [
        "plan a monthly budget for {context} and reduce {expense}",
        "track spending on {expense} while saving for {goal}",
        "create a simple savings rule for {context} costs",
        "review bills and cut {expense} this quarter",
    ],
    "energy_saving": [
        "lower home energy use by adjusting {appliance} schedule",
        "reduce electricity waste from {appliance} at night",
        "set efficient usage targets for {appliance} in winter",
        "audit power draw and optimize {appliance} settings",
    ],
    "wellness_routine": [
        "build a daily routine for {habit} and better sleep",
        "improve consistency with {habit} and hydration reminders",
        "set weekly goals for {habit} and recovery",
        "track progress on {habit} with low stress habits",
    ],
    "transport_planning": [
        "optimize commute route using {mode} and avoid delays",
        "plan affordable trips with {mode} during peak hours",
        "compare {mode} options for reliable arrival times",
        "reduce travel time by coordinating {mode} transfers",
    ],
}

WORD_BANK = {
    "context": [
        "family groceries",
        "rent",
        "student expenses",
        "household basics",
        "medical bills",
    ],
    "expense": [
        "subscriptions",
        "delivery fees",
        "food waste",
        "utility bills",
        "impulse purchases",
    ],
    "goal": [
        "emergency fund",
        "new laptop",
        "debt payoff",
        "annual vacation",
        "home repairs",
    ],
    "appliance": [
        "water heater",
        "dishwasher",
        "air conditioner",
        "space heater",
        "dryer",
    ],
    "habit": [
        "walking",
        "stretching",
        "meal prep",
        "sleep schedule",
        "mindfulness practice",
    ],
    "mode": [
        "bus",
        "bike",
        "train",
        "rideshare",
        "carpool",
    ],
}

NOISE_TERMS = [
    "local",
    "community",
    "weekly",
    "planning",
    "checklist",
    "update",
    "simple",
    "practical",
]


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def generate_dataset(
    seed: int,
    samples_per_label: int,
) -> list[tuple[str, str]]:
    if samples_per_label < 4:
        raise ValueError("samples_per_label must be >= 4.")

    rng = random.Random(seed)
    rows: list[tuple[str, str]] = []
    for label, templates in LABEL_TEMPLATES.items():
        for _ in range(samples_per_label):
            template = rng.choice(templates)
            text = template.format(
                context=rng.choice(WORD_BANK["context"]),
                expense=rng.choice(WORD_BANK["expense"]),
                goal=rng.choice(WORD_BANK["goal"]),
                appliance=rng.choice(WORD_BANK["appliance"]),
                habit=rng.choice(WORD_BANK["habit"]),
                mode=rng.choice(WORD_BANK["mode"]),
            )
            noise_count = rng.randint(0, 2)
            if noise_count:
                noise = " ".join(rng.sample(NOISE_TERMS, noise_count))
                text = f"{text} {noise}"
            rows.append((text, label))
    rng.shuffle(rows)
    return rows


def stratified_split(
    rows: list[tuple[str, str]],
    seed: int,
    test_fraction: float,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    if not (0.0 < test_fraction < 1.0):
        raise ValueError("test_fraction must be in (0.0, 1.0).")

    by_label: dict[str, list[tuple[str, str]]] = {}
    for text, label in rows:
        by_label.setdefault(label, []).append((text, label))

    rng = random.Random(seed + 991)
    train: list[tuple[str, str]] = []
    test: list[tuple[str, str]] = []
    for label_rows in by_label.values():
        label_rows = list(label_rows)
        rng.shuffle(label_rows)
        n_test = max(1, int(len(label_rows) * test_fraction))
        test.extend(label_rows[:n_test])
        train.extend(label_rows[n_test:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def build_tfidf_features(
    train_rows: list[tuple[str, str]],
    all_rows: list[tuple[str, str]],
) -> tuple[dict[str, int], dict[int, float], list[dict[int, float]]]:
    df: Counter[str] = Counter()
    tf_train: list[Counter[str]] = []
    tf_all: list[Counter[str]] = []

    for text, _ in train_rows:
        tokens = tokenize(text)
        counts = Counter(tokens)
        tf_train.append(counts)
        for term in set(tokens):
            df[term] += 1

    vocab_terms = sorted(df.keys())
    vocab = {term: idx for idx, term in enumerate(vocab_terms)}
    n_docs = len(train_rows)
    idf = {
        vocab[term]: math.log((n_docs + 1) / (count + 1)) + 1.0 for term, count in df.items()
    }

    for text, _ in all_rows:
        tf_all.append(Counter(tokenize(text)))

    features: list[dict[int, float]] = []
    for counts in tf_all:
        total = sum(counts.values())
        if total == 0:
            features.append({})
            continue
        row: dict[int, float] = {}
        for term, count in counts.items():
            idx = vocab.get(term)
            if idx is None:
                continue
            row[idx] = (count / total) * idf[idx]
        features.append(row)
    return vocab, idf, features


def softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(value - m) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]


def train_softmax_classifier(
    train_x: list[dict[int, float]],
    train_y: list[int],
    n_classes: int,
    n_features: int,
    steps: int,
    learning_rate: float,
    seed: int,
) -> tuple[list[list[float]], list[float], float]:
    if steps <= 0:
        raise ValueError("steps must be > 0.")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be > 0.")

    weights = [[0.0] * n_features for _ in range(n_classes)]
    bias = [0.0] * n_classes
    rng = random.Random(seed + 4096)
    start = time.perf_counter()

    for _ in range(steps):
        idx = rng.randrange(len(train_x))
        features = train_x[idx]
        target = train_y[idx]

        logits = []
        for c in range(n_classes):
            score = bias[c]
            for j, value in features.items():
                score += weights[c][j] * value
            logits.append(score)
        probs = softmax(logits)

        for c in range(n_classes):
            grad = probs[c] - (1.0 if c == target else 0.0)
            bias[c] -= learning_rate * grad
            for j, value in features.items():
                weights[c][j] -= learning_rate * grad * value

    runtime = time.perf_counter() - start
    return weights, bias, runtime


def predict_class(weights: list[list[float]], bias: list[float], features: dict[int, float]) -> int:
    best_class = 0
    best_score = float("-inf")
    for c, row in enumerate(weights):
        score = bias[c]
        for j, value in features.items():
            score += row[j] * value
        if score > best_score:
            best_score = score
            best_class = c
    return best_class


def compute_macro_f1(
    truth: list[int],
    pred: list[int],
    n_classes: int,
) -> tuple[float, dict[int, float]]:
    per_class_f1: dict[int, float] = {}
    for c in range(n_classes):
        tp = sum(1 for t, p in zip(truth, pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(truth, pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(truth, pred) if t == c and p != c)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0.0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        per_class_f1[c] = f1
    macro = statistics.mean(per_class_f1.values())
    return macro, per_class_f1


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    idx = int(0.95 * (len(values) - 1))
    return sorted(values)[idx]


def run_classification(
    seed: int,
    samples_per_label: int,
    test_fraction: float,
    steps: int,
    learning_rate: float,
    measure_latency: bool = True,
) -> dict[str, object]:
    rows = generate_dataset(seed=seed, samples_per_label=samples_per_label)
    train_rows, test_rows = stratified_split(rows=rows, seed=seed, test_fraction=test_fraction)

    labels = sorted({label for _, label in rows})
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    all_rows = train_rows + test_rows
    vocab, _, features = build_tfidf_features(train_rows=train_rows, all_rows=all_rows)
    train_x = features[: len(train_rows)]
    test_x = features[len(train_rows) :]
    train_y = [label_to_idx[label] for _, label in train_rows]
    test_y = [label_to_idx[label] for _, label in test_rows]

    weights, bias, train_runtime = train_softmax_classifier(
        train_x=train_x,
        train_y=train_y,
        n_classes=len(labels),
        n_features=len(vocab),
        steps=steps,
        learning_rate=learning_rate,
        seed=seed,
    )

    pred: list[int] = []
    latencies_ms: list[float] = []
    for row in test_x:
        t0 = time.perf_counter()
        yhat = predict_class(weights=weights, bias=bias, features=row)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        pred.append(yhat)
        if measure_latency:
            latencies_ms.append(dt_ms)

    hits = sum(1 for t, p in zip(test_y, pred) if t == p)
    accuracy = hits / len(test_y)
    macro_f1, per_class_f1 = compute_macro_f1(truth=test_y, pred=pred, n_classes=len(labels))
    per_label_f1 = {label: per_class_f1[idx] for label, idx in label_to_idx.items()}

    return {
        "schema_version": 1,
        "config": {
            "seed": seed,
            "samples_per_label": samples_per_label,
            "test_fraction": test_fraction,
            "steps": steps,
            "learning_rate": learning_rate,
            "tokenizer": "lower_alnum_regex",
            "vectorizer": "tfidf",
            "model": "softmax_sgd",
        },
        "counts": {
            "labels": labels,
            "train_samples": len(train_rows),
            "test_samples": len(test_rows),
            "vocabulary_size": len(vocab),
        },
        "metrics": {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "per_label_f1": per_label_f1,
            "train_runtime_sec": train_runtime,
            "latency_mean_ms": statistics.mean(latencies_ms) if latencies_ms else 0.0,
            "latency_p95_ms": p95(latencies_ms) if latencies_ms else 0.0,
        },
    }


def summarize(report: dict[str, object]) -> None:
    counts = report["counts"]
    metrics = report["metrics"]
    config = report["config"]
    print("Local classification baseline (CPU-only, dependency-light)\n")
    print(
        f"labels={len(counts['labels'])}, train={counts['train_samples']}, "
        f"test={counts['test_samples']}, vocab={counts['vocabulary_size']}"
    )
    print(
        f"seed={config['seed']}, steps={config['steps']}, "
        f"lr={config['learning_rate']:.3f}, test_fraction={config['test_fraction']:.2f}"
    )
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"Train runtime: {metrics['train_runtime_sec']:.3f} s")
    print(f"Mean latency/sample: {metrics['latency_mean_ms']:.3f} ms")
    print(f"P95 latency/sample: {metrics['latency_p95_ms']:.3f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7, help="Random seed (default: 7).")
    parser.add_argument(
        "--samples-per-label",
        type=int,
        default=60,
        help="Number of generated samples per label (default: 60).",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Test split fraction in (0,1), default: 0.2.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2200,
        help="Training steps for SGD (default: 2200).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.18,
        help="Learning rate for SGD (default: 0.18).",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional path to write machine-readable JSON metrics.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress human-readable summary.",
    )
    args = parser.parse_args()

    report = run_classification(
        seed=args.seed,
        samples_per_label=args.samples_per_label,
        test_fraction=args.test_fraction,
        steps=args.steps,
        learning_rate=args.learning_rate,
        measure_latency=True,
    )
    if not args.quiet:
        summarize(report)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
