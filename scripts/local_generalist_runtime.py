#!/usr/bin/env python3
"""
Local generalist runtime MVP:
- intent routing via a tiny CPU-only softmax model
- local retrieval from repository corpus
- calculator tool execution
- short-term memory across turns
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import local_retrieval_baseline as retrieval


TOKEN_RE = re.compile(r"[a-z0-9]+")
EXPR_RE = re.compile(r"[-+/*().%0-9 ]+")


INTENT_TEMPLATES: dict[str, list[str]] = {
    "memory_store": [
        "remember that my project codename is atlas",
        "remember this note: weekly report due monday",
        "please store this for later: demo starts at 14:00 utc",
        "note this down: use sparse updates",
    ],
    "memory_recall": [
        "what did i ask you to remember",
        "recall my latest note",
        "tell me the note i saved earlier",
        "what was my reminder",
    ],
    "retrieval_lookup": [
        "lookup cpu edge federated learning in corpus",
        "find retrieval note about tfidf vectors",
        "search local docs for index fund savings",
        "retrieve what document mentions ranking vectors",
    ],
    "tool_calculator": [
        "calculate (12 + 8) * 3",
        "compute 7*9",
        "what is 150 / 6",
        "evaluate 19 + 5 - 3",
    ],
    "response_exact": [
        "respond exactly with: ACK READY",
        "reply exactly with: phase gate passed",
        "say exactly: local mode enabled",
        "print exactly: done",
    ],
}

INTENT_NOISE = [
    "please",
    "now",
    "quickly",
    "local",
    "pilot",
    "node",
    "status",
    "update",
]


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def generate_intent_dataset(seed: int, samples_per_intent: int) -> list[tuple[str, str]]:
    if samples_per_intent < 4:
        raise ValueError("samples_per_intent must be >= 4.")
    rng = random.Random(seed)
    rows: list[tuple[str, str]] = []
    for intent, templates in INTENT_TEMPLATES.items():
        for _ in range(samples_per_intent):
            text = rng.choice(templates)
            noise_count = rng.randint(0, 2)
            if noise_count:
                text = f"{text} {' '.join(rng.sample(INTENT_NOISE, noise_count))}"
            rows.append((text, intent))
    rng.shuffle(rows)
    return rows


def vectorize(rows: list[tuple[str, str]]) -> tuple[dict[str, int], list[dict[int, float]]]:
    vocab_terms: set[str] = set()
    tokenized: list[list[str]] = []
    for text, _ in rows:
        toks = tokenize(text)
        tokenized.append(toks)
        vocab_terms.update(toks)

    vocab = {term: idx for idx, term in enumerate(sorted(vocab_terms))}
    features: list[dict[int, float]] = []
    for toks in tokenized:
        if not toks:
            features.append({})
            continue
        counts: dict[int, int] = {}
        for tok in toks:
            idx = vocab.get(tok)
            if idx is None:
                continue
            counts[idx] = counts.get(idx, 0) + 1
        total = sum(counts.values())
        features.append({idx: count / total for idx, count in counts.items()})
    return vocab, features


def softmax(logits: list[float]) -> list[float]:
    peak = max(logits)
    exps = [math.exp(v - peak) for v in logits]
    total = sum(exps)
    return [v / total for v in exps]


def train_softmax(
    x: list[dict[int, float]],
    y: list[int],
    n_classes: int,
    n_features: int,
    steps: int,
    learning_rate: float,
    seed: int,
) -> tuple[list[list[float]], list[float]]:
    if steps <= 0:
        raise ValueError("steps must be > 0.")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be > 0.")

    w = [[0.0] * n_features for _ in range(n_classes)]
    b = [0.0] * n_classes
    rng = random.Random(seed + 913)

    for _ in range(steps):
        idx = rng.randrange(len(x))
        xi = x[idx]
        yi = y[idx]
        logits = []
        for c in range(n_classes):
            score = b[c]
            for j, value in xi.items():
                score += w[c][j] * value
            logits.append(score)
        probs = softmax(logits)
        for c in range(n_classes):
            grad = probs[c] - (1.0 if c == yi else 0.0)
            b[c] -= learning_rate * grad
            for j, value in xi.items():
                w[c][j] -= learning_rate * grad * value
    return w, b


def predict_softmax(
    text: str,
    vocab: dict[str, int],
    labels: list[str],
    weights: list[list[float]],
    bias: list[float],
) -> tuple[str, float]:
    toks = tokenize(text)
    if not toks:
        return labels[0], 0.0
    counts: dict[int, int] = {}
    for tok in toks:
        idx = vocab.get(tok)
        if idx is None:
            continue
        counts[idx] = counts.get(idx, 0) + 1
    total = sum(counts.values())
    if total == 0:
        return labels[0], 0.0
    feat = {idx: count / total for idx, count in counts.items()}

    logits = []
    for c in range(len(labels)):
        score = bias[c]
        for j, value in feat.items():
            score += weights[c][j] * value
        logits.append(score)
    probs = softmax(logits)
    best_idx = max(range(len(labels)), key=lambda idx: probs[idx])
    return labels[best_idx], probs[best_idx]


def sanitize_expression(text: str) -> str:
    matches = [item.strip() for item in EXPR_RE.findall(text)]
    candidates = [item for item in matches if item and any(ch.isdigit() for ch in item)]
    if not candidates:
        return ""
    expr = max(candidates, key=len)
    return expr


def safe_eval_expression(expr: str) -> float:
    if not expr:
        raise ValueError("No expression found.")
    tree = ast.parse(expr, mode="eval")
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.FloorDiv,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"Unsupported expression node: {type(node).__name__}")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError("Expression constants must be numeric.")
    value = eval(compile(tree, filename="<expr>", mode="eval"), {"__builtins__": {}}, {})
    if not isinstance(value, (int, float)):
        raise ValueError("Expression result must be numeric.")
    return float(value)


def normalize_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def extract_exact_response(prompt: str) -> str | None:
    lowered = prompt.lower()
    patterns = [
        "respond exactly with:",
        "reply exactly with:",
        "say exactly:",
        "print exactly:",
    ]
    for marker in patterns:
        idx = lowered.find(marker)
        if idx == -1:
            continue
        return prompt[idx + len(marker) :].strip().strip('"').strip("'")
    return None


def extract_memory_note(prompt: str) -> str | None:
    lowered = prompt.lower()
    markers = [
        "remember that",
        "remember this note:",
        "remember:",
        "store this for later:",
        "note this down:",
    ]
    for marker in markers:
        idx = lowered.find(marker)
        if idx == -1:
            continue
        value = prompt[idx + len(marker) :].strip()
        if value:
            return value
    if lowered.startswith("remember "):
        value = prompt[len("remember ") :].strip()
        if value:
            return value
    return None


def infer_intent_override(prompt: str) -> str | None:
    lowered = prompt.lower()
    if extract_exact_response(prompt):
        return "response_exact"
    if extract_memory_note(prompt):
        return "memory_store"
    if "what did i ask you to remember" in lowered or "recall" in lowered:
        return "memory_recall"
    if any(keyword in lowered for keyword in ["calculate", "compute", "evaluate", "what is "]):
        if any(ch.isdigit() for ch in lowered):
            return "tool_calculator"
    if any(keyword in lowered for keyword in ["lookup", "search", "find", "retrieve", "corpus"]):
        return "retrieval_lookup"
    return None


@dataclass
class MemoryItem:
    note: str
    timestamp_utc: str


class LocalGeneralistRuntime:
    def __init__(
        self,
        corpus_path: str = "data/retrieval_corpus.json",
        top_k: int = 3,
        max_memory_turns: int = 8,
        seed: int = 7,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be > 0.")
        if max_memory_turns <= 0:
            raise ValueError("max_memory_turns must be > 0.")

        self.top_k = top_k
        self.max_memory_turns = max_memory_turns
        self.memory: list[MemoryItem] = []
        self.intent = self._train_intent_router(seed=seed)
        self.corpus = self._load_corpus(corpus_path=corpus_path)

    def _train_intent_router(self, seed: int) -> dict[str, object]:
        rows = generate_intent_dataset(seed=seed, samples_per_intent=24)
        labels = sorted({label for _, label in rows})
        label_to_idx = {label: idx for idx, label in enumerate(labels)}
        vocab, features = vectorize(rows)
        y = [label_to_idx[label] for _, label in rows]
        weights, bias = train_softmax(
            x=features,
            y=y,
            n_classes=len(labels),
            n_features=len(vocab),
            steps=1200,
            learning_rate=0.18,
            seed=seed,
        )
        return {
            "labels": labels,
            "vocab": vocab,
            "weights": weights,
            "bias": bias,
        }

    def _load_corpus(self, corpus_path: str) -> dict[str, object]:
        path = Path(corpus_path)
        if not path.is_absolute():
            path = ROOT / path
        rows = retrieval.load_json(str(path))
        docs: list[tuple[str, str, list[str], dict[str, float], float]] = []
        tokens = [retrieval.tokenize(row["text"]) for row in rows]
        idf = retrieval.compute_idf(tokens)
        for row, doc_tokens in zip(rows, tokens):
            vec = retrieval.tfidf_vector(doc_tokens, idf)
            docs.append((row["id"], row["text"], doc_tokens, vec, retrieval.l2_norm(vec)))
        return {"path": str(path), "idf": idf, "docs": docs}

    def _router_predict(self, prompt: str) -> tuple[str, float]:
        labels = self.intent["labels"]
        vocab = self.intent["vocab"]
        weights = self.intent["weights"]
        bias = self.intent["bias"]
        return predict_softmax(
            text=prompt,
            vocab=vocab,
            labels=labels,
            weights=weights,
            bias=bias,
        )

    def _retrieve(self, query: str, top_k: int) -> list[dict[str, object]]:
        idf = self.corpus["idf"]
        docs = self.corpus["docs"]
        q_vec = retrieval.tfidf_vector(retrieval.tokenize(query), idf)
        scored: list[tuple[str, str, float]] = []
        for doc_id, text, _, vec, vec_norm in docs:
            score = retrieval.cosine(q_vec, vec, vec_norm)
            scored.append((doc_id, text, score))
        scored.sort(key=lambda row: row[2], reverse=True)
        hits = []
        for doc_id, text, score in scored[:top_k]:
            hits.append(
                {
                    "id": doc_id,
                    "score": score,
                    "snippet": text[:180],
                }
            )
        return hits

    def _remember(self, note: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.memory.append(MemoryItem(note=note, timestamp_utc=now))
        if len(self.memory) > self.max_memory_turns:
            self.memory = self.memory[-self.max_memory_turns :]

    def _recall(self, prompt: str) -> str:
        if not self.memory:
            return "No notes are stored yet."
        lowered = prompt.lower()
        if "about" in lowered:
            keyword = lowered.split("about", 1)[1].strip()
            for item in reversed(self.memory):
                if keyword and keyword in item.note.lower():
                    return f"Stored note: {item.note}"
        latest = self.memory[-1]
        return f"Stored note: {latest.note}"

    def respond(self, prompt: str, top_k: int | None = None) -> dict[str, object]:
        start = time.perf_counter()
        desired_top_k = self.top_k if top_k is None else top_k
        if desired_top_k <= 0:
            raise ValueError("top_k must be > 0.")

        overridden = infer_intent_override(prompt)
        model_intent, confidence = self._router_predict(prompt=prompt)
        intent = overridden or model_intent

        answer = ""
        tool_name = ""
        retrieval_hits: list[dict[str, object]] = []

        if intent == "response_exact":
            literal = extract_exact_response(prompt)
            answer = literal or ""
        elif intent == "memory_store":
            note = extract_memory_note(prompt) or prompt.strip()
            self._remember(note=note)
            answer = f"Remembered: {note}"
        elif intent == "memory_recall":
            answer = self._recall(prompt=prompt)
        elif intent == "tool_calculator":
            tool_name = "calculator"
            expr = sanitize_expression(prompt)
            try:
                value = safe_eval_expression(expr)
                answer = normalize_number(value)
            except ValueError as exc:
                answer = f"calculator_error: {exc}"
        elif intent == "retrieval_lookup":
            retrieval_hits = self._retrieve(query=prompt, top_k=desired_top_k)
            if retrieval_hits:
                top = retrieval_hits[0]
                answer = f"{top['id']}: {top['snippet']}"
            else:
                answer = "No retrieval results."
        else:
            answer = "I can help with retrieval lookup, calculator, and memory notes."

        duration_ms = (time.perf_counter() - start) * 1000.0
        return {
            "schema_version": 1,
            "intent": intent,
            "intent_model_prediction": model_intent,
            "intent_confidence": confidence,
            "answer": answer,
            "tool_name": tool_name,
            "retrieval_hits": retrieval_hits,
            "memory_items": len(self.memory),
            "latency_ms": duration_ms,
        }


def run_dialogue(
    runtime: LocalGeneralistRuntime,
    prompts: list[str],
    top_k: int | None = None,
) -> dict[str, object]:
    turns: list[dict[str, object]] = []
    latencies: list[float] = []
    for prompt in prompts:
        result = runtime.respond(prompt=prompt, top_k=top_k)
        turns.append({"prompt": prompt, "response": result})
        latencies.append(float(result["latency_ms"]))
    return {
        "schema_version": 1,
        "turns": turns,
        "summary": {
            "turn_count": len(turns),
            "latency_mean_ms": statistics.mean(latencies) if latencies else 0.0,
            "latency_p95_ms": (
                sorted(latencies)[int(0.95 * (len(latencies) - 1))]
                if len(latencies) > 1
                else (latencies[0] if latencies else 0.0)
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt",
        default="",
        help="Single prompt to process.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top-k retrieval hits (default: 3).",
    )
    parser.add_argument(
        "--max-memory-turns",
        type=int,
        default=8,
        help="Max short-term memory items to keep (default: 8).",
    )
    parser.add_argument(
        "--corpus",
        default="data/retrieval_corpus.json",
        help="Corpus JSON used for retrieval lookup.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Seed for intent router training.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive prompt loop.",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional output path for JSON payload.",
    )
    args = parser.parse_args()

    runtime = LocalGeneralistRuntime(
        corpus_path=args.corpus,
        top_k=args.top_k,
        max_memory_turns=args.max_memory_turns,
        seed=args.seed,
    )

    payload: dict[str, object]
    if args.interactive:
        prompts: list[str] = []
        print("Local generalist runtime interactive mode. Type 'exit' to stop.")
        while True:
            try:
                prompt = input("> ").strip()
            except EOFError:
                break
            if not prompt:
                continue
            if prompt.lower() in {"exit", "quit"}:
                break
            prompts.append(prompt)
            result = runtime.respond(prompt=prompt, top_k=args.top_k)
            print(result["answer"])
        payload = run_dialogue(runtime=runtime, prompts=prompts, top_k=args.top_k)
    else:
        single_prompt = args.prompt.strip() if args.prompt else "What does the corpus say about tfidf?"
        result = runtime.respond(prompt=single_prompt, top_k=args.top_k)
        payload = {
            "schema_version": 1,
            "prompt": single_prompt,
            "response": result,
        }
        print(result["answer"])

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
