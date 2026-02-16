#!/usr/bin/env python3
"""
Local generalist runtime MVP:
- intent routing via a tiny CPU-only softmax model
- local retrieval from repository corpus
- calculator tool execution
- short-term memory across turns
- memory management (list/forget)
- retrieval citations and follow-up controls
- chained calculator workflows with last-result follow-ups
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
TOP_K_RE = re.compile(r"\btop\s+([a-z0-9-]+)\b|\b([a-z0-9-]+)\s+results?\b", re.IGNORECASE)
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
FOLLOWUP_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"\b(?:x|times|multiply(?:\s+that)?(?:\s+by)?|multipl(?:y|ied)(?:\s+that)?(?:\s+by)?)\b(?P<tail>.*)",
            re.IGNORECASE,
        ),
        "mul",
    ),
    (
        re.compile(
            r"\b(?:divide(?:\s+that)?(?:\s+by)?|divided by|over)\b(?P<tail>.*)",
            re.IGNORECASE,
        ),
        "div",
    ),
    (re.compile(r"\b(?:add|plus)\b(?P<tail>.*)", re.IGNORECASE), "add"),
    (re.compile(r"\b(?:subtract|minus)\b(?P<tail>.*)", re.IGNORECASE), "sub"),
]

NUMBER_WORD_UNITS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}
NUMBER_WORD_TEENS = {
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
NUMBER_WORD_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
NUMBER_WORD_SCALES = {
    "hundred": 100,
    "thousand": 1000,
    "million": 1000000,
}
NUMBER_WORD_TOKENS = (
    set(NUMBER_WORD_UNITS)
    | set(NUMBER_WORD_TEENS)
    | set(NUMBER_WORD_TENS)
    | set(NUMBER_WORD_SCALES)
    | {"and", "point", "minus", "negative", "a", "an"}
)
CALC_PREFIXES = [
    "what is",
    "what's",
    "whats",
    "calculate",
    "compute",
    "evaluate",
    "please calculate",
    "please compute",
    "please evaluate",
    "can you calculate",
    "can you compute",
    "could you calculate",
    "could you compute",
]


BASE_INTENT_TEMPLATES: dict[str, list[str]] = {
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

LOCAL_RUNTIME_INTENT_TEMPLATES: dict[str, list[str]] = {
    **BASE_INTENT_TEMPLATES,
    "memory_list": [
        "list my notes",
        "show my stored notes",
        "what notes do you have",
        "show memory list",
    ],
    "memory_forget": [
        "forget the latest note",
        "remove my last note",
        "forget note about rollout",
        "delete the previous memory note",
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


def generate_intent_dataset(
    seed: int,
    samples_per_intent: int,
    intent_templates: dict[str, list[str]] | None = None,
) -> list[tuple[str, str]]:
    if samples_per_intent < 4:
        raise ValueError("samples_per_intent must be >= 4.")
    rng = random.Random(seed)
    rows: list[tuple[str, str]] = []
    templates_map = intent_templates or BASE_INTENT_TEMPLATES
    for intent, templates in templates_map.items():
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


def clean_fragment(text: str) -> str:
    return text.strip().strip(" ,.;:!?").strip('"').strip("'")


def strip_calc_prefix(prompt: str) -> str:
    stripped = prompt.strip()
    lowered = stripped.lower()
    for prefix in sorted(CALC_PREFIXES, key=len, reverse=True):
        marker = f"{prefix} "
        if lowered.startswith(marker):
            return stripped[len(marker) :].strip()
    return stripped


def parse_number_phrase(text: str) -> float | None:
    cleaned = clean_fragment(text).lower().replace("-", " ")
    if not cleaned:
        return None
    numeric_candidate = cleaned.replace(",", "")
    try:
        return float(numeric_candidate)
    except ValueError:
        pass

    tokens = [tok for tok in re.findall(r"[a-z0-9]+", cleaned) if tok]
    if not tokens:
        return None

    negative = False
    if tokens[0] in {"minus", "negative"}:
        negative = True
        tokens = tokens[1:]
    if not tokens:
        return None

    total = 0
    current = 0
    decimal_digits: list[str] = []
    in_decimal = False

    for token in tokens:
        if token == "and" and not in_decimal:
            continue
        if token == "point":
            if in_decimal:
                return None
            in_decimal = True
            continue

        if in_decimal:
            if token in NUMBER_WORD_UNITS:
                decimal_digits.append(str(NUMBER_WORD_UNITS[token]))
                continue
            if token.isdigit() and len(token) == 1:
                decimal_digits.append(token)
                continue
            return None

        if token in {"a", "an"}:
            current += 1
            continue
        if token in NUMBER_WORD_UNITS:
            current += NUMBER_WORD_UNITS[token]
            continue
        if token in NUMBER_WORD_TEENS:
            current += NUMBER_WORD_TEENS[token]
            continue
        if token in NUMBER_WORD_TENS:
            current += NUMBER_WORD_TENS[token]
            continue
        if token == "hundred":
            current = max(1, current) * 100
            continue
        if token in {"thousand", "million"}:
            scale = NUMBER_WORD_SCALES[token]
            total += max(1, current) * scale
            current = 0
            continue
        if token.isdigit():
            current += int(token)
            continue
        return None

    value = float(total + current)
    if decimal_digits:
        value += float(f"0.{''.join(decimal_digits)}")
    if negative:
        value = -value
    return value


def extract_first_number_value(text: str) -> float | None:
    numeric_match = NUMBER_RE.search(text.replace(",", ""))
    if numeric_match:
        try:
            return float(numeric_match.group(0))
        except ValueError:
            return None

    tokens = re.findall(r"[a-z0-9]+", text.lower().replace("-", " "))
    if not tokens:
        return None

    max_span = min(10, len(tokens))
    for start in range(len(tokens)):
        if tokens[start] not in NUMBER_WORD_TOKENS:
            continue
        for end in range(min(len(tokens), start + max_span), start, -1):
            value = parse_number_phrase(" ".join(tokens[start:end]))
            if value is not None:
                return value
    return None


def parse_operand_value(text: str) -> float | None:
    direct = parse_number_phrase(text)
    if direct is not None:
        return direct
    return extract_first_number_value(text)


def parse_binary_calculation(prompt: str) -> tuple[str, float, float] | None:
    lowered = strip_calc_prefix(prompt).lower()
    lowered = re.sub(r"\s+", " ", lowered).strip()
    if not lowered:
        return None

    imperative_patterns = [
        (r"\badd\s+(.+?)\s+to\s+(.+)", "add", True),
        (r"\bsubtract\s+(.+?)\s+from\s+(.+)", "sub", True),
        (r"\bmultiply\s+(.+?)\s+by\s+(.+)", "mul", False),
        (r"\bdivide\s+(.+?)\s+by\s+(.+)", "div", False),
    ]
    for pattern, op_name, reversed_args in imperative_patterns:
        match = re.search(pattern, lowered, flags=re.IGNORECASE)
        if not match:
            continue
        first = parse_operand_value(match.group(1))
        second = parse_operand_value(match.group(2))
        if first is None or second is None:
            continue
        left = second if reversed_args else first
        right = first if reversed_args else second
        return op_name, left, right

    infix_patterns = [
        ("raised to the power of", "pow"),
        ("to the power of", "pow"),
        ("raised to", "pow"),
        ("multiplied by", "mul"),
        ("divided by", "div"),
        ("modulo", "mod"),
        ("times", "mul"),
        ("plus", "add"),
        ("minus", "sub"),
        ("over", "div"),
        ("mod", "mod"),
    ]
    for phrase, op_name in infix_patterns:
        match = re.search(
            rf"(.+?)\b{re.escape(phrase)}\b(.+)",
            lowered,
            flags=re.IGNORECASE,
        )
        if not match:
            continue
        left = parse_operand_value(match.group(1))
        right = parse_operand_value(match.group(2))
        if left is None or right is None:
            continue
        return op_name, left, right

    x_match = re.search(r"(.+?)\sx\s(.+)", lowered, flags=re.IGNORECASE)
    if x_match:
        left = parse_operand_value(x_match.group(1))
        right = parse_operand_value(x_match.group(2))
        if left is not None and right is not None:
            return "mul", left, right
    return None


def evaluate_calculation_prompt(prompt: str) -> float:
    expr = sanitize_expression(prompt)
    if expr:
        return safe_eval_expression(expr)
    parsed = parse_binary_calculation(prompt)
    if parsed is None:
        raise ValueError("No expression found.")
    op_name, left, right = parsed
    return apply_operation(base=left, op_name=op_name, operand=right)


def is_memory_list_request(prompt: str) -> bool:
    lowered = prompt.lower()
    phrases = [
        "list my notes",
        "show my notes",
        "show memory",
        "show stored notes",
        "what notes do you have",
        "what are my notes",
        "show everything you remember",
        "list everything you remember",
        "what do you remember",
    ]
    return any(phrase in lowered for phrase in phrases)


def is_memory_recall_request(prompt: str) -> bool:
    lowered = prompt.lower()
    phrases = [
        "what did i ask you to remember",
        "what did i tell you to remember",
        "what note did i save",
        "what was my reminder",
        "recall",
        "remind me what",
        "do you remember what",
        "what did i ask you to store",
    ]
    return any(phrase in lowered for phrase in phrases)


def is_memory_forget_request(prompt: str) -> bool:
    lowered = prompt.lower()
    verbs = ["forget", "remove", "delete", "erase"]
    nouns = ["note", "memory", "reminder", "latest", "last", "previous"]
    return any(v in lowered for v in verbs) and any(n in lowered for n in nouns)


def is_retrieval_request(prompt: str) -> bool:
    lowered = prompt.lower()
    direct = ["lookup", "search", "find", "retrieve", "corpus", "show sources", "with citations"]
    if any(token in lowered for token in direct):
        return True
    doc_terms = ["docs", "document", "documents", "readme", "repository", "repo", "source", "sources"]
    query_terms = ["what", "which", "where", "show", "tell", "explain", "summarize", "mentions"]
    return any(term in lowered for term in doc_terms) and any(term in lowered for term in query_terms)


def is_calculator_request(prompt: str) -> bool:
    lowered = prompt.lower()
    if parse_binary_calculation(prompt):
        return True
    if "then" in lowered:
        segments = parse_then_segments(prompt)
        if segments and (sanitize_expression(segments[0]) or parse_binary_calculation(segments[0])):
            return True
    if parse_followup_operation(prompt):
        if any(token in lowered for token in ["that", "previous result", "last result"]):
            return True
        if any(token in lowered for token in ["add ", "subtract", "multiply", "divide", "times", "plus", "minus"]):
            return True
    if any(keyword in lowered for keyword in ["calculate", "compute", "evaluate", "what is ", "what's ", "whats "]):
        if sanitize_expression(prompt):
            return True
    return False


def extract_exact_response(prompt: str) -> str | None:
    lowered = prompt.lower()
    patterns = [
        "respond exactly with:",
        "reply exactly with:",
        "answer exactly with:",
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
    if is_memory_recall_request(prompt) or is_memory_list_request(prompt):
        return None

    markers = [
        "remember that",
        "remember this note:",
        "remember:",
        "store this for later:",
        "note this down:",
        "save this for later:",
        "keep in mind that",
        "can you remember that",
        "could you remember that",
    ]
    for marker in markers:
        idx = lowered.find(marker)
        if idx == -1:
            continue
        value = clean_fragment(prompt[idx + len(marker) :])
        if value:
            return value
    if lowered.startswith("remember "):
        value = clean_fragment(prompt[len("remember ") :])
        if value:
            return value
    return None


def extract_forget_keyword(prompt: str) -> str | None:
    lowered = prompt.lower()
    markers = [
        "forget note about",
        "remove note about",
        "delete note about",
        "forget memory about",
        "remove memory about",
        "delete memory about",
    ]
    for marker in markers:
        idx = lowered.find(marker)
        if idx == -1:
            continue
        keyword = prompt[idx + len(marker) :].strip()
        if keyword:
            return keyword.lower()
    return None


def parse_top_k_from_prompt(prompt: str) -> int | None:
    match = TOP_K_RE.search(prompt)
    if not match:
        return None
    raw = match.group(1) or match.group(2)
    if not raw:
        return None
    parsed = parse_number_phrase(raw)
    if parsed is None:
        return None
    value = int(round(parsed))
    if abs(parsed - value) > 1e-9:
        return None
    if value <= 0:
        return None
    return value


def parse_then_segments(prompt: str) -> list[str]:
    lowered = prompt.lower()
    if " then " not in lowered:
        return []
    parts = re.split(r"\bthen\b", prompt, flags=re.IGNORECASE)
    segments = [part.strip(" ,.;") for part in parts if part.strip(" ,.;")]
    if len(segments) < 2:
        return []
    return segments


def parse_followup_operation(prompt: str) -> tuple[str, float] | None:
    for regex, op_name in FOLLOWUP_PATTERNS:
        match = regex.search(prompt)
        if not match:
            continue
        tail = match.group("tail") if "tail" in match.groupdict() else ""
        value = extract_first_number_value(tail)
        if value is None:
            continue
        return op_name, value
    return None


def apply_operation(base: float, op_name: str, operand: float) -> float:
    if op_name == "mul":
        return base * operand
    if op_name == "div":
        if abs(operand) < 1e-12:
            raise ValueError("division by zero")
        return base / operand
    if op_name == "mod":
        if abs(operand) < 1e-12:
            raise ValueError("division by zero")
        return base % operand
    if op_name == "pow":
        return base**operand
    if op_name == "add":
        return base + operand
    if op_name == "sub":
        return base - operand
    raise ValueError(f"unsupported operation: {op_name}")


def infer_intent_override(prompt: str) -> str | None:
    if extract_exact_response(prompt):
        return "response_exact"
    if is_memory_list_request(prompt):
        return "memory_list"
    if is_memory_forget_request(prompt):
        return "memory_forget"
    if is_memory_recall_request(prompt):
        return "memory_recall"
    if extract_memory_note(prompt):
        return "memory_store"
    if is_calculator_request(prompt):
        return "tool_calculator"
    if is_retrieval_request(prompt):
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
        self.last_tool_value: float | None = None
        self.last_retrieval_hits: list[dict[str, object]] = []
        self.last_intent: str = ""
        self.intent = self._train_intent_router(seed=seed)
        self.corpus = self._load_corpus(corpus_path=corpus_path)

    def _train_intent_router(self, seed: int) -> dict[str, object]:
        rows = generate_intent_dataset(
            seed=seed,
            samples_per_intent=24,
            intent_templates=LOCAL_RUNTIME_INTENT_TEMPLATES,
        )
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

    def _list_notes(self) -> str:
        if not self.memory:
            return "No notes are stored yet."
        rows = []
        for idx, item in enumerate(self.memory, start=1):
            rows.append(f"{idx}. {item.note}")
        return "Stored notes:\n" + "\n".join(rows)

    def _forget_note(self, prompt: str) -> str:
        if not self.memory:
            return "No notes are stored yet."
        lowered = prompt.lower()
        if "latest" in lowered or "last" in lowered or "previous" in lowered:
            removed = self.memory.pop()
            return f"Forgot note: {removed.note}"
        keyword = extract_forget_keyword(prompt)
        if keyword:
            for idx in range(len(self.memory) - 1, -1, -1):
                note = self.memory[idx].note
                if keyword in note.lower():
                    removed = self.memory.pop(idx)
                    return f"Forgot note: {removed.note}"
            return f"No stored note matched keyword: {keyword}"
        removed = self.memory.pop()
        return f"Forgot note: {removed.note}"

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

    def _render_retrieval_answer(self, hits: list[dict[str, object]], query: str) -> str:
        if not hits:
            return "No retrieval results."
        lowered = query.lower()
        if any(
            token in lowered
            for token in [
                "all results",
                "show results",
                "show sources",
                "top",
                "with citation",
                "with citations",
                "cite",
            ]
        ):
            lines = []
            for idx, hit in enumerate(hits, start=1):
                lines.append(f"{idx}. {hit['id']} ({hit['score']:.3f}) - {hit['snippet']}")
            return "\n".join(lines)
        top = hits[0]
        return f"{top['id']}: {top['snippet']}"

    def _resolve_followup_calculation(self, prompt: str) -> str | None:
        if self.last_tool_value is None:
            return None
        lowered = prompt.lower()
        if "that" not in lowered and "previous result" not in lowered and "last result" not in lowered:
            return None
        op = parse_followup_operation(prompt)
        if not op:
            return None
        op_name, operand = op
        value = apply_operation(self.last_tool_value, op_name, operand)
        return normalize_number(value)

    def _evaluate_calculation_chain(self, prompt: str) -> str:
        segments = parse_then_segments(prompt)
        if segments:
            value = evaluate_calculation_prompt(segments[0])
            for segment in segments[1:]:
                op = parse_followup_operation(segment)
                if not op:
                    raise ValueError(f"Unsupported chain step: {segment}")
                op_name, operand = op
                value = apply_operation(value, op_name, operand)
            self.last_tool_value = value
            return normalize_number(value)

        followup = self._resolve_followup_calculation(prompt)
        if followup is not None:
            try:
                self.last_tool_value = float(followup)
            except ValueError:
                self.last_tool_value = None
            return followup

        value = evaluate_calculation_prompt(prompt)
        self.last_tool_value = value
        return normalize_number(value)

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
        elif intent == "memory_list":
            answer = self._list_notes()
        elif intent == "memory_forget":
            answer = self._forget_note(prompt=prompt)
        elif intent == "tool_calculator":
            tool_name = "calculator"
            try:
                answer = self._evaluate_calculation_chain(prompt)
            except ValueError as exc:
                answer = f"calculator_error: {exc}"
                self.last_tool_value = None
        elif intent == "retrieval_lookup":
            prompt_top_k = parse_top_k_from_prompt(prompt)
            effective_top_k = desired_top_k if prompt_top_k is None else prompt_top_k
            retrieval_hits = self._retrieve(query=prompt, top_k=effective_top_k)
            self.last_retrieval_hits = retrieval_hits
            answer = self._render_retrieval_answer(retrieval_hits, query=prompt)
        else:
            answer = (
                "I can help with retrieval lookup, calculator workflows, and memory notes "
                "(store, recall, list, forget)."
            )

        duration_ms = (time.perf_counter() - start) * 1000.0
        self.last_intent = intent
        return {
            "schema_version": 1,
            "intent": intent,
            "intent_model_prediction": model_intent,
            "intent_confidence": confidence,
            "answer": answer,
            "tool_name": tool_name,
            "retrieval_hits": retrieval_hits,
            "memory_items": len(self.memory),
            "last_tool_value": (
                normalize_number(self.last_tool_value)
                if self.last_tool_value is not None
                else ""
            ),
            "last_intent": self.last_intent,
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
