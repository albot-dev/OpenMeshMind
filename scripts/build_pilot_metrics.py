#!/usr/bin/env python3
"""
Build pilot metrics payload from experiment artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_cmd(args: list[str], env: dict[str, str] | None = None) -> tuple[int, str]:
    proc = subprocess.run(
        args,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    return proc.returncode, proc.stdout.strip()


def load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"missing artifact: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_env_value(key: str) -> str:
    env_value = os.environ.get(key, "")
    if env_value:
        return env_value

    dotenv_path = ROOT / ".env"
    if not dotenv_path.exists():
        return ""

    with dotenv_path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            env_key, raw_value = stripped.split("=", 1)
            if env_key.strip() != key:
                continue
            value = raw_value.strip().strip('"').strip("'")
            return value
    return ""


def default_repo() -> str:
    code, out = run_cmd(["git", "config", "--get", "remote.origin.url"])
    if code != 0 or not out:
        return ""
    cleaned = out[:-4] if out.endswith(".git") else out
    if cleaned.startswith("https://github.com/"):
        return cleaned.split("https://github.com/", 1)[1]
    if cleaned.startswith("git@github.com:"):
        return cleaned.split("git@github.com:", 1)[1]
    return ""


def github_status(repo: str, token_env_var: str) -> tuple[int, int, bool]:
    if not repo:
        return 0, 0, False
    env = os.environ.copy()
    token = load_env_value(token_env_var)
    if token:
        env[token_env_var] = token
    if token:
        env["GH_TOKEN"] = token

    code_m, out_m = run_cmd(
        ["gh", "api", f"repos/{repo}/milestones?state=open"],
        env=env,
    )
    code_i, out_i = run_cmd(
        ["gh", "api", f"repos/{repo}/issues?state=open&per_page=100"],
        env=env,
    )
    if code_m != 0 or code_i != 0:
        return 0, 0, False
    milestones = json.loads(out_m)
    issues = [item for item in json.loads(out_i) if "pull_request" not in item]
    return len(milestones), len(issues), True


def utility_jain_gain(utility_fairness: dict[str, object] | None) -> float:
    if not utility_fairness:
        return 0.0
    scenarios = utility_fairness.get("scenarios")
    if scenarios:
        gains: list[float] = []
        for scenario in scenarios:
            fp = scenario["methods"]["fedavg_fp32"]["fairness"]
            i8 = scenario["methods"]["fedavg_int8"]["fairness"]
            gains.append(i8["contribution_jain_index_mean"] - fp["contribution_jain_index_mean"])
        return sum(gains) / len(gains) if gains else 0.0
    fp = utility_fairness["methods"]["fedavg_fp32"]["fairness"]
    i8 = utility_fairness["methods"]["fedavg_int8"]["fairness"]
    return i8["contribution_jain_index_mean"] - fp["contribution_jain_index_mean"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-id", default="node-local-1", help="Node identifier.")
    parser.add_argument("--mode", default="reduced", help="Pilot mode label.")
    parser.add_argument("--repo", default="", help="Repository owner/name.")
    parser.add_argument(
        "--token-env-var",
        default="github_token",
        help="Env var for GitHub token (default: github_token).",
    )
    parser.add_argument("--baseline", default="baseline_metrics.json")
    parser.add_argument("--benchmark", default="benchmark_metrics.json")
    parser.add_argument("--classification", default="classification_metrics.json")
    parser.add_argument("--fairness", default="fairness_metrics.json")
    parser.add_argument("--utility", default="utility_fedavg_metrics.json")
    parser.add_argument("--utility-fairness", default="utility_fairness_metrics.json")
    parser.add_argument("--cycle-duration-sec", type=float, default=0.0)
    parser.add_argument("--step-count", type=int, default=0)
    parser.add_argument("--uptime-ratio-24h", type=float, default=0.0)
    parser.add_argument("--last-cycle-ok", action="store_true")
    parser.add_argument("--json-out", default="pilot/pilot_metrics.json")
    args = parser.parse_args()

    try:
        baseline = load_json(ROOT / args.baseline)
        benchmark = load_json(ROOT / args.benchmark)
        classification = load_json(ROOT / args.classification)
        fairness = load_json(ROOT / args.fairness)
        utility = load_json(ROOT / args.utility)
        utility_fairness = load_json(ROOT / args.utility_fairness)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"Failed to build pilot metrics: {exc}")
        return 1

    repo = args.repo or default_repo()
    open_milestones, open_issues, status_collected = github_status(
        repo=repo,
        token_env_var=args.token_env_var,
    )
    _, commit = run_cmd(["git", "rev-parse", "HEAD"])

    baseline_jain = (
        fairness.get("methods", {})
        .get("fedavg_int8", {})
        .get("fairness", {})
        .get("contribution_jain_index_mean", 0.0)
    )
    utility_savings = utility.get("communication_savings_percent", {}).get(
        "int8_vs_fp32_percent", 0.0
    )
    utility_quality = utility.get("methods", {}).get("fedavg_int8", {})

    payload: dict[str, object] = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "node": {
            "node_id": args.node_id,
            "mode": args.mode,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "health": {
            "last_cycle_ok": args.last_cycle_ok,
            "cycle_duration_sec": args.cycle_duration_sec,
            "step_count": args.step_count,
            "uptime_ratio_24h": args.uptime_ratio_24h,
        },
        "quality": {
            "classification_accuracy": classification["metrics"]["accuracy"],
            "classification_macro_f1": classification["metrics"]["macro_f1"],
            "utility_fedavg_int8_accuracy": utility_quality.get("accuracy_mean", 0.0),
            "utility_fedavg_int8_macro_f1": utility_quality.get("macro_f1_mean", 0.0),
        },
        "accessibility": {
            "benchmark_total_runtime_sec": benchmark["summary"]["total_runtime_sec"],
            "max_peak_rss_bytes": benchmark["summary"]["max_peak_rss_bytes"],
            "max_peak_heap_bytes": benchmark["summary"]["max_peak_heap_bytes"],
        },
        "decentralization": {
            "baseline_int8_jain_index": baseline_jain,
            "utility_int8_jain_gain": utility_jain_gain(utility_fairness=utility_fairness),
        },
        "communication": {
            "baseline_int8_reduction_percent": baseline["communication_reduction_percent"],
            "utility_int8_savings_percent": utility_savings,
        },
        "status": {
            "collected": status_collected,
            "open_milestones": open_milestones,
            "open_issues": open_issues,
        },
        "provenance": {
            "repo": repo,
            "commit": commit,
            "decision_log": "DECISION_LOG.md",
            "provenance_template": "PROVENANCE_TEMPLATE.md",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print(f"Pilot metrics written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
