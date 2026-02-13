#!/usr/bin/env python3
"""
Run repeatable pilot node cycles and publish pilot metrics.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def append_log(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line)


def emit(log_path: Path, message: str) -> None:
    stamped = f"[{utc_now_iso()}] {message}"
    print(stamped)
    append_log(log_path, stamped + "\n")


def run_command(command: list[str]) -> tuple[int, float, str]:
    start = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    duration = time.perf_counter() - start
    return proc.returncode, duration, proc.stdout


def load_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if raw.get("schema_version") != 1:
        raise ValueError("config schema_version must be 1")

    steps = raw.get("steps", [])
    if not steps:
        steps = [
            {
                "name": "smoke_check",
                "command": [
                    sys.executable,
                    "scripts/smoke_check.py",
                    "--include-fairness",
                    "--json-out",
                    "smoke_summary.json",
                ],
            }
        ]

    normalized_steps: list[dict[str, object]] = []
    for idx, step in enumerate(steps):
        name = step.get("name", f"step_{idx}")
        command = step.get("command", [])
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"invalid step name at index {idx}")
        if not isinstance(command, list) or not command:
            raise ValueError(f"invalid step command at index {idx}")
        if not all(isinstance(item, str) and item for item in command):
            raise ValueError(f"command values must be non-empty strings at step {name}")
        normalized_steps.append({"name": name, "command": command})

    return {
        "node_id": raw.get("node_id", "volunteer-node-local"),
        "mode": raw.get("mode", "reduced"),
        "cycle_interval_sec": int(raw.get("cycle_interval_sec", 1800)),
        "max_cycles": int(raw.get("max_cycles", 0)),
        "repo": raw.get("repo", ""),
        "token_env_var": raw.get("token_env_var", "github_token"),
        "metrics_out": raw.get("metrics_out", "pilot/pilot_metrics.json"),
        "state_out": raw.get("state_out", "pilot/node_state.json"),
        "log_out": raw.get("log_out", "pilot/node_runner.log"),
        "steps": normalized_steps,
    }


def load_state(path: Path, node_id: str) -> dict[str, object]:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            state = json.load(f)
        state.setdefault("schema_version", 1)
        state.setdefault("node_id", node_id)
        state.setdefault("total_cycles", 0)
        state.setdefault("successful_cycles", 0)
        state.setdefault("uptime_ratio_24h", 0.0)
        state.setdefault("last_cycle_ok", False)
        state.setdefault("last_cycle_utc", "")
        state.setdefault("last_cycle_duration_sec", 0.0)
        state.setdefault("last_error", "")
        return state

    return {
        "schema_version": 1,
        "node_id": node_id,
        "total_cycles": 0,
        "successful_cycles": 0,
        "uptime_ratio_24h": 0.0,
        "last_cycle_ok": False,
        "last_cycle_utc": "",
        "last_cycle_duration_sec": 0.0,
        "last_error": "",
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def run_cycle(
    config: dict[str, object],
    state: dict[str, object],
    log_path: Path,
) -> bool:
    failures: list[str] = []
    started = time.perf_counter()
    steps = config["steps"]

    for step in steps:
        step_name = step["name"]
        command = step["command"]
        emit(log_path, f"Running step: {step_name} -> {' '.join(command)}")
        code, duration, output = run_command(command=command)
        emit(log_path, f"Step result: {step_name} -> {'ok' if code == 0 else 'failed'} ({duration:.2f}s)")
        if output:
            append_log(log_path, output.rstrip() + "\n")
        if code != 0:
            failures.append(f"{step_name} rc={code}")
            break

    cycle_duration = time.perf_counter() - started
    cycle_ok = not failures

    state["total_cycles"] = int(state["total_cycles"]) + 1
    if cycle_ok:
        state["successful_cycles"] = int(state["successful_cycles"]) + 1

    total_cycles = int(state["total_cycles"])
    successful_cycles = int(state["successful_cycles"])
    uptime_ratio = 0.0 if total_cycles == 0 else successful_cycles / total_cycles

    metrics_command = [
        sys.executable,
        "scripts/build_pilot_metrics.py",
        "--node-id",
        str(config["node_id"]),
        "--mode",
        str(config["mode"]),
        "--repo",
        str(config["repo"]),
        "--token-env-var",
        str(config["token_env_var"]),
        "--cycle-duration-sec",
        f"{cycle_duration:.3f}",
        "--step-count",
        str(len(steps)),
        "--uptime-ratio-24h",
        f"{uptime_ratio:.4f}",
        "--json-out",
        str(config["metrics_out"]),
    ]
    if cycle_ok:
        metrics_command.append("--last-cycle-ok")

    emit(log_path, "Refreshing pilot metrics artifact")
    code, duration, output = run_command(command=metrics_command)
    emit(
        log_path,
        f"Pilot metrics refresh -> {'ok' if code == 0 else 'failed'} ({duration:.2f}s)",
    )
    if output:
        append_log(log_path, output.rstrip() + "\n")
    if code != 0:
        failures.append(f"build_pilot_metrics rc={code}")

    cycle_ok = not failures
    state["uptime_ratio_24h"] = uptime_ratio
    state["last_cycle_ok"] = cycle_ok
    state["last_cycle_utc"] = utc_now_iso()
    state["last_cycle_duration_sec"] = cycle_duration
    state["last_error"] = "; ".join(failures)

    write_json(resolve(str(config["state_out"])), state)
    emit(
        log_path,
        (
            "Cycle complete: "
            f"ok={cycle_ok} total_cycles={total_cycles} "
            f"successful_cycles={successful_cycles} uptime_ratio_24h={uptime_ratio:.4f}"
        ),
    )
    return cycle_ok


def health_check(state_path: Path, min_uptime_ratio: float) -> int:
    if not state_path.exists():
        print(f"Node state file not found: {state_path}")
        return 1

    with state_path.open("r", encoding="utf-8") as f:
        state = json.load(f)

    uptime_ratio = state.get("uptime_ratio_24h", 0.0)
    last_cycle_ok = state.get("last_cycle_ok", False)

    print("Pilot node health summary")
    print(f"- node_id: {state.get('node_id')}")
    print(f"- total_cycles: {state.get('total_cycles')}")
    print(f"- successful_cycles: {state.get('successful_cycles')}")
    print(f"- uptime_ratio_24h: {uptime_ratio}")
    print(f"- last_cycle_ok: {last_cycle_ok}")
    print(f"- last_cycle_utc: {state.get('last_cycle_utc')}")
    print(f"- last_error: {state.get('last_error')}")

    if not last_cycle_ok:
        print("\nHealth check failed: last cycle was not successful.")
        return 1
    if uptime_ratio < min_uptime_ratio:
        print(
            "\nHealth check failed: "
            f"uptime_ratio_24h={uptime_ratio} < min_uptime_ratio={min_uptime_ratio}"
        )
        return 1

    print("\nHealth check passed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="pilot/node_config.json",
        help="Path to node config JSON.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run exactly one cycle and exit.",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Override max cycles from config (0 means run continuously).",
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Read node state file and return health status.",
    )
    parser.add_argument(
        "--min-uptime-ratio",
        type=float,
        default=0.90,
        help="Minimum required uptime ratio for --health (default: 0.90).",
    )
    args = parser.parse_args()

    config_path = resolve(args.config)
    if not config_path.exists():
        print(
            f"Config not found: {config_path}. "
            "Create it from pilot/node_config.example.json first."
        )
        return 1

    config = load_config(path=config_path)
    state_path = resolve(str(config["state_out"]))

    if args.health:
        return health_check(state_path=state_path, min_uptime_ratio=args.min_uptime_ratio)

    max_cycles = config["max_cycles"] if args.max_cycles is None else args.max_cycles
    if args.once:
        max_cycles = 1

    log_path = resolve(str(config["log_out"]))
    state = load_state(path=state_path, node_id=str(config["node_id"]))

    cycle_idx = 0
    had_failure = False
    emit(log_path, f"Node runner started with config={config_path}")
    try:
        while True:
            cycle_idx += 1
            emit(log_path, f"Starting cycle #{cycle_idx}")
            ok = run_cycle(config=config, state=state, log_path=log_path)
            if not ok:
                had_failure = True

            if max_cycles and cycle_idx >= max_cycles:
                break

            interval = int(config["cycle_interval_sec"])
            emit(log_path, f"Sleeping for {interval}s before next cycle")
            time.sleep(interval)
    except KeyboardInterrupt:
        emit(log_path, "Received interrupt signal; shutting down node runner")
        return 130

    return 1 if had_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
