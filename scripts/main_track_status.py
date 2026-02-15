#!/usr/bin/env python3
"""
Compute machine-readable status for the current main-track goals and sub-goals.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CommandCheck:
    artifact: str
    command: list[str]


@dataclass(frozen=True)
class SmokeCheck:
    artifact: str


@dataclass
class SubGoalResult:
    id: str
    title: str
    required: bool
    status: str
    detail: str
    artifact: str


@dataclass
class GoalResult:
    id: str
    title: str
    description: str
    status: str
    sub_goals: list[SubGoalResult]


def run_cmd(root: Path, cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.returncode, proc.stdout.strip()


def short_detail(text: str, max_lines: int = 4) -> str:
    if not text:
        return ""
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) <= max_lines:
        return " | ".join(lines)
    return " | ".join(lines[:max_lines]) + " | ..."


def evaluate_sub_goal(
    *,
    root: Path,
    sub_goal_id: str,
    title: str,
    required: bool,
    check: CommandCheck | SmokeCheck,
    runner: Callable[[Path, list[str]], tuple[int, str]],
) -> SubGoalResult:
    artifact_path = root / check.artifact
    if not artifact_path.exists():
        return SubGoalResult(
            id=sub_goal_id,
            title=title,
            required=required,
            status="pending",
            detail=f"missing artifact: {check.artifact}",
            artifact=check.artifact,
        )

    if isinstance(check, SmokeCheck):
        with artifact_path.open("r", encoding="utf-8") as f:
            smoke = json.load(f)
        if smoke.get("ok") is True:
            total = smoke.get("total_duration_sec")
            return SubGoalResult(
                id=sub_goal_id,
                title=title,
                required=required,
                status="done",
                detail=f"smoke_summary ok=true total_duration_sec={total}",
                artifact=check.artifact,
            )
        return SubGoalResult(
            id=sub_goal_id,
            title=title,
            required=required,
            status="blocked",
            detail="smoke_summary ok is not true",
            artifact=check.artifact,
        )

    cmd = [sys.executable, *check.command, check.artifact]
    code, output = runner(root, cmd)
    if code == 0:
        return SubGoalResult(
            id=sub_goal_id,
            title=title,
            required=required,
            status="done",
            detail="validation passed",
            artifact=check.artifact,
        )
    return SubGoalResult(
        id=sub_goal_id,
        title=title,
        required=required,
        status="blocked",
        detail=short_detail(output),
        artifact=check.artifact,
    )


def compute_goal_status(sub_goals: list[SubGoalResult]) -> str:
    required_sub_goals = [item for item in sub_goals if item.required]
    if any(item.status == "blocked" for item in required_sub_goals):
        return "blocked"
    if any(item.status != "done" for item in required_sub_goals):
        return "pending"
    return "done"


def collect_status(
    *,
    root: Path,
    require_fairness: bool,
    require_smoke_summary: bool,
    runner: Callable[[Path, list[str]], tuple[int, str]] = run_cmd,
) -> dict[str, object]:
    goal_specs: list[dict[str, object]] = [
        {
            "id": "g1_cpu_federated_foundation",
            "title": "CPU-First Federated Foundation",
            "description": "Keep federated CPU baselines and adapter proxy quality stable with communication savings.",
            "sub_goals": [
                (
                    "sg1_baseline_gate",
                    "Baseline federated gate passes",
                    True,
                    CommandCheck(
                        artifact="baseline_metrics.json",
                        command=[
                            "scripts/check_baseline.py",
                            "--expected-schema-version",
                            "2",
                        ],
                    ),
                ),
                (
                    "sg2_classification_gate",
                    "Local classification gate passes",
                    True,
                    CommandCheck(
                        artifact="classification_metrics.json",
                        command=[
                            "scripts/check_classification.py",
                            "--expected-schema-version",
                            "1",
                        ],
                    ),
                ),
                (
                    "sg3_adapter_gate",
                    "Adapter intent proxy gate passes",
                    True,
                    CommandCheck(
                        artifact="adapter_intent_metrics.json",
                        command=[
                            "scripts/check_adapter_intent.py",
                            "--expected-schema-version",
                            "1",
                        ],
                    ),
                ),
            ],
        },
        {
            "id": "g2_generality_reproducibility",
            "title": "Generality and Reproducibility",
            "description": "Demonstrate repeatable local generalist capability with explicit metric thresholds.",
            "sub_goals": [
                (
                    "sg1_generality_gate",
                    "Generality gate passes",
                    True,
                    CommandCheck(
                        artifact="generality_metrics.json",
                        command=[
                            "scripts/check_generality.py",
                            "--expected-schema-version",
                            "1",
                        ],
                    ),
                ),
                (
                    "sg2_repro_gate",
                    "Reproducibility sweep gate passes",
                    True,
                    CommandCheck(
                        artifact="reproducibility_metrics.json",
                        command=[
                            "scripts/check_reproducibility.py",
                            "--expected-schema-version",
                            "1",
                        ],
                    ),
                ),
            ],
        },
        {
            "id": "g3_accessibility_ops",
            "title": "Accessibility and Operational Reliability",
            "description": "Keep the low-end smoke path and benchmark envelope healthy for commodity hardware contributors.",
            "sub_goals": [
                (
                    "sg1_benchmark_gate",
                    "Reduced benchmark gate passes",
                    True,
                    CommandCheck(
                        artifact="benchmark_metrics.json",
                        command=[
                            "scripts/check_benchmarks.py",
                            "--expected-schema-version",
                            "1",
                            "--expected-mode",
                            "reduced",
                        ],
                    ),
                ),
                (
                    "sg2_smoke_ok",
                    "Smoke summary reports overall success",
                    require_smoke_summary,
                    SmokeCheck(artifact="smoke_summary.json"),
                ),
            ],
        },
        {
            "id": "g4_decentralization_fairness",
            "title": "Decentralization Fairness Resilience",
            "description": "Track fairness behavior across heterogeneous contributor conditions.",
            "sub_goals": [
                (
                    "sg1_fairness_gate",
                    "Baseline fairness gate passes",
                    require_fairness,
                    CommandCheck(
                        artifact="fairness_metrics.json",
                        command=[
                            "scripts/check_fairness.py",
                            "--expected-schema-version",
                            "2",
                        ],
                    ),
                ),
                (
                    "sg2_utility_fairness_gate",
                    "Utility fairness gate passes",
                    require_fairness,
                    CommandCheck(
                        artifact="utility_fairness_metrics.json",
                        command=[
                            "scripts/check_utility_fairness.py",
                            "--expected-schema-version",
                            "1",
                        ],
                    ),
                ),
            ],
        },
    ]

    goals: list[GoalResult] = []
    for goal_spec in goal_specs:
        sub_results: list[SubGoalResult] = []
        for sub_goal_id, title, required, check in goal_spec["sub_goals"]:
            sub_results.append(
                evaluate_sub_goal(
                    root=root,
                    sub_goal_id=sub_goal_id,
                    title=title,
                    required=bool(required),
                    check=check,
                    runner=runner,
                )
            )
        goals.append(
            GoalResult(
                id=str(goal_spec["id"]),
                title=str(goal_spec["title"]),
                description=str(goal_spec["description"]),
                status=compute_goal_status(sub_results),
                sub_goals=sub_results,
            )
        )

    required_sub_goals = [
        sub
        for goal in goals
        for sub in goal.sub_goals
        if sub.required
    ]
    done_required_sub_goals = [sub for sub in required_sub_goals if sub.status == "done"]
    all_done = all(goal.status == "done" for goal in goals)

    payload: dict[str, object] = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "require_fairness": require_fairness,
        "require_smoke_summary": require_smoke_summary,
        "summary": {
            "total_goals": len(goals),
            "completed_goals": sum(1 for goal in goals if goal.status == "done"),
            "required_sub_goals": len(required_sub_goals),
            "completed_required_sub_goals": len(done_required_sub_goals),
            "all_done": all_done,
        },
        "goals": [
            {
                "id": goal.id,
                "title": goal.title,
                "description": goal.description,
                "status": goal.status,
                "sub_goals": [
                    {
                        "id": sub.id,
                        "title": sub.title,
                        "required": sub.required,
                        "status": sub.status,
                        "artifact": sub.artifact,
                        "detail": sub.detail,
                    }
                    for sub in goal.sub_goals
                ],
            }
            for goal in goals
        ],
    }
    return payload


def render_markdown(status: dict[str, object]) -> str:
    now = status.get("generated_utc", "")
    summary = status.get("summary", {})
    goals = status.get("goals", [])

    lines: list[str] = []
    lines.append("# Main Track Status")
    lines.append("")
    lines.append(f"- Generated: `{now}`")
    lines.append(f"- Require fairness: `{status.get('require_fairness')}`")
    lines.append(f"- Require smoke summary: `{status.get('require_smoke_summary')}`")
    lines.append(
        "- Completed goals: "
        f"`{summary.get('completed_goals')}/{summary.get('total_goals')}`"
    )
    lines.append(
        "- Completed required sub-goals: "
        f"`{summary.get('completed_required_sub_goals')}/{summary.get('required_sub_goals')}`"
    )
    lines.append(f"- All done: `{summary.get('all_done')}`")

    for goal in goals:
        lines.append("")
        lines.append(f"## {goal['title']} ({goal['status']})")
        lines.append("")
        lines.append(f"- {goal['description']}")
        for sub in goal["sub_goals"]:
            mark = "x" if sub["status"] == "done" else " "
            req = "required" if sub["required"] else "optional"
            lines.append(
                f"- [{mark}] {sub['title']} "
                f"(`{sub['status']}`, `{req}`, artifact=`{sub['artifact']}`)"
            )
            lines.append(f"- Detail: {sub['detail']}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        default=str(ROOT),
        help="Repository root path (default: script parent repo root).",
    )
    parser.add_argument(
        "--require-fairness",
        action="store_true",
        help="Treat fairness sub-goals as required.",
    )
    parser.add_argument(
        "--require-smoke-summary",
        action="store_true",
        help="Treat smoke summary sub-goal as required.",
    )
    parser.add_argument(
        "--json-out",
        default="main_track_status.json",
        help="Output JSON path relative to repo root (default: main_track_status.json).",
    )
    parser.add_argument(
        "--md-out",
        default="reports/main_track_status.md",
        help="Output markdown path relative to repo root (default: reports/main_track_status.md).",
    )
    parser.add_argument(
        "--fail-on-incomplete",
        action="store_true",
        help="Exit with non-zero status when required goals are not completed.",
    )
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    status = collect_status(
        root=root,
        require_fairness=args.require_fairness,
        require_smoke_summary=args.require_smoke_summary,
    )

    json_out = root / args.json_out
    json_out.parent.mkdir(parents=True, exist_ok=True)
    with json_out.open("w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, sort_keys=True)

    md_out = root / args.md_out
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(render_markdown(status), encoding="utf-8")

    summary = status["summary"]
    print(
        "Main track goals: "
        f"{summary['completed_goals']}/{summary['total_goals']} complete, "
        f"required sub-goals {summary['completed_required_sub_goals']}/{summary['required_sub_goals']}"
    )
    print(f"Status JSON: {json_out}")
    print(f"Status Markdown: {md_out}")

    if args.fail_on_incomplete and not summary["all_done"]:
        print("Main track incomplete.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
