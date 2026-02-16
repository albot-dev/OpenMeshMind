#!/usr/bin/env python3
"""
Validate machine-verifiable MVP definition-of-done readiness artifacts.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_REQUIRED_METRIC_ARTIFACTS = [
    "baseline_metrics.json",
    "classification_metrics.json",
    "adapter_intent_metrics.json",
    "benchmark_metrics.json",
    "generality_metrics.json",
    "reproducibility_metrics.json",
]
FAIRNESS_METRIC_ARTIFACTS = [
    "fairness_metrics.json",
    "utility_fairness_metrics.json",
]
DEFAULT_REQUIRED_DOCS = [
    "docs/MVP_CRITERIA.md",
    "docs/COMING_GOALS.md",
    "docs/MVP_USER_VALUE_PLAN.md",
    "docs/MVP_DEFINITION_OF_DONE.md",
]
DEFAULT_REQUIRED_GOAL_IDS = [
    "g1_cpu_federated_foundation",
    "g2_generality_reproducibility",
    "g3_accessibility_ops",
]
FAIRNESS_GOAL_ID = "g4_decentralization_fairness"


def _resolve(root: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return root / path


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item in seen:
            continue
        deduped.append(item)
        seen.add(item)
    return deduped


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("expected top-level JSON object")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        default=str(ROOT),
        help="Repository root path (default: script parent repo root).",
    )
    parser.add_argument(
        "--main-track-status",
        default="main_track_status.json",
        help="Path to main track status JSON (default: main_track_status.json).",
    )
    parser.add_argument(
        "--required-metric-artifact",
        action="append",
        default=[],
        help=(
            "Metric artifact required for readiness. Repeat for multiple values. "
            "Defaults to standard MVP artifacts."
        ),
    )
    parser.add_argument(
        "--required-doc",
        action="append",
        default=[],
        help="Documentation artifact required for readiness. Repeat for multiple values.",
    )
    parser.add_argument(
        "--required-goal-id",
        action="append",
        default=[],
        help=(
            "Goal ID that must be marked done in main_track_status. "
            "Defaults to core MVP goals."
        ),
    )
    parser.add_argument(
        "--require-fairness",
        action="store_true",
        help="Require fairness artifacts and the fairness goal.",
    )
    parser.add_argument(
        "--require-all-goals-done",
        action="store_true",
        help="Fail if main_track_status summary.all_done is not true.",
    )
    parser.add_argument(
        "--min-completed-required-sub-goals",
        type=int,
        default=None,
        help=(
            "Minimum completed required sub-goals in main_track_status summary. "
            "Default: require all required sub-goals to be completed."
        ),
    )
    parser.add_argument(
        "--min-completed-goals",
        type=int,
        default=None,
        help="Minimum completed goals in main_track_status summary.",
    )
    parser.add_argument(
        "--max-missing-metrics",
        type=int,
        default=0,
        help="Maximum missing required metric artifacts allowed (default: 0).",
    )
    parser.add_argument(
        "--max-missing-docs",
        type=int,
        default=0,
        help="Maximum missing required docs allowed (default: 0).",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional path to write readiness summary JSON.",
    )
    args = parser.parse_args()

    if args.max_missing_metrics < 0:
        parser.error("--max-missing-metrics must be >= 0")
    if args.max_missing_docs < 0:
        parser.error("--max-missing-docs must be >= 0")
    if args.min_completed_required_sub_goals is not None and args.min_completed_required_sub_goals < 0:
        parser.error("--min-completed-required-sub-goals must be >= 0")
    if args.min_completed_goals is not None and args.min_completed_goals < 0:
        parser.error("--min-completed-goals must be >= 0")

    root = Path(args.repo_root).resolve()
    main_track_path = _resolve(root, args.main_track_status)

    required_metric_artifacts = (
        list(args.required_metric_artifact)
        if args.required_metric_artifact
        else list(DEFAULT_REQUIRED_METRIC_ARTIFACTS)
    )
    if args.require_fairness:
        required_metric_artifacts.extend(FAIRNESS_METRIC_ARTIFACTS)
    required_metric_artifacts = _dedupe(required_metric_artifacts)

    required_docs = (
        list(args.required_doc)
        if args.required_doc
        else list(DEFAULT_REQUIRED_DOCS)
    )
    required_docs = _dedupe(required_docs)

    required_goal_ids = (
        list(args.required_goal_id)
        if args.required_goal_id
        else list(DEFAULT_REQUIRED_GOAL_IDS)
    )
    if args.require_fairness:
        required_goal_ids.append(FAIRNESS_GOAL_ID)
    required_goal_ids = _dedupe(required_goal_ids)

    metric_presence = {
        artifact: _resolve(root, artifact).exists()
        for artifact in required_metric_artifacts
    }
    doc_presence = {
        doc: _resolve(root, doc).exists()
        for doc in required_docs
    }
    missing_metric_artifacts = [
        artifact for artifact, is_present in metric_presence.items() if not is_present
    ]
    missing_docs = [
        doc for doc, is_present in doc_presence.items() if not is_present
    ]

    failures: list[str] = []
    if len(missing_metric_artifacts) > args.max_missing_metrics:
        failures.append(
            "missing metric artifacts "
            f"{len(missing_metric_artifacts)}/{len(required_metric_artifacts)} "
            f"(allowed {args.max_missing_metrics}): {', '.join(missing_metric_artifacts)}"
        )
    if len(missing_docs) > args.max_missing_docs:
        failures.append(
            "missing docs "
            f"{len(missing_docs)}/{len(required_docs)} "
            f"(allowed {args.max_missing_docs}): {', '.join(missing_docs)}"
        )

    main_track_status: dict[str, object] = {}
    if not main_track_path.exists():
        failures.append(f"missing main track status artifact: {args.main_track_status}")
    else:
        try:
            main_track_status = _read_json(main_track_path)
        except (json.JSONDecodeError, ValueError) as exc:
            failures.append(f"invalid main track status JSON: {exc}")

    goal_status_by_id: dict[str, str] = {}
    summary_values: dict[str, object] = {}
    if main_track_status:
        summary = main_track_status.get("summary")
        if not isinstance(summary, dict):
            failures.append("main_track_status.summary missing or not an object")
        else:
            summary_values = summary
            required_sub_goals = summary.get("required_sub_goals")
            completed_required_sub_goals = summary.get("completed_required_sub_goals")
            completed_goals = summary.get("completed_goals")
            total_goals = summary.get("total_goals")
            all_done = summary.get("all_done")

            if not _is_int(required_sub_goals):
                failures.append("main_track_status.summary.required_sub_goals must be an integer")
            if not _is_int(completed_required_sub_goals):
                failures.append(
                    "main_track_status.summary.completed_required_sub_goals must be an integer"
                )
            if not _is_int(completed_goals):
                failures.append("main_track_status.summary.completed_goals must be an integer")
            if not _is_int(total_goals):
                failures.append("main_track_status.summary.total_goals must be an integer")

            if _is_int(required_sub_goals) and _is_int(completed_required_sub_goals):
                required_min_sub_goals = (
                    args.min_completed_required_sub_goals
                    if args.min_completed_required_sub_goals is not None
                    else required_sub_goals
                )
                if completed_required_sub_goals < required_min_sub_goals:
                    failures.append(
                        "required sub-goal completion "
                        f"{completed_required_sub_goals}/{required_sub_goals} "
                        f"below min {required_min_sub_goals}"
                    )

            if args.min_completed_goals is not None and _is_int(completed_goals):
                if completed_goals < args.min_completed_goals:
                    failures.append(
                        f"completed goals {completed_goals} below min {args.min_completed_goals}"
                    )

            if args.require_all_goals_done and all_done is not True:
                failures.append(
                    f"main_track_status.summary.all_done is {all_done!r}; expected True"
                )

        goals = main_track_status.get("goals")
        if not isinstance(goals, list):
            failures.append("main_track_status.goals missing or not an array")
        else:
            for goal in goals:
                if not isinstance(goal, dict):
                    continue
                goal_id = goal.get("id")
                goal_status = goal.get("status")
                if isinstance(goal_id, str):
                    goal_status_by_id[goal_id] = str(goal_status)

            for goal_id in required_goal_ids:
                if goal_id not in goal_status_by_id:
                    failures.append(f"required goal missing in main_track_status: {goal_id}")
                    continue
                if goal_status_by_id[goal_id] != "done":
                    failures.append(
                        f"required goal not done: {goal_id} "
                        f"(status={goal_status_by_id[goal_id]!r})"
                    )

    readiness = not failures
    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "ready": readiness,
        "repo_root": str(root),
        "main_track_status_path": str(main_track_path),
        "thresholds": {
            "max_missing_metrics": args.max_missing_metrics,
            "max_missing_docs": args.max_missing_docs,
            "min_completed_required_sub_goals": args.min_completed_required_sub_goals,
            "min_completed_goals": args.min_completed_goals,
            "require_all_goals_done": args.require_all_goals_done,
        },
        "checks": {
            "required_metric_artifacts": {
                "required": required_metric_artifacts,
                "missing": missing_metric_artifacts,
                "present_count": len(required_metric_artifacts) - len(missing_metric_artifacts),
                "required_count": len(required_metric_artifacts),
            },
            "required_docs": {
                "required": required_docs,
                "missing": missing_docs,
                "present_count": len(required_docs) - len(missing_docs),
                "required_count": len(required_docs),
            },
            "required_goal_ids": {
                "required": required_goal_ids,
                "goal_status": {
                    goal_id: goal_status_by_id.get(goal_id, "missing")
                    for goal_id in required_goal_ids
                },
            },
            "main_track_summary": {
                "required_sub_goals": summary_values.get("required_sub_goals"),
                "completed_required_sub_goals": summary_values.get("completed_required_sub_goals"),
                "completed_goals": summary_values.get("completed_goals"),
                "total_goals": summary_values.get("total_goals"),
                "all_done": summary_values.get("all_done"),
            },
        },
        "failures": failures,
    }

    if args.json_out:
        json_out = _resolve(root, args.json_out)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        with json_out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    print("MVP readiness summary")
    print(
        "- required metrics present: "
        f"{payload['checks']['required_metric_artifacts']['present_count']}/"
        f"{payload['checks']['required_metric_artifacts']['required_count']}"
    )
    print(
        "- required docs present: "
        f"{payload['checks']['required_docs']['present_count']}/"
        f"{payload['checks']['required_docs']['required_count']}"
    )
    summary = payload["checks"]["main_track_summary"]
    print(
        "- required sub-goals complete: "
        f"{summary['completed_required_sub_goals']}/{summary['required_sub_goals']}"
    )
    print(f"- completed goals: {summary['completed_goals']}/{summary['total_goals']}")
    print(f"- all goals done: {summary['all_done']}")
    if args.json_out:
        print(f"- readiness summary json: {_resolve(root, args.json_out)}")

    if failures:
        print("\nReadiness check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nReadiness check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
