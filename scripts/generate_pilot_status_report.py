#!/usr/bin/env python3
"""
Generate a pilot status report from pilot/cohort metrics using a markdown template.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOKEN_PATTERN = re.compile(r"\{\{[^{}]+\}\}")


def resolve(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def load_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_cmd(args: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        args,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.returncode, proc.stdout.strip()


def get(path: dict[str, object], *keys: str, default: object = "n/a") -> object:
    value: object = path
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
        if value is None:
            return default
    return value


def format_metric(value: object, digits: int = 4) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def risk_status(
    cohort_metrics: dict[str, object] | None,
    pilot_metrics: dict[str, object] | None,
) -> str:
    if cohort_metrics:
        health = get(cohort_metrics, "summary", "health", default={})
        status = get(cohort_metrics, "summary", "status", default={})
        if isinstance(health, dict) and isinstance(status, dict):
            uptime = health.get("uptime_ratio_24h_mean", 0.0)
            ok_ratio = health.get("last_cycle_ok_ratio", 0.0)
            open_issues = status.get("open_issues_max", 0)
            if isinstance(uptime, (int, float)) and isinstance(ok_ratio, (int, float)):
                if uptime < 0.85 or ok_ratio < 0.70:
                    return "red"
                if uptime < 0.95 or ok_ratio < 0.90 or int(open_issues) > 0:
                    return "yellow"
                return "green"

    if pilot_metrics:
        if get(pilot_metrics, "health", "last_cycle_ok", default=False) is not True:
            return "red"
        uptime = get(pilot_metrics, "health", "uptime_ratio_24h", default=0.0)
        if isinstance(uptime, (int, float)) and uptime < 0.95:
            return "yellow"
        return "green"

    return "red"


def build_replacements(
    pilot_metrics: dict[str, object] | None,
    cohort_metrics: dict[str, object] | None,
    report_date_utc: str,
    reporting_window: str,
) -> dict[str, str]:
    overall = risk_status(cohort_metrics=cohort_metrics, pilot_metrics=pilot_metrics)

    if cohort_metrics:
        health_uptime = get(cohort_metrics, "summary", "health", "uptime_ratio_24h_mean")
        health_ok = get(cohort_metrics, "summary", "health", "last_cycle_ok_ratio")
        quality_acc = get(cohort_metrics, "summary", "quality", "classification_accuracy_mean")
        quality_f1 = get(cohort_metrics, "summary", "quality", "classification_macro_f1_mean")
        jain_base = get(cohort_metrics, "summary", "decentralization", "baseline_int8_jain_index_mean")
        jain_gain = get(cohort_metrics, "summary", "decentralization", "utility_int8_jain_gain_mean")
        comm_base = get(cohort_metrics, "summary", "communication", "baseline_int8_reduction_percent_mean")
        comm_util = get(cohort_metrics, "summary", "communication", "utility_int8_savings_percent_mean")
        bench_runtime = get(cohort_metrics, "summary", "accessibility", "benchmark_total_runtime_sec_mean")
        peak_rss = get(cohort_metrics, "summary", "accessibility", "max_peak_rss_bytes_max")
        open_milestones = get(cohort_metrics, "summary", "status", "open_milestones_max")
        open_issues = get(cohort_metrics, "summary", "status", "open_issues_max")
        active_nodes = get(cohort_metrics, "cohort", "node_count")
    else:
        health_uptime = get(pilot_metrics or {}, "health", "uptime_ratio_24h")
        health_ok = get(pilot_metrics or {}, "health", "last_cycle_ok")
        quality_acc = get(pilot_metrics or {}, "quality", "classification_accuracy")
        quality_f1 = get(pilot_metrics or {}, "quality", "classification_macro_f1")
        jain_base = get(pilot_metrics or {}, "decentralization", "baseline_int8_jain_index")
        jain_gain = get(pilot_metrics or {}, "decentralization", "utility_int8_jain_gain")
        comm_base = get(pilot_metrics or {}, "communication", "baseline_int8_reduction_percent")
        comm_util = get(pilot_metrics or {}, "communication", "utility_int8_savings_percent")
        bench_runtime = get(pilot_metrics or {}, "accessibility", "benchmark_total_runtime_sec")
        peak_rss = get(pilot_metrics or {}, "accessibility", "max_peak_rss_bytes")
        open_milestones = get(pilot_metrics or {}, "status", "open_milestones")
        open_issues = get(pilot_metrics or {}, "status", "open_issues")
        active_nodes = 1 if pilot_metrics else 0

    key_updates = "Generated from latest pilot and cohort artifacts."
    primary_risks = "No major blockers recorded." if overall == "green" else "Open pilot reliability risks need follow-up."

    return {
        "report_date_utc": report_date_utc,
        "reporting_window": reporting_window,
        "prepared_by": "OpenMeshMind automation",
        "commit": str(get(pilot_metrics or {}, "provenance", "commit", default="n/a")),
        "overall_status": overall,
        "key_updates": key_updates,
        "primary_risks": primary_risks,
        "health_uptime_ratio_24h": format_metric(health_uptime),
        "health_last_cycle_ok": format_metric(health_ok),
        "quality_classification_accuracy": format_metric(quality_acc),
        "quality_classification_macro_f1": format_metric(quality_f1),
        "decentralization_baseline_int8_jain_index": format_metric(jain_base),
        "decentralization_utility_int8_jain_gain": format_metric(jain_gain),
        "communication_baseline_int8_reduction_percent": format_metric(comm_base),
        "communication_utility_int8_savings_percent": format_metric(comm_util),
        "accessibility_benchmark_total_runtime_sec": format_metric(bench_runtime),
        "accessibility_max_peak_rss_bytes": format_metric(peak_rss, digits=0),
        "status_open_milestones": format_metric(open_milestones, digits=0),
        "status_open_issues": format_metric(open_issues, digits=0),
        "active_nodes": format_metric(active_nodes, digits=0),
        "new_nodes": "0",
        "nodes_with_repeated_failures": "0",
        "incident_id": "none",
        "incident_severity": "none",
        "incident_start_end": "n/a",
        "incident_impact_summary": "No incidents recorded in this window.",
        "incident_current_status": "resolved",
        "decision_log_refs": "DECISION_LOG.md",
        "follow_up_issues": "none",
        "owner_due_date": "n/a",
    }


def render_template(template: str, replacements: dict[str, str]) -> str:
    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace("{{" + key + "}}", value)
    return rendered


def bundle_artifacts(
    bundle_path: Path,
    artifacts: list[Path],
) -> list[str]:
    missing = [path for path in artifacts if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"missing expected artifact(s) for bundle: {missing_text}")

    added: list[str] = []
    with tarfile.open(bundle_path, "w:gz") as tar:
        for path in artifacts:
            arc = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name
            tar.add(path, arcname=arc)
            added.append(arc)
    return added


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot-metrics",
        default="pilot/pilot_metrics.json",
        help="Path to single-node pilot metrics JSON.",
    )
    parser.add_argument(
        "--cohort-metrics",
        default="pilot/pilot_cohort_metrics.json",
        help="Path to cohort pilot metrics JSON.",
    )
    parser.add_argument(
        "--template",
        default="reports/PILOT_STATUS_TEMPLATE.md",
        help="Path to markdown template with placeholder tokens.",
    )
    parser.add_argument(
        "--out",
        default="reports/pilot_status.md",
        help="Path to output markdown report.",
    )
    parser.add_argument(
        "--reporting-window",
        default="last 7 days",
        help="Reporting window label.",
    )
    parser.add_argument(
        "--bundle-out",
        default="",
        help="Optional path to artifact bundle .tgz.",
    )
    parser.add_argument(
        "--provenance-out",
        default="pilot/pilot_status_provenance.json",
        help="Output path for provenance manifest JSON.",
    )
    args = parser.parse_args()

    pilot_path = resolve(args.pilot_metrics)
    cohort_path = resolve(args.cohort_metrics)
    template_path = resolve(args.template)

    pilot_metrics = load_json(pilot_path)
    cohort_metrics = load_json(cohort_path)
    if pilot_metrics is None and cohort_metrics is None:
        print("No pilot metrics found. Provide --pilot-metrics and/or --cohort-metrics.")
        return 1
    if not template_path.exists():
        print(f"Template not found: {template_path}")
        return 1

    template = template_path.read_text(encoding="utf-8")
    report_date_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    replacements = build_replacements(
        pilot_metrics=pilot_metrics,
        cohort_metrics=cohort_metrics,
        report_date_utc=report_date_utc,
        reporting_window=args.reporting_window,
    )
    report = render_template(template=template, replacements=replacements)
    unresolved = TOKEN_PATTERN.findall(report)
    if unresolved:
        print("Unresolved template token(s) found after rendering:")
        for token in sorted(set(unresolved)):
            print(f"- {token}")
        return 1

    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"Pilot status report written to: {out_path}")

    provenance_path = resolve(args.provenance_out)
    provenance_cmd = [
        sys.executable,
        "scripts/build_provenance_manifest.py",
        "--label",
        "pilot-status",
        "--out",
        args.provenance_out,
        "--artifact",
        args.pilot_metrics,
        "--artifact",
        args.cohort_metrics,
        "--artifact",
        args.out,
    ]
    prov_code, prov_out = run_cmd(provenance_cmd)
    if prov_out:
        print(prov_out)
    if prov_code != 0:
        print("Failed to generate pilot status provenance manifest.")
        return 1

    if args.bundle_out:
        bundle_path = resolve(args.bundle_out)
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        artifacts = [out_path, provenance_path]
        if pilot_metrics is not None:
            artifacts.append(pilot_path)
        if cohort_metrics is not None:
            artifacts.append(cohort_path)
        try:
            added = bundle_artifacts(
                bundle_path=bundle_path,
                artifacts=artifacts,
            )
        except FileNotFoundError as exc:
            print(exc)
            return 1
        print(f"Pilot artifact bundle written to: {bundle_path}")
        print(f"Bundled files: {len(added)}")
        for item in added:
            print(f"- {item}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
