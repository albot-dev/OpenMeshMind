#!/usr/bin/env python3
"""
Generate final 14-day pilot report from manifest, daily cohort metrics, and incident log.
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def render(template: str, replacements: dict[str, str]) -> str:
    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace("{{" + key + "}}", value)
    return rendered


def fmt(value: object, digits: int = 4) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if value is None:
        return "n/a"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="pilot/cohort_manifest.json")
    parser.add_argument("--onboarding-summary", default="pilot/cohort_onboarding_summary.json")
    parser.add_argument("--daily-cohort-glob", default="pilot/runs/day*/pilot_cohort_metrics.json")
    parser.add_argument("--incident-log", default="pilot/incident_log.json")
    parser.add_argument("--template", default="reports/PILOT_14_DAY_REPORT_TEMPLATE.md")
    parser.add_argument("--out", default="reports/pilot_14_day_report.md")
    parser.add_argument("--window-label", default="14-day pilot window")
    parser.add_argument("--bundle-out", default="")
    parser.add_argument("--provenance-out", default="pilot/pilot_14_day_provenance.json")
    args = parser.parse_args()

    manifest = load_json(resolve(args.manifest)) or {}
    onboarding = load_json(resolve(args.onboarding_summary)) or {}
    incident_log = load_json(resolve(args.incident_log)) or {}

    daily_paths = sorted(glob.glob(str(resolve(args.daily_cohort_glob))))
    daily_reports: list[dict[str, object]] = []
    for path in daily_paths:
        report = load_json(Path(path))
        if report:
            daily_reports.append(report)

    uptime_values: list[float] = []
    last_cycle_ok_values: list[float] = []
    acc_values: list[float] = []
    f1_values: list[float] = []
    jain_gain_values: list[float] = []
    comm_values: list[float] = []

    for report in daily_reports:
        summary = report.get("summary", {})
        health = summary.get("health", {}) if isinstance(summary, dict) else {}
        quality = summary.get("quality", {}) if isinstance(summary, dict) else {}
        decentral = summary.get("decentralization", {}) if isinstance(summary, dict) else {}
        communication = summary.get("communication", {}) if isinstance(summary, dict) else {}

        uptime = health.get("uptime_ratio_24h_mean")
        last_ok = health.get("last_cycle_ok_ratio")
        acc = quality.get("classification_accuracy_mean")
        f1 = quality.get("classification_macro_f1_mean")
        jain_gain = decentral.get("utility_int8_jain_gain_mean")
        comm = communication.get("utility_int8_savings_percent_mean")

        if isinstance(uptime, (int, float)):
            uptime_values.append(float(uptime))
        if isinstance(last_ok, (int, float)):
            last_cycle_ok_values.append(float(last_ok))
        if isinstance(acc, (int, float)):
            acc_values.append(float(acc))
        if isinstance(f1, (int, float)):
            f1_values.append(float(f1))
        if isinstance(jain_gain, (int, float)):
            jain_gain_values.append(float(jain_gain))
        if isinstance(comm, (int, float)):
            comm_values.append(float(comm))

    incidents = incident_log.get("incidents", []) if isinstance(incident_log, dict) else []
    sev1 = 0
    sev2 = 0
    sev3 = 0
    open_count = 0
    follow_up_refs: list[str] = []
    for incident in incidents:
        if not isinstance(incident, dict):
            continue
        severity = incident.get("severity")
        status = incident.get("status")
        if severity == "SEV-1":
            sev1 += 1
        elif severity == "SEV-2":
            sev2 += 1
        elif severity == "SEV-3":
            sev3 += 1
        if status != "resolved":
            open_count += 1
        for issue_num in incident.get("follow_up_issue_numbers", []):
            follow_up_refs.append(f"#{issue_num}")

    uptime_mean = mean(uptime_values)
    overall_status = "green"
    if not daily_reports or len(daily_reports) < 14:
        overall_status = "yellow"
    if uptime_mean < 0.90:
        overall_status = "red"
    if open_count > 0:
        overall_status = "yellow" if overall_status == "green" else overall_status

    _, commit = run_cmd(["git", "rev-parse", "HEAD"])
    template_path = resolve(args.template)
    if not template_path.exists():
        print(f"Template not found: {template_path}")
        return 1
    template = template_path.read_text(encoding="utf-8")

    replacements = {
        "report_date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "window_label": args.window_label,
        "cohort_id": fmt(manifest.get("cohort_id")),
        "commit_or_tag": commit,
        "artifact_bundle_path": args.bundle_out if args.bundle_out else "n/a",
        "overall_status": overall_status,
        "key_outcomes": "Cohort metrics and incident log aggregated for pilot window.",
        "main_risks": "Insufficient day coverage" if len(daily_reports) < 14 else "No critical unresolved risks recorded.",
        "manifest_node_count": fmt(len(manifest.get("nodes", [])) if isinstance(manifest.get("nodes"), list) else 0),
        "onboarding_passed_count": fmt(onboarding.get("passed_count", 0)),
        "onboarding_failed_count": fmt(onboarding.get("failed_count", 0)),
        "onboarding_startup_rate": fmt(onboarding.get("startup_rate", 0.0)),
        "onboarding_failure_rate": fmt(onboarding.get("failure_rate", 0.0)),
        "daily_count": fmt(len(daily_reports)),
        "uptime_mean": fmt(uptime_mean),
        "uptime_min": fmt(min(uptime_values) if uptime_values else 0.0),
        "last_cycle_ok_mean": fmt(mean(last_cycle_ok_values)),
        "classification_accuracy_mean": fmt(mean(acc_values)),
        "classification_macro_f1_mean": fmt(mean(f1_values)),
        "utility_jain_gain_mean": fmt(mean(jain_gain_values)),
        "utility_comm_savings_mean": fmt(mean(comm_values)),
        "incident_total": fmt(len(incidents) if isinstance(incidents, list) else 0),
        "incident_sev1_count": fmt(sev1),
        "incident_sev2_count": fmt(sev2),
        "incident_sev3_count": fmt(sev3),
        "incident_open_count": fmt(open_count),
        "follow_up_issues": ", ".join(sorted(set(follow_up_refs))) if follow_up_refs else "none",
        "follow_up_owners": "Assign in governance cadence (PILOT_GOVERNANCE.md)",
    }

    report = render(template, replacements)
    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"Pilot 14-day report written to: {out_path}")

    provenance_cmd = [
        sys.executable,
        "scripts/build_provenance_manifest.py",
        "--label",
        "pilot-14-day",
        "--out",
        args.provenance_out,
        "--artifact",
        args.manifest,
        "--artifact",
        args.onboarding_summary,
        "--artifact",
        args.incident_log,
        "--artifact",
        args.out,
    ]
    for daily_path in daily_paths:
        rel = str(Path(daily_path).relative_to(ROOT))
        provenance_cmd.extend(["--artifact", rel])
    prov_code, prov_out = run_cmd(provenance_cmd)
    if prov_out:
        print(prov_out)
    if prov_code != 0:
        print("Warning: failed to generate pilot 14-day provenance manifest.")

    if args.bundle_out:
        bundle_path = resolve(args.bundle_out)
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(bundle_path, "w:gz") as tar:
            for rel in [args.manifest, args.onboarding_summary, args.incident_log, args.out]:
                p = resolve(rel)
                if p.exists():
                    arc = str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name
                    tar.add(p, arcname=arc)
            for path in daily_paths:
                p = Path(path)
                if p.exists():
                    arc = str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name
                    tar.add(p, arcname=arc)
            provenance_path = resolve(args.provenance_out)
            if provenance_path.exists():
                arc = (
                    str(provenance_path.relative_to(ROOT))
                    if provenance_path.is_relative_to(ROOT)
                    else provenance_path.name
                )
                tar.add(provenance_path, arcname=arc)
        print(f"Pilot 14-day artifact bundle written to: {bundle_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
