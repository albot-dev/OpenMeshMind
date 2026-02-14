#!/usr/bin/env python3
"""
Generate a public weekly status report and metrics bundle.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ARTIFACTS = [
    "baseline_metrics.json",
    "classification_metrics.json",
    "generality_metrics.json",
    "benchmark_metrics.json",
    "fairness_metrics.json",
    "utility_fedavg_metrics.json",
    "utility_fairness_metrics.json",
    "smoke_summary.json",
    "CHANGELOG.md",
    "DECISION_LOG.md",
    "PROVENANCE_TEMPLATE.md",
]


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


def load_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_git_meta() -> dict[str, str]:
    _, commit = run_cmd(["git", "rev-parse", "HEAD"])
    _, branch = run_cmd(["git", "branch", "--show-current"])
    _, short = run_cmd(["git", "rev-parse", "--short", "HEAD"])
    return {"commit": commit, "branch": branch, "short_commit": short}


def get_default_repo() -> str:
    code, url = run_cmd(["git", "config", "--get", "remote.origin.url"])
    if code != 0 or not url:
        return ""
    cleaned = url
    if cleaned.endswith(".git"):
        cleaned = cleaned[:-4]
    if cleaned.startswith("https://github.com/"):
        return cleaned.split("https://github.com/", 1)[1]
    if cleaned.startswith("git@github.com:"):
        return cleaned.split("git@github.com:", 1)[1]
    return ""


def gh_env(token_env_var: str) -> dict[str, str]:
    env = os.environ.copy()
    token = env.get(token_env_var, "")
    if token:
        env["GH_TOKEN"] = token
    return env


def get_github_status(repo: str, token_env_var: str) -> dict[str, object]:
    if not repo:
        return {"ok": False, "error": "Repository is unknown."}

    env = gh_env(token_env_var)
    code_m, out_m = run_cmd(
        ["gh", "api", f"repos/{repo}/milestones?state=open"],
        env=env,
    )
    code_i, out_i = run_cmd(
        ["gh", "api", f"repos/{repo}/issues?state=open&per_page=100"],
        env=env,
    )
    if code_m != 0 or code_i != 0:
        return {
            "ok": False,
            "error": "GitHub status fetch failed.",
            "milestones_raw": out_m,
            "issues_raw": out_i,
        }

    milestones = json.loads(out_m)
    issues = [
        item
        for item in json.loads(out_i)
        if "pull_request" not in item
    ]
    return {"ok": True, "milestones": milestones, "issues": issues}


def quality_summary(utility: dict[str, object] | None) -> list[str]:
    if not utility:
        return ["utility_fedavg_metrics.json not found."]
    methods = utility.get("methods", {})
    central = methods.get("centralized", {})
    fp32 = methods.get("fedavg_fp32", {})
    int8 = methods.get("fedavg_int8", {})
    sparse = methods.get("fedavg_sparse", {})
    return [
        f"centralized accuracy={central.get('accuracy_mean')}",
        f"fedavg_fp32 accuracy={fp32.get('accuracy_mean')}",
        f"fedavg_int8 accuracy={int8.get('accuracy_mean')}",
        f"fedavg_sparse accuracy={sparse.get('accuracy_mean')}",
    ]


def accessibility_summary(bench: dict[str, object] | None, smoke: dict[str, object] | None) -> list[str]:
    lines: list[str] = []
    if smoke:
        lines.append(f"smoke total duration sec={smoke.get('total_duration_sec')}")
    else:
        lines.append("smoke_summary.json not found.")
    if bench:
        summary = bench.get("summary", {})
        lines.append(f"benchmark total runtime sec={summary.get('total_runtime_sec')}")
        lines.append(f"benchmark max peak RSS bytes={summary.get('max_peak_rss_bytes')}")
    else:
        lines.append("benchmark_metrics.json not found.")
    return lines


def decentralization_summary(
    fairness: dict[str, object] | None,
    utility_fairness: dict[str, object] | None,
) -> list[str]:
    lines: list[str] = []
    if fairness:
        int8 = fairness.get("methods", {}).get("fedavg_int8", {})
        fair = int8.get("fairness", {})
        lines.append(
            "baseline int8 fairness: "
            f"gap={fair.get('contribution_rate_gap_mean')}, "
            f"jain={fair.get('contribution_jain_index_mean')}"
        )
    else:
        lines.append("fairness_metrics.json not found.")

    if utility_fairness:
        scenarios = utility_fairness.get("scenarios")
        if scenarios:
            gains = []
            for scenario in scenarios:
                fp = scenario["methods"]["fedavg_fp32"]["fairness"]
                i8 = scenario["methods"]["fedavg_int8"]["fairness"]
                gains.append(i8["contribution_jain_index_mean"] - fp["contribution_jain_index_mean"])
            lines.append(f"utility fairness Jain gain range={min(gains):.4f}..{max(gains):.4f}")
        else:
            fp = utility_fairness["methods"]["fedavg_fp32"]["fairness"]
            i8 = utility_fairness["methods"]["fedavg_int8"]["fairness"]
            lines.append(
                "utility fairness Jain gain="
                f"{i8['contribution_jain_index_mean'] - fp['contribution_jain_index_mean']:.4f}"
            )
    else:
        lines.append("utility_fairness_metrics.json not found.")
    return lines


def communication_summary(
    baseline: dict[str, object] | None,
    utility: dict[str, object] | None,
) -> list[str]:
    lines: list[str] = []
    if baseline:
        lines.append(
            "baseline int8 comm reduction="
            f"{baseline.get('communication_reduction_percent')}"
        )
    else:
        lines.append("baseline_metrics.json not found.")
    if utility:
        savings = utility.get("communication_savings_percent", {})
        lines.append(f"utility int8 savings vs fp32={savings.get('int8_vs_fp32_percent')}")
        lines.append(f"utility sparse savings vs fp32={savings.get('sparse_vs_fp32_percent')}")
    else:
        lines.append("utility_fedavg_metrics.json not found.")
    return lines


def generality_summary(generality: dict[str, object] | None) -> list[str]:
    if not generality:
        return ["generality_metrics.json not found."]

    tasks = generality.get("tasks", {})
    aggregate = generality.get("aggregate", {})
    lines = []
    cls = tasks.get("classification", {}).get("metrics", {})
    ret = tasks.get("retrieval", {}).get("metrics", {})
    ins = tasks.get("instruction_following", {}).get("metrics", {})
    tool = tasks.get("tool_use", {}).get("metrics", {})
    lines.append(
        "generality core:"
        f" cls_acc={cls.get('accuracy')},"
        f" ret_r@1={ret.get('recall_at_1')},"
        f" instruction_pass={ins.get('pass_rate')},"
        f" tool_pass={tool.get('pass_rate')}"
    )
    lines.append(f"generality overall score={aggregate.get('overall_score')}")
    dist = tasks.get("distributed_reference", {}).get("metrics", {})
    if dist:
        lines.append(
            "distributed reference:"
            f" int8_drop={dist.get('int8_accuracy_drop')},"
            f" int8_comm_savings={dist.get('int8_comm_savings_percent')}"
        )
    return lines


def build_report(
    repo: str,
    gh_status: dict[str, object],
    git_meta: dict[str, str],
) -> str:
    baseline = load_json(ROOT / "baseline_metrics.json")
    classification = load_json(ROOT / "classification_metrics.json")
    benchmark = load_json(ROOT / "benchmark_metrics.json")
    fairness = load_json(ROOT / "fairness_metrics.json")
    utility = load_json(ROOT / "utility_fedavg_metrics.json")
    utility_fairness = load_json(ROOT / "utility_fairness_metrics.json")
    smoke = load_json(ROOT / "smoke_summary.json")
    generality = load_json(ROOT / "generality_metrics.json")

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = []
    lines.append("# Weekly Status Report")
    lines.append("")
    lines.append(f"- Generated: `{now}`")
    lines.append(f"- Repository: `{repo}`")
    lines.append(f"- Branch: `{git_meta['branch']}`")
    lines.append(f"- Commit: `{git_meta['commit']}`")
    lines.append("")
    lines.append("## Provenance References")
    lines.append("")
    lines.append("- `DECISION_LOG.md`")
    lines.append("- `PROVENANCE_TEMPLATE.md`")
    lines.append("- `RELEASE.md`")
    lines.append("")
    lines.append("## Milestone and Issue Status")
    lines.append("")
    if not gh_status.get("ok"):
        lines.append(f"- Unable to fetch GitHub status: {gh_status.get('error')}")
    else:
        milestones = gh_status.get("milestones", [])
        issues = gh_status.get("issues", [])
        lines.append(f"- Open milestones: `{len(milestones)}`")
        for milestone in milestones:
            lines.append(
                f"  - #{milestone['number']} {milestone['title']} "
                f"(open_issues={milestone['open_issues']})"
            )
        lines.append(f"- Open issues: `{len(issues)}`")
        for issue in issues[:20]:
            lines.append(f"  - #{issue['number']} {issue['title']}")
    lines.append("")
    lines.append("## Metrics Summary")
    lines.append("")
    lines.append("### Quality")
    for item in quality_summary(utility):
        lines.append(f"- {item}")
    if classification:
        lines.append(
            f"- local classification accuracy={classification['metrics'].get('accuracy')}, "
            f"macro_f1={classification['metrics'].get('macro_f1')}"
        )
    for item in generality_summary(generality):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("### Accessibility")
    for item in accessibility_summary(benchmark, smoke):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("### Decentralization/Fairness")
    for item in decentralization_summary(fairness, utility_fairness):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("### Communication Efficiency")
    for item in communication_summary(baseline, utility):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Next-Week Focus")
    lines.append("")
    lines.append("- [ ] Ship remaining open milestone issues")
    lines.append("- [ ] Publish weekly status and artifact bundle")
    lines.append("- [ ] Record major decisions in `DECISION_LOG.md`")
    lines.append("")
    return "\n".join(lines)


def bundle_artifacts(bundle_path: Path, report_path: Path) -> list[str]:
    added: list[str] = []
    with tarfile.open(bundle_path, "w:gz") as tar:
        for rel in DEFAULT_ARTIFACTS:
            p = ROOT / rel
            if p.exists():
                tar.add(p, arcname=rel)
                added.append(rel)
        if report_path.exists():
            arc = str(report_path.relative_to(ROOT)) if report_path.is_relative_to(ROOT) else report_path.name
            tar.add(report_path, arcname=arc)
            added.append(arc)
    return added


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default="",
        help="GitHub repository owner/name (default: inferred from origin remote).",
    )
    parser.add_argument(
        "--out",
        default="reports/weekly_status.md",
        help="Output markdown report path (default: reports/weekly_status.md).",
    )
    parser.add_argument(
        "--bundle-out",
        default="reports/weekly_artifacts.tgz",
        help="Output artifacts bundle path (default: reports/weekly_artifacts.tgz).",
    )
    parser.add_argument(
        "--token-env-var",
        default="GITHUB_TOKEN",
        help="Env var used for GitHub auth (default: GITHUB_TOKEN).",
    )
    args = parser.parse_args()

    repo = args.repo or get_default_repo()
    status = get_github_status(repo=repo, token_env_var=args.token_env_var)
    git_meta = get_git_meta()

    report_text = build_report(repo=repo, gh_status=status, git_meta=git_meta)
    report_path = ROOT / args.out
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")

    bundle_path = ROOT / args.bundle_out
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    added = bundle_artifacts(bundle_path=bundle_path, report_path=report_path)

    print(f"Weekly report written to: {report_path}")
    print(f"Artifact bundle written to: {bundle_path}")
    print(f"Bundled files: {len(added)}")
    for item in added:
        print(f"- {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
