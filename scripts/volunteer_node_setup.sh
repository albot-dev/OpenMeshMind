#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/volunteer_node_setup.sh [options]

Options:
  --node-id <id>               Node identifier (default: generated from hostname+timestamp)
  --repo <owner/repo>          GitHub repository slug (default: inferred from git remote)
  --mode <label>               Pilot mode label (default: reduced)
  --region <label>             Node region label (default: unknown)
  --hardware-tier <label>      Node hardware tier label (default: unknown)
  --network-tier <label>       Node network tier label (default: unknown)
  --cycle-interval-sec <int>   Interval between cycles in loop mode (default: 1800)
  --max-cycles <int>           Max cycles for loop mode, 0 means continuous (default: 0)
  --min-uptime-ratio <float>   Health-check threshold (default: 0.90)
  --token-env-var <name>       Token env var key in .env (default: github_token)
  --config <path>              Node config path (default: pilot/node_config.json)
  --python <bin>               Python executable (default: python3)
  --start-loop                 Start continuous runner after onboarding checks
  --skip-bundle                Skip submission bundle creation
  --help                       Show this help

What this script does:
  1) Validates token setup (.env or environment)
  2) Writes/updates node config from template
  3) Runs one onboarding cycle
  4) Runs health and pilot metrics checks
  5) Creates onboarding artifact bundle for submission
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="python3"
CONFIG_PATH="pilot/node_config.json"
TOKEN_ENV_VAR="github_token"
MODE="reduced"
REGION="unknown"
HARDWARE_TIER="unknown"
NETWORK_TIER="unknown"
CYCLE_INTERVAL_SEC="1800"
MAX_CYCLES="0"
MIN_UPTIME_RATIO="0.90"
START_LOOP="0"
SKIP_BUNDLE="0"
REPO_OVERRIDE=""
NODE_ID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --node-id)
      NODE_ID="$2"
      shift 2
      ;;
    --repo)
      REPO_OVERRIDE="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --region)
      REGION="$2"
      shift 2
      ;;
    --hardware-tier)
      HARDWARE_TIER="$2"
      shift 2
      ;;
    --network-tier)
      NETWORK_TIER="$2"
      shift 2
      ;;
    --cycle-interval-sec)
      CYCLE_INTERVAL_SEC="$2"
      shift 2
      ;;
    --max-cycles)
      MAX_CYCLES="$2"
      shift 2
      ;;
    --min-uptime-ratio)
      MIN_UPTIME_RATIO="$2"
      shift 2
      ;;
    --token-env-var)
      TOKEN_ENV_VAR="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --start-loop)
      START_LOOP="1"
      shift 1
      ;;
    --skip-bundle)
      SKIP_BUNDLE="1"
      shift 1
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -f "pilot/node_config.example.json" ]]; then
  echo "Missing template: pilot/node_config.example.json" >&2
  exit 1
fi

read_dotenv_key() {
  local key="$1"
  local dotenv="$2"
  "${PYTHON_BIN}" - "${key}" "${dotenv}" <<'PY'
import sys
from pathlib import Path

key = sys.argv[1]
dotenv = Path(sys.argv[2])
if not dotenv.exists():
    raise SystemExit(0)

value = ""
for raw in dotenv.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    lhs, rhs = line.split("=", 1)
    if lhs.strip() != key:
        continue
    value = rhs.strip().strip('"').strip("'")
print(value)
PY
}

lookup_token_value() {
  local key="$1"
  local value="${!key:-}"
  if [[ -z "${value}" && -f ".env" ]]; then
    value="$(read_dotenv_key "${key}" ".env")"
  fi
  printf '%s' "${value}"
}

TOKEN_VALUE="$(lookup_token_value "${TOKEN_ENV_VAR}")"
if [[ -z "${TOKEN_VALUE}" ]]; then
  for candidate in "github_token" "GITHUB_TOKEN"; do
    TOKEN_VALUE="$(lookup_token_value "${candidate}")"
    if [[ -n "${TOKEN_VALUE}" ]]; then
      TOKEN_ENV_VAR="${candidate}"
      break
    fi
  done
fi
if [[ -z "${TOKEN_VALUE}" ]]; then
  echo "Missing token value for ${TOKEN_ENV_VAR}. Add github_token or GITHUB_TOKEN to .env (or export one)." >&2
  exit 1
fi

infer_repo() {
  local url cleaned
  url="$(git config --get remote.origin.url 2>/dev/null || true)"
  cleaned="${url%.git}"
  if [[ "${cleaned}" == https://github.com/* ]]; then
    echo "${cleaned#https://github.com/}"
    return
  fi
  if [[ "${cleaned}" == git@github.com:* ]]; then
    echo "${cleaned#git@github.com:}"
    return
  fi
  echo ""
}

REPO="${REPO_OVERRIDE}"
if [[ -z "${REPO}" ]]; then
  REPO="$(infer_repo)"
fi
if [[ -z "${REPO}" ]]; then
  echo "Could not infer GitHub repo; pass --repo owner/name" >&2
  exit 1
fi

if [[ -z "${NODE_ID}" ]]; then
  HOST_NAME="$(hostname -s 2>/dev/null || hostname 2>/dev/null || echo node)"
  NODE_ID="${HOST_NAME}-$(date -u +%Y%m%d%H%M%S)"
fi
NODE_ID="$(printf '%s' "${NODE_ID}" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9._-' '-')"
NODE_ROOT_DIR="pilot/nodes/${NODE_ID}"
METRICS_OUT="${NODE_ROOT_DIR}/pilot_metrics.json"
STATE_OUT="${NODE_ROOT_DIR}/node_state.json"
LOG_OUT="${NODE_ROOT_DIR}/node_runner.log"
PROFILE_OUT="${NODE_ROOT_DIR}/node_profile.json"

"${PYTHON_BIN}" - "${ROOT_DIR}" "${CONFIG_PATH}" "${NODE_ID}" "${MODE}" "${CYCLE_INTERVAL_SEC}" "${MAX_CYCLES}" "${REPO}" "${TOKEN_ENV_VAR}" "${METRICS_OUT}" "${STATE_OUT}" "${LOG_OUT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
config_path = root / sys.argv[2]
node_id = sys.argv[3]
mode = sys.argv[4]
cycle_interval_sec = int(sys.argv[5])
max_cycles = int(sys.argv[6])
repo = sys.argv[7]
token_env_var = sys.argv[8]
metrics_out = sys.argv[9]
state_out = sys.argv[10]
log_out = sys.argv[11]

example_path = root / "pilot" / "node_config.example.json"
with example_path.open("r", encoding="utf-8") as f:
    example_cfg = json.load(f)

if config_path.exists():
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
else:
    cfg = example_cfg

cfg["schema_version"] = 1
cfg["node_id"] = node_id
cfg["mode"] = mode
cfg["cycle_interval_sec"] = cycle_interval_sec
cfg["max_cycles"] = max_cycles
cfg["repo"] = repo
cfg["token_env_var"] = token_env_var
cfg["metrics_out"] = metrics_out
cfg["state_out"] = state_out
cfg["log_out"] = log_out
if "steps" not in cfg or not cfg["steps"]:
    cfg["steps"] = example_cfg.get("steps", [])

config_path.parent.mkdir(parents=True, exist_ok=True)
with config_path.open("w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2, sort_keys=True)

print(f"Wrote config: {config_path}")
PY

echo "Running onboarding cycle for node_id=${NODE_ID}"
"${PYTHON_BIN}" scripts/pilot_node_runner.py --config "${CONFIG_PATH}" --once

echo "Running health check"
"${PYTHON_BIN}" scripts/pilot_node_runner.py --config "${CONFIG_PATH}" --health --min-uptime-ratio "${MIN_UPTIME_RATIO}"

echo "Validating pilot metrics artifact"
"${PYTHON_BIN}" scripts/check_pilot_metrics.py "${METRICS_OUT}" --require-status-collected

echo "Writing node profile metadata"
"${PYTHON_BIN}" - "${PROFILE_OUT}" "${NODE_ID}" "${REGION}" "${HARDWARE_TIER}" "${NETWORK_TIER}" <<'PY'
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

profile_out = Path(sys.argv[1])
node_id = sys.argv[2]
region = sys.argv[3]
hardware_tier = sys.argv[4]
network_tier = sys.argv[5]

cpu_cores = os.cpu_count() or 1
memory_gb = 0

try:
    if hasattr(os, "sysconf"):
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        if isinstance(page_size, int) and isinstance(phys_pages, int):
            total_bytes = page_size * phys_pages
            if total_bytes > 0:
                memory_gb = int(math.ceil(total_bytes / (1024 ** 3)))
except (ValueError, OSError, AttributeError):
    pass

if memory_gb <= 0:
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        total_bytes = int(out)
        if total_bytes > 0:
            memory_gb = int(math.ceil(total_bytes / (1024 ** 3)))
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        memory_gb = 1

payload = {
    "schema_version": 1,
    "node_id": node_id,
    "region": region,
    "hardware_tier": hardware_tier,
    "network_tier": network_tier,
    "cpu_cores": int(cpu_cores),
    "memory_gb": int(memory_gb),
    "generated_utc": datetime.now(timezone.utc).isoformat(),
}

profile_out.parent.mkdir(parents=True, exist_ok=True)
with profile_out.open("w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True)

print(f"Wrote node profile: {profile_out}")
PY

if [[ "${SKIP_BUNDLE}" == "0" ]]; then
  mkdir -p pilot/submissions
  TS="$(date -u +%Y%m%dT%H%M%SZ)"
  BUNDLE="pilot/submissions/${NODE_ID}_onboarding_${TS}.tgz"

  FILES=()
  for path in "${CONFIG_PATH}" "${METRICS_OUT}" "${STATE_OUT}" "${LOG_OUT}" "${PROFILE_OUT}"; do
    if [[ -f "${path}" ]]; then
      FILES+=("${path}")
    fi
  done

  if [[ ${#FILES[@]} -gt 0 ]]; then
    tar -czf "${BUNDLE}" "${FILES[@]}"
    echo "Created onboarding bundle: ${BUNDLE}"
  fi
fi

echo "Volunteer node setup complete for ${NODE_ID}."

echo "Next step: submit pilot/submissions/*_onboarding_*.tgz to maintainers."

if [[ "${START_LOOP}" == "1" ]]; then
  echo "Starting continuous runner..."
  exec "${PYTHON_BIN}" scripts/pilot_node_runner.py --config "${CONFIG_PATH}"
fi
