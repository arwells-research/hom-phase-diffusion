#!/usr/bin/env bash
# run_diffusion_suite.sh
#
# Run the phase-diffusion simulation grid used for the diffusion figures/tables.
#
# Produces CSV outputs under:
#   data/generated/
#
# Then runs analysis on each produced CSV using run_diffusion_analysis.py.
#
# Notes:
# - Increase PAIRS for publication-grade runs (e.g. 1000000+).
# - Safe to run from any working directory.
# - Override tools/params via env:
#     PYTHON=python3.13 PAIRS=1000000 ./run_diffusion_suite.sh
#
# Analysis defaults aligned with export_paper_numbers.py:
#   C_MIN=2e-2  MAX_N=80

set -euo pipefail

PYTHON="${PYTHON:-python3}"
PAIRS="${PAIRS:-200000}"        # allow override
DRY_RUN="${DRY_RUN:-0}"         # set DRY_RUN=1 to print commands only

# Analysis knobs (match paper/exporter)
C_MIN="${C_MIN:-2e-2}"
MAX_N="${MAX_N:-80}"

# Resolve repo root from this script location:
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Scripts
PHASE_SCRIPT="${SCRIPT_DIR}/dft_phase_diffusion.py"
ANALYZE_SCRIPT="${SCRIPT_DIR}/run_diffusion_analysis.py"

# Output directories
DATA_DIR="${ROOT_DIR}/data/generated"
OUT_DIR="${ROOT_DIR}/outputs"
LOG_DIR="${OUT_DIR}/logs"

mkdir -p "${DATA_DIR}" "${LOG_DIR}"

# ---- helpers ----

die() { echo "ERROR: $*" >&2; exit 1; }

run_and_tee() {
  # run a command and tee stdout+stderr to a logfile
  local logfile="$1"
  shift
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY_RUN] $*  | tee -a ${logfile}"
  else
    "$@" 2>&1 | tee -a "${logfile}"
  fi
}

check_prereqs() {
  command -v "${PYTHON}" >/dev/null 2>&1 || die "PYTHON not found: ${PYTHON}"
  [[ -f "${PHASE_SCRIPT}" ]] || die "Missing ${PHASE_SCRIPT}"
  [[ -f "${ANALYZE_SCRIPT}" ]] || die "Missing ${ANALYZE_SCRIPT}"
}

fmt_sigma() {
  # Canonical filename tags:
  #   0.05 -> 0p05
  #   0.10 -> 0p1
  #   0.20 -> 0p2
  #   0.005 -> 0p005 (if you ever use it)
  python3 - <<'PY' "$1"
import sys
x = float(sys.argv[1])
s = f"{x:.10f}".rstrip("0").rstrip(".")
print(s.replace(".", "p"))
PY
}

run_case() {
  # Args:
  #   mode out_csv log_file max_delay_steps pairs omega_sigma tag
  local mode="$1"
  local out="$2"
  local log="$3"
  local NMAX="$4"
  local pairs="$5"
  local omega_sigma="$6"
  local tag="${7:-}"

  echo "=== ${mode}: N=${NMAX}, sigma_omega=${omega_sigma}, pairs=${pairs} -> ${out} ==="

  run_and_tee "${log}" \
    "${PYTHON}" "${PHASE_SCRIPT}" \
      --max-delay-steps "${NMAX}" \
      --pairs "${pairs}" \
      --omega0 0.0 \
      --omega-sigma "${omega_sigma}" \
      --out-csv "${out}" \
      --no-plot

  echo "--- Analyzing ${out} (c_min=${C_MIN}, max_N=${MAX_N}) ---"
  run_and_tee "${log}" \
    "${PYTHON}" "${ANALYZE_SCRIPT}" "${out}" \
      --c-min "${C_MIN}" \
      --max-N "${MAX_N}"
  echo
}

# ---- runners ----

run_pure_diff() {
  # Pure diffusion (σ_ω = 0), N=50 and N=200
  for NMAX in 50 200; do
    local out="${DATA_DIR}/phase_diff_pure_diff_N${NMAX}.csv"
    local log="${LOG_DIR}/diffusion_pure_N${NMAX}.log"
    run_case "Pure diffusion" "${out}" "${log}" "${NMAX}" "${PAIRS}" "0.0"
  done
}

run_bandwidth_grid() {
  # Finite bandwidth cases.
  # We use canonical sigma tags in filenames so the exporter/paper stays consistent.
  #
  # Rows: sigma_omega, max_delay_steps, pairs, tag(optional)
  local configs=(
    "0.05 50  ${PAIRS}  "
    "0.05 200 ${PAIRS}  "
    "0.10 50  ${PAIRS}  "
    "0.20 30  ${PAIRS}  "
    "0.10 30  500000    _dense"
    "0.20 200 ${PAIRS}  "
  )

  local cfg sigma NMAX pairs tag sigma_tag out log
  for cfg in "${configs[@]}"; do
    # shellcheck disable=SC2086
    read -r sigma NMAX pairs tag <<< "${cfg}"

    sigma_tag="$(fmt_sigma "${sigma}")"
    out="${DATA_DIR}/phase_diff_omega_sigma_${sigma_tag}_N${NMAX}${tag}.csv"
    log="${LOG_DIR}/diffusion_sigma_${sigma_tag}_N${NMAX}${tag}.log"

    run_case "Bandwidth case" "${out}" "${log}" "${NMAX}" "${pairs}" "${sigma}" "${tag}"
  done
}

main() {
  check_prereqs
  echo "[run_diffusion_suite] ROOT_DIR=${ROOT_DIR}"
  echo "[run_diffusion_suite] DATA_DIR=${DATA_DIR}"
  echo "[run_diffusion_suite] LOG_DIR=${LOG_DIR}"
  echo "[run_diffusion_suite] PYTHON=${PYTHON}"
  echo "[run_diffusion_suite] PAIRS=${PAIRS}"
  echo "[run_diffusion_suite] C_MIN=${C_MIN}"
  echo "[run_diffusion_suite] MAX_N=${MAX_N}"
  [[ "${DRY_RUN}" == "1" ]] && echo "[run_diffusion_suite] DRY_RUN=1"

  run_pure_diff
  run_bandwidth_grid

  echo "All phase diffusion runs complete."
}

main "$@"