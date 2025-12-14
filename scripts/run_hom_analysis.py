#!/usr/bin/env python3
"""
run_hom_analysis.py

Compute simple diagnostic metrics comparing the DFT-generated HOM curve
to the analytic QM reference curve.

Reads:
  data/generated/hom_results.csv   (columns: tau, coincidence_rate)

Writes:
  (no files by default; prints summary metrics to stdout)

Notes:
- The QM reference function is imported from dft_q3a_hom.py.
- Tune tau_c / visibility / classical_level to match the parameters used
  when generating hom_results.csv (or extend this script to fit them).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def find_repo_root(start: Path) -> Path:
    """
    Find repo root by searching upward for the expected data folder.
    This makes the script runnable from different working directories
    and resilient to moving the script within the repo.
    """
    start = start.resolve()
    for p in [start.parent, *start.parents]:
        if (p / "data" / "generated").exists():
            return p
    # Fallback: assume parent directory of this file is the repo root.
    return start.parent


HERE = Path(__file__).resolve()
ROOT = find_repo_root(HERE)

DATA = ROOT / "data" / "generated"
OUT = ROOT / "outputs"

CSV_FILE = DATA / "hom_results.csv"


# Ensure repo root is on sys.path so local imports work no matter where we run from.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dft_q3a_hom import hom_qm_reference  # QM reference curve used in paper
except Exception as e:
    print(f"[run_hom_analysis] ERROR: Failed to import hom_qm_reference from dft_q3a_hom.py: {e}",
          file=sys.stderr)
    print(f"[run_hom_analysis]        ROOT was detected as: {ROOT}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    if not CSV_FILE.exists():
        print(f"[run_hom_analysis] ERROR: Missing CSV: {CSV_FILE}", file=sys.stderr)
        print(f"[run_hom_analysis]        Expected input under: {DATA}", file=sys.stderr)
        sys.exit(1)

    try:
        data = np.genfromtxt(
            CSV_FILE,
            delimiter=",",
            names=True,
            dtype=float,
            encoding="utf-8",
        )
    except Exception as e:
        print(f"[run_hom_analysis] ERROR: Failed to read {CSV_FILE}: {e}", file=sys.stderr)
        sys.exit(1)

    expected = {"tau", "coincidence_rate"}
    got = set(data.dtype.names or [])
    missing = expected - got
    if missing:
        print(f"[run_hom_analysis] ERROR: CSV missing columns: {sorted(missing)}", file=sys.stderr)
        print(f"[run_hom_analysis]        CSV columns present: {sorted(got)}", file=sys.stderr)
        sys.exit(1)

    tau = np.asarray(data["tau"], dtype=float)
    P = np.asarray(data["coincidence_rate"], dtype=float)

    if tau.size == 0 or P.size == 0:
        print("[run_hom_analysis] ERROR: Empty data arrays.", file=sys.stderr)
        sys.exit(1)

    # ---- Reference parameters (keep explicit & documented) ----
    # Tune these to match what you used when generating hom_results.csv.
    tau_c = 1.5            # coherence time / dip width (in same units as tau)
    classical_level = 0.5  # classical coincidence baseline
    visibility = 1.0       # <1 if dip is lifted by noise/imperfections

    P_ref = hom_qm_reference(
        tau,
        tau_c=tau_c,
        visibility=visibility,
        classical_level=classical_level,
    )

    diff = P - P_ref
    rms = float(np.sqrt(np.mean(diff**2)))

    # Guard correlation against constant arrays
    if np.std(P) == 0.0 or np.std(P_ref) == 0.0:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(P, P_ref)[0, 1])

    OUT.mkdir(parents=True, exist_ok=True)

    print(f"[run_hom_analysis] Repo root: {ROOT}")
    print(f"[run_hom_analysis] Input: {CSV_FILE} ({tau.size} rows)")
    print(f"[run_hom_analysis] Reference params: tau_c={tau_c}, visibility={visibility}, classical_level={classical_level}")
    print(f"[run_hom_analysis] RMS difference (DFT vs QM-ref) = {rms:.6f}")
    print(f"[run_hom_analysis] Correlation coefficient        = {corr:.6f}")


if __name__ == "__main__":
    main()