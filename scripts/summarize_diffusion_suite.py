#!/usr/bin/env python3
"""
summarize_diffusion_suite.py

Summarize a batch of phase-diffusion CSV outputs by fitting a mixed variance model

    Var[Δθ](N) ≈ D * N + b * N^2 + c

to each file and printing a compact table (suitable for a paper appendix / Table).

Default behavior:
- Reads diffusion CSVs from:  data/generated/phase_diff_*.csv
- Prints a summary table to stdout

Optional:
- Write a machine-readable summary CSV to outputs/ (use --write-csv)

You may also pass explicit CSV paths as command-line arguments.

Expected CSV columns:
  - N_delay_steps
  - var_dtheta
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def find_repo_root(start: Path) -> Path:
    """
    Find repo root by walking upward and looking for common markers.
    Falls back to the directory containing this script if no markers are found.
    """
    markers = [
        ".git",
        "pyproject.toml",
        "requirements.txt",
        "README.md",
        "data",
    ]
    for p in [start, *start.parents]:
        if any((p / m).exists() for m in markers):
            return p
    return start


HERE = Path(__file__).resolve()
ROOT = find_repo_root(HERE.parent)
DATA = ROOT / "data" / "generated"
OUT = ROOT / "outputs"

DATA.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)


def parse_filename_metadata(path: str) -> Tuple[Optional[float], Optional[int]]:
    """
    Extract sigma_omega and N_max from filenames of the form:
      phase_diff_pure_diff_N50.csv
      phase_diff_omega_sigma_0p05_N200.csv

    Returns:
      (sigma_omega: float|None, N_max: int|None)
    """
    name = Path(path).name

    m_pure = re.match(r"phase_diff_pure_diff_N(\d+)\.csv$", name)
    if m_pure:
        return 0.0, int(m_pure.group(1))

    m_bw = re.match(r"phase_diff_omega_sigma_([0-9p]+)_N(\d+)\.csv$", name)
    if m_bw:
        sigma_tag = m_bw.group(1)  # e.g. "0p05"
        N_max = int(m_bw.group(2))
        sigma_str = sigma_tag.replace("p", ".")
        try:
            sigma = float(sigma_str)
        except ValueError:
            sigma = None
        return sigma, N_max

    return None, None


def fit_mixed_variance(N: np.ndarray, var: np.ndarray) -> Tuple[float, float, float, float, float, float, float]:
    """
    Fit Var ≈ D*N + b*N^2 + c via least squares and estimate 1σ parameter errors.

    Returns:
      (D, b, c, RSS, D_err, b_err, c_err)
    """
    N = np.asarray(N, dtype=float)
    var = np.asarray(var, dtype=float)

    if N.size != var.size:
        raise ValueError("N and var must have the same length.")
    if N.size < 3:
        raise ValueError("Need at least 3 points to fit D, b, c.")

    # Design matrix: [N, N^2, 1]
    X = np.vstack([N, N**2, np.ones_like(N)]).T

    beta, residuals, rank, _s = np.linalg.lstsq(X, var, rcond=None)
    D, b, c = (float(beta[0]), float(beta[1]), float(beta[2]))

    # Residual sum of squares
    if residuals.size > 0:
        rss = float(residuals[0])
    else:
        rss = float(np.sum((var - X @ beta) ** 2))

    # Covariance estimate: σ^2 * (X^T X)^(-1)
    dof = max(int(N.size - X.shape[1]), 1)
    sigma2_hat = rss / dof

    # pinv is safer than inv for near-singular matrices
    XtX_inv = np.linalg.pinv(X.T @ X)
    cov_beta = sigma2_hat * XtX_inv
    errs = np.sqrt(np.maximum(np.diag(cov_beta), 0.0))

    D_err, b_err, c_err = (float(errs[0]), float(errs[1]), float(errs[2]))
    return D, b, c, rss, D_err, b_err, c_err


def summarize_file(path: str, drop_N0: bool = False) -> dict:
    df = pd.read_csv(path)

    required = {"N_delay_steps", "var_dtheta"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"{path}: missing required columns {sorted(required)} "
            f"(found {list(df.columns)})"
        )

    N = df["N_delay_steps"].to_numpy(dtype=float)
    var = df["var_dtheta"].to_numpy(dtype=float)

    if drop_N0:
        mask = N > 0
        N = N[mask]
        var = var[mask]

    sigma_omega, N_max = parse_filename_metadata(path)
    D, b, c, rss, D_err, b_err, c_err = fit_mixed_variance(N, var)

    return {
        "file": str(path),
        "file_short": Path(path).name,
        "sigma_omega": sigma_omega,
        "N_max": N_max,
        "D": D,
        "D_err": D_err,
        "b": b,
        "b_err": b_err,
        "c": c,
        "c_err": c_err,
        "RSS": rss,
        "N_points": int(N.size),
        "drop_N0": bool(drop_N0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Summarize diffusion CSVs with a mixed variance fit Var ≈ D N + b N^2 + c"
    )
    ap.add_argument(
        "files",
        nargs="*",
        help="Optional explicit CSV paths. If omitted, uses data/generated/phase_diff_*.csv",
    )
    ap.add_argument(
        "--drop-N0",
        action="store_true",
        help="Drop the N=0 row before fitting (helps avoid intercept bias when Var(0)=0).",
    )
    ap.add_argument(
        "--write-csv",
        action="store_true",
        help="Write outputs/diffusion_suite_summary.csv in addition to stdout table.",
    )
    ap.add_argument(
        "--out-csv",
        type=str,
        default=str(OUT / "diffusion_suite_summary.csv"),
        help="Output CSV path (used only with --write-csv).",
    )
    args = ap.parse_args()

    if args.files:
        files = [str(Path(f)) for f in args.files]
    else:
        files = sorted(glob.glob(str(DATA / "phase_diff_*.csv")))

    if not files:
        print("No diffusion CSVs found.")
        print(f"Default pattern: {DATA / 'phase_diff_*.csv'}")
        print("Or pass files explicitly, e.g.:")
        print("  python3 summarize_diffusion_suite.py data/generated/phase_diff_*.csv")
        sys.exit(1)

    summaries = []
    for f in files:
        try:
            summaries.append(summarize_file(f, drop_N0=args.drop_N0))
        except Exception as e:
            print(f"[WARN] Skipping {f}: {e}")

    if not summaries:
        print("No valid diffusion files processed.")
        sys.exit(1)

    # Sort by sigma_omega, then N_max (None sorted last)
    def sort_key(s: dict):
        sigma = s["sigma_omega"]
        Nmax = s["N_max"]
        sigma_key = sigma if sigma is not None else 1e9
        Nmax_key = Nmax if Nmax is not None else 1e9
        return (sigma_key, Nmax_key, s["file_short"])

    summaries.sort(key=sort_key)

    # Optionally write CSV (full precision)
    if args.write_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_out = pd.DataFrame(summaries)
        df_out.to_csv(out_path, index=False)
        print(f"[summarize_diffusion_suite] Wrote summary CSV: {out_path}")

    # Print paper-friendly summary table (rounded)
    print("\nPhase Diffusion Summary (mixed variance fit: Var ≈ D N + b N^2 + c)\n")
    header = (
        f"{'file':40s}  {'σ_ω':>7s}  {'N_max':>6s}  "
        f"{'D±err':>16s}  {'b±err':>16s}  {'c±err':>16s}  {'RSS':>10s}  {'N':>5s}"
    )
    print(header)
    print("-" * len(header))

    for s in summaries:
        sigma = s["sigma_omega"]
        sigma_str = f"{sigma:.3f}" if sigma is not None else "-"
        Nmax_str = f"{s['N_max']}" if s["N_max"] is not None else "-"
        print(
            f"{s['file_short']:40s}  "
            f"{sigma_str:>7s}  "
            f"{Nmax_str:>6s}  "
            f"{s['D']:7.4f}±{s['D_err']:7.4f}  "
            f"{s['b']:7.5f}±{s['b_err']:7.5f}  "
            f"{s['c']:7.4f}±{s['c_err']:7.4f}  "
            f"{s['RSS']:10.4f}  "
            f"{s['N_points']:5d}"
        )

    print("\nNotes:")
    print("  • For pure diffusion (σ_ω = 0), the quadratic term b should be ~0 (within error).")
    print("  • For bandwidth runs (σ_ω > 0), b should scale approximately with σ_ω² (check your model’s exact prefactor).")
    print("  • D is the effective linear growth coefficient in Var[Δθ](N); its expected value depends on your normalization.")
    if args.drop_N0:
        print("  • Fits used --drop-N0 (N=0 rows removed before fitting).")
    print()


if __name__ == "__main__":
    main()