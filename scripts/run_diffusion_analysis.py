#!/usr/bin/env python3
"""
run_diffusion_analysis.py

Analyze phase-diffusion CSVs produced by the DFT diffusion suite.

Default input:
  data/generated/phase_diffusion_results.csv

This script:
  1) Fits variance models to Var[Δθ](N)
     - Linear:    Var ≈ a N + c
     - Quadratic: Var ≈ b N^2 + a N + c
     - Mixed:     Var ≈ a N + b N^2 + c   (least-squares)

     Reports:
       - RSS  (raw residual sum of squares)
       - RMSE (sqrt(RSS/n))
       - NRMSE_mean (RMSE / mean(var))
       - R^2

  2) Fits coherence envelope models to c(N) in log-space using only points where
     c > c_min (and optionally N <= max_N):
     - Exponential: c ≈ A exp(-α N)
     - Gaussian:    c ≈ A exp(-β N^2)  (reported via τ = 1/sqrt(β) when β>0)
     - Mixed:       c ≈ A exp(-α N - β N^2)

     Reports:
       - RSS in c-space (same as before)
       - RSS in log-space (diagnostic; highlights floor/overfit issues)

Notes:
- Large RSS values can be *purely a scale effect* when Var becomes large.
  Use RMSE / NRMSE_mean / R^2 for comparisons across runs.
- If coherence fits produce β < 0 in the mixed model, that usually indicates
  you're fitting too far into the numerical floor. Try:
      --c-min 1e-2 --max-N 80
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------
# Paths
# ------------------------

ROOT = Path(__file__).resolve().parents[1]     # repo root (…/hom-phase-diffusion)
DATA = ROOT / "data" / "generated"
OUT = ROOT / "outputs" / "analysis"

DATA.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

DEFAULT_CSV = DATA / "phase_diffusion_results.csv"


def resolve_input_path(p: str) -> Path:
    """
    Resolve input CSV path:
      - If p is an existing path (absolute or relative), use it.
      - Otherwise, treat p as a filename under data/generated/.
    """
    cand = Path(p)
    if cand.exists():
        return cand.resolve()
    cand2 = DATA / p
    if cand2.exists():
        return cand2.resolve()
    return cand.resolve()  # will fail later with clear error


def _fit_metrics(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    """
    Compute scale-aware metrics for a model fit.
    """
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)

    resid = y - yhat
    rss = float(np.sum(resid**2))
    n = int(y.size)

    rmse = float(np.sqrt(rss / max(n, 1)))

    ymean = float(np.mean(y)) if n > 0 else float("nan")
    denom = float(np.mean(y)) if n > 0 else float("nan")
    nrmse_mean = float(rmse / denom) if (n > 0 and denom != 0.0 and np.isfinite(denom)) else float("nan")

    tss = float(np.sum((y - ymean) ** 2))
    r2 = float(1.0 - rss / tss) if tss > 0 else float("nan")

    return {"rss": rss, "rmse": rmse, "nrmse_mean": nrmse_mean, "r2": r2, "n": float(n)}


def fit_variance_models(N, var, var_min_N=None, var_max_N=None):
    """
    Fit linear, quadratic, and mixed (a N + b N^2 + c) models to Var[Δθ](N).

    Optional:
      var_min_N / var_max_N: if provided, fits use only points with
      var_min_N <= N <= var_max_N.

    Returns:
      (lin_dict, quad_dict, mix_dict)
    """
    N = np.asarray(N, dtype=float)
    var = np.asarray(var, dtype=float)

    # Base validity mask
    mask = np.isfinite(N) & np.isfinite(var)

    # Apply optional fit window
    if var_min_N is not None:
        mask &= (N >= float(var_min_N))
    if var_max_N is not None:
        mask &= (N <= float(var_max_N))

    Nf = N[mask]
    vf = var[mask]

    if Nf.size < 3:
        raise ValueError(
            f"Not enough points to fit variance models after filtering: n={Nf.size} "
            f"(var_min_N={var_min_N}, var_max_N={var_max_N})"
        )

    # --------------------
    # Linear: var ≈ a1 * N + c1
    # --------------------
    a1, c1 = np.polyfit(Nf, vf, 1)
    vf_lin = a1 * Nf + c1
    m_lin = _fit_metrics(vf, vf_lin)

    # --------------------
    # Quadratic: var ≈ b2 * N^2 + a2 * N + c2
    # --------------------
    b2, a2, c2 = np.polyfit(Nf, vf, 2)
    vf_quad = b2 * Nf**2 + a2 * Nf + c2
    m_quad = _fit_metrics(vf, vf_quad)

    # --------------------
    # Mixed: var ≈ a * N + b * N^2 + c (least squares)
    # --------------------
    X = np.column_stack([Nf, Nf**2, np.ones_like(Nf)])
    coeffs, *_ = np.linalg.lstsq(X, vf, rcond=None)
    a_m, b_m, c_m = map(float, coeffs)
    vf_mix = a_m * Nf + b_m * Nf**2 + c_m
    m_mix = _fit_metrics(vf, vf_mix)

    fit_window = {
        "min_N": None if var_min_N is None else float(var_min_N),
        "max_N": None if var_max_N is None else float(var_max_N),
        "n_used": int(Nf.size),
    }

    return (
        {"a": float(a1), "b": 0.0,        "c": float(c1), **m_lin,  "fit_window": fit_window},
        {"a": float(a2), "b": float(b2),  "c": float(c2), **m_quad, "fit_window": fit_window},
        {"a": float(a_m), "b": float(b_m),"c": float(c_m), **m_mix, "fit_window": fit_window},
    )


def fit_coherence_models(N, c, c_min=1e-3, max_N=None):
    """
    Fit coherence models to c(N) in log-space and report RSS in both
    c-space and log-space.

        Exponential:  c ≈ A * exp(-α N)
        Gaussian:     c ≈ A * exp(-β N^2)   (β>0 => τ = 1/sqrt(β))
        Mixed:        c ≈ A * exp(-α N - β N^2)

    Only use points where c > c_min and (optionally) N <= max_N.
    """
    N = np.asarray(N, dtype=float)
    c = np.asarray(c, dtype=float)

    # Guard against non-physical values
    c = np.where(c < 0.0, np.nan, c)

    mask = np.isfinite(c) & (c > c_min)
    if max_N is not None:
        mask &= (N <= max_N)

    N_fit = N[mask]
    c_fit = c[mask]

    if N_fit.size < 5:
        return None, None, None  # Not enough data

    ln_c = np.log(c_fit)

    # --- Exponential: ln c = lnA - α N ---
    p1, p0 = np.polyfit(N_fit, ln_c, 1)
    A_exp = float(np.exp(p0))
    alpha_exp = float(-p1)
    c_exp = A_exp * np.exp(-alpha_exp * N_fit)
    rss_exp = float(np.sum((c_fit - c_exp) ** 2))
    rss_exp_log = float(np.sum((ln_c - np.log(np.maximum(c_exp, 1e-300))) ** 2))

    # --- Gaussian: ln c = lnA - β N^2 ---
    N2 = N_fit**2
    p1g, p0g = np.polyfit(N2, ln_c, 1)
    A_gauss = float(np.exp(p0g))
    beta_gauss = float(-p1g)
    tau_gauss = float(1.0 / np.sqrt(beta_gauss)) if beta_gauss > 0 else float("nan")
    c_gauss = A_gauss * np.exp(-beta_gauss * N2)
    rss_gauss = float(np.sum((c_fit - c_gauss) ** 2))
    rss_gauss_log = float(np.sum((ln_c - np.log(np.maximum(c_gauss, 1e-300))) ** 2))

    # --- Mixed: ln c = lnA - α N - β N^2 ---
    X_mix = np.column_stack([N_fit, N_fit**2, np.ones_like(N_fit)])
    coeffs_mix, *_ = np.linalg.lstsq(X_mix, ln_c, rcond=None)
    pN, pN2, pC = map(float, coeffs_mix)  # ln c = pC + pN N + pN2 N^2
    A_mix = float(np.exp(pC))
    alpha_mix = float(-pN)
    beta_mix = float(-pN2)
    c_mix = A_mix * np.exp(-alpha_mix * N_fit - beta_mix * N_fit**2)
    rss_mix = float(np.sum((c_fit - c_mix) ** 2))
    rss_mix_log = float(np.sum((ln_c - np.log(np.maximum(c_mix, 1e-300))) ** 2))

    exp_fit = (A_exp, alpha_exp, rss_exp, rss_exp_log, int(N_fit.size))
    gauss_fit = (A_gauss, tau_gauss, beta_gauss, rss_gauss, rss_gauss_log, int(N_fit.size))
    mixed_fit = (A_mix, alpha_mix, beta_mix, rss_mix, rss_mix_log, int(N_fit.size))

    return exp_fit, gauss_fit, mixed_fit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze phase diffusion CSV produced by phase diffusion scripts"
    )
    parser.add_argument(
        "csv",
        nargs="?",
        default=str(DEFAULT_CSV),
        help=f"CSV file to analyze (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--c-min",
        type=float,
        default=1e-3,
        help="Minimum coherence magnitude to include in fits (default: 1e-3)",
    )
    parser.add_argument(
        "--max-N",
        type=int,
        default=None,
        help="Maximum N_delay_steps to include in coherence fits (default: None = all)",
    )
    parser.add_argument(
        "--write-summary",
        action="store_true",
        help="Write a one-row CSV summary of fit parameters to outputs/analysis/",
    )
    parser.add_argument("--var-min-N", type=int, default=None)
    parser.add_argument("--var-max-N", type=int, default=None)
    args = parser.parse_args()

    csv_path = resolve_input_path(args.csv)
    if not csv_path.exists():
        print(f"[run_diffusion_analysis] ERROR: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[run_diffusion_analysis] ERROR: Failed to read {csv_path}: {e}", file=sys.stderr)
        sys.exit(1)

    required = ["N_delay_steps", "mean_dtheta", "var_dtheta", "coherence_mag"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[run_diffusion_analysis] ERROR: Missing expected columns: {missing}", file=sys.stderr)
        print(f"[run_diffusion_analysis] Found columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    df = df.sort_values("N_delay_steps").reset_index(drop=True)

    N = df["N_delay_steps"].values
    var = df["var_dtheta"].values
    coh = df["coherence_mag"].values

    print(f"[run_diffusion_analysis] Loaded: {csv_path}")
    print(f"[run_diffusion_analysis] Rows: {len(df)}")
    print(f"[run_diffusion_analysis] N range: {np.min(N)} .. {np.max(N)}")
    print()

    # ---------- Variance fits ----------
    print("Variance fits (reporting RSS, RMSE, NRMSE_mean, R^2):")
    lin, quad, mix = fit_variance_models(N, var)

    def _fmt(m: dict[str, float]) -> str:
        return f"RSS={m['rss']:.6g}, RMSE={m['rmse']:.6g}, NRMSE_mean={m['nrmse_mean']:.6g}, R^2={m['r2']:.6g}"

    print(f"  Linear:    Var ≈ {lin['a']:.6g} N + {lin['c']:.6g}, {_fmt(lin)}")
    print(f"  Quadratic: Var ≈ {quad['b']:.6g} N^2 + {quad['a']:.6g} N + {quad['c']:.6g}, {_fmt(quad)}")
    print(f"  Mixed:     Var ≈ {mix['a']:.6g} N + {mix['b']:.6g} N^2 + {mix['c']:.6g}, {_fmt(mix)}")
    print()

    # ---------- Coherence fits ----------
    cond = f"c > {args.c_min:g}" + (f" and N <= {args.max_N}" if args.max_N is not None else "")
    print(f"Coherence fits (using points with {cond}):")

    exp_fit, gauss_fit, mixed_fit = fit_coherence_models(N, coh, c_min=args.c_min, max_N=args.max_N)

    if exp_fit is None:
        print("  Not enough non-tiny coherence points to fit models.")
        return

    A_exp, alpha_exp, rss_exp, rss_exp_log, n_exp = exp_fit
    A_gauss, tau_gauss, beta_gauss, rss_gauss, rss_gauss_log, n_gauss = gauss_fit
    A_mix, alpha_mix, beta_mix, rss_mix, rss_mix_log, n_mix = mixed_fit

    print(
        f"  Exponential: c ≈ {A_exp:.6g} * exp(-{alpha_exp:.6g} N), "
        f"RSS={rss_exp:.6g}, RSS_log={rss_exp_log:.6g} (N_used={n_exp})"
    )
    if np.isfinite(tau_gauss) and beta_gauss > 0:
        print(
            f"  Gaussian:    c ≈ {A_gauss:.6g} * exp(-{beta_gauss:.6g} N^2)  "
            f"(τ≈{tau_gauss:.6g}), RSS={rss_gauss:.6g}, RSS_log={rss_gauss_log:.6g} (N_used={n_gauss})"
        )
    else:
        print(
            f"  Gaussian:    fit yielded β<=0 (τ undefined), "
            f"RSS={rss_gauss:.6g}, RSS_log={rss_gauss_log:.6g} (N_used={n_gauss})"
        )

    beta_note = ""
    if beta_mix < 0:
        beta_note = "  [WARN: β<0; likely fitting into numerical floor—try --c-min 1e-2 --max-N 80]"
    print(
        f"  Mixed:       c ≈ {A_mix:.6g} * exp(-{alpha_mix:.6g} N - {beta_mix:.6g} N^2), "
        f"RSS={rss_mix:.6g}, RSS_log={rss_mix_log:.6g} (N_used={n_mix}){beta_note}"
    )
    print()

    if args.write_summary:
        summary = {
            "input_csv": str(csv_path),
            "c_min": args.c_min,
            "max_N": args.max_N if args.max_N is not None else "",
            "var_lin_a": lin["a"],
            "var_lin_c": lin["c"],
            "var_lin_rss": lin["rss"],
            "var_lin_rmse": lin["rmse"],
            "var_lin_nrmse_mean": lin["nrmse_mean"],
            "var_lin_r2": lin["r2"],

            "var_quad_b": quad["b"],
            "var_quad_a": quad["a"],
            "var_quad_c": quad["c"],
            "var_quad_rss": quad["rss"],
            "var_quad_rmse": quad["rmse"],
            "var_quad_nrmse_mean": quad["nrmse_mean"],
            "var_quad_r2": quad["r2"],

            "var_mix_a": mix["a"],
            "var_mix_b": mix["b"],
            "var_mix_c": mix["c"],
            "var_mix_rss": mix["rss"],
            "var_mix_rmse": mix["rmse"],
            "var_mix_nrmse_mean": mix["nrmse_mean"],
            "var_mix_r2": mix["r2"],

            "coh_exp_A": A_exp,
            "coh_exp_alpha": alpha_exp,
            "coh_exp_rss": rss_exp,
            "coh_exp_rss_log": rss_exp_log,

            "coh_gauss_A": A_gauss,
            "coh_gauss_beta": beta_gauss,
            "coh_gauss_tau": tau_gauss,
            "coh_gauss_rss": rss_gauss,
            "coh_gauss_rss_log": rss_gauss_log,

            "coh_mix_A": A_mix,
            "coh_mix_alpha": alpha_mix,
            "coh_mix_beta": beta_mix,
            "coh_mix_rss": rss_mix,
            "coh_mix_rss_log": rss_mix_log,
        }
        out_path = OUT / (csv_path.stem + "_fit_summary.csv")
        pd.DataFrame([summary]).to_csv(out_path, index=False)
        print(f"[run_diffusion_analysis] Wrote summary: {out_path}")

def analyze_df(df: pd.DataFrame, c_min=1e-3, max_N=None) -> dict:
    required = ["N_delay_steps", "mean_dtheta", "var_dtheta", "coherence_mag"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing} (found {list(df.columns)})")

    df = df.sort_values("N_delay_steps").reset_index(drop=True)

    N = df["N_delay_steps"].values
    var = df["var_dtheta"].values
    coh = df["coherence_mag"].values

    lin, quad, mix = fit_variance_models(N, var, var_min_N=20, var_max_N=120)

    fit_window = lin.get("fit_window") 
    
    lin.pop("fit_window", None)
    quad.pop("fit_window", None)
    mix.pop("fit_window", None)

    exp_fit, gauss_fit, mixed_fit = fit_coherence_models(N, coh, c_min=c_min, max_N=max_N)

    return {
        "N_min": float(np.min(N)),
        "N_max": float(np.max(N)),
        "rows": int(len(df)),
        "variance": {"fit_window": fit_window, "linear": lin, "quadratic": quad, "mixed": mix},
        "coherence": {
            "c_min": float(c_min),
            "max_N": None if max_N is None else int(max_N),
            "exp": None if exp_fit is None else {
                "A": exp_fit[0], "alpha": exp_fit[1], "rss": exp_fit[2], "rss_log": exp_fit[3], "N_used": exp_fit[4]
            },
            "gauss": None if gauss_fit is None else {
                "A": gauss_fit[0], "tau": gauss_fit[1], "beta": gauss_fit[2], "rss": gauss_fit[3], "rss_log": gauss_fit[4], "N_used": gauss_fit[5]
            },
            "mixed": None if mixed_fit is None else {
                "A": mixed_fit[0], "alpha": mixed_fit[1], "beta": mixed_fit[2], "rss": mixed_fit[3], "rss_log": mixed_fit[4], "N_used": mixed_fit[5]
            },
        }
    }

if __name__ == "__main__":
    main()