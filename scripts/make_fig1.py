#!/usr/bin/env python3
"""
make_fig1.py

Generate Figure 1:
Var[Δθ] vs N for pure diffusion (sigma_omega = 0),
with a linear fit and residuals.

Reads:  data/generated/phase_diff_pure_diff_N200.csv
Writes: outputs/figures/fig1_var_vs_N_pure_diffusion.pdf
"""

import matplotlib
# Force non-interactive backend to avoid hangs
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path

# Repo-relative paths (works from any current working directory)
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "generated"
OUT = ROOT / "outputs"
FIGS = OUT / "figures"

DATA.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA / "phase_diff_pure_diff_N200.csv"
OUT_PDF_PATH = FIGS / "fig1_var_vs_N_pure_diffusion.pdf"


def main() -> None:
    print("[make_fig1] Starting figure generation.")
    print(f"[make_fig1] Working directory: {os.getcwd()}")
    print(f"[make_fig1] Input CSV: {CSV_PATH}")
    print(f"[make_fig1] Output PDF: {OUT_PDF_PATH}")

    if not CSV_PATH.exists():
        print(f"[make_fig1] ERROR: CSV file not found: {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"[make_fig1] Loading '{CSV_PATH.name}'...")
    df = pd.read_csv(CSV_PATH)
    print(f"[make_fig1] Columns: {list(df.columns)}")

    # Expect columns: N_delay_steps, mean_dtheta, var_dtheta, coherence_mag
    try:
        N = df["N_delay_steps"].to_numpy(dtype=float)
        var = df["var_dtheta"].to_numpy(dtype=float)
    except KeyError as e:
        print(f"[make_fig1] ERROR: Missing expected column: {e}", file=sys.stderr)
        sys.exit(1)

    # Exclude N=0 for the fit (variance = 0 at N=0 can bias intercept)
    mask = N > 0
    N_fit = N[mask]
    var_fit = var[mask]

    if N_fit.size < 2:
        print("[make_fig1] ERROR: Not enough points with N>0 to fit a line.", file=sys.stderr)
        sys.exit(1)

    print("[make_fig1] Fitting linear model Var ≈ D * N + c ...")
    # Linear fit: var = D * N + c
    D, c = np.polyfit(N_fit, var_fit, deg=1)
    print(f"[make_fig1] Fit result: Var ≈ {D:.6g} * N + {c:.6g}")

    var_pred = D * N_fit + c
    residuals = var_fit - var_pred
    rss = float(np.sum(residuals**2))
    print(f"[make_fig1] Residual sum of squares (RSS) = {rss:.6g}")

    # --- Plotting ---
    print("[make_fig1] Creating plot...")

    fig, (ax_main, ax_resid) = plt.subplots(
        2, 1, figsize=(6.0, 7.0), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
        constrained_layout=True,
    )

    # Main panel: variance + fit
    ax_main.scatter(N, var, s=12, label=r"Simulation $\,\mathrm{Var}[\Delta\theta](N)$")
    # Sort for a clean fitted line (in case CSV row order isn't monotonic in N)
    order = np.argsort(N_fit)
    N_fit_s = N_fit[order]
    var_pred_s = var_pred[order]

    ax_main.plot(
        N_fit_s, var_pred_s, linewidth=1.5,
        label=rf"Linear fit: $\mathrm{{Var}} \approx {D:.3f} N + {c:.3f}$"
    )

    ax_main.set_ylabel(r"$\mathrm{Var}[\Delta\theta]\ \mathrm{(rad^2)}$")
    ax_main.set_title("Phase Diffusion: Pure Intrinsic Variance vs. N")
    ax_main.grid(True, alpha=0.3)
    ax_main.legend(loc="upper left", fontsize=9)

    # Annotation box with D and RSS
    textstr = "\n".join([
        rf"$D \approx {D:.3f}\ \mathrm{{rad^2/step}}$",
        rf"$\mathrm{{RSS}} \approx {rss:.3g}$",
    ])
    ax_main.text(
        0.97, 0.03, textstr,
        transform=ax_main.transAxes,
        fontsize=9,
        va="bottom", ha="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    # Residuals panel
    ax_resid.axhline(0.0, color="black", linewidth=1.0)
    ax_resid.scatter(N_fit, residuals, s=10)
    ax_resid.set_xlabel(r"$N\ \mathrm{(delay\ steps)}$")
    ax_resid.set_ylabel("Residuals")
    ax_resid.grid(True, alpha=0.3)

    print(f"[make_fig1] Saving PDF to '{OUT_PDF_PATH}'...")
    fig.savefig(OUT_PDF_PATH, bbox_inches="tight")
    plt.close(fig)

    print(f"[make_fig1] Done. Wrote '{OUT_PDF_PATH}'.")


if __name__ == "__main__":
    main()