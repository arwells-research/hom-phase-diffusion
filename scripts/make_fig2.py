#!/usr/bin/env python3
"""
make_fig2.py

Generate Figure 2 (two panels/files):

1) Variance vs N for several phase-diffusion bandwidths (sigma_omega),
   with a least-squares quadratic fit:
       Var[Δθ](N) ≈ a N + b N^2 + c

   Reads (from data/generated/):
     - phase_diff_pure_diff_N200.csv
     - phase_diff_omega_sigma_0p05_N200.csv
     - phase_diff_omega_sigma_0p1_N50.csv
     - phase_diff_omega_sigma_0p2_N200.csv

   Writes (to outputs/figures/):
     - fig2_var_vs_N_bandwidths.pdf

2) Quadratic coefficient b vs sigma_omega^2

   Writes (to outputs/figures/):
     - fig2_b_vs_sigma2.pdf

Expected CSV columns:
  - N_delay_steps
  - var_dtheta
(Other columns are ignored.)
"""

import matplotlib
# Force non-interactive backend to avoid hangs (CI/headless)
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]     # repo root (…/hom-phase-diffusion)
DATA = ROOT / "data" / "generated"
OUT  = ROOT / "outputs"
FIGS = OUT / "figures"

DATA.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

FIGS.mkdir(parents=True, exist_ok=True)

FILES = [
    ("phase_diff_pure_diff_N200.csv", 0.0),
    ("phase_diff_omega_sigma_0p05_N200.csv", 0.05),
    ("phase_diff_omega_sigma_0p1_N50.csv", 0.10),
    ("phase_diff_omega_sigma_0p2_N200.csv", 0.20),
]

OUT_PDF_MAIN = FIGS / "fig2_var_vs_N_bandwidths.pdf"
OUT_PDF_INSET = FIGS / "fig2_b_vs_sigma2.pdf"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[make_fig2] ERROR: Missing CSV: {path}", file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(path)


def main() -> None:
    print("[make_fig2] Starting figure generation.")
    print(f"[make_fig2] Repo root: {ROOT}")
    print(f"[make_fig2] Data dir : {DATA}")
    print(f"[make_fig2] Output dir: {FIGS}")

    # ----------------------------
    # Main plot: Var vs N (multiple bandwidths)
    # ----------------------------
    plt.figure(figsize=(7.5, 5.0))

    b_values: list[float] = []
    sigma2: list[float] = []

    for fname, sigma in FILES:
        csv_path = DATA / fname
        print(f"[make_fig2] Loading {csv_path.name} (sigma_omega={sigma}) ...")
        df = _read_csv(csv_path)

        # Expect columns: N_delay_steps, var_dtheta
        try:
            N = df["N_delay_steps"].to_numpy(dtype=float)
            var = df["var_dtheta"].to_numpy(dtype=float)
        except KeyError as e:
            print(f"[make_fig2] ERROR: {csv_path.name} missing column {e}", file=sys.stderr)
            sys.exit(1)

        # Fit var = a*N + b*N^2 + c
        A = np.column_stack([N, N**2, np.ones_like(N)])
        x, *_ = np.linalg.lstsq(A, var, rcond=None)
        a, b, c = (float(x[0]), float(x[1]), float(x[2]))

        b_values.append(b)
        sigma2.append(sigma**2)

        # Scatter + fitted curve
        plt.scatter(N, var, s=10, label=rf"$\sigma_\omega={sigma}$")
        N_sorted = np.sort(N)
        plt.plot(N_sorted, a*N_sorted + b*N_sorted**2 + c, linewidth=1.2)

        print(f"[make_fig2]  Fit for {fname}: a={a:.6g}, b={b:.6g}, c={c:.6g}")

    plt.xlabel(r"$N$")
    plt.ylabel(r"$\mathrm{Var}[\Delta\theta]$")
    plt.title("Variance vs N for Various Bandwidths")
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    print(f"[make_fig2] Saving {OUT_PDF_MAIN.name} ...")
    plt.savefig(OUT_PDF_MAIN, bbox_inches="tight")
    plt.close()

    # ----------------------------
    # Inset plot: b vs sigma^2
    # ----------------------------
    # Sort for a clean connected line (avoids accidental reordering artifacts)
    order = np.argsort(np.asarray(sigma2, dtype=float))
    sigma2_s = np.asarray(sigma2, dtype=float)[order]
    b_values_s = np.asarray(b_values, dtype=float)[order]    
    plt.figure(figsize=(5.0, 4.0))
    plt.scatter(sigma2_s, b_values_s)
    plt.plot(sigma2_s, b_values_s, linewidth=1.2)
    plt.xlabel(r"$\sigma_\omega^2$")
    plt.ylabel(r"Fitted quadratic coefficient $b$")
    plt.title(r"Quadratic coefficient $b$ vs $\sigma_\omega^2$")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    print(f"[make_fig2] Saving {OUT_PDF_INSET.name} ...")
    plt.savefig(OUT_PDF_INSET, bbox_inches="tight")
    plt.close()

    print("[make_fig2] Done.")


if __name__ == "__main__":
    main()