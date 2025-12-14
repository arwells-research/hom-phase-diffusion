#!/usr/bin/env python3
"""
make_fig3.py

Generate Figure 3:
Coherence envelope c(N) on a semi-log plot for several phase-diffusion
bandwidths (sigma_omega).

Reads (from data/generated/):
  - phase_diff_pure_diff_N200.csv
  - phase_diff_omega_sigma_0p05_N200.csv
  - phase_diff_omega_sigma_0p1_N50.csv
  - phase_diff_omega_sigma_0p2_N200.csv

Writes (to outputs/figures/):
  - fig3_coherence_envelope.pdf

Expected CSV columns:
  - N_delay_steps
  - coherence_mag

Notes:
  - Semi-log plotting requires positive values; we drop very small values
    (c <= 1e-3) to avoid log underflow / clutter near numerical floor.
"""

import matplotlib
# Force non-interactive backend to avoid hangs (CI/headless)
matplotlib.use("Agg")

import matplotlib.pyplot as plt
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

OUT_PDF = FIGS / "fig3_coherence_envelope.pdf"

# Numerical floor for semi-log plot readability
MIN_C = 1e-3


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[make_fig3] ERROR: Missing CSV: {path}", file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(path)


def main() -> None:
    print("[make_fig3] Starting figure generation.")
    print(f"[make_fig3] Repo root: {ROOT}")
    print(f"[make_fig3] Data dir : {DATA}")
    print(f"[make_fig3] Output  : {OUT_PDF}")

    plt.figure(figsize=(7.5, 5.0))

    for fname, sigma in FILES:
        csv_path = DATA / fname
        print(f"[make_fig3] Loading {csv_path.name} (sigma_omega={sigma}) ...")
        df = _read_csv(csv_path)

        try:
            N = df["N_delay_steps"].to_numpy(dtype=float)
            c = df["coherence_mag"].to_numpy(dtype=float)
        except KeyError as e:
            print(f"[make_fig3] ERROR: {csv_path.name} missing column {e}", file=sys.stderr)
            sys.exit(1)

        # Semi-log plot: keep only values above a small floor
        mask = (c > MIN_C) & (c > 0.0)
        # also drop NaNs / infs defensively
        mask &= (pd.notnull(c)) & (pd.notnull(N))
        Np = N[mask]
        cp = c[mask]

        if len(Np) == 0:
            print(f"[make_fig3] WARNING: {fname} has no coherence_mag > {MIN_C}. Skipping.")
            continue

        # Ensure monotone x for clean lines
        order = Np.argsort()
        plt.semilogy(Np[order], cp[order], label=rf"$\sigma_\omega={sigma}$")        

    plt.xlabel(r"$N$")
    plt.ylabel(r"$c(N)$")
    plt.title("Coherence Envelope (Semi-log Plot)")
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    print(f"[make_fig3] Saving {OUT_PDF.name} ...")
    plt.savefig(OUT_PDF, bbox_inches="tight")
    plt.close()

    print("[make_fig3] Done.")


if __name__ == "__main__":
    main()