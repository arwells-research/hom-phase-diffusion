#!/usr/bin/env python3
"""
make_fig4.py

Generate Figure 4:
Hong–Ou–Mandel (HOM) interference comparison plot: DFT simulation vs QM reference,
plus the classical limit.

Reads (from data/generated/):
  - hom_dft.csv
  - hom_qm.csv

Writes (to outputs/figures/):
  - fig4_hom_comparison.pdf

Expected CSV columns:
  - delay_steps
  - p_coinc
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

DFT_CSV = DATA / "hom_dft.csv"
QM_CSV  = DATA / "hom_qm.csv"
OUT_PDF = FIGS / "fig4_hom_comparison.pdf"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[make_fig4] ERROR: Missing CSV: {path}", file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(path)


def main() -> None:
    print("[make_fig4] Starting figure generation.")
    print(f"[make_fig4] Repo root: {ROOT}")
    print(f"[make_fig4] Data dir : {DATA}")
    print(f"[make_fig4] Output  : {OUT_PDF}")

    df_dft = _read_csv(DFT_CSV)
    df_qm  = _read_csv(QM_CSV)

    try:
        tau_d = df_dft["delay_steps"].to_numpy(dtype=float)
        P_d   = df_dft["p_coinc"].to_numpy(dtype=float)

        tau_q = df_qm["delay_steps"].to_numpy(dtype=float)
        P_q   = df_qm["p_coinc"].to_numpy(dtype=float)
    except KeyError as e:
        print(f"[make_fig4] ERROR: missing expected column {e}", file=sys.stderr)
        sys.exit(1)

    # Defensive cleanup: drop NaN/inf and sort by delay for clean lines
    def _clean_xy(x, y):
        m = (~pd.isnull(x)) & (~pd.isnull(y))
        x = x[m]
        y = y[m]
        # drop non-finite
        import numpy as np
        m2 = np.isfinite(x) & np.isfinite(y)
        x = x[m2]
        y = y[m2]
        order = x.argsort()
        return x[order], y[order]

    tau_d, P_d = _clean_xy(tau_d, P_d)
    tau_q, P_q = _clean_xy(tau_q, P_q)

    if tau_d.size == 0 or tau_q.size == 0:
        print("[make_fig4] ERROR: No valid numeric rows after cleaning.", file=sys.stderr)
        sys.exit(1)

    plt.figure(figsize=(7.0, 5.0))

    # QM reference: solid
    line_qm, = plt.plot(tau_q, P_q, "-", label="QM reference")

    # DFT simulation: dashed
    line_dft, = plt.plot(tau_d, P_d, "--", label="DFT simulation")

    # Classical limit: horizontal dotted
    line_classical = plt.axhline(
        0.5,
        linestyle=":",
        label="classical limit",
    )

    plt.xlabel(r"Delay $\tau$")
    plt.ylabel("Coincidence Probability")
    plt.title("Hong–Ou–Mandel Interference: DFT vs QM")

    # Use explicit handles so legend styles match the plotted lines
    plt.legend(handles=[line_qm, line_dft, line_classical], fontsize=9)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    print(f"[make_fig4] Saving {OUT_PDF.name} ...")
    plt.savefig(OUT_PDF, bbox_inches="tight")
    plt.close()

    print("[make_fig4] Done.")


if __name__ == "__main__":
    main()