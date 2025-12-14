#!/usr/bin/env python3
"""
make_hom_data.py

Generate synthetic Hong–Ou–Mandel (HOM) coincidence data for Figure 4.

Outputs (to data/generated/):
  - hom_qm.csv  : quantum reference coincidence curve
  - hom_dft.csv : DFT-style synthetic curve (QM curve + small noise)

Both curves use a mixed coherence envelope:

    c(N) = exp(-a |N| - b N^2)

and the standard HOM coincidence form:

    P_coinc(N) = 0.5 * (1 - c(N))

The “DFT” curve is the same functional form with small, optional noise to mimic
finite-sample Monte Carlo variability. You can replace hom_dft.csv later with
true simulation output if desired.

CSV format (both files):
    delay_steps, p_coinc

Reproducibility:
  - This script can write a JSON sidecar file (default: on) capturing all parameters.
  - All paths are repo-relative; you can run from any working directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------
# Paths (repo-relative)
# ------------------------

ROOT = Path(__file__).resolve().parents[1]  # repo root (…/hom-phase-diffusion)
DATA_DIR = ROOT / "data" / "generated"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_QM_OUT = DATA_DIR / "hom_qm.csv"
DEFAULT_DFT_OUT = DATA_DIR / "hom_dft.csv"
DEFAULT_META_OUT = DATA_DIR / "hom_synthetic_params.json"


def coherence_envelope(N: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Mixed envelope: c(N) = exp(-a |N| - b N^2).
    """
    N = np.asarray(N, dtype=float)
    return np.exp(-a * np.abs(N) - b * N**2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate synthetic HOM coincidence CSVs (QM reference + optional noisy DFT curve)."
    )

    # Delay axis
    p.add_argument("--n-max", type=int, default=50,
                   help="Max |N| (delay_steps). Produces 2*n_max+1 samples (default: 50).")
    p.add_argument("--n-step", type=int, default=1,
                   help="Step size in delay_steps (default: 1).")

    # Envelope parameters
    p.add_argument("--a-env", type=float, default=0.07,
                   help="Envelope parameter a in exp(-a|N| - bN^2) (default: 0.07).")
    p.add_argument("--b-env", type=float, default=1.0 / (32.0**2),
                   help="Envelope parameter b in exp(-a|N| - bN^2) (default: 1/(32^2)).")

    # Noise model (applied to DFT curve only)
    p.add_argument("--noise-std", type=float, default=0.001,
                   help="Base noise stddev for synthetic DFT curve (default: 0.001). Use 0 for none.")
    p.add_argument("--gamma-noise", type=float, default=1.0,
                   help="Noise scaling exponent: std(N)=noise_std*c(N)^gamma (default: 1.0).")
    p.add_argument("--seed", type=int, default=12345,
                   help="RNG seed for reproducible noise (default: 12345).")

    # Outputs
    p.add_argument("--qm-out", type=str, default=str(DEFAULT_QM_OUT),
                   help=f"Output CSV for QM reference (default: {DEFAULT_QM_OUT}).")
    p.add_argument("--dft-out", type=str, default=str(DEFAULT_DFT_OUT),
                   help=f"Output CSV for synthetic DFT curve (default: {DEFAULT_DFT_OUT}).")

    # Metadata sidecar
    p.add_argument("--write-meta", action="store_true", default=True,
                   help="Write a JSON sidecar with all parameters (default: enabled).")
    p.add_argument("--no-write-meta", dest="write_meta", action="store_false",
                   help="Disable writing metadata JSON.")
    p.add_argument("--meta-out", type=str, default=str(DEFAULT_META_OUT),
                   help=f"Metadata JSON path (default: {DEFAULT_META_OUT}).")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve outputs (allow users to pass relative paths)
    qm_out = Path(args.qm_out).expanduser()
    dft_out = Path(args.dft_out).expanduser()
    meta_out = Path(args.meta_out).expanduser()

    # If relative, interpret relative to repo root (consistent behavior)
    if not qm_out.is_absolute():
        qm_out = ROOT / qm_out
    if not dft_out.is_absolute():
        dft_out = ROOT / dft_out
    if not meta_out.is_absolute():
        meta_out = ROOT / meta_out

    # Ensure parent dirs exist
    qm_out.parent.mkdir(parents=True, exist_ok=True)
    dft_out.parent.mkdir(parents=True, exist_ok=True)
    if args.write_meta:
        meta_out.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------
    # Sanity checks
    # ------------------------
    if args.n_step <= 0:
        print("[make_hom_data] ERROR: --n-step must be > 0", file=sys.stderr)
        sys.exit(1)
    if args.n_max < 0:
        print("[make_hom_data] ERROR: --n-max must be >= 0", file=sys.stderr)
        sys.exit(1)
    if args.a_env < 0.0 or args.b_env < 0.0:
        print("[make_hom_data] ERROR: --a-env and --b-env must be >= 0", file=sys.stderr)
        sys.exit(1)
    if args.noise_std < 0.0 or args.gamma_noise < 0.0:
        print("[make_hom_data] ERROR: --noise-std and --gamma-noise must be >= 0", file=sys.stderr)
        sys.exit(1)

    print("[make_hom_data] Generating HOM synthetic data")
    print(f"[make_hom_data] Repo root : {ROOT}")
    print(f"[make_hom_data] QM out    : {qm_out}")
    print(f"[make_hom_data] DFT out   : {dft_out}")
    if args.write_meta:
        print(f"[make_hom_data] Meta out  : {meta_out}")

    # ------------------------
    # Build delay axis
    # ------------------------
    delays = np.arange(-args.n_max, args.n_max + 1, args.n_step, dtype=float)
    print(f"[make_hom_data] Delay samples: {len(delays)} (from {-args.n_max} to {args.n_max} step {args.n_step})")

    # Envelope and curves
    c = coherence_envelope(delays, args.a_env, args.b_env)
    p_qm = 0.5 * (1.0 - c)

    rng = np.random.default_rng(seed=int(args.seed))
    if args.noise_std > 0.0:
        local_noise_std = float(args.noise_std) * (c ** float(args.gamma_noise))
        noise = rng.normal(loc=0.0, scale=local_noise_std, size=delays.shape)
        p_dft = p_qm + noise
    else:
        p_dft = p_qm.copy()

    p_dft = np.clip(p_dft, 0.0, 1.0)

    # Write CSVs
    df_qm = pd.DataFrame({"delay_steps": delays, "p_coinc": p_qm})
    df_dft = pd.DataFrame({"delay_steps": delays, "p_coinc": p_dft})

    df_qm.to_csv(qm_out, index=False)
    df_dft.to_csv(dft_out, index=False)

    print(f"[make_hom_data] Wrote {qm_out}  ({len(df_qm)} rows)")
    print(f"[make_hom_data] Wrote {dft_out} ({len(df_dft)} rows)")

    # Metadata sidecar (optional, recommended)
    if args.write_meta:
        meta = {
            "script": "make_hom_data.py",
            "repo_root": str(ROOT),
            "model": {
                "coherence_envelope": "c(N)=exp(-a|N|-bN^2)",
                "coincidence": "P_coinc(N)=0.5*(1-c(N))",
                "dft_curve": "p_dft = p_qm + Normal(0, noise_std*c(N)^gamma)",
                "clamp": "[0,1]",
            },
            "params": {
                "n_max": int(args.n_max),
                "n_step": int(args.n_step),
                "a_env": float(args.a_env),
                "b_env": float(args.b_env),
                "noise_std": float(args.noise_std),
                "gamma_noise": float(args.gamma_noise),
                "seed": int(args.seed),
            },
            "outputs": {
                "qm_csv": str(qm_out),
                "dft_csv": str(dft_out),
            },
        }
        meta_out.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"[make_hom_data] Wrote {meta_out}")

    print("[make_hom_data] Done.")


if __name__ == "__main__":
    main()