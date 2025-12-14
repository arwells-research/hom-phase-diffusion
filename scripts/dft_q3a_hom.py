#!/usr/bin/env python3
"""
dft_q3a_hom.py

DFT-Q3a: Hong–Ou–Mandel (HOM) toy model in a trajectory + phase framework.

Goal
----
Provide a minimal two-photon, two-path sandbox where:

  - "classical" mode reproduces a flat coincidence level (~0.5 for a 50/50 BS).
  - "dft" mode uses a joint T-frame phase constraint at the source plus a
    phase-dependent beamsplitter rule to reproduce a HOM-like coincidence dip.

Important Notes
---------------
- This is a toy / phenomenological model used to generate reproducible figures and
  CSVs. It is NOT presented as a final microscopic derivation.
- The DFT mechanism implemented here is:
    (i) τ-dependent source coherence controlling the relative phase distribution
    (ii) a BS rule P_same(Δθ) = cos^2(Δθ/2), equivalent to the standard HOM rule
         given a relative phase coordinate.
- Outputs are written under:
    data/generated/  (CSV)
    outputs/figures/ (if you enable plotting + saving)

Inputs
------
None (simulation generates data).

Outputs
-------
CSV with columns:
    tau,coincidence_rate
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Use a non-interactive backend to avoid hangs on headless systems
import matplotlib
matplotlib.use("Agg")  # must be set before importing pyplot
import matplotlib.pyplot as plt


# -----------------------------
# Repo paths
# -----------------------------

def find_repo_root(start: Path) -> Path:
    """
    Find repo root by searching upward for an expected marker directory.
    This makes scripts runnable from anywhere inside the repo.
    """
    start = start.resolve()
    for p in [start.parent, *start.parents]:
        if (p / "data" / "generated").exists() or (p / "data").exists():
            return p
    return start.parent


HERE = Path(__file__).resolve()
ROOT = find_repo_root(HERE)
DATA = ROOT / "data" / "generated"
OUT = ROOT / "outputs"
FIGS = OUT / "figures"

# We create output dirs; we do NOT create the input contract dirs silently.
DATA.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Config and state definitions
# -----------------------------

@dataclass
class HOMConfig:
    n_pairs: int = 200_000
    omega: float = 1.0
    tau_min: float = -5.0
    tau_max: float = 5.0
    tau_steps: int = 41
    sigma_phase_noise: float = 0.0
    seed: int = 12345
    mode: str = "dft"
    out_csv: Path = DATA / "hom_results.csv"
    no_plot: bool = False
    tau_c: float = 1.5  # coherence time scale for source constraint (dip width)


@dataclass
class TwoPhotonState:
    theta_A: float
    theta_B: float
    out_A: Optional[int] = None  # 1 or 2 (detector ports)
    out_B: Optional[int] = None  # 1 or 2


# -----------------------------
# Source: pair generation
# -----------------------------

def coherence_factor_gaussian(tau: float, tau_c: float) -> float:
    """
    Coherence factor c(τ) in [0,1]. Here: Gaussian in τ.

        c(τ) = exp(-(τ/τ_c)^2)

    If tau_c <= 0, treat as incoherent (c=0).
    """
    if tau_c <= 0.0:
        return 0.0
    return math.exp(- (tau / tau_c) ** 2)


def generate_pair_classical(cfg: HOMConfig, rng: np.random.Generator) -> TwoPhotonState:
    """
    Classical baseline: no joint phase constraint.
    """
    theta_A = rng.uniform(0.0, 2.0 * math.pi)
    theta_B = rng.uniform(0.0, 2.0 * math.pi)
    return TwoPhotonState(theta_A=theta_A, theta_B=theta_B)


def generate_pair_dft(cfg: HOMConfig, rng: np.random.Generator, tau: float) -> TwoPhotonState:
    """
    DFT-style source with τ-dependent coherence.

    Implemented as a mixture:
      - c(τ) = exp(-(τ/τ_c)^2) (coherence factor)
      - With probability c(τ): correlated phases
          θ_B ≈ θ_A0 + ω τ (+ noise)
      - With probability 1-c(τ): θ_B is independent random
    """
    theta_A0 = rng.uniform(0.0, 2.0 * math.pi)

    # Local phase noise on A (optional)
    if cfg.sigma_phase_noise > 0.0:
        theta_A = (theta_A0 + rng.normal(0.0, cfg.sigma_phase_noise)) % (2.0 * math.pi)
    else:
        theta_A = theta_A0

    c_tau = coherence_factor_gaussian(tau, cfg.tau_c)

    if rng.random() < c_tau:
        # Coherent / correlated branch
        delta_phi = cfg.omega * tau
        if cfg.sigma_phase_noise > 0.0:
            theta_B = (theta_A0 + delta_phi + rng.normal(0.0, cfg.sigma_phase_noise)) % (2.0 * math.pi)
        else:
            theta_B = (theta_A0 + delta_phi) % (2.0 * math.pi)
    else:
        # Incoherent branch: B has independent random phase
        theta_B0 = rng.uniform(0.0, 2.0 * math.pi)
        if cfg.sigma_phase_noise > 0.0:
            theta_B = (theta_B0 + rng.normal(0.0, cfg.sigma_phase_noise)) % (2.0 * math.pi)
        else:
            theta_B = theta_B0

    return TwoPhotonState(theta_A=theta_A, theta_B=theta_B)


# -----------------------------
# Beamsplitter interaction
# -----------------------------

def beamsplitter_classical(state: TwoPhotonState, rng: np.random.Generator) -> None:
    """
    Classical 50/50 beamsplitter:
      Each photon independently exits port 1 or 2 with probability 1/2.
    """
    state.out_A = 1 if rng.random() < 0.5 else 2
    state.out_B = 1 if rng.random() < 0.5 else 2


def beamsplitter_dft_pair(state: TwoPhotonState, rng: np.random.Generator) -> None:
    """
    DFT-inspired pair-level beamsplitter rule.

      P_same(Δθ) = cos^2(Δθ/2)
      P_split     = 1 - P_same

    Implementation:
      - "Same": both photons go to the same random output (1 or 2)
      - "Split": one to port 1, one to port 2 (random assignment to avoid bias)
    """
    dtheta = (state.theta_A - state.theta_B) % (2.0 * math.pi)
    if dtheta > math.pi:
        dtheta -= 2.0 * math.pi

    p_same = math.cos(0.5 * dtheta) ** 2
    p_same = max(0.0, min(1.0, p_same))

    if rng.random() < p_same:
        port = 1 if rng.random() < 0.5 else 2
        state.out_A = port
        state.out_B = port
    else:
        # split: randomize assignment to avoid subtle port bias
        if rng.random() < 0.5:
            state.out_A, state.out_B = 1, 2
        else:
            state.out_A, state.out_B = 2, 1


# -----------------------------
# Simulation core
# -----------------------------

def simulate_hom(cfg: HOMConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scan over τ and compute coincidence rates.

    Coincidence definition:
      one photon at port 1 and the other at port 2  => out_A != out_B
    """
    rng = np.random.default_rng(cfg.seed)

    tau_values = np.linspace(cfg.tau_min, cfg.tau_max, cfg.tau_steps, dtype=float)
    coincidence_rates = np.zeros_like(tau_values, dtype=float)

    for i, tau in enumerate(tau_values):
        coinc = 0
        for _ in range(cfg.n_pairs):
            if cfg.mode == "classical":
                state = generate_pair_classical(cfg, rng)
                beamsplitter_classical(state, rng)
            else:
                state = generate_pair_dft(cfg, rng, tau)
                beamsplitter_dft_pair(state, rng)

            if state.out_A != state.out_B:
                coinc += 1

        coincidence_rates[i] = coinc / float(cfg.n_pairs)

    return tau_values, coincidence_rates


# -----------------------------
# QM reference curve (optional)
# -----------------------------

def hom_qm_reference(
    tau: np.ndarray,
    tau_c: float = 1.0,
    visibility: float = 1.0,
    classical_level: float = 0.5,
) -> np.ndarray:
    """
    Simple QM-inspired HOM reference curve:

      P_coinc(τ) = classical_level * [1 - V * exp(-τ^2 / τ_c^2)]

    This is not used in the simulation; it's just for overlay/diagnostics.
    """
    tau = np.asarray(tau, dtype=float)
    if tau_c <= 0.0:
        return np.full_like(tau, classical_level, dtype=float)
    return classical_level * (1.0 - visibility * np.exp(-(tau ** 2) / (tau_c ** 2)))


# -----------------------------
# CSV and plotting
# -----------------------------

def save_results(cfg: HOMConfig, tau_values: np.ndarray, coincidence_rates: np.ndarray) -> None:
    cfg.out_csv.parent.mkdir(parents=True, exist_ok=True)

    data = np.column_stack([tau_values, coincidence_rates])
    header = "tau,coincidence_rate"
    np.savetxt(cfg.out_csv, data, delimiter=",", header=header, comments="")
    print(f"[dft_q3a_hom] Saved HOM results to {cfg.out_csv}")


def plot_results(cfg: HOMConfig, tau_values: np.ndarray, coincidence_rates: np.ndarray) -> None:
    """
    Save a PDF figure under outputs/figures instead of calling plt.show().
    """
    plt.figure(figsize=(7.0, 5.0))
    plt.plot(tau_values, coincidence_rates, "o-", label=f"{cfg.mode} simulation")

    # Overlay a QM-style reference using cfg.tau_c by default (keeps one knob)
    qm_ref = hom_qm_reference(
        tau_values,
        tau_c=cfg.tau_c,
        visibility=1.0,
        classical_level=0.5,
    )
    plt.plot(tau_values, qm_ref, "--", label="QM-like reference (toy)")

    plt.xlabel("Delay τ (arbitrary units)")
    plt.ylabel("Coincidence rate P_coinc")
    plt.title(f"DFT-Q3a HOM toy ({cfg.mode} mode)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_pdf = FIGS / "hom_toy_curve.pdf"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"[dft_q3a_hom] Saved plot to {out_pdf}")


# -----------------------------
# CLI / main
# -----------------------------

def parse_args() -> HOMConfig:
    p = argparse.ArgumentParser(
        description="DFT-Q3a: HOM toy model in trajectory + phase framework"
    )
    p.add_argument("--mode", choices=["classical", "dft"], default="dft",
                   help="Simulation mode: classical baseline or DFT joint-phase model")
    p.add_argument("--pairs", type=int, default=200_000,
                   help="Number of photon pairs per delay value")
    p.add_argument("--tau-min", type=float, default=-5.0, help="Minimum delay τ")
    p.add_argument("--tau-max", type=float, default=5.0, help="Maximum delay τ")
    p.add_argument("--tau-steps", type=int, default=41,
                   help="Number of τ samples between tau-min and tau-max")
    p.add_argument("--omega", type=float, default=1.0,
                   help="Effective frequency: Δθ = ω τ (correlated branch)")
    p.add_argument("--phase-noise", type=float, default=0.0,
                   help="Stddev of additional Gaussian phase noise (radians)")
    p.add_argument("--seed", type=int, default=12345, help="Random seed")
    p.add_argument("--out-csv", type=str, default=str(DATA / "hom_results.csv"),
                   help="Output CSV path (default: data/generated/hom_results.csv)")
    p.add_argument("--no-plot", action="store_true",
                   help="Disable plotting (only CSV output)")
    p.add_argument("--tau-c", type=float, default=1.5,
                   help="Coherence time scale τ_c for DFT joint phase (HOM width)")

    args = p.parse_args()

    return HOMConfig(
        n_pairs=args.pairs,
        omega=args.omega,
        tau_min=args.tau_min,
        tau_max=args.tau_max,
        tau_steps=args.tau_steps,
        sigma_phase_noise=args.phase_noise,
        seed=args.seed,
        mode=args.mode,
        out_csv=Path(args.out_csv),
        no_plot=args.no_plot,
        tau_c=args.tau_c,
    )


def main() -> None:
    cfg = parse_args()

    print(f"[dft_q3a_hom] Repo root: {ROOT}")
    print(f"[dft_q3a_hom] Mode={cfg.mode!r}, pairs/τ={cfg.n_pairs}, "
          f"τ∈[{cfg.tau_min}, {cfg.tau_max}] with {cfg.tau_steps} steps, "
          f"tau_c={cfg.tau_c}, omega={cfg.omega}, phase_noise={cfg.sigma_phase_noise}")

    tau_values, coincidence_rates = simulate_hom(cfg)

    # Basic diagnostics
    mid = len(tau_values) // 2
    for idx in (0, mid, len(tau_values) - 1):
        print(f"[dft_q3a_hom]  τ={tau_values[idx]: .3f}, P_coinc={coincidence_rates[idx]: .6f}")

    save_results(cfg, tau_values, coincidence_rates)

    if not cfg.no_plot:
        plot_results(cfg, tau_values, coincidence_rates)


if __name__ == "__main__":
    main()