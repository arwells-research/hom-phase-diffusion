#!/usr/bin/env python3
"""
DFT-Q3a: Hong–Ou–Mandel toy model in a trajectory + phase framework.

Goal:
  - Provide a minimal two-photon, two-path setup where:
      * "Classical" mode reproduces flat coincidence rate (~0.25).
      * "DFT" mode uses joint T-frame phase constraints and a
        phase-dependent beamsplitter rule to reproduce a HOM-like dip.

This is a toy, NOT a final derivation. The key is to have a clean
experimental sandbox to test DFT assumptions about:
  - joint phase constraints at the source
  - beamsplitter as a phase-conditioning interface
  - how coincidence statistics depend on relative phase Δθ and delay τ
"""

import math
import argparse
import random
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


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
    out_csv: str = "hom_results.csv"
    no_plot: bool = False
    tau_c: float = 1.5   # coherence time scale for joint phase (NEW)


@dataclass
class TwoPhotonState:
    theta_A: float
    theta_B: float
    out_A: Optional[int] = None  # 1 or 2 (detector ports)
    out_B: Optional[int] = None  # 1 or 2


# -----------------------------
# Source: pair generation
# -----------------------------

def generate_pair_classical(cfg: HOMConfig,
                            rng: random.Random,
                            tau: float) -> TwoPhotonState:
    """
    Classical baseline: no joint phase constraint.
    Phases are independent and irrelevant; we include them just for symmetry.
    """
    theta_A = rng.uniform(0.0, 2.0 * math.pi)
    theta_B = rng.uniform(0.0, 2.0 * math.pi)
    return TwoPhotonState(theta_A=theta_A, theta_B=theta_B)


def generate_pair_dft(cfg: HOMConfig,
                      rng: random.Random,
                      tau: float) -> TwoPhotonState:
    """
    DFT-style source with τ-dependent coherence.

    Idea (Option C):
      - For small |τ|: strong joint phase constraint, θ_B ≈ θ_A + ω τ (up to noise).
      - For large |τ|: constraint washed out; θ_B becomes effectively independent.

    Implemented as a mixture:
      - c(τ) = exp(-(τ/τ_c)^2)  (coherence factor)
      - With probability c(τ): correlated phases.
      - With probability 1 - c(τ): independent phase for B.
    """
    theta_A0 = rng.uniform(0.0, 2.0 * math.pi)

    # Local phase noise on A
    noise_A = rng.gauss(0.0, cfg.sigma_phase_noise)
    theta_A = (theta_A0 + noise_A) % (2.0 * math.pi)

    # Coherence factor as a function of delay
    if cfg.tau_c > 0.0:
        c_tau = math.exp(-(tau / cfg.tau_c) ** 2)
    else:
        c_tau = 0.0  # if tau_c <= 0, treat as fully incoherent

    if rng.random() < c_tau:
        # Coherent / correlated branch
        delta_phi = cfg.omega * tau
        noise_B = rng.gauss(0.0, cfg.sigma_phase_noise)
        theta_B = (theta_A0 + delta_phi + noise_B) % (2.0 * math.pi)
    else:
        # Incoherent branch: B has independent random phase
        theta_B0 = rng.uniform(0.0, 2.0 * math.pi)
        noise_B = rng.gauss(0.0, cfg.sigma_phase_noise)
        theta_B = (theta_B0 + noise_B) % (2.0 * math.pi)

    return TwoPhotonState(theta_A=theta_A, theta_B=theta_B)


# -----------------------------
# Beamsplitter interaction
# -----------------------------

def beamsplitter_classical(state: TwoPhotonState,
                           cfg: HOMConfig,
                           rng: random.Random) -> None:
    """
    Classical 50/50 beamsplitter:
      - Each photon independently has 50% chance to go to D1 or D2.
      - Phases play no role.
    """
    state.out_A = 1 if rng.random() < 0.5 else 2
    state.out_B = 1 if rng.random() < 0.5 else 2


def beamsplitter_dft_pair(state: TwoPhotonState,
                          cfg: HOMConfig,
                          rng: random.Random) -> None:
    """
    DFT-inspired pair-level beamsplitter rule.

    Idea:
      - The beamsplitter is a T-frame phase-conditioning interface that
        "sees" the relative phase Δθ between the two incoming trajectories.
      - When Δθ ≈ 0, they are indistinguishable and bunch:
          P_same ≈ 1, P_split ≈ 0
      - When Δθ is random (large |τ|, noise), P_same → 0.5, P_split → 0.5

    Minimal rule (HOM-like):
      P_same(Δθ)  = cos^2(Δθ/2)
      P_split     = 1 - P_same

    Implementation:
      - "Same": both photons go to the same random output (1 or 2).
      - "Split": one to D1, one to D2.
    """
    dtheta = (state.theta_A - state.theta_B) % (2.0 * math.pi)

    # Map Δθ into [-π, π] for symmetry
    if dtheta > math.pi:
        dtheta -= 2.0 * math.pi

    p_same = math.cos(0.5 * dtheta) ** 2
    # Clamp for numerical safety
    p_same = max(0.0, min(1.0, p_same))

    if rng.random() < p_same:
        # both exit same port
        port = 1 if rng.random() < 0.5 else 2
        state.out_A = port
        state.out_B = port
    else:
        # split across detectors
        # (assignment of who goes where doesn't matter statistically)
        state.out_A = 1
        state.out_B = 2


# -----------------------------
# Simulation core
# -----------------------------

def simulate_hom(cfg: HOMConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scan over τ and compute coincidence rates.

    For each τ:
      - Generate cfg.n_pairs two-photon states from the chosen source model.
      - Pass each pair through the appropriate beamsplitter rule.
      - Count how often one photon ends at D1 and the other at D2.

    Returns:
      tau_values, coincidence_rates
    """
    rng = random.Random(cfg.seed)

    tau_values = np.linspace(cfg.tau_min, cfg.tau_max, cfg.tau_steps)
    coincidence_rates = np.zeros_like(tau_values, dtype=float)

    for i, tau in enumerate(tau_values):
        coinc = 0
        for _ in range(cfg.n_pairs):
            if cfg.mode == "classical":
                state = generate_pair_classical(cfg, rng, tau)
                beamsplitter_classical(state, cfg, rng)
            else:
                state = generate_pair_dft(cfg, rng, tau)
                beamsplitter_dft_pair(state, cfg, rng)

            # Coincidence: one at D1, one at D2
            if state.out_A is not None and state.out_B is not None:
                if state.out_A != state.out_B:
                    coinc += 1

        coincidence_rates[i] = coinc / float(cfg.n_pairs)

    return tau_values, coincidence_rates


# -----------------------------
# QM reference curve (optional)
# -----------------------------

def hom_qm_reference(tau: np.ndarray,
                     tau_c: float = 1.0,
                     visibility: float = 1.0,
                     classical_level: float = 0.5) -> np.ndarray:
    """
    Simple QM-inspired HOM reference curve (up to scaling):

      P_coinc(τ) = classical_level * [1 - V * exp(-τ^2 / τ_c^2)]

    Defaults:
      classical_level = 0.5  (what DFT toy approaches for large |τ|)
      V = visibility  (0..1)
      τ_c sets width of dip.

    This is not used in the simulation; it's just for overlay/diagnostics.
    """
    return classical_level * (1.0 - visibility * np.exp(-(tau ** 2) / (tau_c ** 2)))


# -----------------------------
# Plotting and CSV I/O
# -----------------------------

def save_results(cfg: HOMConfig,
                 tau_values: np.ndarray,
                 coincidence_rates: np.ndarray) -> None:
    data = np.column_stack([tau_values, coincidence_rates])
    header = "tau,coincidence_rate"
    np.savetxt(cfg.out_csv, data, delimiter=",", header=header, comments="")
    print(f"Saved HOM results to {cfg.out_csv}")


def plot_results(cfg: HOMConfig,
                 tau_values: np.ndarray,
                 coincidence_rates: np.ndarray) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(tau_values, coincidence_rates, "o-", label=f"{cfg.mode} simulation")

    # Overlay a simple QM-style reference (tuned by eye)
    tau_c_guess = 1.5
    qm_ref = hom_qm_reference(tau_values, tau_c=tau_c_guess,
                              visibility=1.0, classical_level=0.5)
    plt.plot(tau_values, qm_ref, "--", label="QM-like reference (toy)")

    plt.xlabel("Delay τ (arbitrary units)")
    plt.ylabel("Coincidence rate P_coinc")
    plt.title(f"DFT-Q3a HOM toy ({cfg.mode} mode)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------
# CLI / main
# -----------------------------

def parse_args() -> HOMConfig:
    p = argparse.ArgumentParser(
        description="DFT-Q3a: HOM toy model in trajectory + phase framework"
    )
    p.add_argument("--mode", choices=["classical", "dft"],
                   default="dft",
                   help="Simulation mode: classical baseline or DFT joint-phase model")
    p.add_argument("--pairs", type=int, default=200_000,
                   help="Number of photon pairs per delay value")
    p.add_argument("--tau-min", type=float, default=-5.0,
                   help="Minimum delay τ")
    p.add_argument("--tau-max", type=float, default=5.0,
                   help="Maximum delay τ")
    p.add_argument("--tau-steps", type=int, default=41,
                   help="Number of τ samples between tau-min and tau-max")
    p.add_argument("--omega", type=float, default=1.0,
                   help="Effective frequency: Δθ = ω τ")
    p.add_argument("--phase-noise", type=float, default=0.0,
                   help="Stddev of additional Gaussian phase noise (radians)")
    p.add_argument("--seed", type=int, default=12345,
                   help="Random seed")
    p.add_argument("--out-csv", type=str, default="hom_results.csv",
                   help="Output CSV filename")
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
        out_csv=args.out_csv,
        no_plot=args.no_plot,
        tau_c=args.tau_c,      
    )


def main() -> None:
    cfg = parse_args()
    print(f"Running HOM simulation in {cfg.mode!r} mode...")
    print(f"Pairs per τ: {cfg.n_pairs}, τ ∈ [{cfg.tau_min}, {cfg.tau_max}] "
          f"with {cfg.tau_steps} steps")

    tau_values, coincidence_rates = simulate_hom(cfg)

    # Basic diagnostics
    print("Some sample points:")
    for idx in (0, len(tau_values)//2, len(tau_values)-1):
        print(f"  τ = {tau_values[idx]: .3f}, P_coinc ≈ {coincidence_rates[idx]: .4f}")

    save_results(cfg, tau_values, coincidence_rates)

    if not cfg.no_plot:
        plot_results(cfg, tau_values, coincidence_rates)


if __name__ == "__main__":
    main()