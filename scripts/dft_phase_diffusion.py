#!/usr/bin/env python3
"""
dft_phase_diffusion.py

DFT phase diffusion experiment with motion budget and a distribution of
deterministic phase increments ω across pairs.

Per extra step on the delayed arm:
    Δθ_step = ω_pair + Δθ_noise

where:
  - ω_pair ~ Normal(omega0, omega_sigma) drawn ONCE per pair
  - Δθ_noise is from (Δx)^2 + (Δθ_noise)^2 = 1, with Δx chosen from dx_choices,
    and sign ± equiprobable.

For each delay N_delay, we simulate an ensemble of n_pairs:
    Δθ_pair(N_delay) = - Σ_{k=1..N_delay} (ω_pair + noise_k)

We measure:
  mean_dtheta(N)      = <Δθ>
  var_dtheta(N)       = Var[Δθ]   (population variance, ddof=0)
  coherence_mag(N)    = |<exp(iΔθ)>|

Outputs CSV with columns:
  N_delay_steps,mean_dtheta,var_dtheta,coherence_mag

Implementation notes:
- Uses NumPy vectorization + chunking.
- Runs in TWO passes:
    (1) accumulate sums for mean/variance
    (2) replay with same RNG seed to accumulate complex coherence safely
  This avoids storing all Δθ samples in memory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import hashlib, inspect


# ------------------------
# Config
# ------------------------

@dataclass(frozen=True)
class DiffusionConfig:
    max_delay_steps: int = 50
    n_pairs: int = 100_000
    dx_choices: Tuple[float, ...] = (-0.15, 0.0, 0.15)

    omega0: float = 0.0
    omega_sigma: float = 0.0

    seed: int = 12345
    out_csv: str = "data/generated/phase_diffusion_results.csv"
    no_plot: bool = False

    # Process pairs in chunks to control memory footprint.
    chunk_pairs: int = 250_000


# ------------------------
# Core helpers
# ------------------------

def _noise_dtheta_magnitudes(dx_choices: np.ndarray) -> np.ndarray:
    dx2 = dx_choices * dx_choices
    if np.any(dx2 > 1.0):
        raise ValueError("dx_choices must satisfy |dx| <= 1")
    return np.sqrt(1.0 - dx2)


def _simulate_chunk_deltas(
    rng: np.random.Generator,
    *,
    m: int,
    Nmax: int,
    dx_choices: np.ndarray,
    dtheta_mags: np.ndarray,
    sign_choices: np.ndarray,
    omega0: float,
    omega_sigma: float,
) -> np.ndarray:
    """
    Return Δθ for a chunk for all N=1..Nmax as an array of shape (m, Nmax),
    where deltas[:, n-1] is Δθ at delay N=n.

    If Nmax==0, returns shape (m, 0).
    """
    # ω_pair per pair (shape: m)
    if omega_sigma > 0.0:
        omega_pair = rng.normal(loc=omega0, scale=omega_sigma, size=m)
    else:
        omega_pair = np.full(m, omega0, dtype=float)

    if Nmax <= 0:
        return np.empty((m, 0), dtype=float)

    # Noise: choose dx index and sign for each (pair, step)
    dx_idx = rng.integers(0, len(dx_choices), size=(m, Nmax), endpoint=False)
    mags = dtheta_mags[dx_idx]
    sign = rng.choice(sign_choices, size=(m, Nmax))
    dtheta_noise = mags * sign

    # Step increments: ω_pair broadcast + noise
    dtheta_step = dtheta_noise + omega_pair[:, None]

    # Cumulative extra phase and Δθ = -cumsum
    cum_extra = np.cumsum(dtheta_step, axis=1)
    deltas = -cum_extra
    return deltas


def simulate_phase_diffusion(cfg: DiffusionConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns arrays:
      delay_steps (int)
      mean_dtheta (float)
      var_dtheta  (float)
      coherence_mag (float)
    """
    Nmax = int(cfg.max_delay_steps)
    n_pairs_total = int(cfg.n_pairs)

    delay_steps = np.arange(Nmax + 1, dtype=int)
    mean_dtheta = np.zeros(Nmax + 1, dtype=float)
    var_dtheta = np.zeros(Nmax + 1, dtype=float)
    coh_mag = np.zeros(Nmax + 1, dtype=float)

    # Handle degenerate case: no pairs
    if n_pairs_total <= 0:
        # Δθ undefined; keep zeros, but coherence at N=0 is 1 by convention.
        coh_mag[0] = 1.0
        return delay_steps, mean_dtheta, var_dtheta, coh_mag

    # Precompute noise magnitudes
    dx_choices = np.asarray(cfg.dx_choices, dtype=float)
    dtheta_mags = _noise_dtheta_magnitudes(dx_choices)
    sign_choices = np.array([-1.0, 1.0], dtype=float)

    # N=0 is trivial in this unwrapped model
    coh_mag[0] = 1.0

    chunk = max(1, int(cfg.chunk_pairs))

    # --------------------------
    # Pass 1: mean/variance sums
    # --------------------------
    rng1 = np.random.default_rng(cfg.seed)
    for start in range(0, n_pairs_total, chunk):
        m = min(chunk, n_pairs_total - start)

        deltas = _simulate_chunk_deltas(
            rng1,
            m=m,
            Nmax=Nmax,
            dx_choices=dx_choices,
            dtheta_mags=dtheta_mags,
            sign_choices=sign_choices,
            omega0=cfg.omega0,
            omega_sigma=cfg.omega_sigma,
        )

        for N in range(1, Nmax + 1):
            d = deltas[:, N - 1]
            mean_dtheta[N] += float(np.sum(d))
            var_dtheta[N] += float(np.sum(d * d))

    # Normalize to mean/variance (population variance ddof=0)
    mean_dtheta[1:] /= n_pairs_total
    Ed2 = var_dtheta[1:] / n_pairs_total
    var_dtheta[1:] = Ed2 - mean_dtheta[1:] ** 2
    var_dtheta[1:] = np.maximum(var_dtheta[1:], 0.0)  # guard tiny negative roundoff

    # --------------------------
    # Pass 2: coherence (complex)
    # --------------------------
    rng2 = np.random.default_rng(cfg.seed)
    coh_sum = np.zeros(Nmax + 1, dtype=np.complex128)
    coh_sum[0] = n_pairs_total + 0j

    for start in range(0, n_pairs_total, chunk):
        m = min(chunk, n_pairs_total - start)

        deltas = _simulate_chunk_deltas(
            rng2,
            m=m,
            Nmax=Nmax,
            dx_choices=dx_choices,
            dtheta_mags=dtheta_mags,
            sign_choices=sign_choices,
            omega0=cfg.omega0,
            omega_sigma=cfg.omega_sigma,
        )

        for N in range(1, Nmax + 1):
            d = deltas[:, N - 1]
            coh_sum[N] += np.sum(np.exp(1j * d))

    coh_mag = np.abs(coh_sum / n_pairs_total).astype(float)

    return delay_steps, mean_dtheta, var_dtheta, coh_mag


# ------------------------
# I/O
# ------------------------

def save_results(
    delay_steps: np.ndarray,
    mean_dtheta: np.ndarray,
    var_dtheta: np.ndarray,
    coh_mag: np.ndarray,
    out_csv: str,
) -> None:
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.column_stack([delay_steps, mean_dtheta, var_dtheta, coh_mag])
    header = "N_delay_steps,mean_dtheta,var_dtheta,coherence_mag"
    np.savetxt(out_path, data, delimiter=",", header=header, comments="")
    print(f"[dft_phase_diffusion] Wrote: {out_path.resolve()}")


def plot_results(delay_steps: np.ndarray, var_dtheta: np.ndarray, coh_mag: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(delay_steps, var_dtheta, marker="o")
    plt.xlabel("Delay steps N_delay")
    plt.ylabel("Var[Δθ]")
    plt.title("Phase variance vs delay (unwrapped Δθ)")
    plt.grid(True)
    plt.tight_layout()

    plt.figure(figsize=(8, 5))
    plt.plot(delay_steps, coh_mag, marker="o")
    plt.xlabel("Delay steps N_delay")
    plt.ylabel("|⟨e^{iΔθ}⟩| (coherence)")
    plt.title("Coherence magnitude vs delay")
    plt.grid(True)
    plt.tight_layout()

    plt.show()


# ------------------------
# CLI
# ------------------------

def _parse_dx_choices(s: str) -> Tuple[float, ...]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError("dx_choices cannot be empty")
    return tuple(float(p) for p in parts)


def parse_args(argv: Sequence[str] | None = None) -> DiffusionConfig:
    repo_root = Path(__file__).resolve().parents[1]
    default_out = repo_root / "data" / "generated" / "phase_diffusion_results.csv"

    p = argparse.ArgumentParser(description="DFT phase diffusion experiment (variance + coherence vs delay).")
    p.add_argument("--max-delay-steps", type=int, default=50)
    p.add_argument("--pairs", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--omega0", type=float, default=0.0)
    p.add_argument("--omega-sigma", type=float, default=0.0)
    p.add_argument("--dx-choices", type=str, default="-0.15,0.0,0.15")
    p.add_argument("--out-csv", type=str, default=str(default_out))
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--chunk-pairs", type=int, default=250_000,
                   help="Process pairs in chunks to limit memory (default: 250000)")
    args = p.parse_args(argv)

    return DiffusionConfig(
        max_delay_steps=args.max_delay_steps,
        n_pairs=args.pairs,
        dx_choices=_parse_dx_choices(args.dx_choices),
        omega0=args.omega0,
        omega_sigma=args.omega_sigma,
        seed=args.seed,
        out_csv=args.out_csv,
        no_plot=args.no_plot,
        chunk_pairs=args.chunk_pairs,
    )


def main() -> None:
    cfg = parse_args()
    print(
        "[dft_phase_diffusion] Running with:",
        f"max_delay_steps={cfg.max_delay_steps}, pairs={cfg.n_pairs},",
        f"omega0={cfg.omega0}, omega_sigma={cfg.omega_sigma},",
        f"seed={cfg.seed}, dx_choices={cfg.dx_choices},",
        f"out_csv={cfg.out_csv}",
    )

    src = Path(inspect.getsourcefile(simulate_phase_diffusion)).read_bytes()
    print("[dft_phase_diffusion] CODE_SHA256=", hashlib.sha256(src).hexdigest()[:16])
    delay_steps, mean_dtheta, var_dtheta, coh_mag = simulate_phase_diffusion(cfg)
    save_results(delay_steps, mean_dtheta, var_dtheta, coh_mag, cfg.out_csv)

    if not cfg.no_plot:
        plot_results(delay_steps, var_dtheta, coh_mag)


if __name__ == "__main__":
    main()