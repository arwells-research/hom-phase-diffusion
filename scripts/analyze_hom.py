#!/usr/bin/env python3
import numpy as np
from dft_q3a_hom import hom_qm_reference  # import from your main script

def main():
    data = np.loadtxt("hom_results.csv", delimiter=",", skiprows=1)
    tau = data[:, 0]
    P = data[:, 1]

    # Tune these by eye / trial:
    tau_c = 1.5       # match what you used in the sim
    classical_level = 0.5
    visibility = 1.0  # or <1 if your dip is lifted by noise

    P_ref = hom_qm_reference(tau,
                             tau_c=tau_c,
                             visibility=visibility,
                             classical_level=classical_level)

    diff = P - P_ref
    rms = np.sqrt(np.mean(diff**2))
    corr = np.corrcoef(P, P_ref)[0, 1]

    print(f"RMS difference (DFT vs QM-ref) = {rms:.4f}")
    print(f"Correlation coefficient        = {corr:.4f}")

if __name__ == "__main__":
    main()