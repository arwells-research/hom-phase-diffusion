# HOM Phase Diffusion (DFT)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17931012.svg)](https://doi.org/10.5281/zenodo.17931012)

**Reproducible Research Artifact — Deterministic, Physics-Verified**

Code and generated datasets supporting the paper:

**Two-Photon Bunching from Geometric Phase Diffusion:  
Hong–Ou–Mandel Interference without Wavefunctions**

This repository implements a trajectory- and phase-based model within
**Dual-Frame Theory (DFT)** that reproduces Hong–Ou–Mandel (HOM) interference
*without* wavefunctions, Hilbert-space tensor products, or phenomenological
interference terms.

All figures, tables, and quoted numerical values in the paper are
**fully reproducible** from the scripts provided here.

---

## Reproducibility Status

**Reproducibility:** ✔ Deterministic (fixed RNG seeds; byte-for-byte reproducibility) 
**Data provenance:** ✔ Generated entirely by this repository  
**Physics checks:** ✔ Verified against analytic expectations  
**External data:** ✘ None required  

A full end-to-end verification script is provided (`verify.sh`).

---

## Quickstart

Install dependencies:

    pip install -r requirements.txt

Run the full phase-diffusion simulation suite:

    bash scripts/run_diffusion_suite.sh

Export paper-ready numerical tables:

    python scripts/export_paper_numbers.py

Generate figures:

    python scripts/make_fig1.py
    python scripts/make_fig2.py
    python scripts/make_fig3.py
    python scripts/make_fig4.py

---

## Reproducibility Verification (Strongly Recommended)

Running individual figure scripts reproduces plots; running verify.sh additionally 
confirms that all quoted numerical values and scaling laws are internally consistent 
and physically correct.

To reproduce *all* numerical results **and** verify physical correctness:

    bash verify.sh

A successful run ends with:

    ================= VERIFY COMPLETE: PASS =================

### What `verify.sh` guarantees

`verify.sh` performs the following checks automatically:

1. **Environment sanity**
   - Python version
   - Script executability
   - No CRLF / shebang issues

2. **Deterministic data generation**
   - Regenerates all phase-diffusion CSV datasets
   - Confirms byte-for-byte identity across repeated runs

3. **Paper number integrity**
   - Confirms JSON ↔ LaTeX table consistency
   - Ensures no duplicated or missing parameter cases

4. **Intrinsic diffusion physics**
   - Verifies linear variance scaling for σ₍ω₎ = 0
   - Confirms diffusion coefficient D ≈ 0.985 within Monte–Carlo uncertainty

5. **Bandwidth-dependent quadratic scaling**
   - Verifies Var[Δθ](N) = DN + σ²₍ω₎ N²
   - Confirms b ≈ σ²₍ω₎ to percent-level accuracy

6. **Coherence envelope behavior**
   - Confirms monotonic decay of fitted coherence
   - Verifies mixed exponential–Gaussian form
   - Ensures sufficient data support for all fits

If `verify.sh` passes, the numerical claims of the paper are reproduced
*without manual intervention*.

---

## Repository Structure

    hom-phase-diffusion/
    ├── data/
    │   └── generated/              # Generated CSV datasets (simulation output)
    │
    ├── scripts/                    # Reproducibility, analysis & figure-generation
    │   ├── analyze_hom.py
    │   ├── dft_phase_diffusion.py
    │   ├── dft_q3a_hom.py
    │   ├── run_diffusion_suite.sh
    │   ├── export_paper_numbers.py
    │   ├── make_fig1.py
    │   ├── make_fig2.py
    │   ├── make_fig3.py
    │   ├── make_fig4.py
    │   ├── make_hom_data.py
    │   ├── run_diffusion_analysis.py
    │   ├── run_hom_analysis.py
    │   └── summarize_diffusion_suite.py
    │
    ├── src/
    │   └── hom_model.py
    │
    ├── outputs/                    # Created at runtime
    │   ├── figures/                # Generated PDF figures
    │   ├── logs/                   # Verification & run logs
    │   └── paper/                  # Auto-generated tables & numbers
    │
    ├── paper/                      # LaTeX sources for the manuscript
    │   ├── dft_hom_phase_geometry_v1.tex
    │   └── make_fig5.tex
    │
    ├── verify.sh                   # End-to-end reproducibility & physics checks
    ├── requirements.txt
    ├── environment.yml
    ├── LICENSE
    └── README.md

---

## Reproducing Phase-Diffusion Results (Figures 1–3)

1) Run the diffusion simulation grid:

    bash scripts/run_diffusion_suite.sh

Outputs:

- CSV datasets → data/generated/
- Logs → outputs/logs/
- Analysis summaries → stdout

All published results were generated with PAIRS = 200,000–500,000 per delay. Increasing PAIRS further reduces long-N regression noise but does not change extracted short-delay diffusion coefficients.

2) Generate diffusion figures:

    python scripts/make_fig1.py   # Intrinsic diffusion (σ₍ω₎ = 0)
    python scripts/make_fig2.py   # Bandwidth-dependent quadratic scaling
    python scripts/make_fig3.py   # Coherence envelope

Figures are written to:

    outputs/figures/

---

## Reproducing HOM Interference Results (Figure 4)

1) Generate HOM datasets:

    python scripts/make_hom_data.py

This produces:

- hom_qm.csv   — analytic QM reference
- hom_dft.csv  — DFT simulation curve

Written to:

    data/generated/

2) Generate HOM comparison figure:

    python scripts/make_fig4.py

Output:

    outputs/figures/fig4_hom_comparison.pdf

3) Optional quantitative comparison:

    python scripts/run_hom_analysis.py

This reports RMS deviation and Pearson correlation relative to the QM reference.

---

## Scientific Context

This repository supports a broader research program in **Dual-Frame Theory (DFT)**,
where interference and correlation phenomena arise from constrained joint phase
geometry rather than wavefunctions or nonlocal hidden variables.

The present codebase demonstrates that:

- Phase diffusion arises from a discrete geometric motion budget
- Coherence envelopes follow directly from measured phase statistics
- Hong–Ou–Mandel interference emerges without phenomenological assumptions

The simulations are intentionally minimal, deterministic, and transparent.

---

## Citation

If you use this code or the associated datasets, please cite the corresponding paper:

**Two-Photon Bunching from Geometric Phase Diffusion:  
Hong–Ou–Mandel Interference without Wavefunctions**  
A. R. Wells (2025).

The code and reproducible data are archived on Zenodo:

**Concept DOI (all versions):**  
https://doi.org/10.5281/zenodo.17931012

**Version DOI (v1.0.0):**  
https://doi.org/10.5281/zenodo.17931013

This is an open scholarly release. Reuse and extension are permitted with attribution.