# Simulation-Based Inference for an Adaptive-Network Epidemic Model

This repository contains the complete implementation and the final technical report for the ST3247 Computational Statistics project. The study focuses on inferring unknown parameters of a stochastic SIR (Susceptible-Infected-Recovered) model on an adaptive network using advanced Simulation-Based Inference (SBI) techniques.

## Project Overview
In adaptive epidemic models, individuals modify their social contacts in response to the spread of disease (e.g., breaking links with infected neighbors). This behavioral feedback creates a complex loop that makes the likelihood function analytically intractable. We utilize **Sequential Monte Carlo ABC (SMC-ABC)** combined with advanced distance metrics and post-processing adjustments to estimate:
- **$\beta$**: Infection probability per S-I edge per step
- **$\gamma$**: Recovery probability per infected agent per step
- **$\rho$**: Rewiring probability per S-I edge per step

## Key Features
- **High-Performance Simulation**: The core epidemic engine is optimized using **Numba** to achieve the computational speed required for the thousands of forward simulations necessary for SBI.
- **Mahalanobis Distance Metric**: To resolve the parameter non-identifiability (specifically the $\beta$-$\rho$ trade-off), we implemented the Mahalanobis distance to decorrelate the summary statistics space.
- **Local Linear Regression Adjustment**: A post-processing step based on Beaumont et al. (2002) was applied to correct for the bias introduced by non-zero tolerance levels, significantly sharpening the posterior distribution.
- **Validation**: The model's predictive performance was validated using **Posterior Predictive Checks (PPC)**, demonstrating 100% coverage of observed temporal and structural data.

## Repository Structure
```text
.
├── main.pdf                   # Final Technical Report
├── main.tex                   # LaTeX source code
├── references.bib             # Bibliography file (APA style)
├── requirements.txt           # Python dependency list
├── simulator_fast.py          # Numba-accelerated SIR simulator engine
├── 01_baseline_abc.ipynb      # Step 1: Baseline Rejection Sampling
├── 02_smc_abc.ipynb           # Step 2: Initial SMC-ABC implementation
├── 03_smc_abc.ipynb           # Step 3: Feature exploration and sensitivity
├── 04_smc_abc_final.ipynb     # Step 4: Final Pipeline (Mahalanobis + Regression + PPC)
├── data/                      # Directory containing provided observed datasets
├── plots/                     # Output figures and visualizations
└── sections/                  # LaTeX sub-files for the report
```

## Getting Started
### Prerequisites

This project requires Python 3.9+. You can install all necessary dependencies using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Reproducing Results

To reproduce the findings and figures presented in the report, please execute the notebooks in the following order:

1. `01_baseline_abc.ipynb`: Run this to understand the baseline performance and the motivation for sequential methods.

2. `04_smc_abc_final.ipynb`: Run this for the primary results, including the Mahalanobis-based SMC-ABC, the local linear regression adjustment, and the final Posterior Predictive Checks.

## Methodology Summary
1. **Summary Statistic Design**: We utilize 5 informative features, including peak infection fraction, rewiring ratios, and final degree variance.

2. **Sequential Refinement**: The SMC-ABC algorithm uses an adaptive tolerance schedule across 4 generations to efficiently narrow down the parameter space.

3. **Accuracy Enhancement**: The use of inverse-covariance weighting (Mahalanobis) and regression-based "shrinkage" allows for high precision despite the stochastic nature of the model.

## Author
Yohei Kiguchi (A0308628X, e1399068@u.nus.edu)

## Acknowledgments
This project was developed for the ST3247 Computational Statistics module. Generative AI tools (Gemini/Notebook LM) were utilized for LaTeX document structuring and refactoring Python code for performance optimization.