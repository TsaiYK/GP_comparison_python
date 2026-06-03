# Gaussian Process Regression: Package Comparison

A Jupyter notebook benchmarking three Python GP libraries on a 1D regression test problem.

## Test Problem

**Function:** `f(x) = sin(2πx) + 0.3·sin(4πx)` with Gaussian noise (σ = 0.15)

- 20 training points sampled uniformly from [0, 1]
- 300 test points for evaluation
- All models use a **Matérn-5/2 kernel** so differences reflect optimizer behavior, not modeling power

## Packages Compared

| Package | Version | Optimizer | GPU |
|---------|---------|-----------|-----|
| [scikit-learn](https://scikit-learn.org/stable/modules/gaussian_process.html) | ≥ 1.0 | L-BFGS-B (5 restarts) | No |
| [GPy](https://sheffieldml.github.io/GPy/) | ≥ 1.10 | `optimize_restarts` (5 restarts) | No |
| [GPyTorch](https://gpytorch.ai/) | ≥ 1.11 | Adam (200 iterations, MLL loss) | Yes |

## Notebook Contents

1. **Data generation** — test function and noisy observations
2. **scikit-learn GP** — `GaussianProcessRegressor` with `Matern + WhiteKernel`
3. **GPy** — `GPRegression` with `Matern52`
4. **GPyTorch** — `ExactGP` with training loss curve
5. **Side-by-side plots** — mean prediction and ±2σ uncertainty bands
6. **Metrics table** — RMSE, NLL, and fit time per package
7. **Calibration curve** — actual vs. expected coverage (reliability of uncertainty estimates)

## Outputs

| File | Description |
|------|-------------|
| `gp_comparison.png` | Side-by-side prediction plots |
| `gp_calibration.png` | Uncertainty calibration curves |

## Installation

Install only the packages you need — each section handles `ImportError` gracefully.

```bash
pip install scikit-learn matplotlib numpy   # minimum (sklearn only)
pip install GPy                             # add GPy
pip install torch gpytorch                 # add GPyTorch
```

## Usage

```bash
jupyter notebook gaussian_process_comparison.ipynb
```

Or open directly in VS Code with the Jupyter extension.

## Package Trade-offs

| Package | Strengths | Weaknesses |
|---------|-----------|------------|
| **scikit-learn** | Zero extra deps, clean API, stable | No GPU, limited kernels, slow for N > 5k |
| **GPy** | Rich kernel library, good diagnostics | No GPU, slower development pace |
| **GPyTorch** | GPU-ready, sparse/deep GPs, active | Verbose API, requires PyTorch |
