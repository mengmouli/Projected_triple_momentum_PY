# Projected Triple Momentum Method: Numerical Example

This repository contains a numerical example for the paper:

> Mengmou Li, Ioannis Lestas, and Masaaki Nagahara,  
> **First-Order Projected Algorithms With the Same Linear Convergence Rate Bounds as Their Unconstrained Counterparts**,  
> arXiv:2503.13965.  
> https://arxiv.org/abs/2503.13965

The code generates and compares projected first-order algorithms, with a focus on the **projected triple momentum method (projected TMM)**. The example illustrates how a first-order method designed for an unconstrained strongly convex problem can be combined with projection while preserving the theoretical linear convergence-rate bound certified from the unconstrained dynamics.

## Overview

The repository includes two main numerical scripts.

- `run_comparison.py`  
  Runs the projected numerical example. It compares:
  - projected gradient descent,
  - TMM with Euclidean projection,
  - TMM with Lyapunov-matrix-norm projection,
  - the theoretical convergence-rate bound.

- `unconstrained_simulation.py`  
  Runs the corresponding unconstrained simulation. It compares:
  - triple momentum method,
  - gradient descent,
  - the theoretical convergence-rate bound.

The implementation utilities are collected in:

- `projected_algorithms.py`  
  Contains the projected TMM routines, projection operators, quadratic objective utilities, IQC/LMI certificate construction, and simulation helpers.

## Mathematical problem

The numerical example considers a strongly convex quadratic objective

```math
f(x) = \frac{1}{2} x^\top F x + p^\top x,
```

with

```python
F = [[100, -1],
     [-1,   1]],
p = [1, 10].
```

The projected example uses the ellipsoidal constraint

```math
x^\top Q x \le r^2,
```

where

```python
Q = [[1, 0],
     [0, 2]],
r^2 = 5.
```

The constrained optimum is computed analytically via the KKT multiplier equation for the ellipsoid-constrained quadratic program.

## Main features

- Construction of triple momentum method parameters from the strong convexity and smoothness constants.
- IQC/LMI-based verification of a nominal linear convergence-rate certificate.
- Euclidean projection onto an ellipsoid.
- Lyapunov-matrix-norm projection correction for projected TMM.
- Comparison with projected gradient descent.
- Generation of distance and objective-gap plots.

## Repository structure

```text
.
├── projected_algorithms.py        # Core algorithms, projections, IQC/LMI utilities
├── run_comparison.py              # Projected TMM numerical example
├── unconstrained_simulation.py    # Unconstrained TMM baseline example
├── requirements.txt               # Python dependencies
├── .gitignore
└── outputs/                       # Generated figures, created by the user if absent
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate          # Windows PowerShell
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The default SDP solver in `run_comparison.py` is `CLARABEL`. MOSEK can also be used if it is installed and a valid license is available.

## Usage

Create the output directory first:

```bash
mkdir -p outputs
```

Run the projected example:

```bash
python run_comparison.py
```

This generates:

```text
outputs/Distance_Plot.png
outputs/Objective_Plot.png
```

Run the unconstrained baseline example:

```bash
python unconstrained_simulation.py
```

This generates:

```text
outputs/Distance_Plot_Unconstrained.png
outputs/Objective_Plot_Unconstrained.png
```

## Expected outputs

The projected example prints the strong convexity and smoothness constants, the theoretical TMM rate, the LMI certificate status, the Lyapunov matrix, and the projected optimum. It then saves two plots:

- distance to the constrained optimizer,
- objective gap relative to the constrained optimum.

The unconstrained example prints the unconstrained optimizer and final iterate errors, then saves analogous plots for the unconstrained problem.

## Notes

- The example is intentionally low-dimensional so that the projected trajectories and convergence behavior are easy to inspect.
- The LMI certificate is used to obtain the Lyapunov matrix that defines the norm used in the projected TMM construction.
- Numerical solver tolerances may slightly affect the reported certificate margins.

## Citation

If this code is useful for your research, please cite:

```bibtex
@article{li2025first,
  title        = {First-Order Projected Algorithms With the Same Linear Convergence Rate Bounds as Their Unconstrained Counterparts},
  author       = {Li, Mengmou and Lestas, Ioannis and Nagahara, Masaaki},
  year         = {2025},
  journal      = {arXiv preprint arXiv:2503.13965}
}
```
