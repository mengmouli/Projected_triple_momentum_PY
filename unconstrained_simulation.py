"""Simulate the unconstrained triple-momentum method and a gradient-descent baseline.

The triple-momentum method was proposed in:

    Van Scoy, B., Freeman, R. A., and Lynch, K. M.
    "The fastest known globally convergent first-order method for minimizing
    strongly convex functions."
    IEEE Control Systems Letters, 2(1), 49--54, 2018.

This script runs the unconstrained version of the algorithm in transformed
coordinates and generates distance and objective-value plots together with
the corresponding theoretical convergence-rate bounds.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from projected_algorithms import (
    triple_momentum_matrices,
    projected_triple_momentum_Euclidean,
    proj_gradient,
    quadratic_objective,
    analyze_quadratic_function,
)


def main() -> None:
    Path("outputs").mkdir(exist_ok=True)
    # Problem setup
    d = 2
    F = np.array([[100.0, -1.0], [-1.0, 1.0]])
    p = np.array([1.0, 10.0])
    f = quadratic_objective(F, p)
    grad_f, m, L = analyze_quadratic_function(F, p)
    print(f"m = {m:.12g}, L = {L:.12g}")

    # Triple-momentum algorithm matrices
    A, B, C, D, pars = triple_momentum_matrices(m, L)
    print("rho (theory) =", pars["rho"]) 

    # Initial condition and simulation length
    rng = np.random.default_rng(4)
    x0 = rng.random(2 * d)
    num_steps = 300

    # Identity projection (unconstrained)
    proj_fun = lambda x: x

    # Simulate triple-momentum (unconstrained: projection is identity)
    results_tm = projected_triple_momentum_Euclidean(
        A, B, C, D, x0, num_steps, grad_f, proj_fun
    )

    # Simulate plain projected gradient (unconstrained gradient descent)
    A_g = 1.0
    B_g = -2.0 / (L + m)
    C_g = 1.0
    D_g = 0.0
    x0_g = x0[:d]
    results_g = proj_gradient(A_g, B_g, C_g, D_g, x0_g, num_steps, grad_f, proj_fun)

    # Unconstrained analytic optimum
    x_opt = -np.linalg.solve(F, p)
    f_opt = f(x_opt)

    # Extract final iterates and compute distances
    if results_tm.y is None:
        raise RuntimeError("expected y record from triple-momentum simulation")

    final_tm = results_tm.y[:, -1]
    final_g = results_g.x[:, -1]

    dist_tm = np.linalg.norm(final_tm - x_opt)
    dist_g = np.linalg.norm(final_g - x_opt)

    print("Unconstrained optimum:", x_opt)
    print("Final triple-momentum y_T:", final_tm)
    print("Final gradient x_T:", final_g)
    print(f"Distance to optimum: triple-momentum={dist_tm:.6e}, gradient={dist_g:.6e}")

    # Compute theoretical rate bounds using rho from theory
    rho_theory = pars["rho"]
    
    # Initial distance and objective gap
    y0_tm = results_tm.y[:, 0]
    C_dist = np.linalg.norm(y0_tm - x_opt, 2)
    dist_rate_bound = C_dist * rho_theory ** np.arange(num_steps)

    f_0_tm = f(y0_tm)
    C_f = abs(f_0_tm - f_opt)
    f_rate_bound = C_f * (rho_theory ** np.arange(num_steps)) ** 2

    # Record distances and objectives over iterations
    dist_tm_traj = np.linalg.norm(results_tm.y - x_opt[:, None], axis=0)
    dist_g_traj = np.linalg.norm(results_g.x - x_opt[:, None], axis=0)

    f_tm_traj = np.array([f(results_tm.y[:, k]) for k in range(num_steps)])
    f_g_traj = np.array([f(results_g.x[:, k]) for k in range(num_steps + 1)])

    # Distance plot
    plt.figure()
    plt.semilogy(dist_tm_traj, linewidth=2, label="Triple momentum method")
    plt.semilogy(dist_g_traj, "--",linewidth=2, label="Gradient descent")
    plt.semilogy(dist_rate_bound, ":", linewidth=1.5, label="Theoretical bound")
    plt.grid(True)
    plt.xlim([0, num_steps])
    plt.ylim([1e-15, 1e1])
    plt.legend(fontsize=11)
    plt.xlabel("Iteration Step", fontsize=15)
    plt.ylabel(r"$\| y_k - y^{\mathrm{opt}} \|_2$", fontsize=15)
    plt.tight_layout()
    plt.savefig("outputs/Distance_Plot_Unconstrained.png", dpi=200)
    print("Saved outputs/Distance_Plot_Unconstrained.png")

    # Objective plot
    plt.figure()
    plt.semilogy(np.abs(f_tm_traj - f_opt), linewidth=2, label="Triple momentum method")
    plt.semilogy(np.abs(f_g_traj - f_opt), "--", linewidth=2, label="Gradient descent")
    plt.semilogy(f_rate_bound, ":", linewidth=1.5, label="Theoretical bound")
    plt.grid(True)
    plt.xlim([0, num_steps])
    plt.ylim([1e-16, 1e2])
    plt.legend(fontsize=11)
    plt.xlabel("Iteration Step", fontsize=15)
    plt.ylabel(r"$| f(y_k) - f(y^{\mathrm{opt}}) |$", fontsize=15)
    plt.tight_layout()
    plt.savefig("outputs/Objective_Plot_Unconstrained.png", dpi=200)
    print("Saved outputs/Objective_Plot_Unconstrained.png")


if __name__ == "__main__":
    main()
