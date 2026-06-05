"""Run the projected triple-momentum numerical example.
The example compares projected gradient descent, triple momentum with Euclidean
projection, and triple momentum with Lyapunov-matrix-norm projection on a
strongly convex quadratic objective over an ellipsoidal constraint set. 
It also solves the associated IQC/LMI certificate and generates distance and objective
plots together with theoretical convergence-rate bounds.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from projected_algorithms import (
    EllipsoidConstraint,
    build_projection,
    projected_triple_momentum_Euclidean,
    projected_triple_momentum_P_norm,
    quadratic_objective,
    analyze_quadratic_function,
    proj_gradient,
    solve_ellipsoid_qp,
    triple_momentum_matrices,
    solve_iqc_LMI,
    search_best_rho,
)


def main() -> None:
    Path("outputs").mkdir(exist_ok=True)
    solver = "CLARABEL"
    # solver = "MOSEK"  # fallback to CLARABEL/SCS if installed and MOSEK is not available
    verbose = 0

    opts = {
        "solver": solver,
        "verbose": verbose,
        "epsP": 1e-5,
        "epsLMI": 1e-9,
        "epsCheck": 1e-9,
        "rhoLo": 1e-6,
        "rhoHi": 1.0,
        "tol": 1e-6,
        "maxIter": 35,
        "use_sector": True,
        "use_obo": True,
        "use_weighted": True,
        "maximize_margins": False,
        # "rho_bar": 0.95, # Optional: fix rho_bar independently of rho. If absent, rho_bar=rho.
    }

    # Problem setup
    d = 2
    F = np.array([[100.0, -1.0], [-1.0, 1.0]])
    p = np.array([1.0, 10.0])
    f = quadratic_objective(F, p)
    grad_f, m, L = analyze_quadratic_function(F, p)
    print(f"m = {m:.12g}, L = {L:.12g}")

    # Triple momentum realization in the transformed coordinate.
    A, B, C, D, pars = triple_momentum_matrices(m, L)
    rho_theory = pars["rho"]
    print("rho (table/theory) =", rho_theory)

    # Verify a nominal rho.
    rho_cert = min(0.999999, rho_theory * 1.0001)
    rho_bar = opts.get("rho_bar", rho_cert)
    nominal_verify = solve_iqc_LMI(
        A, B, C, D, m, L, rho=rho_cert, rho_bar=rho_bar, opts=opts
    )
    print("Verified nominal rho:", nominal_verify["rho"], "feasible:", nominal_verify["feasible"])
    print("rho_bar:", nominal_verify["rho_bar"])
    print("status:", nominal_verify["status"])
    print("alpha =", nominal_verify["alpha"])
    print("P shape:", nominal_verify["P"].shape)
    print("minEigP:", nominal_verify["minEigP"])
    print("maxEigLMI:", nominal_verify["maxEigLMI"])
    h = np.linalg.solve(nominal_verify["P"][1:, 1:], nominal_verify["P"][0, 1:].reshape(-1, 1)).reshape(-1)
    print("P =", nominal_verify["P"]);
    print("Shift =", h[0]);

    # P = nominal_verify["P"]
    # P22 = P[1:, 1:]
    # print("eig(P)   =", np.linalg.eigvalsh(P))
    # print("eig(P22) =", np.linalg.eigvalsh(P22))
    # print("cond(P)  =", np.linalg.cond(P))
    # print("cond(P22)=", np.linalg.cond(P22))
    # print("shift    =", np.linalg.solve(P22, P[0, 1:].reshape(-1, 1)).reshape(-1))

    if not nominal_verify["feasible"]:
        print("Certificate was not feasible; stopping before projection simulation.")
        return
    
    # Optional best-rate search. This can be slow depending on the SDP solver.
    # best = search_best_rho(A, B, C, D, m, L, opts)
    # print("Best rho:", best.get("rho"))
    # print("bounds:", best.get("rhoLowerBound"), best.get("rhoUpperBound"))
    # print("status:", best.get("status"))
    # print("alpha:", best.get("alpha"))


    # Constraint: y.T Q y <= r2.
    constraint = EllipsoidConstraint(Q=np.array([[1.0, 0.0], [0.0, 2.0]]), r2=5.0)
    proj_fun = build_projection(constraint)

    rng = np.random.default_rng(4)
    x0 = rng.random(2 * d)
    num_steps = 300

    results_proj_1 = projected_triple_momentum_Euclidean(
        A, B, C, D, x0, num_steps, grad_f, proj_fun
    )
    P = nominal_verify["P"]
    results_proj_2 = projected_triple_momentum_P_norm(
        A, B, C, D, x0, num_steps, grad_f, proj_fun, P
    )

    # Projected gradient descent baseline.
    A_g = 1.0
    B_g = -2.0 / (L + m)
    C_g = 1.0
    D_g = 0.0
    x0_g = x0[:d]
    results_proj_g = proj_gradient(A_g, B_g, C_g, D_g, x0_g, num_steps, grad_f, proj_fun)

    # Constrained optimum.
    x_optimal_proj = solve_ellipsoid_qp(F, p, constraint)
    f_optimal_proj = f(x_optimal_proj)
    print("x_optimal_proj =", x_optimal_proj)
    print("f_optimal_proj =", f_optimal_proj)

    y0 = results_proj_g.x[:, 0]
    Cx = np.linalg.norm(y0 - x_optimal_proj, 2)
    x_rate_bound = Cx * rho_theory ** np.arange(num_steps + 1)

    f_0 = f(results_proj_g.x[:, 0])
    Cf = abs(f_0 - f_optimal_proj)
    f_rate_bound = Cf * (rho_theory ** np.arange(num_steps + 1)) ** 2

    if results_proj_1.y is None or results_proj_2.y is None:
        raise RuntimeError("Projection results missing y arrays.")

    dist_g = np.linalg.norm(results_proj_g.x - x_optimal_proj[:, None], axis=0)
    dist_1 = np.linalg.norm(results_proj_1.y - x_optimal_proj[:, None], axis=0)
    dist_2 = np.linalg.norm(results_proj_2.y - x_optimal_proj[:, None], axis=0)

    plt.figure()
    plt.semilogy(dist_g, "-.", linewidth=2)
    plt.semilogy(dist_1, "--", linewidth=2)
    plt.semilogy(dist_2, linewidth=2)
    plt.semilogy(x_rate_bound, ":", linewidth=2)
    plt.grid(True)
    plt.xlim([0, num_steps])
    plt.ylim([1e-15, 1e1])
    plt.legend(
        [
            "Projected gradient descent",
            "TMM with Euclidean projection",
            "TMM with Lyapunov-matrix-norm projection",
            "Theoretical bound",
        ],
        fontsize=10,
    )
    plt.xlabel("Iteration Step", fontsize=15)
    plt.ylabel(r"$\| y_k - y_{\Omega}^{\mathrm{opt}} \|_2$", fontsize=15)
    plt.tight_layout()
    plt.savefig("outputs/Distance_Plot.png", dpi=200)

    f_record_g = np.array([f(results_proj_g.x[:, k]) for k in range(num_steps + 1)])
    f_record_1 = np.array([f(results_proj_1.y[:, k]) for k in range(num_steps)])
    f_record_2 = np.array([f(results_proj_2.y[:, k]) for k in range(num_steps)])

    plt.figure()
    plt.semilogy(np.abs(f_record_g - f_optimal_proj), "-.", linewidth=2)
    plt.semilogy(np.abs(f_record_1 - f_optimal_proj), "--", linewidth=2)
    plt.semilogy(np.abs(f_record_2 - f_optimal_proj), linewidth=2)
    plt.semilogy(f_rate_bound, ":", linewidth=2)
    plt.grid(True)
    plt.xlim([0, num_steps])
    plt.ylim([1e-15, 1e2])
    plt.legend(
        [
            "Projected gradient descent",
            "TMM with Euclidean projection",
            "TMM with Lyapunov-matrix-norm projection",
            "Theoretical bound",
        ],
        fontsize=10,
    )
    plt.xlabel("Iteration Step", fontsize=15)
    plt.ylabel(r"$| f(y_k) - f(y_{\Omega}^{\mathrm{opt}}) |$", fontsize=15)
    plt.tight_layout()
    plt.savefig("outputs/Objective_Plot.png", dpi=200)

    print("Saved outputs/Distance_Plot.png and outputs/Objective_Plot.png")


if __name__ == "__main__":
    main()
