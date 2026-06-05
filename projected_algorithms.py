from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, Tuple, runtime_checkable
import warnings

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq

Array = np.ndarray


# -----------------------------------------------------------------------------
# Small containers and utilities
# -----------------------------------------------------------------------------


def as_col(x: ArrayLike) -> Array:
    """Return x as a 1-D float array."""
    return np.asarray(x, dtype=float).reshape(-1)


def _as_matrix(x: ArrayLike) -> Array:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def _sym(M: Array) -> Array:
    return 0.5 * (M + M.T)


@dataclass
class Results:
    x: Array
    y: Optional[Array] = None
    u: Optional[Array] = None
    x_temp: Optional[Array] = None


@runtime_checkable
class ProjectionSet(Protocol):
    def project(self, x: Array) -> Array: ...

    def contains(self, x: Array, tol: float = 1e-10) -> bool: ...


@dataclass
class IqcAugmentedSystem:
    """Plant plus IQC filters for sector, off-by-one, and weighted off-by-one IQCs."""

    Ahat: Array
    Bhat: Array
    Chat: Array
    Dhat: Array
    rho_bar: float
    m: float
    L: float


# -----------------------------------------------------------------------------
# Projection and quadratic objective utilities
# -----------------------------------------------------------------------------


@dataclass
class EllipsoidConstraint:
    """Constraint x.T @ Q @ x <= r2.

    This reproduces the MATLAB example
        Q = [1 0; 0 2];  y' Q y <= 5
    without numerical computation of the projection.

    The Euclidean projection solves
        min_y ||y-x||^2  s.t. y.T Q y <= r2.
    For positive definite Q, the KKT solution is
        y(lambda) = (I + lambda Q)^(-1) x,
    with lambda >= 0 chosen so y(lambda).T Q y(lambda) = r2 when x is outside.
    """
    Q: Array
    r2: float

    def __post_init__(self) -> None:
        self.Q = np.asarray(self.Q, dtype=float)
        if self.Q.ndim != 2 or self.Q.shape[0] != self.Q.shape[1]:
            raise ValueError("Q must be a square matrix.")
        if not np.allclose(self.Q, self.Q.T, atol=1e-12):
            raise ValueError("Q must be symmetric.")
        evals = np.linalg.eigvalsh(self.Q)
        if np.min(evals) <= 0:
            raise ValueError("Q must be positive definite for this analytic projector.")
        self._evals, self._U = np.linalg.eigh(self.Q)

    def contains(self, x: Array, tol: float = 1e-10) -> bool:
        x = as_col(x)
        return float(x @ self.Q @ x) <= self.r2 + tol

    def project(self, x: Array) -> Array:
        x = as_col(x)
        if self.contains(x):
            return x.copy()

        # Work in the eigenbasis of Q.
        z = self._U.T @ x
        q = self._evals

        def phi(lam: float) -> float:
            y = z / (1.0 + lam * q)
            return float(np.sum(q * y * y) - self.r2)

        lo, hi = np.float64(0.0), np.float64(1.0)
        while phi(float(hi)) > 0.0:
            hi *= np.float64(2.0)
        result = brentq(phi, lo, hi, xtol=np.float64(1e-13), rtol=np.float64(1e-13), maxiter=200, full_output=False)
        lam = float(result[0] if isinstance(result, tuple) else result)
        y = z / (1.0 + lam * q)
        return self._U @ y



def build_projection(
    constraint: Optional[ProjectionSet | Callable[[Array], Array]],
) -> Callable[[Array], Array]:
    """Return a projection function from None, a ProjectionSet, or a callable."""
    if constraint is None:
        return lambda x: as_col(x)
    if isinstance(constraint, ProjectionSet):
        return lambda x: as_col(constraint.project(as_col(x)))
    if callable(constraint):
        return lambda x: as_col(constraint(as_col(x)))
    raise TypeError("constraint must be None, a ProjectionSet, or a callable projector.")


def analyze_quadratic_function(
    F: ArrayLike,
    p: ArrayLike,
) -> Tuple[Callable[[Array], Array], float, float]:
    """For f(x)=0.5*x.T*F*x+p.T*x, return grad_f, m, L."""
    F = np.asarray(F, dtype=float)
    p = as_col(p)
    if F.shape != (p.size, p.size):
        raise ValueError("F must be dxd and p must have length d.")
    Fs = _sym(F)
    eigs = np.linalg.eigvalsh(Fs)

    def grad_f(x: Array) -> Array:
        return Fs @ as_col(x) + p

    return grad_f, float(eigs[0]), float(eigs[-1])


def quadratic_objective(F: ArrayLike, p: ArrayLike) -> Callable[[Array], float]:
    F = np.asarray(F, dtype=float)
    p = as_col(p)

    def f(x: Array) -> float:
        x = as_col(x)
        return float(0.5 * x @ F @ x + p @ x)

    return f


def solve_ellipsoid_qp(
    H: ArrayLike,
    c: ArrayLike,
    constraint: EllipsoidConstraint,
) -> Array:
    """Solve min_y 0.5*y.T@H@y+c.T@y subject to y.T@Q@y <= r2."""
    H = _sym(np.asarray(H, dtype=float))
    c = as_col(c)
    Q = _sym(np.asarray(constraint.Q, dtype=float))
    r2 = float(constraint.r2)

    def solve_for_lam(lam: float) -> Array:
        return -np.linalg.solve(H + 2.0 * lam * Q, c).reshape(-1)

    def ell_value(y: Array) -> float:
        return float(as_col(y) @ Q @ as_col(y))

    try:
        y0 = solve_for_lam(0.0)
        if ell_value(y0) <= r2 + 1e-10:
            return as_col(y0)
    except np.linalg.LinAlgError:
        pass

    lo, hi = 0.0, 1.0
    while ell_value(solve_for_lam(hi)) > r2:
        hi *= 2.0
        if hi > 1e12:
            raise RuntimeError("Failed to bracket ellipsoid QP multiplier.")

    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if ell_value(solve_for_lam(mid)) > r2:
            lo = mid
        else:
            hi = mid
    return as_col(solve_for_lam(hi))


# -----------------------------------------------------------------------------
# IQC certificate construction
# -----------------------------------------------------------------------------


def build_augmented_from_plant_ss(
    A: ArrayLike,
    B: ArrayLike,
    C: ArrayLike,
    D: ArrayLike,
    m: float,
    L: float,
    rho_bar: float,
) -> IqcAugmentedSystem:
    """Build the augmented system for sector/off-by-one/weighted off-by-one IQCs.

    The plant is scalar-input/scalar-output:
        xi_{k+1} = A xi_k + B u_k,
        y_k      = C xi_k + D u_k.

    The augmented output z stacks three two-dimensional IQC outputs:
        1. sector: [alpha_k; beta_k]
        2. off-by-one: [alpha_k-alpha_{k-1}; beta_k]
        3. weighted off-by-one: [alpha_k-rho_bar^2 alpha_{k-1}; beta_k]
    where alpha_k = L y_k - u_k and beta_k = u_k - m y_k.
    """
    A = np.asarray(A, dtype=float)
    B = _as_matrix(B)
    C = np.asarray(C, dtype=float)
    D = _as_matrix(D)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square.")
    n = A.shape[0]
    if B.shape != (n, 1):
        raise ValueError(f"B must have shape ({n}, 1); got {B.shape}.")
    if C.shape != (1, n):
        raise ValueError(f"C must have shape (1, {n}); got {C.shape}.")
    if D.shape != (1, 1):
        raise ValueError(f"D must have shape (1, 1); got {D.shape}.")
    if not (L > m > 0):
        raise ValueError("Require L > m > 0.")
    if not (0.0 <= rho_bar <= 1.0):
        raise ValueError("Require 0 <= rho_bar <= 1 for the lifted weighted off-by-one IQC.")

    b = float(rho_bar) ** 2
    zcol = np.zeros((n, 1))
    zrow = np.zeros((1, 1))

    # Both dynamic filters store zeta_{k+1} = -alpha_k = -L y_k + u_k.
    Bphi_y = np.array([[-L]])
    Bphi_u = np.array([[1.0]])

    Ahat = np.block(
        [
            [A, zcol, zcol],
            [Bphi_y @ C, np.zeros((1, 1)), zrow],
            [Bphi_y @ C, zrow, np.zeros((1, 1))],
        ]
    )
    Bhat = np.vstack([B, Bphi_y @ D + Bphi_u, Bphi_y @ D + Bphi_u])

    Dphi_y = np.array([[L], [-m]])
    Dphi_u = np.array([[-1.0], [1.0]])

    Chat = np.vstack(
        [
            np.hstack([Dphi_y @ C, np.zeros((2, 1)), np.zeros((2, 1))]),
            np.hstack([Dphi_y @ C, np.array([[1.0], [0.0]]), np.zeros((2, 1))]),
            np.hstack([Dphi_y @ C, np.zeros((2, 1)), np.array([[b], [0.0]])]),
        ]
    )
    Dhat = np.vstack([Dphi_y @ D + Dphi_u] * 3)

    return IqcAugmentedSystem(Ahat, Bhat, Chat, Dhat, float(rho_bar), float(m), float(L))


def build_M_big(alpha: ArrayLike) -> Array:
    """Return kron(diag(alpha), [[0,1],[1,0]])."""
    alpha = np.asarray(alpha, dtype=float).reshape(3)
    return np.kron(np.diag(alpha), np.array([[0.0, 1.0], [1.0, 0.0]]))


def _failed_result(status: str, nx: int, nu: int, rho: float, rho_bar: float) -> dict[str, Any]:
    return {
        "feasible": False,
        "status": status,
        "P": np.full((nx, nx), np.nan),
        "K": np.full((nx + nu, nx + nu), np.nan),
        "alpha": np.full(3, np.nan),
        "delta": np.nan,
        "epsi": np.nan,
        "minEigP": np.nan,
        "maxEigLMI": np.nan,
        "rho": rho,
        "rho_bar": rho_bar,
    }


def solve_iqc_LMI(
    A: ArrayLike,
    B: ArrayLike,
    C: ArrayLike,
    D: ArrayLike,
    m: float,
    L: float,
    rho: Optional[float] = None,
    rho_bar: Optional[float] = None,
    opts: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Verify a candidate rate by solving the sector/Lemma8/Lemma10 IQC LMI."""
    opts = {} if opts is None else dict(opts)
    rho_default = 1.0 - 1.0 / np.sqrt(float(L) / float(m))
    rho = float(rho_default if rho is None else rho)
    rho_bar = float(rho if rho_bar is None else rho_bar)
    if rho <= 0:
        raise ValueError("rho must be positive.")

    sys = build_augmented_from_plant_ss(A, B, C, D, m, L, rho_bar)
    Ahat, Bhat, Chat, Dhat = sys.Ahat, sys.Bhat, sys.Chat, sys.Dhat
    nx, nu = Ahat.shape[0], Bhat.shape[1]
    if nu != 1:
        raise ValueError("This implementation expects scalar algorithm input.")

    epsP = float(opts.get("epsP", 1e-8))
    epsLMI = float(opts.get("epsLMI", 1e-9))
    epsCheck = float(opts.get("epsCheck", 1e-7))
    solver = str(opts.get("solver", "CLARABEL")).upper()
    verbose = bool(int(opts.get("verbose", 0)))
    maximize_margins = bool(opts.get("maximize_margins", False))

    use_flags = [
        bool(opts.get("use_sector", True)),
        bool(opts.get("use_obo", True)),
        bool(opts.get("use_weighted", True)),
    ]
    if not any(use_flags):
        raise ValueError("At least one IQC must be active.")

    try:
        import cvxpy as cp  # type: ignore
    except Exception as exc:
        warnings.warn(
            f"cvxpy is not available: {exc}",
            category=UserWarning,
            stacklevel=2,
        )
        return _failed_result(f"cvxpy-unavailable: {exc}", nx, nu, rho, rho_bar)

    P = cp.Variable((nx, nx), symmetric=True)
    alpha = cp.Variable(3, nonneg=True)
    constraints: list[Any] = [alpha[i] == 0 for i, active in enumerate(use_flags) if not active]

    Z = np.hstack([Chat, Dhat])
    M2 = np.array([[0.0, 1.0], [1.0, 0.0]])
    Mbig = cp.kron(cp.diag(alpha), M2)

    K = cp.bmat(
        [
            [Ahat.T @ P @ Ahat - (rho**2) * P, Ahat.T @ P @ Bhat],
            [Bhat.T @ P @ Ahat, Bhat.T @ P @ Bhat],
        ]
    ) + Z.T @ Mbig @ Z
    Ksym = 0.5 * (K + K.T)

    delta = epsi = None
    if maximize_margins:
        delta = cp.Variable(nonneg=True)
        epsi = cp.Variable(nonneg=True)
        constraints += [
            P >> delta * np.eye(nx),
            Ksym << -epsi * np.eye(nx + nu),
            cp.trace(P) == nx,
        ]
        objective = cp.Maximize(delta + epsi)
    else:
        constraints += [P >> epsP * np.eye(nx), Ksym << -epsLMI * np.eye(nx + nu)]
        objective = cp.Minimize(0)

    prob = cp.Problem(objective, constraints)
    installed = {s.upper() for s in cp.installed_solvers()}
    candidates = [s for s in [solver, "MOSEK", "CLARABEL", "SCS"] if s in installed]

    used_solver = None
    last_error = None
    for cand in candidates:
        try:
            if cand == "SCS":
                prob.solve(solver=cand, verbose=verbose, eps=1e-6, max_iters=200000)
            else:
                prob.solve(solver=cand, verbose=verbose)
        except Exception as exc:
            last_error = exc
            continue
        if prob.status in {"optimal", "optimal_inaccurate"}:
            used_solver = cand
            break

    if used_solver is None or P.value is None or alpha.value is None:
        status = str(prob.status)
        if last_error is not None:
            status += f"; last_error={last_error}"
        return _failed_result(status, nx, nu, rho, rho_bar)

    Pnum = _sym(np.asarray(P.value, dtype=float))
    anum = np.asarray(alpha.value, dtype=float).reshape(3)
    Knum = _sym(
        np.block(
            [
                [Ahat.T @ Pnum @ Ahat - (rho**2) * Pnum, Ahat.T @ Pnum @ Bhat],
                [Bhat.T @ Pnum @ Ahat, Bhat.T @ Pnum @ Bhat],
            ]
        )
        + Z.T @ build_M_big(anum) @ Z
    )

    minEigP = float(np.min(np.linalg.eigvalsh(Pnum)))
    maxEigLMI = float(np.max(np.linalg.eigvalsh(Knum)))
    feasible = bool(
        prob.status in {"optimal", "optimal_inaccurate"}
        and minEigP >= epsP - epsCheck
        and maxEigLMI <= -epsLMI + epsCheck
    )

    return {
        "feasible": feasible,
        "status": str(prob.status),
        "P": Pnum,
        "K": Knum,
        "alpha": anum,
        "delta": float(delta.value) if delta is not None and delta.value is not None else np.nan,
        "epsi": float(epsi.value) if epsi is not None and epsi.value is not None else np.nan,
        "minEigP": minEigP,
        "maxEigLMI": maxEigLMI,
        "rho": rho,
        "rho_bar": rho_bar,
        "Ahat": Ahat,
        "Bhat": Bhat,
        "Chat": Chat,
        "Dhat": Dhat,
        "Mbig": build_M_big(anum),
        "used_solver": used_solver,
    }


def search_best_rho(
    A: ArrayLike,
    B: ArrayLike,
    C: ArrayLike,
    D: ArrayLike,
    m: float,
    L: float,
    opts: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Bisection search for the smallest feasible rho."""
    opts = {} if opts is None else dict(opts)
    rhoLo = float(opts.get("rhoLo", 1e-6))
    rhoHi = float(opts.get("rhoHi", 1.0))
    tol = float(opts.get("tol", 1e-5))
    maxIter = int(opts.get("maxIter", 40))
    verbose = int(opts.get("verbose", 0))
    rho_bar_fixed = opts.get("rho_bar", None)

    if not (0 < rhoLo < rhoHi <= 1.0):
        raise ValueError("Require 0 < rhoLo < rhoHi <= 1.")

    def check(r: float) -> dict[str, Any]:
        rb = float(r if rho_bar_fixed is None else rho_bar_fixed)
        return solve_iqc_LMI(A, B, C, D, m, L, rho=r, rho_bar=rb, opts=opts)

    hi_result = check(rhoHi)
    if not hi_result["feasible"]:
        out = dict(hi_result)
        out.update({"rho": None, "rhoLowerBound": rhoLo, "rhoUpperBound": rhoHi})
        return out

    lo, hi, best = rhoLo, rhoHi, hi_result
    for k in range(maxIter):
        mid = 0.5 * (lo + hi)
        res = check(mid)
        if res["feasible"]:
            hi, best = mid, res
        else:
            lo = mid
        if verbose:
            print(
                f"iter={k + 1:02d}, rho={mid:.10f}, feasible={int(res['feasible'])}, "
                f"interval=[{lo:.10f},{hi:.10f}], status={res['status']}"
            )
        if hi - lo <= tol:
            break
    best = dict(best)
    best.update({"rhoLowerBound": lo, "rhoUpperBound": hi})
    return best


# -----------------------------------------------------------------------------
# Algorithm matrices and simulations
# -----------------------------------------------------------------------------


def triple_momentum_matrices(m: float, L: float) -> Tuple[Array, Array, Array, Array, dict[str, float]]:
    """Triple momentum matrices in the canonical coordinates used by projection code."""
    rho = 1.0 - 1.0 / np.sqrt(L / m)
    alpha = (1.0 + rho) / L
    beta = rho**2 / (2.0 - rho)
    gamma = rho**2 / ((1.0 + rho) * (2.0 - rho))

    A0 = np.array([[1.0 + beta, -beta], [1.0, 0.0]])
    B0 = np.array([[-alpha], [0.0]])
    C0 = np.array([[1.0 + gamma, -gamma]])
    T = np.array([[1.0 + gamma, -gamma], [0.0, 1.0]])
    V = np.linalg.inv(T)

    return T @ A0 @ V, T @ B0, C0 @ V, np.zeros((1, 1)), {
        "rho": rho,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
    }


def proj_gradient(
    A: float | ArrayLike,
    B: float | ArrayLike,
    C: float | ArrayLike,
    D: float | ArrayLike,
    x0: ArrayLike,
    num_steps: int,
    grad_f: Callable[[Array], Array],
    proj_fun: Callable[[Array], Array],
) -> Results:
    x0 = as_col(x0)
    d = x0.size
    Ad = np.asarray(A, dtype=float).reshape(1, 1) @ np.eye(1)
    Bd = np.asarray(B, dtype=float).reshape(1, 1) @ np.eye(1)
    Cd = np.asarray(C, dtype=float).reshape(1, 1) @ np.eye(1)
    Ad, Bd, Cd = np.kron(Ad, np.eye(d)), np.kron(Bd, np.eye(d)), np.kron(Cd, np.eye(d))

    x = np.zeros((d, num_steps + 1))
    x_temp = np.zeros_like(x)
    u = np.zeros((d, num_steps))
    x[:, 0] = x_temp[:, 0] = x0

    for k in range(num_steps):
        u[:, k] = grad_f(Cd @ x[:, k])
        x_temp[:, k + 1] = Ad @ x[:, k] + Bd @ u[:, k]
        x[:, k + 1] = proj_fun(x_temp[:, k + 1])
    return Results(x=x, x_temp=x_temp, u=u)


def _prepare_two_state_sim(A: ArrayLike, B: ArrayLike, C: ArrayLike, x0: ArrayLike) -> tuple[Array, Array, Array, Array, int]:
    x0 = as_col(x0)
    if x0.size % 2 != 0:
        raise ValueError("x0 length must be 2*d.")
    d = x0.size // 2
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    return np.kron(A, np.eye(d)), np.kron(B, np.eye(d)), np.kron(C, np.eye(d)), x0, d


def projected_triple_momentum_Euclidean(
    A: ArrayLike,
    B: ArrayLike,
    C: ArrayLike,
    D: ArrayLike,
    x0: ArrayLike,
    num_steps: int,
    grad_f: Callable[[Array], Array],
    proj_fun: Callable[[Array], Array],
) -> Results:
    Ad, Bd, Cd, x0, d = _prepare_two_state_sim(A, B, C, x0)
    n, m_in, p_out = Ad.shape[0], Bd.shape[1], Cd.shape[0]
    x = np.zeros((n, num_steps + 1))
    x_temp = np.zeros_like(x)
    y = np.zeros((p_out, num_steps))
    u = np.zeros((m_in, num_steps))
    x[:, 0] = x_temp[:, 0] = x0

    for k in range(num_steps):
        u[:, k] = grad_f(Cd @ x[:, k])
        x_temp[:, k + 1] = Ad @ x[:, k] + Bd @ u[:, k]
        y_proj = proj_fun(x_temp[:d, k + 1])
        x[:d, k + 1] = y_proj
        x[d:, k + 1] = x_temp[d:, k + 1]
        y[:, k] = y_proj
    return Results(x=x, y=y, u=u, x_temp=x_temp)


def projected_triple_momentum_P_norm(
    A: ArrayLike,
    B: ArrayLike,
    C: ArrayLike,
    D: ArrayLike,
    x0: ArrayLike,
    num_steps: int,
    grad_f: Callable[[Array], Array],
    proj_fun: Callable[[Array], Array],
    P: ArrayLike,
) -> Results:
    Ad, Bd, Cd, x0, d = _prepare_two_state_sim(A, B, C, x0)
    P = np.asarray(P, dtype=float)
    if P.shape[0] < 2:
        raise ValueError("P must contain at least the two algorithmic-state coordinates.")

    h = np.linalg.solve(P[1:, 1:], P[0, 1:].reshape(-1, 1)).reshape(-1)
    H = np.kron(h.reshape(-1, 1), np.eye(d))

    n, m_in, p_out = Ad.shape[0], Bd.shape[1], Cd.shape[0]
    x = np.zeros((n, num_steps + 1))
    x_temp = np.zeros_like(x)
    y = np.zeros((p_out, num_steps))
    u = np.zeros((m_in, num_steps))
    x[:, 0] = x_temp[:, 0] = x0

    for k in range(num_steps):
        u[:, k] = grad_f(Cd @ x[:, k])
        x_temp[:, k + 1] = Ad @ x[:, k] + Bd @ u[:, k]
        y_proj = proj_fun(x_temp[:d, k + 1])
        x[:d, k + 1] = y_proj
        shift = H @ (y_proj - x_temp[:d, k + 1])
        x[d:, k + 1] = x_temp[d:, k + 1] - shift[:d]
        y[:, k] = y_proj
    return Results(x=x, y=y, u=u, x_temp=x_temp)


__all__ = [
    "Array",
    "Results",
    "EllipsoidConstraint",
    "IqcAugmentedSystem",
    "as_col",
    "build_projection",
    "analyze_quadratic_function",
    "quadratic_objective",
    "solve_ellipsoid_qp",
    "build_augmented_from_plant_ss",
    "build_M_big",
    "solve_iqc_LMI",
    "search_best_rho",
    "triple_momentum_matrices",
    "proj_gradient",
    "projected_triple_momentum_Euclidean",
    "projected_triple_momentum_P_norm",
]
