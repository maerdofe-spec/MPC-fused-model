"""Convex QP backends used by the MPC module.

It solves

    min 0.5*x.T@P@x + q.T@x
    subject to lower <= A@x <= upper

OSQP is preferred for real-time control.  A NumPy-only ADMM implementation is
kept as a portable fallback for simulation and environments where OSQP has not
yet been installed.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

try:  # Optional at import time; ``create_qp_solver`` selects it when present.
    import osqp
    from scipy import sparse
except ImportError:  # pragma: no cover - availability depends on deployment.
    osqp = None
    sparse = None


@dataclass
class QPSolverSettings:
    backend: str = "auto"
    rho: float = 2.0
    sigma: float = 1e-8
    max_iterations: int = 4000
    absolute_tolerance: float = 1e-5
    relative_tolerance: float = 2e-4
    adaptive_rho: bool = True
    adaptive_interval: int = 25
    adaptive_tolerance: float = 5.0
    rho_min: float = 1e-4
    rho_max: float = 1e5
    time_limit_seconds: float = 0.040
    check_termination_interval: int = 10


@dataclass
class QPSolution:
    x: np.ndarray
    status: str
    iterations: int
    objective: float
    primal_residual: float
    dual_residual: float

    @property
    def solved(self) -> bool:
        return self.status in {"solved", "solved_inaccurate"}


def _validated_problem(P, q, A, lower, upper):
    P = np.asarray(P, dtype=float)
    q = np.asarray(q, dtype=float).reshape(-1)
    A = np.asarray(A, dtype=float)
    lower = np.asarray(lower, dtype=float).reshape(-1)
    upper = np.asarray(upper, dtype=float).reshape(-1)

    n = q.size
    m = lower.size
    if P.shape != (n, n) or A.shape != (m, n) or upper.shape != (m,):
        raise ValueError("incompatible QP dimensions")
    if np.any(lower > upper):
        raise ValueError("lower constraint bound exceeds upper bound")
    if not all(np.all(np.isfinite(v)) for v in (P, q, A)):
        raise ValueError("P, q, and A must be finite")
    return 0.5 * (P + P.T), q, A, lower, upper


class OSQPQPSolver:
    """Sparse real-time QP backend with a hard per-call time budget."""

    backend_name = "osqp"

    @staticmethod
    def available() -> bool:
        return osqp is not None and sparse is not None

    def __init__(self, settings: QPSolverSettings | None = None) -> None:
        if not self.available():
            raise ImportError("OSQP backend requires the osqp and scipy packages")
        self.settings = settings or QPSolverSettings()

    def solve(self, P, q, A, lower, upper, warm_start=None) -> QPSolution:
        P, q, A, lower, upper = _validated_problem(P, q, A, lower, upper)
        settings = self.settings
        if settings.rho <= 0.0 or settings.sigma <= 0.0:
            raise ValueError("rho and sigma must be positive")

        problem = osqp.OSQP()
        setup_settings = dict(
            verbose=False,
            warm_starting=True,
            polishing=False,
            rho=float(settings.rho),
            sigma=float(settings.sigma),
            max_iter=int(settings.max_iterations),
            eps_abs=float(settings.absolute_tolerance),
            eps_rel=float(settings.relative_tolerance),
            adaptive_rho=bool(settings.adaptive_rho),
            adaptive_rho_interval=int(settings.adaptive_interval),
            adaptive_rho_tolerance=float(settings.adaptive_tolerance),
            check_termination=int(settings.check_termination_interval),
        )
        if settings.time_limit_seconds > 0.0:
            setup_settings["time_limit"] = float(settings.time_limit_seconds)
        problem.setup(
            P=sparse.csc_matrix(np.triu(P)),
            q=q,
            A=sparse.csc_matrix(A),
            l=lower,
            u=upper,
            **setup_settings,
        )
        if warm_start is not None:
            candidate = np.asarray(warm_start, dtype=float).reshape(-1)
            if candidate.shape == q.shape and np.all(np.isfinite(candidate)):
                problem.warm_start(x=candidate)
        result = problem.solve(raise_error=False)
        status = str(result.info.status).strip().lower().replace(" ", "_")
        x = np.asarray(result.x, dtype=float) if result.x is not None else np.zeros_like(q)
        if x.shape != q.shape or not np.all(np.isfinite(x)):
            x = np.zeros_like(q)
            status = "numerical_error"
        return QPSolution(
            x=x,
            status=status,
            iterations=int(result.info.iter),
            objective=float(result.info.obj_val),
            primal_residual=float(result.info.prim_res),
            dual_residual=float(result.info.dual_res),
        )


class DenseADMMQPSolver:
    backend_name = "numpy_admm"

    def __init__(self, settings: QPSolverSettings | None = None) -> None:
        self.settings = settings or QPSolverSettings()

    def solve(self, P, q, A, lower, upper, warm_start=None) -> QPSolution:
        started = time.perf_counter()
        P, q, A, lower, upper = _validated_problem(P, q, A, lower, upper)
        n = q.size
        m = lower.size
        settings = self.settings
        rho = float(settings.rho)
        if rho <= 0.0 or settings.sigma <= 0.0:
            raise ValueError("rho and sigma must be positive")

        AtA = A.T @ A

        def factorize(current_rho):
            regularized = P + current_rho * AtA + settings.sigma * np.eye(n)
            try:
                # Compute the linear solution operator once per rho value.
                # The previous implementation called two generic dense solves
                # during every ADMM iteration, which missed a 20 Hz deadline.
                return np.linalg.solve(regularized, np.eye(n))
            except np.linalg.LinAlgError as error:
                raise ValueError("QP KKT matrix is not positive definite") from error

        factor = factorize(rho)

        if warm_start is None:
            x = np.zeros(n)
        else:
            x = np.asarray(warm_start, dtype=float).reshape(-1).copy()
            if x.shape != (n,) or not np.all(np.isfinite(x)):
                x = np.zeros(n)

        Ax = A @ x
        projected = np.minimum(np.maximum(Ax, lower), upper)
        dual = np.zeros(m)
        primal_norm = np.inf
        dual_norm = np.inf

        def linear_solve(rhs):
            return factor @ rhs

        status = "maximum_iterations"
        iterations = settings.max_iterations
        for iteration in range(1, settings.max_iterations + 1):
            rhs = -q + rho * A.T @ (projected - dual)
            x = linear_solve(rhs)

            Ax = A @ x
            previous_projected = projected.copy()
            projected = np.minimum(np.maximum(Ax + dual, lower), upper)
            dual += Ax - projected

            primal = Ax - projected
            dual_residual = rho * A.T @ (projected - previous_projected)
            primal_norm = float(np.linalg.norm(primal, np.inf))
            dual_norm = float(np.linalg.norm(dual_residual, np.inf))

            primal_tolerance = settings.absolute_tolerance + settings.relative_tolerance * max(
                float(np.linalg.norm(Ax, np.inf)),
                float(np.linalg.norm(projected, np.inf)),
            )
            dual_tolerance = settings.absolute_tolerance + settings.relative_tolerance * max(
                float(np.linalg.norm(P @ x, np.inf)),
                float(np.linalg.norm(A.T @ (rho * dual), np.inf)),
                float(np.linalg.norm(q, np.inf)),
            )

            if primal_norm <= primal_tolerance and dual_norm <= dual_tolerance:
                status = "solved"
                iterations = iteration
                break

            if (
                settings.time_limit_seconds > 0.0
                and iteration % settings.check_termination_interval == 0
                and time.perf_counter() - started >= settings.time_limit_seconds
            ):
                status = "time_limit"
                iterations = iteration
                break

            if (
                settings.adaptive_rho
                and iteration % settings.adaptive_interval == 0
                and primal_norm > 0.0
                and dual_norm > 0.0
            ):
                old_rho = rho
                if primal_norm > settings.adaptive_tolerance * dual_norm:
                    rho = min(rho * 2.0, settings.rho_max)
                elif dual_norm > settings.adaptive_tolerance * primal_norm:
                    rho = max(rho / 2.0, settings.rho_min)
                if rho != old_rho:
                    # dual stores the scaled multiplier y/rho.
                    dual *= old_rho / rho
                    factor = factorize(rho)

        objective = float(0.5 * x @ P @ x + q @ x)
        if not np.all(np.isfinite(x)):
            status = "numerical_error"

        return QPSolution(
            x=x,
            status=status,
            iterations=iterations,
            objective=objective,
            primal_residual=primal_norm,
            dual_residual=dual_norm,
        )


def create_qp_solver(settings: QPSolverSettings | None = None):
    """Select OSQP when installed, otherwise use the NumPy ADMM fallback."""
    settings = settings or QPSolverSettings()
    backend = settings.backend.strip().lower()
    if backend not in {"auto", "osqp", "numpy_admm"}:
        raise ValueError("QP backend must be 'auto', 'osqp', or 'numpy_admm'")
    if backend in {"auto", "osqp"} and OSQPQPSolver.available():
        return OSQPQPSolver(settings)
    if backend == "osqp":
        raise ImportError("OSQP was requested but osqp/scipy are not installed")
    return DenseADMMQPSolver(settings)

