"""Constrained MPC for the absolute-force model-2 dynamics.

Model 2 is intentionally kept separate from the dual-model controller.  Its
prediction is exactly

    x[k+1] = A_d x[k] + B_d tau[k]

``tau_base`` is not part of this prediction or its input cost.  The inherited
controller machinery is used only for the common QP constraints and solver;
the prediction matrices and cost are rebuilt here so that a future change in
the fusion controller cannot silently change model 2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from MPC_model1.mpc_controller import (
    MPCConfig as _Model1MPCConfig,
    MPCResult,
    RelativeMPCController as _Model1Controller,
    _block_diagonal,
)


@dataclass
class MPCConfig(_Model1MPCConfig):
    """Model-2 weights and constraints.

    Independent terminal position and velocity multipliers are optional in
    the mathematical structure.  Keeping both at ``4`` reproduces the old
    single terminal multiplier, while exposing them separately lets tuning
    distinguish terminal position accuracy from terminal velocity damping.
    """

    terminal_position_weight_scale: float | None = None
    terminal_velocity_weight_scale: float | None = None

    def normalized(self) -> "MPCConfig":
        super().normalized()
        if self.terminal_position_weight_scale is None:
            self.terminal_position_weight_scale = self.terminal_weight_scale
        if self.terminal_velocity_weight_scale is None:
            self.terminal_velocity_weight_scale = self.terminal_weight_scale
        if (
            self.terminal_position_weight_scale <= 0.0
            or self.terminal_velocity_weight_scale <= 0.0
        ):
            raise ValueError("terminal position and velocity scales must be positive")
        return self


class RelativeMPCController(_Model1Controller):
    """Finite-horizon QP controller using pure model-2 prediction."""

    def safe_force(
        self,
        tau_previous,
        target_force: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return to zero force without consulting ``tau_base``."""
        previous = np.asarray(tau_previous, dtype=float).reshape(3)
        target = (
            np.zeros(3)
            if target_force is None
            else np.asarray(target_force, dtype=float).reshape(3)
        )
        return self._safe_fallback(previous, target)

    def solve(
        self,
        state,
        tau_previous,
        reference_position=None,
    ) -> MPCResult:
        """Solve model 2; zero is used only for the safety fallback."""
        return super().solve(
            state=state,
            tau_previous=tau_previous,
            tau_base=np.zeros(3),
            reference_position=reference_position,
        )

    def _build_prediction_matrices(self) -> None:
        """Build ``x+ = A_d*x + B_d*tau`` with a rolling input state."""
        physical_A = self.model.A_d
        physical_B = self.model.B_d

        # z=[x, tau_previous].  tau_previous is needed for rate constraints,
        # but has zero influence on the model-2 physical state.
        A = np.zeros((9, 9))
        B = np.zeros((9, 3))
        A[:6, :6] = physical_A
        B[:6] = physical_B
        B[6:] = np.eye(3)

        output = np.hstack((np.eye(6), np.zeros((6, 3))))
        N = self.config.horizon
        nz, nu = A.shape[0], B.shape[1]
        nx = 6

        self.Sx = np.zeros((N * nx, nz))
        self.Su = np.zeros((N * nx, N * nu))
        powers = [np.eye(nz)]
        for _ in range(N):
            powers.append(powers[-1] @ A)
        for i in range(N):
            self.Sx[i * nx : (i + 1) * nx] = output @ powers[i + 1]
            for j in range(i + 1):
                self.Su[
                    i * nx : (i + 1) * nx,
                    j * nu : (j + 1) * nu,
                ] = output @ powers[i - j] @ B

        self.rate_matrix = np.zeros((N * nu, N * nu))
        for i in range(N):
            self.rate_matrix[
                i * nu : (i + 1) * nu,
                i * nu : (i + 1) * nu,
            ] = np.eye(nu)
            if i > 0:
                self.rate_matrix[
                    i * nu : (i + 1) * nu,
                    (i - 1) * nu : i * nu,
                ] = -np.eye(nu)

        # Model 2 penalizes absolute force, while the separate S term keeps
        # force changes smooth.  Do not use tau_base as a force reference.
        self.effective_force_matrix = np.eye(N * nu)

    def _cost(
        self,
        free_prediction: np.ndarray,
        reference_position: np.ndarray,
        tau_previous: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        N = cfg.horizon
        Q_stage = np.diag(
            np.concatenate((cfg.position_weights, cfg.velocity_weights))
        )
        Q_terminal = np.diag(
            np.concatenate(
                (
                    cfg.position_weights * cfg.terminal_position_weight_scale,
                    cfg.velocity_weights * cfg.terminal_velocity_weight_scale,
                )
            )
        )
        Q_bar = _block_diagonal(
            [Q_terminal if i == N - 1 else Q_stage for i in range(N)]
        )
        R_bar = np.kron(np.eye(N), np.diag(cfg.force_weights))
        S_bar = np.kron(np.eye(N), np.diag(cfg.delta_force_weights))

        reference = self._reference_stack(reference_position)
        error_free = free_prediction - reference
        rate_reference = np.zeros(3 * N)
        rate_reference[:3] = tau_previous
        # The absolute-force penalty is referenced to zero.  A nonzero
        # tau_base belongs to model 1/fallback handling, not model 2.
        effective_force_reference = np.zeros(3 * N)

        P_force = 2.0 * (
            self.Su.T @ Q_bar @ self.Su
            + self.effective_force_matrix.T
            @ R_bar
            @ self.effective_force_matrix
            + self.rate_matrix.T @ S_bar @ self.rate_matrix
        )
        q_force = 2.0 * (
            self.Su.T @ Q_bar @ error_free
            - self.effective_force_matrix.T
            @ R_bar
            @ effective_force_reference
            - self.rate_matrix.T @ S_bar @ rate_reference
        )

        n_force = 3 * N
        n_slack = self.SLACKS_PER_STEP * N
        P = np.zeros((n_force + n_slack, n_force + n_slack))
        q = np.zeros(n_force + n_slack)
        P[:n_force, :n_force] = P_force
        P[n_force:, n_force:] = (
            2.0 * cfg.slack_quadratic_weight * np.eye(n_slack)
        )
        q[:n_force] = q_force
        q[n_force:] = cfg.slack_linear_weight
        P += 1e-9 * np.eye(P.shape[0])
        return P, q


__all__ = ["MPCConfig", "MPCResult", "RelativeMPCController"]
