"""Rotation-aware dual-model translation MPC.

Yaw is not a QP decision.  A yaw state machine and cascaded PID first freeze a
future yaw trajectory; this controller then solves a three-input convex QP for
translation force increments only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from MPC_dual_model.dense_qp import QPSolverSettings, create_qp_solver
from MPC_dual_model.fossen_fixed_dl_model import vector3
from MPC_dual_model.mpc_controller import _block_diagonal

from .yaw_controller import YawPrediction
from .yaw_relative_model import (
    RotationAwareRelativeModel,
    finite_scalar,
    rotation_body_from_previous,
    visibility_frame_geometry,
)


Array = np.ndarray


def _positive_vector3(value, name: str) -> Array:
    result = vector3(value, name)
    if np.any(result < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return result


@dataclass
class YawMPCConfig:
    """Translation-QP weights and constraints in physical SI units."""

    horizon: int = 10
    reference_position: object = (0.60, 0.0, 0.0)

    position_weights: object = (50.0, 80.0, 100.0)
    velocity_weights: object = (8.0, 10.0, 12.0)
    terminal_weight_scale: float = 4.0
    force_weights: object = (0.04, 0.06, 0.06)
    delta_force_weights: object = (0.8, 1.0, 1.0)

    force_min: object = (-20.0, -15.0, -15.0)
    force_max: object = (20.0, 15.0, 15.0)
    delta_force_min: object = (-4.0, -3.0, -3.0)
    delta_force_max: object = (4.0, 3.0, 3.0)

    # Optional actuator model.  A measured planar matrix has rows
    # [F_frd, N_yaw]; a full matrix may additionally include roll/pitch rows
    # and then has rows [F_frd, M_roll, M_pitch, N_yaw].  Supporting the
    # planar form avoids inventing roll/pitch moment arms when only the real
    # M1..M8 translation directions and yaw arms are known.
    thruster_wrench_matrix: object | None = None
    thruster_force_min: object | None = None
    thruster_force_max: object | None = None
    thruster_force_regularization: float = 1.0e-5

    forward_distance_min: float = 0.25
    forward_distance_max: float = 1.50
    horizontal_half_fov_deg: float = 42.0
    vertical_half_fov_deg: float = 30.0
    fov_margin_deg: float = 5.0
    forward_axis: int = 0
    horizontal_axis: int = 1
    vertical_axis: int = 2

    # FOV/range limits are evaluated at the camera optical centre. The
    # visibility frame order is [camera forward, camera right, camera down].
    rotation_visibility_from_body: object = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    camera_origin_in_body: object = (0.0, 0.0, 0.0)

    slack_quadratic_weight: float = 2.0e4
    slack_linear_weight: float = 50.0
    slack_max: float = 5.0

    solver_settings: QPSolverSettings = field(
        default_factory=lambda: QPSolverSettings(
            backend="osqp",
            rho=10.0,
            sigma=1.0e-8,
            max_iterations=1800,
            absolute_tolerance=2.0e-5,
            relative_tolerance=3.0e-4,
        )
    )

    def normalized(self) -> "YawMPCConfig":
        if self.horizon < 1:
            raise ValueError("horizon must be at least 1")
        self.reference_position = vector3(self.reference_position, "reference_position")
        self.position_weights = _positive_vector3(
            self.position_weights, "position_weights"
        )
        self.velocity_weights = _positive_vector3(
            self.velocity_weights, "velocity_weights"
        )
        self.force_weights = _positive_vector3(self.force_weights, "force_weights")
        self.delta_force_weights = _positive_vector3(
            self.delta_force_weights, "delta_force_weights"
        )
        self.force_min = vector3(self.force_min, "force_min")
        self.force_max = vector3(self.force_max, "force_max")
        self.delta_force_min = vector3(self.delta_force_min, "delta_force_min")
        self.delta_force_max = vector3(self.delta_force_max, "delta_force_max")
        if self.thruster_wrench_matrix is not None:
            matrix = np.asarray(self.thruster_wrench_matrix, dtype=float)
            if (
                matrix.ndim != 2
                or matrix.shape[0] not in (4, 6)
                or matrix.shape[1] < matrix.shape[0]
                or not np.all(np.isfinite(matrix))
            ):
                raise ValueError(
                    "thruster_wrench_matrix must have shape (4, n) or (6, n)"
                )
            if np.linalg.matrix_rank(matrix) < matrix.shape[0]:
                raise ValueError(
                    "thruster_wrench_matrix must span its configured wrench space"
                )
            self.thruster_wrench_matrix = matrix
        if self.thruster_wrench_matrix is None:
            if (
                self.thruster_force_min is not None
                or self.thruster_force_max is not None
            ):
                raise ValueError(
                    "thruster force bounds require thruster_wrench_matrix"
                )
        else:
            thruster_count = self.thruster_wrench_matrix.shape[1]
            self.thruster_force_min = self._normalize_thruster_bound(
                self.thruster_force_min,
                -1.0,
                thruster_count,
                "thruster_force_min",
            )
            self.thruster_force_max = self._normalize_thruster_bound(
                self.thruster_force_max,
                1.0,
                thruster_count,
                "thruster_force_max",
            )
            if np.any(self.thruster_force_min >= 0.0) or np.any(
                self.thruster_force_max <= 0.0
            ):
                raise ValueError("each thruster interval must contain zero")
        if (
            not np.isfinite(self.thruster_force_regularization)
            or self.thruster_force_regularization < 0.0
        ):
            raise ValueError("thruster_force_regularization must be finite and nonnegative")
        if np.any(self.force_min >= self.force_max):
            raise ValueError("force_min must be smaller than force_max")
        if np.any(self.delta_force_min >= self.delta_force_max):
            raise ValueError("delta_force_min must be smaller than delta_force_max")
        if not 0.0 < self.forward_distance_min < self.forward_distance_max:
            raise ValueError("invalid forward distance limits")
        if not 0.0 < self.fov_margin_deg < min(
            self.horizontal_half_fov_deg, self.vertical_half_fov_deg
        ):
            raise ValueError("invalid FOV margin")
        axes = {self.forward_axis, self.horizontal_axis, self.vertical_axis}
        if axes != {0, 1, 2}:
            raise ValueError("forward/horizontal/vertical axes must permute 0,1,2")
        self.rotation_visibility_from_body, self.camera_origin_in_body = (
            visibility_frame_geometry(
                self.rotation_visibility_from_body,
                self.camera_origin_in_body,
            )
        )
        if self.terminal_weight_scale <= 0.0:
            raise ValueError("terminal_weight_scale must be positive")
        if self.slack_quadratic_weight <= 0.0 or self.slack_linear_weight < 0.0:
            raise ValueError("invalid slack weights")
        if self.slack_max <= 0.0:
            raise ValueError("slack_max must be positive")
        return self

    @staticmethod
    def _normalize_thruster_bound(
        value,
        default: float,
        size: int,
        name: str,
    ) -> Array:
        if value is None:
            return np.full(size, default, dtype=float)
        result = np.asarray(value, dtype=float).reshape(-1)
        if result.shape != (size,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must contain {size} finite values")
        return result


@dataclass
class YawMPCResult:
    force: Array
    yaw_moment: float
    force_sequence: Array
    delta_force_sequence: Array
    thruster_force_sequence: Array
    predicted_states: Array
    predicted_augmented_states: Array
    frozen_yaw_angles: Array
    frozen_yaw_rates: Array
    frozen_delta_yaw: Array
    frozen_yaw_moments: Array
    slacks: Array
    status: str
    iterations: int
    objective: float
    used_fallback: bool
    model1_weight: Array
    model2_weight: Array

    @property
    def input_sequence(self) -> Array:
        """Compatibility alias: the QP now returns absolute force sequence."""
        return self.force_sequence

    @property
    def thruster_force(self) -> Array:
        """First exact per-thruster allocation selected by the QP."""
        return self.thruster_force_sequence[0]


class RotationAwareMPCController:
    """Frozen-rotation LTV MPC with delta-force decision variables.

    The internal state is x=[e, v, tau_previous], where e=p-p_d. For each
    frozen R_j and model-1 weight W, the maintained recursion is

        v+ = R F v + R G tau
             - W R G tau_base - (I-W) R G tau_h
        e+ = R e + dt v+ + (R-I) p_d
        tau+ = tau_previous + delta_tau.

    ``tau_base`` is the achieved force preceding this solve and is fixed for
    the entire horizon. ``tau_h`` is the immutable Fossen restoring force.
    The objective penalizes absolute total force and adjacent force changes.
    """

    AUGMENTED_DIM = 9
    INPUT_DIM = 3
    SLACKS_PER_STEP = 6

    def __init__(
        self,
        model: RotationAwareRelativeModel,
        config: YawMPCConfig | None = None,
    ) -> None:
        self.model = model
        self.config = (config or YawMPCConfig()).normalized()
        self.solver = create_qp_solver(self.config.solver_settings)
        self._warm_start: Array | None = None
        self._last_feasible_force: Array | None = None
        self._last_feasible_thruster_force: Array | None = None

    def reset(self) -> None:
        self._warm_start = None
        self._last_feasible_force = None
        self._last_feasible_thruster_force = None

    @property
    def thruster_count(self) -> int:
        matrix = self.config.thruster_wrench_matrix
        return 0 if matrix is None else int(matrix.shape[1])

    def _decision_layout(self) -> tuple[int, int, int, int]:
        n_input = self.INPUT_DIM * self.config.horizon
        n_thruster = self.thruster_count * self.config.horizon
        n_slack = self.SLACKS_PER_STEP * self.config.horizon
        return n_input, n_thruster, n_slack, n_input + n_thruster + n_slack

    def safe_force(
        self,
        force_previous,
        force_target=None,
        yaw_moment: float = 0.0,
    ) -> Array:
        previous = vector3(force_previous, "force_previous")
        target = (
            self.model.translation.restoring_force
            if force_target is None
            else vector3(force_target, "force_target")
        )
        return self._safe_fallback(previous, target, yaw_moment=yaw_moment)

    def _validate_yaw_prediction(self, prediction: YawPrediction) -> None:
        N = self.config.horizon
        expected = (
            (prediction.angles, (N + 1,), "angles"),
            (prediction.rates, (N + 1,), "rates"),
            (prediction.delta_angles, (N,), "delta_angles"),
            (prediction.moments, (N,), "moments"),
        )
        for values, shape, name in expected:
            array = np.asarray(values, dtype=float)
            if array.shape != shape or not np.all(np.isfinite(array)):
                raise ValueError(f"yaw prediction {name} must have shape {shape}")

    def _build_prediction_matrices(
        self,
        reference_position: Array,
        model1_weight: Array,
        yaw_prediction: YawPrediction,
        tau_base: Array,
    ) -> None:
        cfg = self.config
        N = cfg.horizon
        nz = self.AUGMENTED_DIM
        nu = self.INPUT_DIM
        W = np.diag(model1_weight)
        I = np.eye(3)
        F = self.model.translation.F
        G = self.model.translation.G
        tau_h = self.model.translation.restoring_force
        dt = self.model.dt

        transitions: list[Array] = []
        inputs: list[Array] = []
        constants: list[Array] = []
        for step in range(N):
            rotation = rotation_body_from_previous(
                yaw_prediction.delta_angles[step]
            )
            rotated_F = rotation @ F
            rotated_G = rotation @ G
            weighted_baseline_gain = W @ rotated_G
            weighted_restoring_gain = (I - W) @ rotated_G

            A = np.zeros((nz, nz))
            B = np.zeros((nz, nu))
            c = np.zeros(nz)
            A[:3, :3] = rotation
            A[:3, 3:6] = dt * rotated_F
            A[:3, 6:9] = dt * rotated_G
            A[3:6, 3:6] = rotated_F
            A[3:6, 6:9] = rotated_G
            A[6:9, 6:9] = np.eye(3)
            B[:3] = dt * rotated_G
            B[3:6] = rotated_G
            B[6:9] = np.eye(3)
            c[:3] = (
                (rotation - I) @ reference_position
                - dt * weighted_baseline_gain @ tau_base
                - dt * weighted_restoring_gain @ tau_h
            )
            c[3:6] = (
                -weighted_baseline_gain @ tau_base
                - weighted_restoring_gain @ tau_h
            )
            transitions.append(A)
            inputs.append(B)
            constants.append(c)

        self.Sx = np.zeros((N * nz, nz))
        self.Su = np.zeros((N * nz, N * nu))
        self.Sc = np.zeros(N * nz)
        state_gain = np.eye(nz)
        input_gain = np.zeros((nz, N * nu))
        affine = np.zeros(nz)
        for step, (A, B, c) in enumerate(zip(transitions, inputs, constants)):
            state_gain = A @ state_gain
            input_gain = A @ input_gain
            input_gain[:, step * nu : (step + 1) * nu] += B
            affine = A @ affine + c
            rows = slice(step * nz, (step + 1) * nz)
            self.Sx[rows] = state_gain
            self.Su[rows] = input_gain
            self.Sc[rows] = affine

    def _cost(self, free_prediction: Array) -> tuple[Array, Array]:
        cfg = self.config
        N = cfg.horizon
        Q_stage = np.diag(
            np.concatenate(
                (cfg.position_weights, cfg.velocity_weights, cfg.force_weights)
            )
        )
        Q_terminal = np.diag(
            np.concatenate(
                (
                    cfg.terminal_weight_scale * cfg.position_weights,
                    cfg.terminal_weight_scale * cfg.velocity_weights,
                    cfg.force_weights,
                )
            )
        )
        Q_bar = _block_diagonal(
            [Q_terminal if step == N - 1 else Q_stage for step in range(N)]
        )
        S_bar = np.kron(np.eye(N), np.diag(cfg.delta_force_weights))
        hessian = self.Su.T @ Q_bar @ self.Su + S_bar
        gradient = self.Su.T @ Q_bar @ free_prediction

        n_input, n_thruster, n_slack, n_variable = self._decision_layout()
        thruster_start = n_input
        slack_start = n_input + n_thruster
        P = np.zeros((n_variable, n_variable))
        q = np.zeros(n_variable)
        P[:n_input, :n_input] = 2.0 * hessian
        if n_thruster:
            P[thruster_start:slack_start, thruster_start:slack_start] = (
                2.0 * cfg.thruster_force_regularization * np.eye(n_thruster)
            )
        P[slack_start:, slack_start:] = (
            2.0 * cfg.slack_quadratic_weight * np.eye(n_slack)
        )
        q[:n_input] = 2.0 * gradient
        q[slack_start:] = cfg.slack_linear_weight
        P += 1.0e-9 * np.eye(P.shape[0])
        return P, q

    def _constraints(
        self,
        free_prediction: Array,
        reference_position: Array,
        yaw_prediction: YawPrediction,
        rotation_visibility_from_body: Array,
        camera_origin_in_body: Array,
    ) -> tuple[Array, Array, Array]:
        cfg = self.config
        N = cfg.horizon
        nz = self.AUGMENTED_DIM
        n_input, n_thruster, n_slack, n_variable = self._decision_layout()
        thruster_start = n_input
        slack_start_global = n_input + n_thruster
        rows: list[Array] = []
        lower: list[float] = []
        upper: list[float] = []

        def append_row(coefficients, lower_bound, upper_bound) -> None:
            rows.append(np.asarray(coefficients, dtype=float))
            lower.append(float(lower_bound))
            upper.append(float(upper_bound))

        # Delta-force limits are direct bounds on the QP variables.
        for index in range(n_input):
            row = np.zeros(n_variable)
            row[index] = 1.0
            axis = index % 3
            append_row(row, cfg.delta_force_min[axis], cfg.delta_force_max[axis])

        # Absolute force limits use tau_j from the augmented prediction.
        for step in range(N):
            for axis in range(3):
                state_index = nz * step + 6 + axis
                row = np.zeros(n_variable)
                row[:n_input] = self.Su[state_index]
                constant = free_prediction[state_index]
                append_row(
                    row,
                    cfg.force_min[axis] - constant,
                    cfg.force_max[axis] - constant,
                )

        # Actuator reachable set. Thruster forces remain free QP variables;
        # the wrench equalities retain actuator null-space freedom instead of
        # committing to one pseudoinverse allocation.  The 4-row form uses
        # only the measured/identified [force, yaw] space.  The 6-row form
        # additionally requests zero roll/pitch moment.
        if cfg.thruster_wrench_matrix is not None:
            thruster_count = self.thruster_count
            for step in range(N):
                force_rows = slice(nz * step + 6, nz * step + 9)
                force_gain = self.Su[force_rows]
                force_free = free_prediction[force_rows]

                step_thruster_start = thruster_start + step * thruster_count
                for thruster_index in range(thruster_count):
                    row = np.zeros(n_variable)
                    row[step_thruster_start + thruster_index] = 1.0
                    append_row(
                        row,
                        cfg.thruster_force_min[thruster_index],
                        cfg.thruster_force_max[thruster_index],
                    )

                wrench_row_count = cfg.thruster_wrench_matrix.shape[0]
                for wrench_axis in range(wrench_row_count):
                    row = np.zeros(n_variable)
                    row[
                        step_thruster_start : step_thruster_start + thruster_count
                    ] = cfg.thruster_wrench_matrix[wrench_axis]
                    if wrench_axis < 3:
                        row[:n_input] -= force_gain[wrench_axis]
                        target = force_free[wrench_axis]
                    elif wrench_row_count == 6 and wrench_axis < 5:
                        target = 0.0
                    else:
                        target = yaw_prediction.moments[step]
                    append_row(row, target, target)

        for index in range(n_slack):
            row = np.zeros(n_variable)
            row[slack_start_global + index] = 1.0
            append_row(row, 0.0, cfg.slack_max)

        horizontal_gain = np.tan(
            np.deg2rad(cfg.horizontal_half_fov_deg - cfg.fov_margin_deg)
        )
        vertical_gain = np.tan(
            np.deg2rad(cfg.vertical_half_fov_deg - cfg.fov_margin_deg)
        )
        for step in range(N):
            offset = nz * step
            position_gain_body = self.Su[offset : offset + 3]
            position_free_body = (
                free_prediction[offset : offset + 3] + reference_position
            )
            position_gain_visibility = (
                rotation_visibility_from_body @ position_gain_body
            )
            p_gains = [position_gain_visibility[axis] for axis in range(3)]
            p_free = rotation_visibility_from_body @ (
                position_free_body - camera_origin_in_body
            )
            forward_gain = p_gains[cfg.forward_axis]
            horizontal_state_gain = p_gains[cfg.horizontal_axis]
            vertical_state_gain = p_gains[cfg.vertical_axis]
            forward_free = p_free[cfg.forward_axis]
            horizontal_free = p_free[cfg.horizontal_axis]
            vertical_free = p_free[cfg.vertical_axis]
            slack_start = slack_start_global + self.SLACKS_PER_STEP * step

            def soft_upper(coefficients, constant, slack_local, limit=0.0) -> None:
                row = np.zeros(n_variable)
                row[:n_input] = coefficients
                row[slack_start + slack_local] = -1.0
                append_row(row, -np.inf, limit - constant)

            soft_upper(
                horizontal_state_gain - horizontal_gain * forward_gain,
                horizontal_free - horizontal_gain * forward_free,
                0,
            )
            soft_upper(
                -horizontal_state_gain - horizontal_gain * forward_gain,
                -horizontal_free - horizontal_gain * forward_free,
                1,
            )
            soft_upper(
                vertical_state_gain - vertical_gain * forward_gain,
                vertical_free - vertical_gain * forward_free,
                2,
            )
            soft_upper(
                -vertical_state_gain - vertical_gain * forward_gain,
                -vertical_free - vertical_gain * forward_free,
                3,
            )
            soft_upper(-forward_gain, -forward_free, 4, -cfg.forward_distance_min)
            soft_upper(forward_gain, forward_free, 5, cfg.forward_distance_max)

        return np.vstack(rows), np.asarray(lower), np.asarray(upper)

    def _shift_warm_start(self, decision: Array) -> Array:
        N = self.config.horizon
        n_input, n_thruster, _, _ = self._decision_layout()
        thruster_start = n_input
        slack_start = n_input + n_thruster
        delta = decision[:n_input].reshape(N, 3)
        parts = [np.vstack((delta[1:], np.zeros((1, 3)))).ravel()]
        if n_thruster:
            thruster = decision[thruster_start:slack_start].reshape(
                N, self.thruster_count
            )
            parts.append(np.vstack((thruster[1:], thruster[-1:])).ravel())
        slack = decision[slack_start:].reshape(N, self.SLACKS_PER_STEP)
        parts.append(np.vstack((slack[1:], slack[-1:])).ravel())
        return np.concatenate(parts)

    def _project_reachable_wrench(
        self,
        previous: Array,
        target: Array,
        yaw_moment: float,
        *,
        lock_force: bool = False,
    ) -> tuple[Array, Array]:
        cfg = self.config
        yaw = finite_scalar(yaw_moment, "yaw_moment")
        previous = np.clip(previous, cfg.force_min, cfg.force_max)
        low = np.maximum(cfg.force_min, previous + cfg.delta_force_min)
        high = np.minimum(cfg.force_max, previous + cfg.delta_force_max)
        target = np.minimum(np.maximum(target, low), high)
        if lock_force:
            low = target.copy()
            high = target.copy()

        matrix = cfg.thruster_wrench_matrix
        if matrix is None:
            return target, np.zeros(0)

        thruster_count = self.thruster_count
        n_variable = 3 + thruster_count
        P = np.zeros((n_variable, n_variable))
        q = np.zeros(n_variable)
        P[:3, :3] = 2.0 * np.eye(3)
        q[:3] = -2.0 * target
        P[3:, 3:] = 2.0 * max(
            cfg.thruster_force_regularization, 1.0e-9
        ) * np.eye(thruster_count)

        rows = []
        lower = []
        upper = []
        force_bounds = np.zeros((3, n_variable))
        force_bounds[:, :3] = np.eye(3)
        rows.append(force_bounds)
        lower.extend(low)
        upper.extend(high)
        thruster_bounds = np.zeros((thruster_count, n_variable))
        thruster_bounds[:, 3:] = np.eye(thruster_count)
        rows.append(thruster_bounds)
        lower.extend(cfg.thruster_force_min)
        upper.extend(cfg.thruster_force_max)

        wrench_row_count = matrix.shape[0]
        wrench_equalities = np.zeros((wrench_row_count, n_variable))
        wrench_equalities[:, 3:] = matrix
        wrench_equalities[:3, :3] = -np.eye(3)
        desired = (
            np.array([0.0, 0.0, 0.0, yaw])
            if wrench_row_count == 4
            else np.array([0.0, 0.0, 0.0, 0.0, 0.0, yaw])
        )
        rows.append(wrench_equalities)
        lower.extend(desired)
        upper.extend(desired)
        solution = self.solver.solve(
            P,
            q,
            np.vstack(rows),
            np.asarray(lower),
            np.asarray(upper),
        )
        if solution.solved:
            return solution.x[:3].copy(), solution.x[3:].copy()

        # Deterministic emergency path. The exact minimum-norm inverse is only
        # used after a solver failure and is scaled toward zero translation.
        inverse = np.linalg.pinv(matrix, rcond=1.0e-10)
        for scale in np.linspace(1.0, 0.0, 101):
            force = np.minimum(np.maximum(scale * target, low), high)
            desired[:3] = force
            thruster = inverse @ desired
            if np.all(thruster >= cfg.thruster_force_min - 1.0e-9) and np.all(
                thruster <= cfg.thruster_force_max + 1.0e-9
            ):
                return force, thruster
        return np.zeros(3), np.zeros(thruster_count)

    def _safe_fallback(
        self,
        previous: Array,
        target: Array,
        yaw_moment: float = 0.0,
    ) -> Array:
        force, thruster = self._project_reachable_wrench(
            previous,
            target,
            finite_scalar(yaw_moment, "yaw_moment"),
        )
        self._last_feasible_thruster_force = thruster.copy()
        return force

    def thruster_utilization_from_allocation(
        self,
        thruster_force,
        row_indices=None,
    ) -> float:
        """Return direction-aware utilization of an explicit allocation."""
        if self.config.thruster_wrench_matrix is None:
            return 0.0
        values = np.asarray(thruster_force, dtype=float).reshape(-1)
        if values.shape != (self.thruster_count,) or not np.all(np.isfinite(values)):
            raise ValueError(
                f"thruster_force must contain {self.thruster_count} finite values"
            )
        lower = self.config.thruster_force_min
        upper = self.config.thruster_force_max
        if row_indices is not None:
            indices = np.asarray(row_indices, dtype=int).reshape(-1)
            values = values[indices]
            lower = lower[indices]
            upper = upper[indices]
        scales = np.where(values >= 0.0, upper, -lower)
        return float(np.max(np.abs(values) / scales))

    def thruster_utilization(
        self,
        force,
        yaw_moment: float = 0.0,
        row_indices=None,
    ) -> float:
        """Return largest direction-aware joint wrench utilization."""
        if self.config.thruster_wrench_matrix is None:
            return 0.0
        requested_force = vector3(force, "force")
        _, thruster = self._project_reachable_wrench(
            requested_force,
            requested_force,
            finite_scalar(yaw_moment, "yaw_moment"),
            lock_force=True,
        )
        return self.thruster_utilization_from_allocation(thruster, row_indices)

    def solve(
        self,
        state,
        force_previous,
        yaw_prediction: YawPrediction,
        reference_position=None,
        model1_weight=None,
        rotation_visibility_from_body=None,
        camera_origin_in_body=None,
    ) -> YawMPCResult:
        state = np.asarray(state, dtype=float).reshape(-1)
        if state.shape != (6,) or not np.all(np.isfinite(state)):
            raise ValueError("state must be [p_rel(3), v_rel(3)]")
        previous = vector3(force_previous, "force_previous")
        # PDF rolling operating point: latch tau[k-1] once at the start of the
        # solve, then keep it unchanged through all N prediction stages.
        baseline = previous.copy()
        self._validate_yaw_prediction(yaw_prediction)
        reference = (
            self.config.reference_position
            if reference_position is None
            else vector3(reference_position, "reference_position")
        )
        weight1 = (
            np.ones(3)
            if model1_weight is None
            else np.clip(vector3(model1_weight, "model1_weight"), 0.0, 1.0)
        )
        visibility_rotation, camera_origin = visibility_frame_geometry(
            self.config.rotation_visibility_from_body
            if rotation_visibility_from_body is None
            else rotation_visibility_from_body,
            self.config.camera_origin_in_body
            if camera_origin_in_body is None
            else camera_origin_in_body,
        )
        self._build_prediction_matrices(
            reference,
            weight1,
            yaw_prediction,
            baseline,
        )
        augmented_state = np.concatenate((state[:3] - reference, state[3:], previous))
        free_prediction = self.Sx @ augmented_state + self.Sc
        P, q = self._cost(free_prediction)
        constraint_matrix, lower, upper = self._constraints(
            free_prediction,
            reference,
            yaw_prediction,
            visibility_rotation,
            camera_origin,
        )
        solution = self.solver.solve(
            P,
            q,
            constraint_matrix,
            lower,
            upper,
            warm_start=self._warm_start,
        )

        N = self.config.horizon
        n_input, n_thruster, _, _ = self._decision_layout()
        thruster_start = n_input
        slack_start = n_input + n_thruster
        if solution.solved:
            decision = solution.x
            delta_sequence = decision[:n_input].reshape(N, 3)
            thruster_sequence = decision[thruster_start:slack_start].reshape(
                N, self.thruster_count
            )
            slacks = np.clip(
                decision[slack_start:].reshape(N, self.SLACKS_PER_STEP),
                0.0,
                self.config.slack_max,
            )
            predicted_augmented = (
                free_prediction + self.Su @ delta_sequence.ravel()
            ).reshape(N, self.AUGMENTED_DIM)
            force_sequence = predicted_augmented[:, 6:9].copy()
            force = force_sequence[0].copy()
            self._warm_start = self._shift_warm_start(decision)
            self._last_feasible_force = force.copy()
            self._last_feasible_thruster_force = thruster_sequence[0].copy()
            used_fallback = False
            status = solution.status
        else:
            if self._last_feasible_force is None:
                fallback_target = self.model.translation.restoring_force
                fallback_source = "restoring_force"
            else:
                fallback_target = self._last_feasible_force
                fallback_source = "last_feasible"
            force = self._safe_fallback(
                previous,
                fallback_target,
                yaw_moment=yaw_prediction.moments[0],
            )
            delta_sequence = np.zeros((N, 3))
            delta_sequence[0] = force - previous
            predicted_augmented = (
                free_prediction + self.Su @ delta_sequence.ravel()
            ).reshape(N, self.AUGMENTED_DIM)
            force_sequence = predicted_augmented[:, 6:9].copy()
            thruster_sequence = np.zeros((N, self.thruster_count))
            if self.thruster_count:
                thruster_sequence[0] = self._last_feasible_thruster_force
            slacks = np.zeros((N, self.SLACKS_PER_STEP))
            self._warm_start = None
            used_fallback = True
            status = f"fallback:{fallback_source}:{solution.status}"

        predicted_states = np.hstack(
            (predicted_augmented[:, :3] + reference, predicted_augmented[:, 3:6])
        )
        return YawMPCResult(
            force=force.copy(),
            yaw_moment=float(yaw_prediction.moments[0]),
            force_sequence=force_sequence,
            delta_force_sequence=delta_sequence,
            thruster_force_sequence=thruster_sequence,
            predicted_states=predicted_states,
            predicted_augmented_states=predicted_augmented,
            frozen_yaw_angles=np.asarray(yaw_prediction.angles, dtype=float).copy(),
            frozen_yaw_rates=np.asarray(yaw_prediction.rates, dtype=float).copy(),
            frozen_delta_yaw=np.asarray(yaw_prediction.delta_angles, dtype=float).copy(),
            frozen_yaw_moments=np.asarray(yaw_prediction.moments, dtype=float).copy(),
            slacks=slacks,
            status=status,
            iterations=solution.iterations,
            objective=solution.objective,
            used_fallback=used_fallback,
            model1_weight=weight1.copy(),
            model2_weight=1.0 - weight1,
        )
