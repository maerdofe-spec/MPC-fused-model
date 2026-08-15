"""Constrained three-axis MPC for relative target tracking."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    from .camera_transform import rotation_state_body_from_previous
    from .dense_qp import QPSolverSettings, create_qp_solver
    from .fossen_fixed_dl_model import FixedLinearDampingRelativeModel, vector3
except ImportError:
    from camera_transform import rotation_state_body_from_previous
    from dense_qp import QPSolverSettings, create_qp_solver
    from fossen_fixed_dl_model import FixedLinearDampingRelativeModel, vector3


Array = np.ndarray


def _positive_vector3(value, name: str, allow_zero: bool = True) -> Array:
    result = vector3(value, name)
    minimum = 0.0 if allow_zero else np.finfo(float).eps
    if np.any(result < minimum):
        raise ValueError(f"{name} must be nonnegative")
    return result


def _block_diagonal(blocks: list[Array]) -> Array:
    rows = sum(block.shape[0] for block in blocks)
    columns = sum(block.shape[1] for block in blocks)
    result = np.zeros((rows, columns))
    row = 0
    column = 0
    for block in blocks:
        r, c = block.shape
        result[row : row + r, column : column + c] = block
        row += r
        column += c
    return result


@dataclass
class MPCConfig:
    """Weights and constraints. All force quantities use newtons."""

    horizon: int = 10
    reference_position: object = (0.60, 0.0, 0.0)

    position_weights: object = (50.0, 100.0, 100.0)
    velocity_weights: object = (8.0, 12.0, 12.0)
    terminal_weight_scale: float = 4.0
    # Absolute total propulsion effort. The separate delta-force term still
    # penalizes consecutive planned physical-force changes.
    force_weights: object = (0.04, 0.06, 0.06)
    delta_force_weights: object = (0.8, 1.0, 1.0)

    # Optional upper-computer actuator model. Disabled by default until an
    # offline force/RPM replay validates the candidate delay.
    actuator_model_enabled: bool = False
    actuator_pure_delay_s: float = 0.0
    actuator_time_constant_s: float = 0.0

    force_min: object = (-20.0, -15.0, -15.0)
    force_max: object = (20.0, 15.0, 15.0)
    delta_force_min: object = (-4.0, -3.0, -3.0)
    delta_force_max: object = (4.0, 3.0, 3.0)

    # Optional per-thruster map. Each row maps body force [forward,right,down]
    # to one thruster value. Bounds may be asymmetric and use the same unit as
    # the mapped value (normally newtons). The scalar limit remains as a
    # backward-compatible symmetric default.
    thruster_command_matrix: object | None = None
    thruster_command_limit: float = 1.0
    thruster_command_min: object | None = None
    thruster_command_max: object | None = None

    forward_distance_min: float = 0.25
    forward_distance_max: float = 1.50
    horizontal_half_fov_deg: float = 42.0
    vertical_half_fov_deg: float = 30.0
    fov_margin_deg: float = 5.0

    # Six nonnegative slacks per prediction step:
    # horizontal +/-; vertical +/-; too-near; too-far.
    slack_quadratic_weight: float = 2.0e4
    slack_linear_weight: float = 50.0
    slack_max: float = 5.0

    # State axes are body forward/right/down by default.
    forward_axis: int = 0
    horizontal_axis: int = 1
    vertical_axis: int = 2

    # Visibility constraints are expressed in the physical camera frame. The
    # visibility frame order is [camera forward, camera right, camera down].
    rotation_visibility_from_body: object = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    camera_origin_in_body: object = (0.0, 0.0, 0.0)

    solver_settings: QPSolverSettings = field(
        default_factory=lambda: QPSolverSettings(
            rho=10.0,
            sigma=1e-8,
            max_iterations=1500,
            absolute_tolerance=2e-5,
            relative_tolerance=3e-4,
        )
    )

    def normalized(self) -> "MPCConfig":
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
        if not isinstance(self.actuator_model_enabled, (bool, np.bool_)):
            raise ValueError("actuator_model_enabled must be boolean")
        if (
            not np.isfinite(self.actuator_pure_delay_s)
            or self.actuator_pure_delay_s < 0.0
        ):
            raise ValueError("actuator_pure_delay_s must be nonnegative and finite")
        if (
            not np.isfinite(self.actuator_time_constant_s)
            or self.actuator_time_constant_s < 0.0
        ):
            raise ValueError(
                "actuator_time_constant_s must be nonnegative and finite"
            )
        if self.actuator_model_enabled and self.actuator_time_constant_s <= 0.0:
            raise ValueError(
                "actuator_time_constant_s must be positive when actuator model is enabled"
            )
        self.force_min = vector3(self.force_min, "force_min")
        self.force_max = vector3(self.force_max, "force_max")
        self.delta_force_min = vector3(self.delta_force_min, "delta_force_min")
        self.delta_force_max = vector3(self.delta_force_max, "delta_force_max")
        if self.thruster_command_matrix is not None:
            matrix = np.asarray(self.thruster_command_matrix, dtype=float)
            if (
                matrix.ndim != 2
                or matrix.shape[1] != 3
                or matrix.shape[0] < 1
                or not np.all(np.isfinite(matrix))
            ):
                raise ValueError("thruster_command_matrix must have shape (n, 3)")
            self.thruster_command_matrix = matrix
        if self.thruster_command_limit <= 0.0 or not np.isfinite(
            self.thruster_command_limit
        ):
            raise ValueError("thruster_command_limit must be positive and finite")
        if self.thruster_command_matrix is None:
            if self.thruster_command_min is not None or self.thruster_command_max is not None:
                raise ValueError("thruster bounds require thruster_command_matrix")
        else:
            row_count = self.thruster_command_matrix.shape[0]
            self.thruster_command_min = self._normalize_thruster_bound(
                self.thruster_command_min,
                -self.thruster_command_limit,
                row_count,
                "thruster_command_min",
            )
            self.thruster_command_max = self._normalize_thruster_bound(
                self.thruster_command_max,
                self.thruster_command_limit,
                row_count,
                "thruster_command_max",
            )
            if np.any(self.thruster_command_min >= 0.0) or np.any(
                self.thruster_command_max <= 0.0
            ):
                raise ValueError("each thruster interval must contain zero")
        if np.any(self.force_min >= self.force_max):
            raise ValueError("force_min must be smaller than force_max")
        if np.any(self.delta_force_min >= self.delta_force_max):
            raise ValueError("delta_force_min must be smaller than delta_force_max")
        if not 0.0 < self.forward_distance_min < self.forward_distance_max:
            raise ValueError("invalid forward distance limits")
        if not 0.0 < self.fov_margin_deg < min(
            self.horizontal_half_fov_deg, self.vertical_half_fov_deg
        ):
            raise ValueError("FOV margin must be positive and smaller than each half-FOV")
        if self.terminal_weight_scale <= 0.0:
            raise ValueError("terminal_weight_scale must be positive")
        if self.slack_quadratic_weight <= 0.0 or self.slack_linear_weight < 0.0:
            raise ValueError("slack weights are invalid")
        if self.slack_max <= 0.0:
            raise ValueError("slack_max must be positive")
        axes = {self.forward_axis, self.horizontal_axis, self.vertical_axis}
        if axes != {0, 1, 2}:
            raise ValueError("forward/horizontal/vertical axes must be a permutation of 0,1,2")
        rotation = np.asarray(self.rotation_visibility_from_body, dtype=float)
        origin = np.asarray(self.camera_origin_in_body, dtype=float).reshape(-1)
        if rotation.shape != (3, 3) or origin.shape != (3,):
            raise ValueError(
                "camera visibility rotation must be 3x3 and origin must be a 3-vector"
            )
        if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(origin)):
            raise ValueError("camera visibility geometry must be finite")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-5):
            raise ValueError("rotation_visibility_from_body must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-5):
            raise ValueError("rotation_visibility_from_body must be a proper rotation")
        self.rotation_visibility_from_body = rotation
        self.camera_origin_in_body = origin
        return self

    @staticmethod
    def _normalize_thruster_bound(value, default: float, size: int, name: str) -> Array:
        if value is None:
            return np.full(size, default, dtype=float)
        result = np.asarray(value, dtype=float).reshape(-1)
        if result.shape != (size,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must contain {size} finite values")
        return result


@dataclass
class MPCResult:
    force: Array
    force_sequence: Array
    delta_force_sequence: Array
    # Retained for trace/API compatibility.  The absolute-force cost has no
    # equilibrium-centering reference, so this is always the zero vector.
    force_reference: Array
    model1_base_force: Array
    predicted_states: Array
    slacks: Array
    status: str
    iterations: int
    objective: float
    used_fallback: bool
    model1_weight: Array
    model2_weight: Array
    predicted_delta_yaw_rad: Array
    predicted_actuator_force_sequence: Array | None = None


class RelativeMPCController:
    """Finite-horizon QP controller using the fixed-D_L relative model."""

    SLACKS_PER_STEP = 6

    def __init__(
        self,
        model: FixedLinearDampingRelativeModel,
        config: MPCConfig | None = None,
    ) -> None:
        self.model = model
        self.config = (config or MPCConfig()).normalized()
        self.solver = create_qp_solver(self.config.solver_settings)
        self._warm_start = None
        self._last_feasible_force = None
        self._prediction_weight = None
        self._prediction_yaw_rate = None
        self._actuator_delay_steps = 0
        self._actuator_alpha = 1.0
        # The command queue represents commands that have already been sent
        # to the lower controller but whose force has not reached the vehicle
        # yet.  It must survive successive MPC solves; rebuilding it from the
        # latest achieved force would erase the very delay we are modelling.
        self._actuator_command_queue: list[Array] | None = None
        if self.config.actuator_model_enabled:
            self._actuator_delay_steps = int(
                np.ceil(
                    self.config.actuator_pure_delay_s / self.model.dt
                    - 1.0e-12
                )
            )
            self._actuator_alpha = float(
                np.exp(-self.model.dt / self.config.actuator_time_constant_s)
            )
            self._augmented_state_size = 12 + 3 * self._actuator_delay_steps
        else:
            self._augmented_state_size = 9
        self._build_prediction_matrices(np.ones(3), 0.0)

    def reset(self) -> None:
        """Clear optimizer history when mode or tracked target changes."""
        self._warm_start = None
        self._last_feasible_force = None
        self._actuator_command_queue = None

    def safe_force(self, tau_previous, target_force=None) -> Array:
        """Rate-limit a safe return toward fixed Fossen balance force."""
        previous = vector3(tau_previous, "tau_previous")
        target = (
            self.model.restoring_force
            if target_force is None
            else vector3(target_force, "target_force")
        )
        return self._safe_fallback(previous, target)

    def _build_prediction_matrices(
        self,
        model1_weight,
        yaw_rate_rad_s: float = 0.0,
    ) -> None:
        """Build predictions for the two-model, fixed-per-solve-base formulation.

        Without the optional actuator model the augmented state is
        ``z=[x, tau_previous]``. When enabled it additionally contains the
        achieved actuator force, a quantized command-delay queue, and the
        preceding command. The tracker supplies the separately slew-limited
        only for rate costs/constraints. The tracker supplies the separately
        slew-limited model-1 operating force ``tau_base,k``, which is held
        fixed over the complete horizon. With W containing model-1 weights,
        the fused physical-state equation is

            x+ = R_yaw (A_d x + B_d tau)
                 - W R_yaw B_d tau_base,k
                 - (I-W) R_yaw B_d tau_h.

        a1=1 is the fixed matched-speed model; a1=0 is the target-stationary
        Fossen model with fixed restoring/environment balance force ``tau_h``.
        """
        weight = np.clip(vector3(model1_weight, "model1_weight"), 0.0, 1.0)
        yaw_rate = float(yaw_rate_rad_s)
        if not np.isfinite(yaw_rate):
            raise ValueError("yaw_rate_rad_s must be finite")
        if (
            self._prediction_weight is not None
            and np.allclose(
                weight, self._prediction_weight, atol=1e-12, rtol=0.0
            )
            and self._prediction_yaw_rate is not None
            and np.isclose(
                yaw_rate,
                self._prediction_yaw_rate,
                atol=1.0e-12,
                rtol=0.0,
            )
        ):
            return
        # Constant-yaw-rate is the only future attitude assumption available
        # from current telemetry. Every model step rotates the preceding body
        # coordinates into the newly predicted body frame.
        rotation6 = rotation_state_body_from_previous(yaw_rate * self.model.dt)
        physical_A = rotation6 @ self.model.A_d
        physical_B = rotation6 @ self.model.B_d
        W = np.diag(np.concatenate((weight, weight)))
        if not self.config.actuator_model_enabled:
            A = np.zeros((9, 9))
            B = np.zeros((9, 3))
            A[:6, :6] = physical_A
            B[:6] = physical_B
            B[6:] = np.eye(3)
        else:
            delay_steps = self._actuator_delay_steps
            alpha = self._actuator_alpha
            actuator_start = 6
            queue_start = actuator_start + 3
            previous_command_start = queue_start + 3 * delay_steps
            A = np.zeros(
                (self._augmented_state_size, self._augmented_state_size)
            )
            B = np.zeros((self._augmented_state_size, 3))
            A[:6, :6] = physical_A
            A[
                actuator_start : actuator_start + 3,
                actuator_start : actuator_start + 3,
            ] = alpha * np.eye(3)
            if delay_steps:
                delayed_gain = (1.0 - alpha) * np.eye(3)
                A[
                    actuator_start : actuator_start + 3,
                    queue_start : queue_start + 3,
                ] = delayed_gain
                for queue_index in range(delay_steps - 1):
                    A[
                        queue_start + 3 * queue_index : queue_start + 3 * (queue_index + 1),
                        queue_start + 3 * (queue_index + 1) : queue_start + 3 * (queue_index + 2),
                    ] = np.eye(3)
                B[
                    queue_start + 3 * (delay_steps - 1) : queue_start + 3 * delay_steps
                ] = np.eye(3)
                A[:6, actuator_start : actuator_start + 3] += (
                    physical_B @ (alpha * np.eye(3))
                )
                A[:6, queue_start : queue_start + 3] += (
                    physical_B @ delayed_gain
                )
            else:
                current_gain = (1.0 - alpha) * np.eye(3)
                B[actuator_start : actuator_start + 3] = current_gain
                A[:6, actuator_start : actuator_start + 3] += (
                    physical_B @ (alpha * np.eye(3))
                )
                B[:6] += physical_B @ current_gain
            B[previous_command_start : previous_command_start + 3] = np.eye(3)
        baseline_transition = np.zeros((A.shape[0], 3))
        baseline_transition[:6] = -W @ physical_B
        restoring_transition = np.zeros((A.shape[0], 3))
        restoring_transition[:6] = (
            -(np.eye(6) - W) @ physical_B
        )
        output = np.hstack((np.eye(6), np.zeros((6, A.shape[0] - 6))))
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

        # Constant baseline contribution for each prediction stage. The
        # baseline is supplied at solve time, so this remains affine and the
        # QP stays linear.
        self.baseline_effect_matrix = np.zeros((N * nx, 3))
        self.restoring_effect_matrix = np.zeros((N * nx, 3))
        for i in range(N):
            accumulated_baseline = np.zeros((6, 3))
            accumulated_restoring = np.zeros((6, 3))
            for j in range(i + 1):
                accumulated_baseline += (
                    output @ powers[j] @ baseline_transition
                )
                accumulated_restoring += (
                    output @ powers[j] @ restoring_transition
                )
            self.baseline_effect_matrix[
                i * nx : (i + 1) * nx
            ] = accumulated_baseline
            self.restoring_effect_matrix[
                i * nx : (i + 1) * nx
            ] = accumulated_restoring

        self.rate_matrix = np.zeros((N * nu, N * nu))
        for i in range(N):
            self.rate_matrix[i * nu : (i + 1) * nu, i * nu : (i + 1) * nu] = np.eye(nu)
            if i > 0:
                self.rate_matrix[
                    i * nu : (i + 1) * nu,
                    (i - 1) * nu : i * nu,
                ] = -np.eye(nu)
        self._prediction_weight = weight.copy()
        self._prediction_yaw_rate = yaw_rate

    def _reference_stack(self, reference_position: Array) -> Array:
        state_reference = np.concatenate((reference_position, np.zeros(3)))
        return np.tile(state_reference, self.config.horizon)

    def _cost(
        self,
        free_prediction: Array,
        reference_position: Array,
        tau_previous: Array,
        force_reference: Array | None = None,
    ) -> tuple[Array, Array]:
        # ``force_reference`` is retained for callers from the previous
        # equilibrium-centered formulation; absolute-force cost intentionally
        # ignores it.
        cfg = self.config
        N = cfg.horizon
        Q_stage = np.diag(np.concatenate((cfg.position_weights, cfg.velocity_weights)))
        Q_terminal = cfg.terminal_weight_scale * Q_stage
        Q_bar = _block_diagonal(
            [Q_terminal if i == N - 1 else Q_stage for i in range(N)]
        )
        R_bar = np.kron(np.eye(N), np.diag(cfg.force_weights))
        S_bar = np.kron(np.eye(N), np.diag(cfg.delta_force_weights))

        reference = self._reference_stack(reference_position)
        error_free = free_prediction - reference
        rate_reference = np.zeros(3 * N)
        rate_reference[:3] = tau_previous

        P_force = 2.0 * (
            self.Su.T @ Q_bar @ self.Su
            + R_bar
            + self.rate_matrix.T @ S_bar @ self.rate_matrix
        )
        q_force = 2.0 * (
            self.Su.T @ Q_bar @ error_free
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

    def _constraints(
        self,
        free_prediction: Array,
        tau_previous: Array,
    ) -> tuple[Array, Array, Array]:
        cfg = self.config
        N = cfg.horizon
        n_force = 3 * N
        n_slack = self.SLACKS_PER_STEP * N
        n_variable = n_force + n_slack
        rows: list[Array] = []
        lower: list[float] = []
        upper: list[float] = []

        def append_row(coefficients, lower_bound, upper_bound) -> None:
            rows.append(np.asarray(coefficients, dtype=float))
            lower.append(float(lower_bound))
            upper.append(float(upper_bound))

        # Absolute force limits.
        for index in range(n_force):
            row = np.zeros(n_variable)
            row[index] = 1.0
            axis = index % 3
            append_row(row, cfg.force_min[axis], cfg.force_max[axis])

        # Couple force axes through the real per-thruster command limits.
        if cfg.thruster_command_matrix is not None:
            for step in range(N):
                force_start = 3 * step
                for row_index, command_row in enumerate(cfg.thruster_command_matrix):
                    row = np.zeros(n_variable)
                    row[force_start : force_start + 3] = command_row
                    append_row(
                        row,
                        cfg.thruster_command_min[row_index],
                        cfg.thruster_command_max[row_index],
                    )

        # Force-rate limits. First difference is tau[0] - tau_previous.
        for index in range(n_force):
            row = np.zeros(n_variable)
            row[:n_force] = self.rate_matrix[index]
            axis = index % 3
            if index < 3:
                append_row(
                    row,
                    tau_previous[axis] + cfg.delta_force_min[axis],
                    tau_previous[axis] + cfg.delta_force_max[axis],
                )
            else:
                append_row(row, cfg.delta_force_min[axis], cfg.delta_force_max[axis])

        # Slack bounds.
        for index in range(n_slack):
            row = np.zeros(n_variable)
            row[n_force + index] = 1.0
            append_row(row, 0.0, cfg.slack_max)

        horizontal_gain = np.tan(
            np.deg2rad(cfg.horizontal_half_fov_deg - cfg.fov_margin_deg)
        )
        vertical_gain = np.tan(
            np.deg2rad(cfg.vertical_half_fov_deg - cfg.fov_margin_deg)
        )

        # FOV and forward-distance constraints on each predicted target ray in
        # the physical camera frame, not on body-origin coordinates.
        visibility_rotation = cfg.rotation_visibility_from_body
        camera_origin = cfg.camera_origin_in_body
        for step in range(N):
            state_offset = 6 * step
            position_force_gain = visibility_rotation @ self.Su[
                state_offset : state_offset + 3
            ]
            position_free = visibility_rotation @ (
                free_prediction[state_offset : state_offset + 3]
                - camera_origin
            )
            force_forward = position_force_gain[cfg.forward_axis]
            force_horizontal = position_force_gain[cfg.horizontal_axis]
            force_vertical = position_force_gain[cfg.vertical_axis]
            free_forward = position_free[cfg.forward_axis]
            free_horizontal = position_free[cfg.horizontal_axis]
            free_vertical = position_free[cfg.vertical_axis]

            slack_start = n_force + self.SLACKS_PER_STEP * step

            def soft_upper(force_coeff, constant, slack_local, limit=0.0) -> None:
                row = np.zeros(n_variable)
                row[:n_force] = force_coeff
                row[slack_start + slack_local] = -1.0
                append_row(row, -np.inf, limit - constant)

            # +/- horizontal <= tan(FOV_h-margin) * forward + slack
            soft_upper(
                force_horizontal - horizontal_gain * force_forward,
                free_horizontal - horizontal_gain * free_forward,
                0,
            )
            soft_upper(
                -force_horizontal - horizontal_gain * force_forward,
                -free_horizontal - horizontal_gain * free_forward,
                1,
            )

            # +/- vertical <= tan(FOV_v-margin) * forward + slack
            soft_upper(
                force_vertical - vertical_gain * force_forward,
                free_vertical - vertical_gain * free_forward,
                2,
            )
            soft_upper(
                -force_vertical - vertical_gain * force_forward,
                -free_vertical - vertical_gain * free_forward,
                3,
            )

            # forward + near_slack >= minimum distance
            soft_upper(
                -force_forward,
                -free_forward,
                4,
                -cfg.forward_distance_min,
            )
            # forward - far_slack <= maximum distance
            soft_upper(
                force_forward,
                free_forward,
                5,
                cfg.forward_distance_max,
            )

        return np.vstack(rows), np.asarray(lower), np.asarray(upper)

    def _shift_warm_start(self, decision: Array) -> Array:
        N = self.config.horizon
        n_force = 3 * N
        force = decision[:n_force].reshape(N, 3)
        slack = decision[n_force:].reshape(N, self.SLACKS_PER_STEP)
        force_shifted = np.vstack((force[1:], force[-1:]))
        slack_shifted = np.vstack((slack[1:], slack[-1:]))
        return np.concatenate((force_shifted.ravel(), slack_shifted.ravel()))

    def _predict_actuator_force_sequence(
        self,
        tau_previous: Array,
        force_sequence: Array,
        initial_queue: list[Array] | None = None,
    ) -> Array:
        """Replay the configured actuator state for diagnostics."""
        commands = np.asarray(force_sequence, dtype=float).reshape(-1, 3)
        if not self.config.actuator_model_enabled:
            return commands.copy()
        actuator = vector3(tau_previous, "tau_previous").copy()
        if initial_queue is None:
            queue = [actuator.copy() for _ in range(self._actuator_delay_steps)]
        else:
            if len(initial_queue) != self._actuator_delay_steps:
                raise ValueError("initial_queue has the wrong actuator delay length")
            queue = [vector3(item, "initial_queue_item").copy() for item in initial_queue]
        predicted: list[Array] = []
        for command in commands:
            if queue:
                applied = queue.pop(0)
                queue.append(command.copy())
            else:
                applied = command
            actuator = (
                self._actuator_alpha * actuator
                + (1.0 - self._actuator_alpha) * applied
            )
            predicted.append(actuator.copy())
        return np.asarray(predicted)

    def _actuator_queue_for_solve(self, tau_previous: Array) -> list[Array]:
        """Return the queued commands, initializing from measured force once.

        The measured achieved force is used to re-anchor the current actuator
        state at every visual update.  The command queue, however, is retained
        across updates so a command sent during the previous interval remains
        in the prediction until its configured delay has elapsed.
        """

        if not self.config.actuator_model_enabled:
            return []
        previous = vector3(tau_previous, "tau_previous")
        if self._actuator_command_queue is None:
            self._actuator_command_queue = [
                previous.copy() for _ in range(self._actuator_delay_steps)
            ]
        if len(self._actuator_command_queue) != self._actuator_delay_steps:
            raise RuntimeError("actuator command queue length changed at runtime")
        return [item.copy() for item in self._actuator_command_queue]

    def _advance_actuator_queue(self, command: Array) -> None:
        """Latch the command actually sent for the next MPC solve."""

        if not self.config.actuator_model_enabled:
            return
        if self._actuator_delay_steps == 0:
            self._actuator_command_queue = []
            return
        if self._actuator_command_queue is None:
            raise RuntimeError("actuator queue was not initialized before advance")
        self._actuator_command_queue.pop(0)
        self._actuator_command_queue.append(vector3(command, "command").copy())

    def _safe_fallback(self, tau_previous: Array, target_force: Array) -> Array:
        cfg = self.config
        previous = np.clip(tau_previous, cfg.force_min, cfg.force_max)
        low = np.maximum(cfg.force_min, previous + cfg.delta_force_min)
        high = np.minimum(cfg.force_max, previous + cfg.delta_force_max)
        force = np.minimum(np.maximum(target_force, low), high)
        if cfg.thruster_command_matrix is None:
            return force

        # Alternating projections keep the safety fallback inside both the
        # per-axis/rate box and the joint thruster envelope.
        for _ in range(8):
            force = np.minimum(np.maximum(force, low), high)
            for row_index, command_row in enumerate(cfg.thruster_command_matrix):
                value = float(command_row @ force)
                lower_bound = cfg.thruster_command_min[row_index]
                upper_bound = cfg.thruster_command_max[row_index]
                norm_sq = max(
                    float(command_row @ command_row), np.finfo(float).eps
                )
                if value > upper_bound:
                    force -= (value - upper_bound) / norm_sq * command_row
                elif value < lower_bound:
                    force += (lower_bound - value) / norm_sq * command_row
        return np.minimum(np.maximum(force, low), high)

    def thruster_utilization(self, force, row_indices=None) -> float:
        """Return largest direction-aware utilization for a body force."""
        if self.config.thruster_command_matrix is None:
            return 0.0
        values = self.config.thruster_command_matrix @ vector3(force, "force")
        lower = self.config.thruster_command_min
        upper = self.config.thruster_command_max
        if row_indices is not None:
            indices = np.asarray(row_indices, dtype=int).reshape(-1)
            values = values[indices]
            lower = lower[indices]
            upper = upper[indices]
        scales = np.where(values >= 0.0, upper, -lower)
        return float(np.max(np.abs(values) / scales))

    def solve(
        self,
        state,
        tau_previous,
        tau_base=None,
        reference_position=None,
        model1_weight=None,
        yaw_rate_rad_s: float = 0.0,
    ) -> MPCResult:
        state = np.asarray(state, dtype=float).reshape(-1)
        if state.shape != (6,) or not np.all(np.isfinite(state)):
            raise ValueError("state must be [p_rel(3), v_rel(3)]")
        tau_previous = vector3(tau_previous, "tau_previous")
        # The actual preceding achieved force remains the first force-rate
        # reference.  Model 1 may use a separately slew-limited operating
        # force, which is still fixed over this complete solve horizon.
        baseline = (
            tau_previous.copy()
            if tau_base is None
            else vector3(tau_base, "tau_base")
        )
        restoring = self.model.restoring_force.copy()
        weight1 = (
            np.ones(3)
            if model1_weight is None
            else np.clip(vector3(model1_weight, "model1_weight"), 0.0, 1.0)
        )
        # Penalize absolute total propulsion effort.  The model-specific
        # operating forces remain in the state prediction only; they must not
        # create a linear incentive to maintain a large baseline force.
        force_reference = np.zeros(3)
        self._build_prediction_matrices(weight1, yaw_rate_rad_s)
        reference = (
            self.config.reference_position
            if reference_position is None
            else vector3(reference_position, "reference_position")
        )

        actuator_queue = self._actuator_queue_for_solve(tau_previous)
        if self.config.actuator_model_enabled:
            augmented_state = np.zeros(self._augmented_state_size)
            augmented_state[:6] = state
            augmented_state[6:9] = tau_previous
            queue_start = 9
            for index, queued_command in enumerate(actuator_queue):
                augmented_state[
                    queue_start + 3 * index : queue_start + 3 * (index + 1)
                ] = queued_command
            augmented_state[queue_start + 3 * self._actuator_delay_steps :] = (
                tau_previous
            )
        else:
            augmented_state = np.concatenate((state, tau_previous))
        free_prediction = (
            self.Sx @ augmented_state
            + self.baseline_effect_matrix @ baseline
            + self.restoring_effect_matrix @ restoring
        )
        P, q = self._cost(
            free_prediction,
            reference,
            tau_previous,
            force_reference,
        )
        constraint_matrix, lower, upper = self._constraints(
            free_prediction, tau_previous
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
        n_force = 3 * N

        if solution.solved:
            decision = solution.x
            force_sequence = decision[:n_force].reshape(N, 3)
            slacks = np.clip(
                decision[n_force:].reshape(N, self.SLACKS_PER_STEP),
                0.0,
                self.config.slack_max,
            )
            low = np.maximum(
                self.config.force_min,
                tau_previous + self.config.delta_force_min,
            )
            high = np.minimum(
                self.config.force_max,
                tau_previous + self.config.delta_force_max,
            )
            force = np.minimum(np.maximum(force_sequence[0], low), high)
            if self.config.thruster_command_matrix is not None:
                # The QP already enforces this.  The projection only removes
                # numerical solver residuals introduced by the final clip.
                force = self._safe_fallback(tau_previous, force)
            self._warm_start = self._shift_warm_start(decision)
            self._last_feasible_force = force.copy()
            used_fallback = False
            status = solution.status
        else:
            if self._last_feasible_force is None:
                fallback_target = restoring
                fallback_source = "restoring_force"
            else:
                fallback_target = self._last_feasible_force
                fallback_source = "last_feasible"
            force = self._safe_fallback(tau_previous, fallback_target)
            force_sequence = np.tile(force, (N, 1))
            slacks = np.zeros((N, self.SLACKS_PER_STEP))
            self._warm_start = None
            used_fallback = True
            status = f"fallback:{fallback_source}:{solution.status}"

        predicted = (
            self.Sx @ augmented_state
            + self.baseline_effect_matrix @ baseline
            + self.restoring_effect_matrix @ restoring
            + self.Su @ force_sequence.ravel()
        ).reshape(N, 6)
        rate_reference = np.zeros(3 * N)
        rate_reference[:3] = tau_previous
        delta_force_sequence = (
            self.rate_matrix @ force_sequence.ravel() - rate_reference
        ).reshape(N, 3)
        predicted_actuator_force_sequence = self._predict_actuator_force_sequence(
            tau_previous,
            force_sequence,
            initial_queue=actuator_queue,
        )
        # The first command may have been projected into the joint thruster
        # envelope after the QP solve.  Retain that command in the delay queue,
        # because it is the value that will actually be sent to the vehicle.
        self._advance_actuator_queue(force)
        return MPCResult(
            force=force,
            force_sequence=force_sequence,
            delta_force_sequence=delta_force_sequence,
            force_reference=force_reference.copy(),
            model1_base_force=baseline.copy(),
            predicted_states=predicted,
            slacks=slacks,
            status=status,
            iterations=solution.iterations,
            objective=solution.objective,
            used_fallback=used_fallback,
            model1_weight=weight1.copy(),
            model2_weight=1.0 - weight1,
            predicted_delta_yaw_rad=np.full(
                N,
                float(yaw_rate_rad_s) * self.model.dt,
            ),
            predicted_actuator_force_sequence=predicted_actuator_force_sequence,
        )
