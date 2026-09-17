"""Model-1-only constrained MPC for relative target tracking.

The prediction is fixed to

    x[k+1] = A_d x[k] + B_d (tau[k] - tau[k-1]).

The previous force rolls forward as an augmented state.  This module contains
no model-2 candidate, fusion weight, or dependency on the dual-model package.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    from .dense_qp import QPSolverSettings, create_qp_solver
    from .fossen_fixed_dl_model import FixedLinearDampingRelativeModel, vector3
except ImportError:
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
    # In model 1 the physical action is the rolling force increment
    # tau[k] - tau[k-1], so this term does not penalize a matched hold force.
    force_weights: object = (0.04, 0.06, 0.06)
    delta_force_weights: object = (0.8, 1.0, 1.0)

    force_min: object = (-20.0, -15.0, -15.0)
    force_max: object = (20.0, 15.0, 15.0)
    delta_force_min: object = (-4.0, -3.0, -3.0)
    delta_force_max: object = (4.0, 3.0, 3.0)

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
        self.force_min = vector3(self.force_min, "force_min")
        self.force_max = vector3(self.force_max, "force_max")
        self.delta_force_min = vector3(self.delta_force_min, "delta_force_min")
        self.delta_force_max = vector3(self.delta_force_max, "delta_force_max")
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
        return self


@dataclass
class MPCResult:
    force: Array
    force_sequence: Array
    predicted_states: Array
    slacks: Array
    status: str
    iterations: int
    objective: float
    used_fallback: bool


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
        self._build_prediction_matrices()

    def reset(self) -> None:
        """Clear optimizer history when mode or tracked target changes."""
        self._warm_start = None
        self._last_feasible_force = None

    def safe_force(self, tau_previous, target_force=None) -> Array:
        """Rate-limit a safe return toward target_force (baseline by default)."""
        previous = vector3(tau_previous, "tau_previous")
        target = (
            self.model.tau_base
            if target_force is None
            else vector3(target_force, "target_force")
        )
        return self._safe_fallback(previous, target)

    def _build_prediction_matrices(self) -> None:
        """Build rolling-force predictions for pure model 1.

        The augmented state is z=[x, tau_previous]:

            x+ = A_d x + B_d (tau - tau_previous)
            tau_previous+ = tau
        """
        physical_A = self.model.A_d
        physical_B = self.model.B_d
        A = np.zeros((9, 9))
        B = np.zeros((9, 3))
        A[:6, :6] = physical_A
        A[:6, 6:] = -physical_B
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
        self.effective_force_matrix = self.rate_matrix.copy()

    def _reference_stack(self, reference_position: Array) -> Array:
        state_reference = np.concatenate((reference_position, np.zeros(3)))
        return np.tile(state_reference, self.config.horizon)

    def _cost(
        self,
        free_prediction: Array,
        reference_position: Array,
        tau_previous: Array,
    ) -> tuple[Array, Array]:
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
        effective_force_reference = rate_reference.copy()

        P_force = 2.0 * (
            self.Su.T @ Q_bar @ self.Su
            + self.effective_force_matrix.T @ R_bar @ self.effective_force_matrix
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

        # FOV and forward-distance constraints on each predicted position.
        for step in range(N):
            state_offset = 6 * step
            force_forward = self.Su[state_offset + cfg.forward_axis]
            force_horizontal = self.Su[state_offset + cfg.horizontal_axis]
            force_vertical = self.Su[state_offset + cfg.vertical_axis]
            free_forward = free_prediction[state_offset + cfg.forward_axis]
            free_horizontal = free_prediction[state_offset + cfg.horizontal_axis]
            free_vertical = free_prediction[state_offset + cfg.vertical_axis]

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

    def _safe_fallback(self, tau_previous: Array, tau_base: Array) -> Array:
        cfg = self.config
        previous = np.clip(tau_previous, cfg.force_min, cfg.force_max)
        low = np.maximum(cfg.force_min, previous + cfg.delta_force_min)
        high = np.minimum(cfg.force_max, previous + cfg.delta_force_max)
        return np.minimum(np.maximum(tau_base, low), high)

    def solve(
        self,
        state,
        tau_previous,
        tau_base=None,
        reference_position=None,
    ) -> MPCResult:
        state = np.asarray(state, dtype=float).reshape(-1)
        if state.shape != (6,) or not np.all(np.isfinite(state)):
            raise ValueError("state must be [p_rel(3), v_rel(3)]")
        tau_previous = vector3(tau_previous, "tau_previous")
        # The rolling previous force drives model 1.  The separately latched
        # baseline is reserved for a safe optimizer-failure fallback.
        fallback_baseline = (
            self.model.tau_base
            if tau_base is None
            else vector3(tau_base, "tau_base")
        )
        reference = (
            self.config.reference_position
            if reference_position is None
            else vector3(reference_position, "reference_position")
        )

        augmented_state = np.concatenate((state, tau_previous))
        free_prediction = self.Sx @ augmented_state
        P, q = self._cost(free_prediction, reference, tau_previous)
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
            self._warm_start = self._shift_warm_start(decision)
            self._last_feasible_force = force.copy()
            used_fallback = False
            status = solution.status
        else:
            if self._last_feasible_force is None:
                fallback_target = fallback_baseline
                fallback_source = "baseline"
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
            + self.Su @ force_sequence.ravel()
        ).reshape(N, 6)
        return MPCResult(
            force=force,
            force_sequence=force_sequence,
            predicted_states=predicted,
            slacks=slacks,
            status=status,
            iterations=solution.iterations,
            objective=solution.objective,
            used_fallback=used_fallback,
        )

