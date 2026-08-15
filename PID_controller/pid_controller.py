"""Three-axis relative-position controller implemented only with PID.

The controller uses no vehicle model, state observer, optimizer, or prediction
horizon.  Position error is differentiated internally and low-pass filtered.
All quantities follow the FineSUB body FRD convention: forward, right, down.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


def vector3(value: object, name: str) -> Array:
    result = np.asarray(value, dtype=float).reshape(-1)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector with shape (3,)")
    return result


def _nonnegative_vector3(value: object, name: str) -> Array:
    result = vector3(value, name)
    if np.any(result < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return result


@dataclass
class PIDConfig:
    """PID gains and physical output limits.

    ``kp``, ``ki`` and ``kd`` act independently on body forward/right/down.
    The integral state is measured in metre-seconds.  When a thruster matrix
    is supplied, every final command also satisfies its asymmetric bounds.
    """

    dt: float = 0.05
    reference_position: object = (0.80, 0.0, 0.0)

    kp: object = (28.0, 38.0, 50.0)
    ki: object = (1.0, 1.5, 2.0)
    kd: object = (16.0, 22.0, 26.0)
    derivative_filter_time_constant: float = 0.12
    integral_limit: object = (2.0, 1.5, 1.2)
    anti_windup_gain: float = 1.0
    error_deadband: object = (0.005, 0.005, 0.008)

    force_min: object = (-16.0, -16.0, -23.0)
    force_max: object = (16.0, 16.0, 29.0)
    delta_force_min: object = (-4.0, -4.0, -5.6)
    delta_force_max: object = (4.0, 4.0, 5.6)

    thruster_force_matrix: object | None = None
    thruster_force_min: object | None = None
    thruster_force_max: object | None = None

    def normalized(self) -> "PIDConfig":
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be positive and finite")
        self.reference_position = vector3(
            self.reference_position, "reference_position"
        )
        self.kp = _nonnegative_vector3(self.kp, "kp")
        self.ki = _nonnegative_vector3(self.ki, "ki")
        self.kd = _nonnegative_vector3(self.kd, "kd")
        self.integral_limit = _nonnegative_vector3(
            self.integral_limit, "integral_limit"
        )
        self.error_deadband = _nonnegative_vector3(
            self.error_deadband, "error_deadband"
        )
        self.force_min = vector3(self.force_min, "force_min")
        self.force_max = vector3(self.force_max, "force_max")
        self.delta_force_min = vector3(self.delta_force_min, "delta_force_min")
        self.delta_force_max = vector3(self.delta_force_max, "delta_force_max")
        if np.any(self.force_min >= self.force_max):
            raise ValueError("force_min must be smaller than force_max")
        if np.any(self.delta_force_min >= 0.0) or np.any(
            self.delta_force_max <= 0.0
        ):
            raise ValueError("each delta-force interval must contain zero")
        if (
            not np.isfinite(self.derivative_filter_time_constant)
            or self.derivative_filter_time_constant < 0.0
        ):
            raise ValueError("derivative_filter_time_constant must be nonnegative")
        if not np.isfinite(self.anti_windup_gain) or self.anti_windup_gain < 0.0:
            raise ValueError("anti_windup_gain must be nonnegative")
        self._normalize_thruster_envelope()
        return self

    def _normalize_thruster_envelope(self) -> None:
        supplied = (
            self.thruster_force_matrix is not None,
            self.thruster_force_min is not None,
            self.thruster_force_max is not None,
        )
        if any(supplied) and not all(supplied):
            raise ValueError("thruster matrix and both bounds must be supplied together")
        if not all(supplied):
            return
        matrix = np.asarray(self.thruster_force_matrix, dtype=float)
        if (
            matrix.ndim != 2
            or matrix.shape[1] != 3
            or matrix.shape[0] == 0
            or not np.all(np.isfinite(matrix))
        ):
            raise ValueError("thruster_force_matrix must have shape (n, 3)")
        lower = np.asarray(self.thruster_force_min, dtype=float).reshape(-1)
        upper = np.asarray(self.thruster_force_max, dtype=float).reshape(-1)
        if lower.shape != (matrix.shape[0],) or upper.shape != lower.shape:
            raise ValueError("thruster bounds must match the matrix row count")
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("thruster bounds must be finite")
        if np.any(lower >= 0.0) or np.any(upper <= 0.0):
            raise ValueError("every thruster-force interval must contain zero")
        self.thruster_force_matrix = matrix
        self.thruster_force_min = lower
        self.thruster_force_max = upper


@dataclass(frozen=True)
class PIDResult:
    force: Array
    requested_force: Array
    error: Array
    error_derivative: Array
    integral_state: Array
    proportional_term: Array
    integral_term: Array
    derivative_term: Array
    saturated: Array
    status: str


class RelativePIDController:
    """Independent three-axis PID plus coupled actuator safety limiting."""

    def __init__(self, config: PIDConfig | None = None) -> None:
        self.config = (config or PIDConfig()).normalized()
        self._integral = np.zeros(3)
        self._derivative = np.zeros(3)
        self._previous_error: Array | None = None
        self._baseline_force = np.zeros(3)

    @property
    def baseline_force(self) -> Array:
        return self._baseline_force.copy()

    def reset(self, *, keep_baseline: bool = True) -> None:
        """Clear all PID history, optionally including the latched baseline."""
        self._integral.fill(0.0)
        self._derivative.fill(0.0)
        self._previous_error = None
        if not keep_baseline:
            self._baseline_force.fill(0.0)

    def latch_baseline(self, force: object) -> None:
        """Capture the current hold force and start with fresh PID history."""
        self._baseline_force = self._project_absolute(
            vector3(force, "force"), np.zeros(3)
        )
        self.reset(keep_baseline=True)

    def set_reference(self, reference_position: object, *, reset_history: bool = False) -> None:
        self.config.reference_position = vector3(
            reference_position, "reference_position"
        )
        if reset_history:
            self.reset(keep_baseline=True)

    def update(
        self,
        position: object,
        previous_force: object,
        reference_position: object | None = None,
    ) -> PIDResult:
        """Compute one PID update from a new relative-position measurement."""
        measured = vector3(position, "position")
        previous = self._project_absolute(
            vector3(previous_force, "previous_force"), np.zeros(3)
        )
        reference = (
            self.config.reference_position
            if reference_position is None
            else vector3(reference_position, "reference_position")
        )

        raw_error = measured - reference
        error = np.where(
            np.abs(raw_error) <= self.config.error_deadband, 0.0, raw_error
        )
        if self._previous_error is None:
            raw_derivative = np.zeros(3)
        else:
            raw_derivative = (error - self._previous_error) / self.config.dt
        alpha = self.config.dt / (
            self.config.derivative_filter_time_constant + self.config.dt
        )
        self._derivative += alpha * (raw_derivative - self._derivative)

        candidate_integral = np.clip(
            self._integral + error * self.config.dt,
            -self.config.integral_limit,
            self.config.integral_limit,
        )
        p_term = self.config.kp * error
        i_term = self.config.ki * candidate_integral
        d_term = self.config.kd * self._derivative
        requested = self._baseline_force + p_term + i_term + d_term
        limited = self._apply_output_limits(requested, previous)

        # Back-calculation prevents the integrator from storing an actuator
        # request that cannot be produced due to force, slew, or motor limits.
        correction = np.zeros(3)
        nonzero_ki = self.config.ki > np.finfo(float).eps
        correction[nonzero_ki] = (
            self.config.anti_windup_gain
            * (limited[nonzero_ki] - requested[nonzero_ki])
            / self.config.ki[nonzero_ki]
            * self.config.dt
        )
        self._integral = np.clip(
            candidate_integral + correction,
            -self.config.integral_limit,
            self.config.integral_limit,
        )
        self._previous_error = error.copy()

        saturated = ~np.isclose(limited, requested, atol=1e-9, rtol=0.0)
        return PIDResult(
            force=limited.copy(),
            requested_force=requested.copy(),
            error=error.copy(),
            error_derivative=self._derivative.copy(),
            integral_state=self._integral.copy(),
            proportional_term=p_term.copy(),
            integral_term=(self.config.ki * self._integral).copy(),
            derivative_term=d_term.copy(),
            saturated=saturated,
            status="limited" if np.any(saturated) else "ok",
        )

    def safe_force(self, previous_force: object) -> Array:
        """Rate-limit output back to the captured baseline after target loss."""
        previous = self._project_absolute(
            vector3(previous_force, "previous_force"), np.zeros(3)
        )
        self.reset(keep_baseline=True)
        return self._apply_output_limits(self._baseline_force, previous)

    def _apply_output_limits(self, requested: Array, previous: Array) -> Array:
        desired = self._project_absolute(requested, np.zeros(3))
        delta = np.clip(
            desired - previous,
            self.config.delta_force_min,
            self.config.delta_force_max,
        )
        candidate = previous + delta
        # Axis-wise slew clipping can leave a coupled polytope. Project along
        # the line from the known feasible previous force when necessary.
        return self._project_absolute(candidate, previous)

    def _project_absolute(self, requested: Array, origin: Array) -> Array:
        target = np.clip(requested, self.config.force_min, self.config.force_max)
        if self.config.thruster_force_matrix is None:
            return target
        start = np.clip(origin, self.config.force_min, self.config.force_max)
        if not self._inside_thruster_envelope(start):
            start = self._radial_thruster_projection(start)
        direction = target - start
        scale = self._maximum_feasible_scale(start, direction)
        return start + scale * direction

    def _inside_thruster_envelope(self, force: Array) -> bool:
        values = self.config.thruster_force_matrix @ force
        return bool(
            np.all(values >= self.config.thruster_force_min - 1e-10)
            and np.all(values <= self.config.thruster_force_max + 1e-10)
        )

    def _radial_thruster_projection(self, force: Array) -> Array:
        scale = self._maximum_feasible_scale(np.zeros(3), force)
        return scale * force

    def _maximum_feasible_scale(self, start: Array, direction: Array) -> float:
        matrix = self.config.thruster_force_matrix
        if matrix is None:
            return 1.0
        start_values = matrix @ start
        changes = matrix @ direction
        maximum = 1.0
        for value, change, lower, upper in zip(
            start_values,
            changes,
            self.config.thruster_force_min,
            self.config.thruster_force_max,
        ):
            if change > 1e-12:
                maximum = min(maximum, (upper - value) / change)
            elif change < -1e-12:
                maximum = min(maximum, (lower - value) / change)
        return float(np.clip(maximum, 0.0, 1.0))
