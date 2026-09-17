"""Yaw state machine, cascaded PID, and frozen yaw prediction.

Yaw control is intentionally outside the translation QP.  At each camera
update this module selects a HOLD/TURN/SETTLE goal, produces the current yaw
moment, and freezes a future (psi, omega, N) trajectory for one MPC solve.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .yaw_relative_model import LinearYawDynamics, finite_scalar, wrap_angle


Array = np.ndarray


class YawMode(str, Enum):
    HOLD = "HOLD"
    TURN = "TURN"
    SETTLE = "SETTLE"


@dataclass
class YawControlConfig:
    """Yaw thresholds, cascaded-PID gains, and physical limits.

    Angles use radians, angular rates use rad/s, and moments use N*m.  Every
    numerical default is only a simulation placeholder and must be calibrated
    before hardware use.
    """

    alpha_on: float = np.deg2rad(25.0)
    alpha_off: float = np.deg2rad(10.0)
    alpha_emergency: float = np.deg2rad(35.0)
    trigger_frames: int = 3
    settle_frames: int = 5
    require_outward_motion_to_trigger: bool = False

    yaw_tolerance: float = np.deg2rad(2.0)
    omega_tolerance: float = np.deg2rad(2.0)

    outer_kp: float = 1.5
    outer_ki: float = 0.0
    outer_kd: float = 0.20
    outer_integral_limit: float = np.deg2rad(30.0)

    inner_kp: float = 3.0
    inner_ki: float = 0.20
    inner_kd: float = 0.0
    inner_integral_limit: float = np.deg2rad(60.0)

    omega_command_max: float = np.deg2rad(35.0)
    omega_command_acceleration_max: float = np.deg2rad(60.0)
    yaw_moment_min: float = -4.0
    yaw_moment_max: float = 4.0
    delta_yaw_moment_min: float = -0.8
    delta_yaw_moment_max: float = 0.8
    use_dynamics_feedforward: bool = False

    def normalized(self) -> "YawControlConfig":
        scalar_names = (
            "alpha_on",
            "alpha_off",
            "alpha_emergency",
            "yaw_tolerance",
            "omega_tolerance",
            "outer_kp",
            "outer_ki",
            "outer_kd",
            "outer_integral_limit",
            "inner_kp",
            "inner_ki",
            "inner_kd",
            "inner_integral_limit",
            "omega_command_max",
            "omega_command_acceleration_max",
            "yaw_moment_min",
            "yaw_moment_max",
            "delta_yaw_moment_min",
            "delta_yaw_moment_max",
        )
        for name in scalar_names:
            setattr(self, name, finite_scalar(getattr(self, name), name))
        if not 0.0 < self.alpha_off < self.alpha_on < self.alpha_emergency < np.pi / 2:
            raise ValueError("require 0 < alpha_off < alpha_on < alpha_emergency < pi/2")
        if self.trigger_frames < 1 or self.settle_frames < 1:
            raise ValueError("trigger_frames and settle_frames must be positive")
        positive_names = (
            "yaw_tolerance",
            "omega_tolerance",
            "outer_integral_limit",
            "inner_integral_limit",
            "omega_command_max",
            "omega_command_acceleration_max",
        )
        if any(getattr(self, name) <= 0.0 for name in positive_names):
            raise ValueError("yaw tolerances and limits must be positive")
        gain_names = ("outer_kp", "outer_ki", "outer_kd", "inner_kp", "inner_ki", "inner_kd")
        if any(getattr(self, name) < 0.0 for name in gain_names):
            raise ValueError("PID gains must be nonnegative")
        if self.yaw_moment_min >= self.yaw_moment_max:
            raise ValueError("invalid yaw moment limits")
        if self.delta_yaw_moment_min >= self.delta_yaw_moment_max:
            raise ValueError("invalid yaw moment-rate limits")
        return self


@dataclass(frozen=True)
class YawPrediction:
    """Yaw trajectory frozen for one translation-QP solve."""

    angles: Array
    rates: Array
    delta_angles: Array
    moments: Array
    goal_angle: float
    mode: YawMode

    @property
    def horizon(self) -> int:
        return int(self.moments.size)


@dataclass(frozen=True)
class YawControlResult:
    mode: YawMode
    hold_angle: float
    goal_angle: float
    alpha: float
    alpha_rate: float
    angle_error: float
    omega_command: float
    yaw_moment: float
    prediction: YawPrediction


@dataclass
class _ControlMemory:
    outer_integral: float = 0.0
    inner_integral: float = 0.0
    previous_inner_error: float = 0.0
    omega_command: float = 0.0
    previous_moment: float = 0.0

    def copy(self) -> "_ControlMemory":
        return _ControlMemory(
            outer_integral=self.outer_integral,
            inner_integral=self.inner_integral,
            previous_inner_error=self.previous_inner_error,
            omega_command=self.omega_command,
            previous_moment=self.previous_moment,
        )


class YawStateController:
    """HOLD/TURN/SETTLE state machine with angle/omega cascaded PID."""

    def __init__(
        self,
        model: LinearYawDynamics,
        config: YawControlConfig | None = None,
    ) -> None:
        self.model = model
        self.config = (config or YawControlConfig()).normalized()
        self.reset()

    def reset(self, yaw_angle: float | None = None) -> None:
        self.initialized = yaw_angle is not None
        initial = 0.0 if yaw_angle is None else finite_scalar(yaw_angle, "yaw_angle")
        self.hold_angle = wrap_angle(initial)
        self.goal_angle = self.hold_angle
        self.mode = YawMode.HOLD
        self._trigger_count = 0
        self._settle_count = 0
        self._previous_alpha: float | None = None
        self._memory = _ControlMemory(previous_moment=self.model.yaw_moment_base)

    def safe_moment(self, previous_moment: float, target_moment: float | None = None) -> float:
        cfg = self.config
        previous = np.clip(
            finite_scalar(previous_moment, "previous_moment"),
            cfg.yaw_moment_min,
            cfg.yaw_moment_max,
        )
        target = self.model.yaw_moment_base if target_moment is None else finite_scalar(
            target_moment, "target_moment"
        )
        low = max(cfg.yaw_moment_min, previous + cfg.delta_yaw_moment_min)
        high = min(cfg.yaw_moment_max, previous + cfg.delta_yaw_moment_max)
        return float(np.clip(target, low, high))

    def _initialize_if_needed(self, yaw_angle: float, previous_moment: float) -> None:
        if not self.initialized:
            self.reset(yaw_angle)
            self._memory.previous_moment = previous_moment

    def _set_turn_goal(self, yaw_angle: float, alpha: float) -> None:
        self.goal_angle = wrap_angle(yaw_angle + alpha)
        self.mode = YawMode.TURN
        self._settle_count = 0

    def _update_mode(
        self,
        yaw_angle: float,
        yaw_rate: float,
        alpha: float,
        alpha_rate: float,
    ) -> None:
        cfg = self.config
        outward = alpha * alpha_rate > 0.0
        if self.mode is YawMode.HOLD:
            trigger = abs(alpha) >= cfg.alpha_on and (
                outward or not cfg.require_outward_motion_to_trigger
            )
            self._trigger_count = self._trigger_count + 1 if trigger else 0
            if self._trigger_count >= cfg.trigger_frames:
                self._set_turn_goal(yaw_angle, alpha)
                self._trigger_count = 0
            return

        angle_close = abs(wrap_angle(self.goal_angle - yaw_angle)) <= cfg.yaw_tolerance
        rate_close = abs(yaw_rate) <= cfg.omega_tolerance

        if self.mode is YawMode.TURN:
            if abs(alpha) >= cfg.alpha_emergency and outward:
                self.goal_angle = wrap_angle(yaw_angle + alpha)
            elif angle_close and rate_close and abs(alpha) > cfg.alpha_off:
                # The target moved during the turn; continue from the current
                # state toward a newly measured endpoint.
                self.goal_angle = wrap_angle(yaw_angle + alpha)
            if abs(alpha) <= cfg.alpha_off:
                self.mode = YawMode.SETTLE
                self._settle_count = 0

        if self.mode is YawMode.SETTLE:
            if abs(alpha) >= cfg.alpha_on:
                self._set_turn_goal(yaw_angle, alpha)
                return
            finished = angle_close and rate_close and abs(alpha) <= cfg.alpha_off
            self._settle_count = self._settle_count + 1 if finished else 0
            if self._settle_count >= cfg.settle_frames:
                self.hold_angle = self.goal_angle
                self.goal_angle = self.hold_angle
                self.mode = YawMode.HOLD
                self._settle_count = 0
                self._memory.outer_integral = 0.0
                self._memory.inner_integral = 0.0

    def _command(
        self,
        yaw_angle: float,
        yaw_rate: float,
        previous_moment: float,
        memory: _ControlMemory,
    ) -> tuple[float, float]:
        cfg = self.config
        dt = self.model.dt
        angle_error = wrap_angle(self.goal_angle - yaw_angle)

        outer_candidate = float(
            np.clip(
                memory.outer_integral + dt * angle_error,
                -cfg.outer_integral_limit,
                cfg.outer_integral_limit,
            )
        )
        omega_raw = (
            cfg.outer_kp * angle_error
            + cfg.outer_ki * outer_candidate
            - cfg.outer_kd * yaw_rate
        )
        omega_limited = float(
            np.clip(omega_raw, -cfg.omega_command_max, cfg.omega_command_max)
        )
        max_delta_omega = cfg.omega_command_acceleration_max * dt
        omega_command = memory.omega_command + float(
            np.clip(
                omega_limited - memory.omega_command,
                -max_delta_omega,
                max_delta_omega,
            )
        )
        if np.isclose(omega_raw, omega_limited) or angle_error * (omega_raw - omega_limited) < 0.0:
            memory.outer_integral = outer_candidate

        inner_error = omega_command - yaw_rate
        derivative = (inner_error - memory.previous_inner_error) / dt
        inner_candidate = float(
            np.clip(
                memory.inner_integral + dt * inner_error,
                -cfg.inner_integral_limit,
                cfg.inner_integral_limit,
            )
        )
        feedforward = 0.0
        if cfg.use_dynamics_feedforward:
            acceleration_command = (omega_command - memory.omega_command) / dt
            feedforward = (
                self.model.effective_inertia * acceleration_command
                + self.model.linear_damping * omega_command
                + self.model.quadratic_damping * abs(omega_command) * omega_command
            )
        moment_raw = (
            feedforward
            + cfg.inner_kp * inner_error
            + cfg.inner_ki * inner_candidate
            + cfg.inner_kd * derivative
        )
        moment_absolute = float(
            np.clip(moment_raw, cfg.yaw_moment_min, cfg.yaw_moment_max)
        )
        previous = float(
            np.clip(previous_moment, cfg.yaw_moment_min, cfg.yaw_moment_max)
        )
        low = max(cfg.yaw_moment_min, previous + cfg.delta_yaw_moment_min)
        high = min(cfg.yaw_moment_max, previous + cfg.delta_yaw_moment_max)
        moment_command = float(np.clip(moment_absolute, low, high))
        if np.isclose(moment_raw, moment_command) or inner_error * (moment_raw - moment_command) < 0.0:
            memory.inner_integral = inner_candidate

        memory.previous_inner_error = inner_error
        memory.omega_command = omega_command
        memory.previous_moment = moment_command
        return moment_command, angle_error

    def _prediction(
        self,
        yaw_angle: float,
        yaw_rate: float,
        current_moment: float,
        memory_after_current: _ControlMemory,
        horizon: int,
    ) -> YawPrediction:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        angles = np.zeros(horizon + 1)
        rates = np.zeros(horizon + 1)
        deltas = np.zeros(horizon)
        moments = np.zeros(horizon)
        angles[0] = yaw_angle
        rates[0] = yaw_rate
        moments[0] = current_moment
        prediction_memory = memory_after_current.copy()

        for step in range(horizon):
            if step > 0:
                moments[step], _ = self._command(
                    angles[step],
                    rates[step],
                    moments[step - 1],
                    prediction_memory,
                )
            angles[step + 1], rates[step + 1], deltas[step] = (
                self.model.predict_yaw_step(
                    angles[step], rates[step], moments[step]
                )
            )
        return YawPrediction(
            angles=angles,
            rates=rates,
            delta_angles=deltas,
            moments=moments,
            goal_angle=self.goal_angle,
            mode=self.mode,
        )

    def update(
        self,
        yaw_angle: float,
        yaw_rate: float,
        alpha: float,
        previous_achieved_moment: float,
        horizon: int,
    ) -> YawControlResult:
        """Advance real controller state and freeze one future yaw trajectory."""
        psi = wrap_angle(yaw_angle)
        omega = finite_scalar(yaw_rate, "yaw_rate")
        alpha = wrap_angle(alpha)
        previous_moment = finite_scalar(
            previous_achieved_moment, "previous_achieved_moment"
        )
        self._initialize_if_needed(psi, previous_moment)

        alpha_rate = 0.0
        if self._previous_alpha is not None:
            alpha_rate = wrap_angle(alpha - self._previous_alpha) / self.model.dt
        self._previous_alpha = alpha
        self._update_mode(psi, omega, alpha, alpha_rate)

        moment, angle_error = self._command(
            psi, omega, previous_moment, self._memory
        )
        prediction = self._prediction(
            psi, omega, moment, self._memory, horizon
        )
        return YawControlResult(
            mode=self.mode,
            hold_angle=self.hold_angle,
            goal_angle=self.goal_angle,
            alpha=alpha,
            alpha_rate=alpha_rate,
            angle_error=angle_error,
            omega_command=self._memory.omega_command,
            yaw_moment=moment,
            prediction=prediction,
        )
