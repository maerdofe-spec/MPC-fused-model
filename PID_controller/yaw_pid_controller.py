"""Pure PID controller for the FineSUB yaw angle only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def finite_scalar(value: object, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def wrap_angle(angle: object) -> float:
    """Wrap an angle in radians to [-pi, pi)."""
    value = finite_scalar(angle, "angle")
    return float((value + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass
class YawPIDConfig:
    """Yaw-angle PID parameters; output is yaw moment in N*m."""

    dt: float = 0.05
    kp: float = 1.8
    ki: float = 0.12
    kd: float = 0.55
    integral_limit: float = np.deg2rad(35.0)
    derivative_filter_time_constant: float = 0.10
    angle_deadband: float = np.deg2rad(0.7)
    anti_windup_gain: float = 1.0
    yaw_moment_min: float = -1.5
    yaw_moment_max: float = 1.5
    delta_yaw_moment_min: float = -0.25
    delta_yaw_moment_max: float = 0.25

    def normalized(self) -> "YawPIDConfig":
        for name in (
            "dt",
            "kp",
            "ki",
            "kd",
            "integral_limit",
            "derivative_filter_time_constant",
            "angle_deadband",
            "anti_windup_gain",
            "yaw_moment_min",
            "yaw_moment_max",
            "delta_yaw_moment_min",
            "delta_yaw_moment_max",
        ):
            setattr(self, name, finite_scalar(getattr(self, name), name))
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if min(self.kp, self.ki, self.kd) < 0.0:
            raise ValueError("PID gains must be nonnegative")
        if min(
            self.integral_limit,
            self.derivative_filter_time_constant,
            self.angle_deadband,
            self.anti_windup_gain,
        ) < 0.0:
            raise ValueError("yaw limits/filter constants must be nonnegative")
        if self.yaw_moment_min >= 0.0 or self.yaw_moment_max <= 0.0:
            raise ValueError("yaw moment interval must contain zero")
        if self.delta_yaw_moment_min >= 0.0 or self.delta_yaw_moment_max <= 0.0:
            raise ValueError("yaw moment-rate interval must contain zero")
        return self


@dataclass(frozen=True)
class YawPIDResult:
    yaw_angle: float
    reference_yaw: float
    angle_error: float
    error_derivative: float
    integral_state: float
    proportional_term: float
    integral_term: float
    derivative_term: float
    requested_yaw_moment: float
    yaw_moment: float
    saturated: bool
    status: str


class YawPIDController:
    """Scalar yaw-angle PID with wrapping, filtering, and anti-windup."""

    def __init__(self, config: YawPIDConfig | None = None) -> None:
        self.config = (config or YawPIDConfig()).normalized()
        self._integral = 0.0
        self._error_derivative = 0.0
        self._previous_error: float | None = None
        self._baseline_moment = 0.0
        self._hold_angle: float | None = None

    @property
    def hold_angle(self) -> float | None:
        return self._hold_angle

    def reset(self, yaw_angle: float | None = None) -> None:
        self._integral = 0.0
        self._error_derivative = 0.0
        self._previous_error = None
        self._hold_angle = None if yaw_angle is None else wrap_angle(yaw_angle)

    def latch_baseline(self, yaw_moment: float, yaw_angle: float | None = None) -> None:
        self._baseline_moment = float(
            np.clip(
                finite_scalar(yaw_moment, "yaw_moment"),
                self.config.yaw_moment_min,
                self.config.yaw_moment_max,
            )
        )
        self.reset(yaw_angle)

    def update(
        self,
        yaw_angle: float,
        previous_yaw_moment: float,
        reference_yaw: float | None = None,
        yaw_rate: float | None = None,
    ) -> YawPIDResult:
        """Control yaw angle; positive yaw turns the nose right in FRD."""
        angle = wrap_angle(yaw_angle)
        previous = float(
            np.clip(
                finite_scalar(previous_yaw_moment, "previous_yaw_moment"),
                self.config.yaw_moment_min,
                self.config.yaw_moment_max,
            )
        )
        if reference_yaw is None:
            if self._hold_angle is None:
                self._hold_angle = angle
            reference = self._hold_angle
        else:
            reference = wrap_angle(reference_yaw)

        raw_error = wrap_angle(reference - angle)
        error = 0.0 if abs(raw_error) <= self.config.angle_deadband else raw_error
        if yaw_rate is None:
            raw_derivative = (
                0.0
                if self._previous_error is None
                else wrap_angle(error - self._previous_error) / self.config.dt
            )
        else:
            # D on the measured yaw angle avoids a reference-step kick.
            raw_derivative = -finite_scalar(yaw_rate, "yaw_rate")
        alpha = self.config.dt / (
            self.config.derivative_filter_time_constant + self.config.dt
        )
        self._error_derivative += alpha * (
            raw_derivative - self._error_derivative
        )

        candidate_integral = float(
            np.clip(
                self._integral + error * self.config.dt,
                -self.config.integral_limit,
                self.config.integral_limit,
            )
        )
        p_term = self.config.kp * error
        i_term = self.config.ki * candidate_integral
        d_term = self.config.kd * self._error_derivative
        requested = self._baseline_moment + p_term + i_term + d_term
        absolute_limited = float(
            np.clip(
                requested,
                self.config.yaw_moment_min,
                self.config.yaw_moment_max,
            )
        )
        moment = float(
            np.clip(
                absolute_limited,
                previous + self.config.delta_yaw_moment_min,
                previous + self.config.delta_yaw_moment_max,
            )
        )

        if self.config.ki > np.finfo(float).eps:
            candidate_integral += (
                self.config.anti_windup_gain
                * (moment - requested)
                / self.config.ki
                * self.config.dt
            )
        self._integral = float(
            np.clip(
                candidate_integral,
                -self.config.integral_limit,
                self.config.integral_limit,
            )
        )
        self._previous_error = error
        saturated = not np.isclose(moment, requested, atol=1e-10, rtol=0.0)
        return YawPIDResult(
            yaw_angle=angle,
            reference_yaw=reference,
            angle_error=error,
            error_derivative=self._error_derivative,
            integral_state=self._integral,
            proportional_term=p_term,
            integral_term=self.config.ki * self._integral,
            derivative_term=d_term,
            requested_yaw_moment=requested,
            yaw_moment=moment,
            saturated=bool(saturated),
            status="limited" if saturated else "ok",
        )

    def safe_moment(self, previous_yaw_moment: float) -> float:
        """Rate-limit yaw output back to its latched baseline moment."""
        previous = float(
            np.clip(
                finite_scalar(previous_yaw_moment, "previous_yaw_moment"),
                self.config.yaw_moment_min,
                self.config.yaw_moment_max,
            )
        )
        moment = float(
            np.clip(
                self._baseline_moment,
                previous + self.config.delta_yaw_moment_min,
                previous + self.config.delta_yaw_moment_max,
            )
        )
        self.reset()
        return moment
