"""Position-measurement Kalman filter for relative position and velocity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from .fossen_fixed_dl_model import FixedLinearDampingRelativeModel, vector3
except ImportError:
    from fossen_fixed_dl_model import FixedLinearDampingRelativeModel, vector3


@dataclass
class KalmanConfig:
    position_std: object = (0.015, 0.015, 0.025)
    acceleration_std: object = (0.35, 0.35, 0.45)
    initial_position_std: object = (0.04, 0.04, 0.06)
    initial_velocity_std: object = (0.30, 0.30, 0.40)


class RelativePositionKalmanFilter:
    """Estimate x=[p_rel, v_rel] using relative-position measurements only."""

    def __init__(
        self,
        model: FixedLinearDampingRelativeModel,
        config: KalmanConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or KalmanConfig()
        self.H = np.hstack((np.eye(3), np.zeros((3, 3))))
        self.R = np.diag(vector3(self.config.position_std, "position_std") ** 2)
        self.Q = self._process_covariance()
        self.x = np.zeros(6)
        self.P = np.eye(6)
        self.initialized = False

    def _process_covariance(self) -> np.ndarray:
        dt = self.model.dt
        variance = vector3(self.config.acceleration_std, "acceleration_std") ** 2
        diagonal = np.diag(variance)
        Q = np.zeros((6, 6))
        Q[:3, :3] = 0.25 * dt**4 * diagonal
        Q[:3, 3:] = 0.5 * dt**3 * diagonal
        Q[3:, :3] = 0.5 * dt**3 * diagonal
        Q[3:, 3:] = dt**2 * diagonal
        return Q

    def initialize(self, position, velocity=(0.0, 0.0, 0.0)) -> np.ndarray:
        self.x = np.concatenate(
            (vector3(position, "position"), vector3(velocity, "velocity"))
        )
        p_std = vector3(self.config.initial_position_std, "initial_position_std")
        v_std = vector3(self.config.initial_velocity_std, "initial_velocity_std")
        self.P = np.diag(np.concatenate((p_std**2, v_std**2)))
        self.initialized = True
        return self.x.copy()

    def predict(self, tau_achieved, tau_base=None) -> np.ndarray:
        if not self.initialized:
            raise RuntimeError("filter must be initialized before predict")
        base = self.model.tau_base if tau_base is None else vector3(tau_base, "tau_base")
        delta_tau = vector3(tau_achieved, "tau_achieved") - base
        self.x = self.model.A_d @ self.x + self.model.B_d @ delta_tau
        self.P = self.model.A_d @ self.P @ self.model.A_d.T + self.Q
        self.P = 0.5 * (self.P + self.P.T)
        return self.x.copy()

    def predict_mean(self, predicted_state) -> np.ndarray:
        """Advance covariance with A_d while supplying a fused predicted mean."""
        if not self.initialized:
            raise RuntimeError("filter must be initialized before predict")
        mean = np.asarray(predicted_state, dtype=float).reshape(-1)
        if mean.shape != (6,) or not np.all(np.isfinite(mean)):
            raise ValueError("predicted_state must be finite with shape (6,)")
        self.x = mean.copy()
        self.P = self.model.A_d @ self.P @ self.model.A_d.T + self.Q
        self.P = 0.5 * (self.P + self.P.T)
        return self.x.copy()

    def update(self, position_measurement) -> np.ndarray:
        measurement = vector3(position_measurement, "position_measurement")
        if not self.initialized:
            return self.initialize(measurement)

        innovation = measurement - self.H @ self.x
        innovation_covariance = self.H @ self.P @ self.H.T + self.R
        gain = np.linalg.solve(innovation_covariance, self.H @ self.P).T
        self.x = self.x + gain @ innovation

        # Joseph form preserves a positive-semidefinite covariance better.
        identity = np.eye(6)
        correction = identity - gain @ self.H
        self.P = correction @ self.P @ correction.T + gain @ self.R @ gain.T
        self.P = 0.5 * (self.P + self.P.T)
        return self.x.copy()

    def step(self, position_measurement, tau_achieved, tau_base=None) -> np.ndarray:
        if not self.initialized:
            return self.initialize(position_measurement)
        self.predict(tau_achieved, tau_base)
        return self.update(position_measurement)

