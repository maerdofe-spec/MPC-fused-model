"""Rotation-aware position-only Kalman filter for [p_rel, v_rel]."""

from __future__ import annotations

import numpy as np

from MPC_dual_model.fossen_fixed_dl_model import vector3
from MPC_dual_model.relative_kalman import KalmanConfig

from .yaw_relative_model import RotationAwareRelativeModel, rotation_state_matrix


class RotationAwareKalmanFilter:
    """Estimate body-frame relative position and translational velocity.

    The IMU yaw increment is treated as a measured coordinate transformation,
    not as process noise and not as target translation.
    """

    def __init__(
        self,
        model: RotationAwareRelativeModel,
        config: KalmanConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or KalmanConfig()
        self.H = np.hstack((np.eye(3), np.zeros((3, 3))))
        self.R = np.diag(vector3(self.config.position_std, "position_std") ** 2)
        self.Q = self._process_covariance()
        self.reset()

    def reset(self) -> None:
        """Discard the previous target track before a fresh acquisition."""
        self.x = np.zeros(6)
        self.P = np.eye(6)
        self.initialized = False

    def _process_covariance(self) -> np.ndarray:
        dt = self.model.dt
        variance = vector3(self.config.acceleration_std, "acceleration_std") ** 2
        diagonal = np.diag(variance)
        covariance = np.zeros((6, 6))
        covariance[:3, :3] = 0.25 * dt**4 * diagonal
        covariance[:3, 3:] = 0.5 * dt**3 * diagonal
        covariance[3:, :3] = 0.5 * dt**3 * diagonal
        covariance[3:, 3:] = dt**2 * diagonal
        return covariance

    def initialize(self, position, velocity=(0.0, 0.0, 0.0)) -> np.ndarray:
        self.x = np.concatenate(
            (vector3(position, "position"), vector3(velocity, "velocity"))
        )
        p_std = vector3(self.config.initial_position_std, "initial_position_std")
        v_std = vector3(self.config.initial_velocity_std, "initial_velocity_std")
        self.P = np.diag(np.concatenate((p_std**2, v_std**2)))
        self.initialized = True
        return self.x.copy()

    def predict(self, effective_force, delta_yaw_rad: float) -> np.ndarray:
        if not self.initialized:
            raise RuntimeError("filter must be initialized before predict")
        transition, input_matrix = self.model.rotation_aware_matrices(delta_yaw_rad)
        self.x = transition @ self.x + input_matrix @ vector3(
            effective_force, "effective_force"
        )
        self._predict_covariance(transition, delta_yaw_rad)
        return self.x.copy()

    def predict_mean(self, predicted_state, delta_yaw_rad: float) -> np.ndarray:
        """Advance covariance with actual yaw while using a fused mean."""
        if not self.initialized:
            raise RuntimeError("filter must be initialized before predict")
        mean = np.asarray(predicted_state, dtype=float).reshape(-1)
        if mean.shape != (6,) or not np.all(np.isfinite(mean)):
            raise ValueError("predicted_state must be finite with shape (6,)")
        transition, _ = self.model.rotation_aware_matrices(delta_yaw_rad)
        self.x = mean.copy()
        self._predict_covariance(transition, delta_yaw_rad)
        return self.x.copy()

    def _predict_covariance(
        self,
        transition: np.ndarray,
        delta_yaw_rad: float,
    ) -> None:
        rotation = rotation_state_matrix(delta_yaw_rad)
        process_covariance = rotation @ self.Q @ rotation.T
        self.P = transition @ self.P @ transition.T + process_covariance
        self.P = 0.5 * (self.P + self.P.T)

    def update(self, position_measurement) -> np.ndarray:
        measurement = vector3(position_measurement, "position_measurement")
        if not self.initialized:
            return self.initialize(measurement)

        innovation = measurement - self.H @ self.x
        innovation_covariance = self.H @ self.P @ self.H.T + self.R
        gain = np.linalg.solve(innovation_covariance, self.H @ self.P).T
        self.x = self.x + gain @ innovation

        identity = np.eye(6)
        correction = identity - gain @ self.H
        self.P = correction @ self.P @ correction.T + gain @ self.R @ gain.T
        self.P = 0.5 * (self.P + self.P.T)
        return self.x.copy()
