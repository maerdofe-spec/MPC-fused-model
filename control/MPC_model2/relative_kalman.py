"""Position-measurement Kalman estimator with model-2 prediction."""

import numpy as np

from MPC_dual_model.relative_kalman import (
    KalmanConfig,
    RelativePositionKalmanFilter as _SharedRelativePositionKalmanFilter,
)

from .fossen_fixed_dl_model import vector3


class RelativePositionKalmanFilter(_SharedRelativePositionKalmanFilter):
    """Estimate [p_rel,v_rel] using absolute force in the time update."""

    def predict(self, tau_achieved) -> np.ndarray:
        if not self.initialized:
            raise RuntimeError("filter must be initialized before predict")
        tau = vector3(tau_achieved, "tau_achieved")
        self.x = self.model.A_d @ self.x + self.model.B_d @ tau
        self.P = self.model.A_d @ self.P @ self.model.A_d.T + self.Q
        self.P = 0.5 * (self.P + self.P.T)
        return self.x.copy()

    def step(self, position_measurement, tau_achieved) -> np.ndarray:
        if not self.initialized:
            return self.initialize(position_measurement)
        self.predict(tau_achieved)
        return self.update(position_measurement)

__all__ = ["KalmanConfig", "RelativePositionKalmanFilter"]
