import unittest

import numpy as np

from MPC_model1.device_adapter import ForceCommandAdapter
from MPC_model1.fossen_fixed_dl_model import (
    FixedLinearDampingRelativeModel,
)
from MPC_model1.mpc_controller import (
    MPCConfig,
    RelativeMPCController,
)
from MPC_model1.mpc_tracker import MPCTracker
from MPC_model1.relative_kalman import (
    RelativePositionKalmanFilter,
)


class Model1MPCTest(unittest.TestCase):
    def build(self):
        model = FixedLinearDampingRelativeModel(
            M_t=np.diag([20.0, 25.0, 30.0]),
            D_L=np.diag([8.0, 10.0, 12.0]),
            dt=0.1,
        )
        controller = RelativeMPCController(
            model,
            MPCConfig(
                horizon=4,
                force_min=(-12.0, -10.0, -10.0),
                force_max=(12.0, 10.0, 10.0),
                delta_force_min=(-3.0, -2.0, -2.0),
                delta_force_max=(3.0, 2.0, 2.0),
            ),
        )
        return model, controller

    def test_prediction_uses_rolling_previous_force(self):
        model, controller = self.build()
        state = np.zeros(6)
        previous = np.array([2.0, -1.0, 0.5])
        force_sequence = np.array(
            [[3.0, -0.5, 0.0], [4.0, 0.0, -0.5], [4.0, 0.0, -0.5], [4.0, 0.0, -0.5]]
        )
        predicted = (
            controller.Sx @ np.concatenate((state, previous))
            + controller.Su @ force_sequence.ravel()
        ).reshape(4, 6)

        expected = []
        x = state.copy()
        tau_before = previous.copy()
        for tau in force_sequence:
            x = model.A_d @ x + model.B_d @ (tau - tau_before)
            expected.append(x.copy())
            tau_before = tau
        np.testing.assert_allclose(predicted, expected, atol=1e-12)

    def test_public_result_has_no_fusion_weights(self):
        _, controller = self.build()
        result = controller.solve(
            state=np.array([1.0, 0.1, -0.05, 0.0, 0.0, 0.0]),
            tau_previous=np.zeros(3),
        )
        self.assertFalse(result.used_fallback, result.status)
        self.assertFalse(hasattr(result, "model1_weight"))
        self.assertFalse(hasattr(result, "model2_weight"))

    def test_tracker_predicts_only_model1(self):
        model, controller = self.build()
        estimator = RelativePositionKalmanFilter(model)
        tracker = MPCTracker(
            model,
            estimator,
            controller,
            ForceCommandAdapter(positive_force_at_limit=(12.0, 10.0, 10.0)),
        )
        tracker.latch_baseline((1.0, 0.0, 0.0))
        tracker.update((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        old = estimator.x.copy()
        tau_now = np.array([2.0, 0.0, 0.0])
        expected_prior = model.A_d @ old + model.B_d @ np.array([1.0, 0.0, 0.0])
        tracker.update(expected_prior[:3], tau_now)
        np.testing.assert_allclose(estimator.x, expected_prior, atol=2e-7)


if __name__ == "__main__":
    unittest.main()
