import unittest

import numpy as np

from MPC_dual_model.device_adapter import (
    FineSUBThrusterAllocator,
    ForceCommandAdapter,
)
from MPC_model2.fossen_fixed_dl_model import (
    FixedLinearDampingRelativeModel,
)
from MPC_model2.mpc_controller import (
    MPCConfig,
    RelativeMPCController,
)
from MPC_model2.mpc_tracker import MPCTracker
from MPC_model2.relative_kalman import (
    RelativePositionKalmanFilter,
)


class Model2MPCTest(unittest.TestCase):
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

    def test_prediction_uses_absolute_force_only(self):
        model, controller = self.build()
        state = np.zeros(6)
        previous_a = np.array([2.0, -1.0, 0.5])
        previous_b = np.array([-4.0, 3.0, -2.0])
        force_sequence = np.array(
            [
                [3.0, -0.5, 0.0],
                [4.0, 0.0, -0.5],
                [4.0, 0.0, -0.5],
                [4.0, 0.0, -0.5],
            ]
        )

        predicted_a = (
            controller.Sx @ np.concatenate((state, previous_a))
            + controller.Su @ force_sequence.ravel()
        ).reshape(4, 6)
        predicted_b = (
            controller.Sx @ np.concatenate((state, previous_b))
            + controller.Su @ force_sequence.ravel()
        ).reshape(4, 6)

        expected = []
        x = state.copy()
        for tau in force_sequence:
            x = model.A_d @ x + model.B_d @ tau
            expected.append(x.copy())

        np.testing.assert_allclose(predicted_a, expected, atol=1e-12)
        np.testing.assert_allclose(predicted_b, expected, atol=1e-12)

    def test_controller_prediction_and_cost_ignore_tau_base(self):
        model, controller = self.build()
        state = np.array([1.0, 0.1, -0.05, 0.02, -0.01, 0.0])
        previous = np.array([2.0, -1.0, 0.5])

        model.set_tau_base((0.0, 0.0, 0.0))
        result_zero = controller.solve(state, previous)
        controller.reset()
        model.set_tau_base((9.0, -7.0, 5.0))
        result_nonzero = controller.solve(state, previous)

        np.testing.assert_allclose(result_nonzero.force, result_zero.force, atol=1e-8)
        np.testing.assert_allclose(
            result_nonzero.predicted_states,
            result_zero.predicted_states,
            atol=1e-8,
        )
        self.assertAlmostEqual(result_nonzero.objective, result_zero.objective, places=7)

    def test_terminal_position_and_velocity_weights_are_independent(self):
        model, _ = self.build()
        position_heavy = RelativeMPCController(
            model,
            MPCConfig(
                horizon=4,
                terminal_position_weight_scale=9.0,
                terminal_velocity_weight_scale=1.0,
            ),
        )
        velocity_heavy = RelativeMPCController(
            model,
            MPCConfig(
                horizon=4,
                terminal_position_weight_scale=1.0,
                terminal_velocity_weight_scale=9.0,
            ),
        )
        free = np.zeros(24)
        reference = np.array([0.8, 0.0, 0.0])
        previous = np.zeros(3)
        P_position, _ = position_heavy._cost(free, reference, previous)
        P_velocity, _ = velocity_heavy._cost(free, reference, previous)

        self.assertFalse(np.allclose(P_position, P_velocity))

    def test_legacy_terminal_scale_populates_both_terminal_weights(self):
        config = MPCConfig(terminal_weight_scale=7.0).normalized()
        self.assertEqual(config.terminal_position_weight_scale, 7.0)
        self.assertEqual(config.terminal_velocity_weight_scale, 7.0)

    def test_public_model_and_filter_ignore_fallback_baseline(self):
        model, _ = self.build()
        model.set_tau_base((9.0, -4.0, 2.0))
        p_next, v_next = model.predict(
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
        )
        expected = model.A_d @ np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        expected += model.B_d @ np.array([2.0, 0.0, 0.0])
        np.testing.assert_allclose(np.concatenate((p_next, v_next)), expected)

        estimator = RelativePositionKalmanFilter(model)
        estimator.initialize((1.0, 0.0, 0.0))
        predicted = estimator.predict((2.0, 0.0, 0.0))
        np.testing.assert_allclose(predicted, expected)

    def test_public_result_has_no_fusion_weights(self):
        _, controller = self.build()
        result = controller.solve(
            state=np.array([1.0, 0.1, -0.05, 0.0, 0.0, 0.0]),
            tau_previous=np.zeros(3),
        )
        self.assertFalse(result.used_fallback, result.status)
        self.assertFalse(hasattr(result, "model1_weight"))
        self.assertFalse(hasattr(result, "model2_weight"))

    def test_tracker_predicts_only_model2(self):
        model, controller = self.build()
        estimator = RelativePositionKalmanFilter(model)
        tracker = MPCTracker(
            model,
            estimator,
            controller,
            ForceCommandAdapter(positive_force_at_limit=(12.0, 10.0, 10.0)),
        )
        tracker.latch_baseline((7.0, 0.0, 0.0))
        tracker.update((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        old = estimator.x.copy()
        tau_now = np.array([2.0, 0.0, 0.0])
        expected_prior = model.A_d @ old + model.B_d @ tau_now
        tracker.update(expected_prior[:3], tau_now)
        np.testing.assert_allclose(estimator.x, expected_prior, atol=2e-7)

    def test_tracker_outputs_finesub_allocation(self):
        model, controller = self.build()
        tracker = MPCTracker(
            model,
            RelativePositionKalmanFilter(model),
            controller,
            ForceCommandAdapter(positive_force_at_limit=(12.0, 10.0, 10.0)),
            thruster_allocator=FineSUBThrusterAllocator(
                positive_force_at_limit=(12.0, 10.0, 10.0)
            ),
        )
        output = tracker.update((1.0, 0.1, -0.05), (0.0, 0.0, 0.0))
        self.assertIsNotNone(output.thruster_allocation)
        self.assertEqual(output.thruster_allocation.throttles.shape, (8,))


if __name__ == "__main__":
    unittest.main()
