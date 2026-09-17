import unittest

import numpy as np

from MPC_dual_model_yaw.yaw_controller import (
    YawControlConfig,
    YawMode,
    YawStateController,
)
from MPC_dual_model_yaw.yaw_relative_model import LinearYawDynamics


class YawStateControllerTest(unittest.TestCase):
    def build(self):
        model = LinearYawDynamics(2.0, 0.8, 0.1, quadratic_damping=0.2)
        config = YawControlConfig(
            alpha_on=0.20,
            alpha_off=0.08,
            alpha_emergency=0.50,
            trigger_frames=2,
            settle_frames=2,
            omega_command_max=0.6,
            omega_command_acceleration_max=1.0,
            yaw_moment_min=-1.0,
            yaw_moment_max=1.0,
            delta_yaw_moment_min=-0.25,
            delta_yaw_moment_max=0.25,
        )
        return model, YawStateController(model, config)

    def test_hold_turn_hysteresis_and_positive_direction(self) -> None:
        _, controller = self.build()
        first = controller.update(0.0, 0.0, 0.30, 0.0, 4)
        self.assertEqual(first.mode, YawMode.HOLD)
        second = controller.update(0.0, 0.0, 0.30, first.yaw_moment, 4)
        self.assertEqual(second.mode, YawMode.TURN)
        self.assertAlmostEqual(second.goal_angle, 0.30)
        self.assertGreater(second.omega_command, 0.0)
        self.assertGreater(second.yaw_moment, 0.0)

    def test_moment_and_moment_rate_are_limited(self) -> None:
        _, controller = self.build()
        previous = 0.0
        for _ in range(10):
            output = controller.update(0.0, 0.0, 0.45, previous, 4)
            self.assertLessEqual(output.yaw_moment, 1.0)
            self.assertGreaterEqual(output.yaw_moment, -1.0)
            self.assertLessEqual(output.yaw_moment - previous, 0.25 + 1.0e-12)
            self.assertGreaterEqual(output.yaw_moment - previous, -0.25 - 1.0e-12)
            previous = output.yaw_moment

    def test_one_prediction_freezes_goal_and_has_consistent_dimensions(self) -> None:
        _, controller = self.build()
        controller.update(0.0, 0.0, 0.30, 0.0, 5)
        output = controller.update(0.0, 0.0, 0.30, 0.0, 5)
        prediction = output.prediction
        self.assertEqual(prediction.angles.shape, (6,))
        self.assertEqual(prediction.rates.shape, (6,))
        self.assertEqual(prediction.delta_angles.shape, (5,))
        self.assertEqual(prediction.moments.shape, (5,))
        self.assertAlmostEqual(prediction.goal_angle, output.goal_angle)
        np.testing.assert_allclose(
            prediction.delta_angles,
            0.05 * (prediction.rates[:-1] + prediction.rates[1:]),
            atol=1.0e-12,
        )

if __name__ == "__main__":
    unittest.main()
