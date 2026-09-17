import unittest

import numpy as np

from MPC_dual_model.fossen_fixed_dl_model import (
    FixedLinearDampingRelativeModel,
)
from MPC_dual_model_yaw.yaw_kalman import (
    RotationAwareKalmanFilter,
)
from MPC_dual_model_yaw.yaw_relative_model import (
    LinearYawDynamics,
    RotationAwareRelativeModel,
    rotation_body_from_previous,
    rotation_compensated_velocity,
    wrap_angle,
)


class RotationModelTest(unittest.TestCase):
    def build(self, restoring_force=(0.0, 0.0, 0.0)):
        translation = FixedLinearDampingRelativeModel(
            np.diag([20.0, 25.0, 30.0]),
            np.diag([8.0, 10.0, 12.0]),
            0.05,
            restoring_force=restoring_force,
        )
        yaw = LinearYawDynamics(2.5, 1.2, translation.dt)
        return translation, RotationAwareRelativeModel(translation, yaw)

    def test_positive_vehicle_yaw_rotates_coordinates_negative(self) -> None:
        rotated = rotation_body_from_previous(np.pi / 2.0) @ [1.0, 0.0, 0.0]
        np.testing.assert_allclose(rotated, [0.0, -1.0, 0.0], atol=1.0e-12)

    def test_pure_rotation_is_not_translational_velocity(self) -> None:
        dt = 0.05
        delta_yaw = 0.08
        previous = np.array([1.2, 0.3, -0.1])
        current = rotation_body_from_previous(delta_yaw) @ previous
        velocity = rotation_compensated_velocity(
            current, previous, delta_yaw, dt
        )
        np.testing.assert_allclose(velocity, np.zeros(3), atol=1.0e-12)

    def test_yaw_increment_is_continuous_across_pi_boundary(self) -> None:
        previous = np.deg2rad(179.0)
        current = np.deg2rad(-179.0)
        self.assertAlmostEqual(wrap_angle(current - previous), np.deg2rad(2.0))

    def test_zero_yaw_uses_pdf_end_velocity_position_update(self) -> None:
        translation, model = self.build()
        state = np.array([1.0, 0.2, -0.1, 0.1, -0.03, 0.02])
        force = np.array([2.0, -1.0, 0.5])
        velocity_next = translation.F @ state[3:] + translation.G @ force
        expected = np.concatenate(
            (state[:3] + translation.dt * velocity_next, velocity_next)
        )
        actual = model.predict_model2(state, force, delta_yaw_rad=0.0)
        np.testing.assert_allclose(actual, expected, atol=1.0e-12)

    def test_model1_subtracts_tau_base_instead_of_previous_force(self) -> None:
        _, model = self.build()
        state = np.array([1.0, 0.2, -0.1, 0.1, -0.03, 0.02])
        force = np.array([3.0, -1.0, 0.5])
        tau_base = np.array([1.2, 0.4, -0.2])
        expected = model.predict_translation(
            state,
            force - tau_base,
            delta_yaw_rad=0.03,
        )
        actual = model.predict_model1(
            state,
            force,
            tau_base,
            delta_yaw_rad=0.03,
        )
        np.testing.assert_allclose(actual, expected, atol=1.0e-12)

    def test_model2_subtracts_fixed_fossen_restoring_force(self) -> None:
        restoring = np.array([0.0, 0.0, 0.80729])
        _, model = self.build(restoring)
        state = np.array([1.0, 0.2, -0.1, 0.1, -0.03, 0.02])
        force = np.array([0.4, -0.2, 1.1])
        expected = model.predict_translation(
            state,
            force - restoring,
            delta_yaw_rad=0.03,
        )
        actual = model.predict_model2(state, force, delta_yaw_rad=0.03)
        np.testing.assert_allclose(actual, expected, atol=1.0e-12)

    def test_yaw_step_uses_trapezoidal_integration(self) -> None:
        _, model = self.build()
        psi_next, omega_next, delta = model.yaw.predict_yaw_step(0.2, 0.1, 0.4)
        self.assertAlmostEqual(delta, 0.5 * model.dt * (0.1 + omega_next))
        self.assertAlmostEqual(psi_next, wrap_angle(0.2 + delta))

    def test_filter_keeps_zero_translation_during_pure_yaw(self) -> None:
        _, model = self.build()
        estimator = RotationAwareKalmanFilter(model)
        position = np.array([1.0, 0.25, 0.0])
        estimator.initialize(position)
        delta_yaw = 0.03
        for _ in range(20):
            position = rotation_body_from_previous(delta_yaw) @ position
            predicted = model.predict_model2(
                estimator.x, np.zeros(3), delta_yaw
            )
            estimator.predict_mean(predicted, delta_yaw)
            estimator.update(position)
        self.assertLess(np.linalg.norm(estimator.x[3:]), 1.0e-10)


if __name__ == "__main__":
    unittest.main()
