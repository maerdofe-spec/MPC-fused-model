import unittest

import numpy as np

from MPC_dual_model.dense_qp import QPSolverSettings
from MPC_dual_model.device_adapter import (
    finesub_planar_wrench_thruster_force_matrix,
    finesub_six_dof_wrench_matrix_unity,
)
from MPC_dual_model.fossen_fixed_dl_model import FixedLinearDampingRelativeModel
from MPC_dual_model_yaw.yaw_controller import (
    YawControlConfig,
    YawStateController,
)
from MPC_dual_model_yaw.yaw_mpc_controller import (
    RotationAwareMPCController,
    YawMPCConfig,
)
from MPC_dual_model_yaw.yaw_relative_model import (
    LinearYawDynamics,
    RotationAwareRelativeModel,
)


class YawMPCTest(unittest.TestCase):
    def build(self, restoring_force=(0.0, 0.0, 0.0)):
        translation = FixedLinearDampingRelativeModel(
            np.diag([20.0, 25.0, 30.0]),
            np.diag([8.0, 10.0, 12.0]),
            0.1,
            restoring_force=restoring_force,
        )
        yaw = LinearYawDynamics(2.0, 0.8, translation.dt)
        model = RotationAwareRelativeModel(translation, yaw)
        controller = RotationAwareMPCController(
            model,
            YawMPCConfig(
                horizon=5,
                reference_position=(1.0, 0.0, 0.0),
                position_weights=(5.0, 5.0, 5.0),
                force_min=(-30.0, -30.0, -30.0),
                force_max=(30.0, 30.0, 30.0),
                delta_force_min=(-10.0, -10.0, -10.0),
                delta_force_max=(10.0, 10.0, 10.0),
                solver_settings=QPSolverSettings(
                    backend="numpy_admm",
                    rho=10.0,
                    max_iterations=5000,
                    time_limit_seconds=2.0,
                ),
            ),
        )
        yaw_controller = YawStateController(
            yaw,
            YawControlConfig(
                alpha_on=0.10,
                alpha_off=0.05,
                alpha_emergency=0.60,
                trigger_frames=1,
                settle_frames=2,
                omega_command_acceleration_max=10.0,
                yaw_moment_min=-2.0,
                yaw_moment_max=2.0,
                delta_yaw_moment_min=-0.5,
                delta_yaw_moment_max=0.5,
            ),
        )
        return model, controller, yaw_controller

    def solve(self, position):
        _, controller, yaw_controller = self.build()
        alpha = float(np.arctan2(position[1], position[0]))
        yaw_control = yaw_controller.update(0.0, 0.0, alpha, 0.0, 5)
        result = controller.solve(
            state=np.concatenate((position, np.zeros(3))),
            force_previous=np.zeros(3),
            yaw_prediction=yaw_control.prediction,
            model1_weight=(0.8, 0.8, 0.8),
        )
        return controller, yaw_control, result

    def test_yaw_is_generated_outside_translation_qp(self) -> None:
        _, yaw_control, result = self.solve(np.array([1.0, 0.35, 0.0]))
        self.assertGreater(yaw_control.yaw_moment, 0.0)
        self.assertAlmostEqual(result.yaw_moment, yaw_control.yaw_moment)
        self.assertEqual(result.force_sequence.shape, (5, 3))
        self.assertEqual(result.delta_force_sequence.shape, (5, 3))
        self.assertEqual(result.predicted_states.shape, (5, 6))
        self.assertEqual(result.frozen_yaw_rates.shape, (6,))
        self.assertEqual(result.frozen_delta_yaw.shape, (5,))

    def test_centred_target_has_no_yaw_bias(self) -> None:
        _, yaw_control, result = self.solve(np.array([1.0, 0.0, 0.0]))
        self.assertAlmostEqual(yaw_control.yaw_moment, 0.0, places=12)
        self.assertAlmostEqual(result.yaw_moment, 0.0, places=12)

    def test_frozen_yaw_uses_trapezoidal_angle_increment(self) -> None:
        _, yaw_control, result = self.solve(np.array([1.0, 0.35, 0.0]))
        expected = 0.5 * 0.1 * (
            result.frozen_yaw_rates[:-1] + result.frozen_yaw_rates[1:]
        )
        np.testing.assert_allclose(result.frozen_delta_yaw, expected, atol=1.0e-12)
        np.testing.assert_allclose(
            result.frozen_yaw_moments,
            yaw_control.prediction.moments,
            atol=0.0,
        )

    def test_fused_prediction_uses_tau_base_and_fossen_tau_h(self) -> None:
        restoring = np.array([0.2, -0.1, 0.7])
        model, controller, yaw_controller = self.build(restoring)
        reference = controller.config.reference_position
        previous = np.array([3.0, -2.0, 1.0])
        tau_base = np.array([1.0, 0.5, -0.4])
        weight = np.array([0.8, 0.3, 0.6])
        yaw_prediction = yaw_controller.update(
            0.0, 0.0, 0.0, 0.0, controller.config.horizon
        ).prediction
        controller._build_prediction_matrices(
            reference,
            weight,
            yaw_prediction,
            tau_base,
        )
        augmented = np.concatenate((np.zeros(6), previous))
        free = (
            controller.Sx @ augmented + controller.Sc
        ).reshape(controller.config.horizon, controller.AUGMENTED_DIM)

        expected_velocity = model.translation.G @ (
            previous - weight * tau_base - (1.0 - weight) * restoring
        )
        np.testing.assert_allclose(free[0, 3:6], expected_velocity, atol=1.0e-12)
        np.testing.assert_allclose(
            free[0, :3],
            model.dt * expected_velocity,
            atol=1.0e-12,
        )

    def test_solve_latches_previous_force_as_fixed_tau_base(self) -> None:
        _, controller, yaw_controller = self.build(restoring_force=(0.0, 0.0, 0.8))
        previous = np.array([0.3, -0.2, 0.9])
        prediction = yaw_controller.update(
            0.0, 0.0, 0.0, 0.0, controller.config.horizon
        ).prediction
        captured = []
        original = controller._build_prediction_matrices

        def capture(reference, weight, yaw, tau_base):
            captured.append(np.asarray(tau_base).copy())
            return original(reference, weight, yaw, tau_base)

        controller._build_prediction_matrices = capture
        controller.solve(
            state=np.concatenate((controller.config.reference_position, np.zeros(3))),
            force_previous=previous,
            yaw_prediction=prediction,
            model1_weight=(0.8, 0.8, 0.8),
        )
        self.assertEqual(len(captured), 1)
        np.testing.assert_allclose(captured[0], previous)

    def test_force_cost_is_always_absolute_total_force(self) -> None:
        _, controller, yaw_controller = self.build()
        self.assertFalse(hasattr(controller.config, "force_cost_mode"))
        prediction = yaw_controller.update(
            0.0, 0.0, 0.0, 0.0, controller.config.horizon
        ).prediction
        controller._build_prediction_matrices(
            controller.config.reference_position,
            np.array([0.8, 0.8, 0.8]),
            prediction,
            np.zeros(3),
        )
        free = np.zeros(controller.config.horizon * controller.AUGMENTED_DIM)
        for step in range(controller.config.horizon):
            free[step * controller.AUGMENTED_DIM + 6] = 2.0
        _, gradient = controller._cost(free)
        self.assertGreater(gradient[0], 0.0)

    def test_planar_allocator_reproduces_wrench_without_roll_or_pitch(self) -> None:
        allocation = finesub_planar_wrench_thruster_force_matrix()
        geometry = finesub_six_dof_wrench_matrix_unity()
        expected = np.zeros((6, 4))
        expected[0, 0] = 1.0
        expected[2, 1] = 1.0
        expected[1, 2] = -1.0
        expected[4, 3] = -1.0

        self.assertEqual(allocation.shape, (8, 4))
        np.testing.assert_allclose(geometry @ allocation, expected, atol=1.0e-12)

if __name__ == "__main__":
    unittest.main()
