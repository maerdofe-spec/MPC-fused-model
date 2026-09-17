import unittest

import numpy as np

from MPC_model1.camera_transform import (
    camera_to_body_position,
)
from MPC_model1.dense_qp import QPSolution
from MPC_model1.device_adapter import (
    FineSUBThrusterAllocator,
    ForceCommandAdapter,
)
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


class IntegrationTest(unittest.TestCase):
    def build(self):
        model = FixedLinearDampingRelativeModel(
            M_t=np.diag([20.0, 25.0, 30.0]),
            D_L=np.diag([8.0, 10.0, 12.0]),
            dt=0.1,
        )
        controller = RelativeMPCController(
            model,
            MPCConfig(
                horizon=5,
                force_min=(-12.0, -10.0, -10.0),
                force_max=(12.0, 10.0, 10.0),
                delta_force_min=(-3.0, -2.0, -2.0),
                delta_force_max=(3.0, 2.0, 2.0),
            ),
        )
        return model, controller

    def test_camera_coordinate_order(self) -> None:
        np.testing.assert_allclose(
            camera_to_body_position((0.2, -0.1, 1.3)),
            (1.3, 0.2, -0.1),
        )

    def test_force_rate_fov_and_output_shapes(self) -> None:
        _, controller = self.build()
        result = controller.solve(
            state=np.array([1.2, 0.15, -0.10, 0.0, 0.0, 0.0]),
            tau_previous=np.zeros(3),
        )
        self.assertFalse(result.used_fallback, result.status)
        upper = np.array([3.0, 2.0, 2.0])
        lower = -upper
        self.assertTrue(np.all(result.force <= upper + 1e-8))
        self.assertTrue(np.all(result.force >= lower - 1e-8))
        self.assertEqual(result.predicted_states.shape, (5, 6))
        self.assertEqual(result.force_sequence.shape, (5, 3))
        self.assertEqual(result.slacks.shape, (5, 6))

    def test_tracker_and_finesub_output_format(self) -> None:
        model, controller = self.build()
        tracker = MPCTracker(
            model=model,
            estimator=RelativePositionKalmanFilter(model),
            controller=controller,
            adapter=ForceCommandAdapter(
                positive_force_at_limit=(12.0, 10.0, 10.0)
            ),
            thruster_allocator=FineSUBThrusterAllocator(
                positive_force_at_limit=(12.0, 10.0, 10.0),
                deadband=0.0,
            ),
        )
        output = tracker.update((1.0, 0.1, -0.05), (0.0, 0.0, 0.0))
        self.assertEqual(output.estimated_state.shape, (6,))
        self.assertEqual(output.mpc.force.shape, (3,))
        self.assertEqual(output.thruster_allocation.throttles.shape, (8,))
        self.assertFalse(hasattr(output.mpc, "model1_weight"))
        self.assertFalse(hasattr(output.mpc, "model2_weight"))

    def test_solver_failure_returns_to_baseline(self) -> None:
        model, controller = self.build()
        model.set_tau_base((5.0, 1.0, -1.0))

        class AlwaysFailSolver:
            def solve(self, P, q, A, lower, upper, warm_start=None):
                return QPSolution(
                    x=np.zeros_like(q),
                    status="time_limit",
                    iterations=0,
                    objective=np.inf,
                    primal_residual=np.inf,
                    dual_residual=np.inf,
                )

        controller.solver = AlwaysFailSolver()
        result = controller.solve(
            state=np.array([1.2, 0.0, 0.0, 0.0, 0.0, 0.0]),
            tau_previous=np.zeros(3),
        )
        self.assertTrue(result.used_fallback)
        self.assertEqual(result.status, "fallback:baseline:time_limit")
        np.testing.assert_allclose(result.force, (3.0, 1.0, -1.0))

    def test_finesub_forward_mixer_motor_order(self) -> None:
        allocation = FineSUBThrusterAllocator(deadband=0.0).allocate(
            (20.0, 0.0, 0.0)
        )
        expected = np.array(
            [-0.35, 0.0, 0.0, -0.35, 0.35, 0.0, 0.0, 0.35]
        )
        np.testing.assert_allclose(allocation.throttles, expected)


if __name__ == "__main__":
    unittest.main()
