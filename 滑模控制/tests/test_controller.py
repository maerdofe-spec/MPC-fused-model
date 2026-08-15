from __future__ import annotations

import math
import unittest

from openauv_smc import (
    OpenAUVState,
    build_default_controller,
    build_openauv_model,
    saturation,
    wrap_angle,
)


class UtilityTests(unittest.TestCase):
    def test_saturation_is_continuous_boundary_layer(self) -> None:
        self.assertEqual(saturation(-2.0), -1.0)
        self.assertAlmostEqual(saturation(-0.25), -0.25)
        self.assertAlmostEqual(saturation(0.25), 0.25)
        self.assertEqual(saturation(2.0), 1.0)

    def test_angle_wrap_takes_short_path(self) -> None:
        error = wrap_angle(math.radians(-179.0) - math.radians(179.0))
        self.assertAlmostEqual(math.degrees(error), 2.0, places=10)


class ClosedLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = build_openauv_model()
        self.controller = build_default_controller(self.model)

    def test_model_damping_opposes_motion(self) -> None:
        positive = self.model.heave.damping_force(0.3)
        negative = self.model.heave.damping_force(-0.3)
        self.assertGreater(positive, 0.0)
        self.assertLess(negative, 0.0)
        self.assertAlmostEqual(positive, -negative)

    def test_controller_respects_input_limits(self) -> None:
        state = OpenAUVState(depth=-20.0, yaw=math.radians(-170.0))
        output = self.controller.compute(
            state,
            depth_reference=20.0,
            yaw_reference=math.radians(170.0),
            dt=0.01,
        )
        self.assertLessEqual(abs(output.heave_force), 80.0)
        self.assertLessEqual(abs(output.yaw_moment), 12.0)

    def test_nominal_step_tracking(self) -> None:
        state = OpenAUVState()
        dt = 0.01
        depth_reference = 1.2
        yaw_reference = math.radians(45.0)

        for _ in range(int(25.0 / dt)):
            output = self.controller.compute(
                state, depth_reference, yaw_reference, dt
            )
            state = self.model.step_rk4(
                state,
                output.heave_force,
                output.yaw_moment,
                dt,
            )

        self.assertLess(abs(depth_reference - state.depth), 0.02)
        self.assertLess(abs(wrap_angle(yaw_reference - state.yaw)), math.radians(0.5))
        self.assertLess(abs(state.heave_velocity), 0.01)
        self.assertLess(abs(state.yaw_rate), math.radians(0.5))


if __name__ == "__main__":
    unittest.main()

