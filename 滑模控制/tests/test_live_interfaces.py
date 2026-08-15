from __future__ import annotations

import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

MPC_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(MPC_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(MPC_WORKSPACE_ROOT))

from MPC_dual_model.finesub_protocol import FineSUBHardwareAdapter

from openauv_smc import (
    BBoxTargetEstimator,
    FineSUBCommandMapper,
    OpenAUVState,
    TelemetryStateEstimator,
    VisionConfig,
    build_default_controller,
    build_openauv_model,
)


class VisionInterfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.estimator = BBoxTargetEstimator(VisionConfig(width_filter_length=1))

    def test_centered_reference_box_matches_range_calibration(self) -> None:
        observation = self.estimator.update_bbox(
            (275.0, 195.0, 90.0, 90.0),
            640,
            480,
            timestamp=10.0,
        )
        self.assertAlmostEqual(observation.forward_m, 0.50)
        self.assertAlmostEqual(observation.right_m, 0.0)
        self.assertAlmostEqual(observation.down_m, 0.0)

    def test_bbox_generates_depth_yaw_and_range_commands(self) -> None:
        observation = self.estimator.update_bbox(
            (390.0, 260.0, 45.0, 60.0),
            640,
            480,
            timestamp=10.0,
        )
        reference = self.estimator.make_reference(
            observation,
            current_depth_m=1.2,
            current_yaw_rad=0.1,
        )
        self.assertGreater(reference.depth_reference_m, 1.2)
        self.assertGreater(reference.yaw_reference_rad, 0.1)
        self.assertGreater(reference.forward_force_n, 0.0)

    def test_measurement_freshness_has_hard_timeout(self) -> None:
        observation = self.estimator.update_bbox(
            (275.0, 195.0, 90.0, 90.0),
            640,
            480,
            timestamp=10.0,
        )
        self.assertTrue(self.estimator.is_fresh(observation, now=10.20))
        self.assertFalse(self.estimator.is_fresh(observation, now=10.30))


class FineSUBInterfaceTests(unittest.TestCase):
    def test_telemetry_builds_smc_state_and_filters_depth_rate(self) -> None:
        estimator = TelemetryStateEstimator(
            rate_filter_tau_s=0.10,
            max_abs_heave_rate_m_s=1.0,
        )
        first = SimpleNamespace(
            depth_m=1.0,
            received_monotonic=5.0,
            yaw_rad=0.2,
            yaw_rate_rad_s=0.3,
        )
        second = SimpleNamespace(
            depth_m=1.05,
            received_monotonic=5.1,
            yaw_rad=0.22,
            yaw_rate_rad_s=0.25,
        )
        initial = estimator.update(first)
        state = estimator.update(second)
        self.assertEqual(initial.heave_velocity, 0.0)
        self.assertGreater(state.heave_velocity, 0.0)
        self.assertLess(state.heave_velocity, 0.5)
        self.assertAlmostEqual(state.yaw, 0.22)
        self.assertAlmostEqual(state.yaw_rate, 0.25)

    def test_command_mapper_uses_forward_down_yaw_direct(self) -> None:
        class FakeHardwareAdapter:
            def __init__(self) -> None:
                self.call = None

            def convert(self, force, moment, *, armed, yaw_direct):
                self.call = (tuple(force), moment, armed, yaw_direct)
                return self.call

        model = build_openauv_model()
        controller = build_default_controller(model)
        control = controller.compute(
            state=model.step_rk4(
                state=OpenAUVState(),
                heave_force=0.0,
                yaw_moment=0.0,
                dt=0.01,
            ),
            depth_reference=1.0,
            yaw_reference=math.radians(10.0),
            dt=0.01,
        )
        hardware = FakeHardwareAdapter()
        mapper = FineSUBCommandMapper(hardware)
        mapper.convert(control, forward_force_n=3.0, armed=True)
        force, moment, armed, yaw_direct = hardware.call
        self.assertEqual(force[0], 3.0)
        self.assertEqual(force[1], 0.0)
        self.assertEqual(force[2], control.heave_force)
        self.assertEqual(moment, control.yaw_moment)
        self.assertTrue(armed)
        self.assertTrue(yaw_direct)

    def test_real_finesub_adapter_preserves_calibrated_channel_scales(self) -> None:
        hardware = FineSUBHardwareAdapter(
            positive_force_at_limit=(20.0, 15.0, 15.0),
            translation_channel_limits=(0.35, 0.35, 0.50),
            positive_yaw_moment_at_limit=4.0,
            yaw_channel_limit=0.20,
        )
        command = hardware.convert(
            (20.0, 0.0, 15.0),
            4.0,
            armed=True,
            yaw_direct=True,
        )
        self.assertAlmostEqual(command.forward, 0.35)
        self.assertAlmostEqual(command.right, 0.0)
        self.assertAlmostEqual(command.down, 0.50)
        self.assertAlmostEqual(command.yaw, 0.20)
        self.assertTrue(command.armed)
        self.assertTrue(command.yaw_direct)


if __name__ == "__main__":
    unittest.main()
