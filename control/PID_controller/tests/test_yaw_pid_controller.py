from __future__ import annotations

import numpy as np
import pytest

from live_integration_example import build_tracker, one_control_update
from yaw_pid_controller import YawPIDConfig, YawPIDController, wrap_angle


def test_angle_wrap_uses_shortest_turn() -> None:
    controller = YawPIDController(
        YawPIDConfig(kp=1.0, ki=0.0, kd=0.0)
    )
    result = controller.update(
        yaw_angle=np.deg2rad(179.0),
        reference_yaw=np.deg2rad(-179.0),
        previous_yaw_moment=0.0,
    )
    assert result.angle_error == pytest.approx(np.deg2rad(2.0))
    assert result.yaw_moment > 0.0


def test_yaw_moment_slew_limit() -> None:
    controller = YawPIDController(
        YawPIDConfig(kp=20.0, ki=0.0, kd=0.0)
    )
    result = controller.update(
        yaw_angle=0.0,
        reference_yaw=np.pi / 2.0,
        previous_yaw_moment=0.0,
    )
    assert result.yaw_moment == pytest.approx(0.25)
    assert result.saturated


def test_tracker_turns_right_toward_target_bearing() -> None:
    tracker = build_tracker()
    tracker.latch_baseline(np.zeros(3), 0.0, yaw_rad=0.0)
    # Camera [right, down, forward]: target is to the right of the nose.
    output = one_control_update(
        tracker,
        position_camera_xyz=(0.4, 0.0, 1.0),
        last_achieved_force_body=np.zeros(3),
        imu_yaw_rad=0.0,
        imu_yaw_rate_rad_s=0.0,
        last_achieved_yaw_moment=0.0,
    )
    assert output.yaw_pid is not None
    assert output.yaw_pid.reference_yaw == pytest.approx(np.arctan2(0.4, 1.0))
    assert output.yaw_pid.yaw_moment > 0.0
    assert output.yaw_channel > 0.0
    np.testing.assert_allclose(
        output.thruster_allocation.attitude_channels[:2], 0.0
    )
    assert output.thruster_allocation.attitude_channels[2] > 0.0


def test_explicit_yaw_reference_overrides_target_bearing() -> None:
    tracker = build_tracker()
    output = one_control_update(
        tracker,
        position_camera_xyz=(0.5, 0.0, 1.0),
        last_achieved_force_body=np.zeros(3),
        imu_yaw_rad=0.2,
        imu_yaw_rate_rad_s=0.0,
        reference_yaw_rad=-0.3,
    )
    assert output.yaw_pid is not None
    assert output.yaw_pid.reference_yaw == pytest.approx(-0.3)
    assert output.yaw_pid.yaw_moment < 0.0


def test_missing_imu_preserves_translation_only_compatibility() -> None:
    tracker = build_tracker()
    output = one_control_update(
        tracker,
        position_camera_xyz=(0.2, 0.0, 1.0),
        last_achieved_force_body=np.zeros(3),
    )
    assert output.yaw_pid is None
    assert output.yaw_channel == 0.0


def test_frozen_yaw_emits_no_yaw_output() -> None:
    tracker = build_tracker()
    tracker.freeze_yaw()
    output = one_control_update(
        tracker,
        position_camera_xyz=(0.5, 0.2, 1.0),
        last_achieved_force_body=np.zeros(3),
        imu_yaw_rad=0.0,
        imu_yaw_rate_rad_s=1.0,
    )
    assert output.yaw_pid is None
    assert output.yaw_channel == 0.0
    assert output.thruster_allocation is not None
    assert output.thruster_allocation.attitude_channels[2] == 0.0


def test_wrap_angle_range() -> None:
    assert wrap_angle(3.0 * np.pi) == pytest.approx(-np.pi)
    assert -np.pi <= wrap_angle(123.0) < np.pi


def test_nominal_yaw_closed_loop_converges() -> None:
    tracker = build_tracker()
    yaw = 0.0
    yaw_rate = 0.0
    yaw_moment = 0.0
    force = np.zeros(3)
    target_world_yaw = 0.6
    tracker.latch_baseline(force, yaw_moment, yaw)
    for _ in range(240):
        bearing = wrap_angle(target_world_yaw - yaw)
        position_body = np.array([np.cos(bearing), np.sin(bearing), 0.0])
        output = tracker.update(
            position_body,
            force,
            yaw_rad=yaw,
            yaw_rate_rad_s=yaw_rate,
            achieved_yaw_moment_previous=yaw_moment,
        )
        force = output.pid.force.copy()
        yaw_moment = output.yaw_pid.yaw_moment
        yaw_rate += (yaw_moment - 0.8 * yaw_rate) / 0.8 * 0.05
        yaw = wrap_angle(yaw + yaw_rate * 0.05)
    assert abs(wrap_angle(target_world_yaw - yaw)) < np.deg2rad(1.0)
