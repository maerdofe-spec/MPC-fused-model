"""Minimal integration pattern for rov_track_control3.py.

This file does not open sockets or start propellers. Copy the marked calls into
the existing video/control loop after calibration.
"""

from __future__ import annotations

import numpy as np

try:
    from .camera_transform import camera_to_body_position
    from .dense_qp import QPSolverSettings
    from .device_adapter import (
        FINESUB_V4_PRO1_FORCE_NEGATIVE_N,
        FINESUB_V4_PRO1_FORCE_POSITIVE_N,
        FineSUBThrusterAllocator,
        ForceCommandAdapter,
        finesub_translation_force_bounds,
        finesub_translation_thruster_force_matrix,
    )
    from .finesub_protocol import (
        FineSUBHardwareAdapter,
        build_default_hardware_adapter,
    )
    from .fossen_fixed_dl_model import FixedLinearDampingRelativeModel
    from .mpc_controller import MPCConfig, RelativeMPCController
    from .mpc_tracker import (
        BaselineAdaptationConfig,
        MPCTracker,
        build_default_staircase_fusion,
    )
    from .relative_kalman import KalmanConfig, RelativePositionKalmanFilter
except ImportError:
    from camera_transform import camera_to_body_position
    from dense_qp import QPSolverSettings
    from device_adapter import (
        FINESUB_V4_PRO1_FORCE_NEGATIVE_N,
        FINESUB_V4_PRO1_FORCE_POSITIVE_N,
        FineSUBThrusterAllocator,
        ForceCommandAdapter,
        finesub_translation_force_bounds,
        finesub_translation_thruster_force_matrix,
    )
    from finesub_protocol import FineSUBHardwareAdapter, build_default_hardware_adapter
    from fossen_fixed_dl_model import FixedLinearDampingRelativeModel
    from mpc_controller import MPCConfig, RelativeMPCController
    from mpc_tracker import (
        BaselineAdaptationConfig,
        MPCTracker,
        build_default_staircase_fusion,
    )
    from relative_kalman import KalmanConfig, RelativePositionKalmanFilter


def build_hardware_adapter() -> FineSUBHardwareAdapter:
    return build_default_hardware_adapter()


def to_finesub_command(output, *, armed: bool, adapter=None):
    """Convert fused translation output and retain the MCU's local yaw hold."""

    hardware = adapter or build_hardware_adapter()
    force = output.mpc.force if hasattr(output, "mpc") else output.force
    return hardware.convert(force, 0.0, armed=armed, yaw_direct=False)


def build_tracker() -> MPCTracker:
    force_min, force_max = finesub_translation_force_bounds()
    model = FixedLinearDampingRelativeModel(
        # Identified in UnderwaterVision with direct +/-6 N and +/-12 N steps.
        # Axis order is body forward/right/down = Unity X/Z/-Y.
        M_t=np.diag([26.07276, 26.79684, 26.07276]),
        D_L=np.diag([93.88006, 143.69195, 280.86849]),
        dt=0.05,
        # Replace this with the identified h(0) for a real vehicle.  It stays
        # independent from the model-1 gated-EMA operating force below.
        restoring_force=np.zeros(3),
    )
    config = MPCConfig(
        horizon=10,
        reference_position=(0.80, 0.0, 0.0),
        # Tuned for horizontal tracking while retaining the exact V4 Pro1
        # canonical per-thruster force envelope.
        position_weights=(10000.0, 14000.0, 25000.0),
        velocity_weights=(2.0, 20.0, 12.0),
        force_weights=(0.003, 0.002, 0.04),
        delta_force_weights=(0.01, 0.01, 0.5),
        force_min=force_min,
        force_max=force_max,
        delta_force_min=(-4.0, -4.0, -5.6),
        delta_force_max=(4.0, 4.0, 5.6),
        thruster_command_matrix=finesub_translation_thruster_force_matrix(),
        thruster_command_min=-np.asarray(FINESUB_V4_PRO1_FORCE_NEGATIVE_N),
        thruster_command_max=np.asarray(FINESUB_V4_PRO1_FORCE_POSITIVE_N),
        horizontal_half_fov_deg=42.0,
        vertical_half_fov_deg=30.0,
        fov_margin_deg=5.0,
        slack_quadratic_weight=5.0e4,
        slack_linear_weight=100.0,
        solver_settings=QPSolverSettings(
            rho=10.0,
            sigma=1e-8,
            max_iterations=1500,
            absolute_tolerance=2e-5,
            relative_tolerance=3e-4,
            # Leave 15 ms of the 50 ms control cycle for ROS and safety logic.
            time_limit_seconds=0.035,
        ),
    )
    estimator = RelativePositionKalmanFilter(
        model,
        KalmanConfig(position_std=(0.015, 0.015, 0.025)),
    )
    controller = RelativeMPCController(model, config)
    adapter = ForceCommandAdapter(
        positive_force_at_limit=force_max,
        # TODO: flip individual signs after a dry test if required.
        signs=(1.0, 1.0, 1.0),
        command_limits=(99.0, 99.0, 45.0),
    )
    fusion = build_default_staircase_fusion()
    allocator = FineSUBThrusterAllocator(
        positive_force_at_limit=(20.0, 15.0, 15.0),
        # True enables the intended depth channel. Set False only to reproduce
        # the checked source line that passed 0.0 into the depth mixer.
        enable_depth=True,
    )
    return MPCTracker(
        model,
        estimator,
        controller,
        adapter,
        fusion=fusion,
        thruster_allocator=allocator,
        baseline_adaptation=BaselineAdaptationConfig(
            enabled=True,
            adaptation_rate=0.02,
            transient_adaptation_rate=0.08,
            steady_position_error_tolerance=0.03,
            steady_velocity_tolerance=0.02,
            position_error_tolerance=0.20,
            velocity_tolerance=0.20,
        ),
    )


def one_control_update(
    tracker: MPCTracker,
    position_camera_xyz,
    last_achieved_force_body,
):
    """Call this when a new stereo 3-D position is available."""
    position_body = camera_to_body_position(position_camera_xyz)
    return tracker.update(
        position_body=position_body,
        tau_achieved_previous=last_achieved_force_body,
    )


# Integration outline inside rov_track_control3.py:
#
# tracker = build_tracker()
# last_force_body = np.zeros(3)
#
# When AUTO changes from OFF to ON:
# tracker.latch_baseline(last_force_body)
#
# When stereo returns position_camera = [x_right, y_down, z_forward] in metres:
# output = one_control_update(tracker, position_camera, last_force_body)
# cmd = output.device_command
# motor_throttles = output.thruster_allocation.throttles
# send_xy_2(cmd.planar_forward, cmd.planar_right)
# send_pkt(PROPELLER_FRAME_HEAD, CMD_SET_DEPTH_FORCE, i16le(cmd.depth_force))
# last_force_body = output.mpc.force.copy()  # approximation without force feedback
#
# IMPORTANT: use either the existing high-level cmd path above (the MCU mixes),
# or motor_throttles for a direct eight-ESC path. Do not apply both mixers.
#
# When the target is lost:
# safe = tracker.target_lost(last_force_body)
# send_xy_2(safe.device_command.planar_forward, safe.device_command.planar_right)
# send_pkt(PROPELLER_FRAME_HEAD, CMD_SET_DEPTH_FORCE,
#          i16le(safe.device_command.depth_force))
# last_force_body = safe.force.copy()
