"""Minimal integration pattern for the model-1-only tracker."""

from __future__ import annotations

import numpy as np

from MPC_dual_model.finesub_protocol import (
    FineSUBHardwareAdapter,
    build_default_hardware_adapter,
)

try:
    from .camera_transform import camera_to_body_position
    from .device_adapter import FineSUBThrusterAllocator, ForceCommandAdapter
    from .fossen_fixed_dl_model import FixedLinearDampingRelativeModel
    from .mpc_controller import MPCConfig, RelativeMPCController
    from .mpc_tracker import MPCTracker
    from .relative_kalman import KalmanConfig, RelativePositionKalmanFilter
except ImportError:
    from camera_transform import camera_to_body_position
    from device_adapter import FineSUBThrusterAllocator, ForceCommandAdapter
    from fossen_fixed_dl_model import FixedLinearDampingRelativeModel
    from mpc_controller import MPCConfig, RelativeMPCController
    from mpc_tracker import MPCTracker
    from relative_kalman import KalmanConfig, RelativePositionKalmanFilter


def build_hardware_adapter() -> FineSUBHardwareAdapter:
    return build_default_hardware_adapter()


def to_finesub_command(output, *, armed: bool, adapter=None):
    """Convert model-1 output while retaining the MCU's local yaw hold."""

    hardware = adapter or build_hardware_adapter()
    force = output.mpc.force if hasattr(output, "mpc") else output.force
    return hardware.convert(force, 0.0, armed=armed, yaw_direct=False)


def build_tracker() -> MPCTracker:
    model = FixedLinearDampingRelativeModel(
        # TODO: replace placeholders with pool-identification results.
        M_t=np.diag([20.0, 25.0, 30.0]),
        D_L=np.diag([8.0, 10.0, 12.0]),
        dt=0.05,
        tau_base=np.zeros(3),
    )
    config = MPCConfig(
        horizon=10,
        reference_position=(0.60, 0.0, 0.0),
        force_min=(-20.0, -15.0, -15.0),
        force_max=(20.0, 15.0, 15.0),
        delta_force_min=(-3.0, -2.0, -2.0),
        delta_force_max=(3.0, 2.0, 2.0),
        horizontal_half_fov_deg=42.0,
        vertical_half_fov_deg=30.0,
        fov_margin_deg=5.0,
    )
    estimator = RelativePositionKalmanFilter(
        model,
        KalmanConfig(position_std=(0.015, 0.015, 0.025)),
    )
    controller = RelativeMPCController(model, config)
    adapter = ForceCommandAdapter(
        positive_force_at_limit=(20.0, 15.0, 15.0),
        signs=(1.0, 1.0, 1.0),
        command_limits=(99.0, 99.0, 45.0),
    )
    allocator = FineSUBThrusterAllocator(
        positive_force_at_limit=(20.0, 15.0, 15.0),
        enable_depth=True,
    )
    return MPCTracker(
        model,
        estimator,
        controller,
        adapter,
        thruster_allocator=allocator,
    )


def one_control_update(
    tracker: MPCTracker,
    position_camera_xyz,
    last_achieved_force_body,
):
    """Run one update from OpenCV camera coordinates [right, down, forward]."""
    position_body = camera_to_body_position(position_camera_xyz)
    return tracker.update(
        position_body=position_body,
        tau_achieved_previous=last_achieved_force_body,
    )


# Integration outline for rov_track_control3.py:
#
# tracker = build_tracker()
# last_force_body = np.zeros(3)
# tracker.latch_baseline(last_force_body)  # AUTO off -> on
# output = one_control_update(tracker, position_camera_xyz, last_force_body)
# cmd = output.device_command
# last_force_body = output.mpc.force.copy()
#
# Send either cmd through the MCU mixer or output.thruster_allocation.throttles
# directly to the ESC path. Never apply both mixers.
