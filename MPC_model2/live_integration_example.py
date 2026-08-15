"""Minimal integration pattern for the model-2-only tracker."""

from __future__ import annotations

import numpy as np

from MPC_dual_model.finesub_protocol import (
    FineSUBHardwareAdapter,
    build_default_hardware_adapter,
)

from MPC_dual_model.camera_transform import camera_to_body_position
from MPC_dual_model.device_adapter import (
    FineSUBThrusterAllocator,
    ForceCommandAdapter,
)
from .fossen_fixed_dl_model import FixedLinearDampingRelativeModel
from .mpc_controller import MPCConfig, RelativeMPCController
from .mpc_tracker import MPCTracker
from .relative_kalman import KalmanConfig, RelativePositionKalmanFilter


def build_hardware_adapter() -> FineSUBHardwareAdapter:
    return build_default_hardware_adapter()


def to_finesub_command(output, *, armed: bool, adapter=None):
    """Convert model-2 output while retaining the MCU's local yaw hold."""

    hardware = adapter or build_hardware_adapter()
    force = output.mpc.force if hasattr(output, "mpc") else output.force
    return hardware.convert(force, 0.0, armed=armed, yaw_direct=False)


def build_tracker() -> MPCTracker:
    model = FixedLinearDampingRelativeModel(
        # Identified from UnderwaterVision direct-force step responses.
        # Axis order is body forward/right/down = Unity X/Z/-Y.
        M_t=np.diag([26.07276, 26.79684, 26.07276]),
        D_L=np.diag([93.88006, 143.69195, 280.86849]),
        dt=0.05,
        # The shared model type carries a compatibility tau_base field, but
        # model 2 never reads it in prediction, estimation, or normal control.
    )
    config = MPCConfig(
        horizon=10,
        reference_position=(0.80, 0.0, 0.0),
        # Selected after Unity straight-ping-pong tuning.  These values are
        # intentionally aggressive because pure model 2 has no tau_base.
        position_weights=(60000.0, 10000.0, 120000.0),
        velocity_weights=(0.2, 0.5, 0.5),
        terminal_position_weight_scale=4.0,
        terminal_velocity_weight_scale=4.0,
        force_weights=(0.0005, 0.0015, 0.003),
        delta_force_weights=(0.02, 0.06, 0.12),
        force_min=(-19.799, -19.799, -28.0),
        force_max=(19.799, 19.799, 28.0),
        delta_force_min=(-4.0, -4.0, -5.6),
        delta_force_max=(4.0, 4.0, 5.6),
        horizontal_half_fov_deg=42.0,
        vertical_half_fov_deg=30.0,
        fov_margin_deg=5.0,
        slack_quadratic_weight=5.0e4,
        slack_linear_weight=100.0,
    )
    estimator = RelativePositionKalmanFilter(
        model,
        KalmanConfig(position_std=(0.015, 0.015, 0.025)),
    )
    controller = RelativeMPCController(model, config)
    adapter = ForceCommandAdapter(
        positive_force_at_limit=(19.799, 19.799, 28.0),
        signs=(1.0, 1.0, 1.0),
        command_limits=(99.0, 99.0, 45.0),
    )
    allocator = FineSUBThrusterAllocator(
        positive_force_at_limit=(19.799, 19.799, 28.0),
        enable_depth=True,
    )
    return MPCTracker(
        model,
        estimator,
        controller,
        adapter,
        thruster_allocator=allocator,
    )


def one_control_update(tracker, position_camera_xyz, last_achieved_force_body):
    """Run one update for camera [right, down, forward] position in metres."""
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
