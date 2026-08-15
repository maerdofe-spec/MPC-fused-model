"""Read-only construction helpers for the real rotation-aware MPC.

This module opens no camera, transport, serial device, or propeller output.
It intentionally delegates to the same strict builder as the guarded AUTO
runtime so examples cannot silently retain a different translation model,
actuator order, yaw controller, or parameter set.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from MPC_dual_model.camera_transform import (
    ALIGNED_OPENCV_TO_BODY,
    camera_to_body_position,
)
from MPC_dual_model.experimental_auto import build_experimental_runtime_config
from MPC_dual_model.finesub_protocol import (
    FineSUBHardwareAdapter,
    build_runtime_hardware_adapter,
)

from .auto_tracker import build_auto_tracker
from .yaw_tracker import RotationAwareMPCTracker


DEFAULT_RUNTIME_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "MPC_dual_model"
    / "finesub_v4pro1_mpc.json"
)


def load_runtime_config(path: str | Path = DEFAULT_RUNTIME_CONFIG) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def build_hardware_adapter(
    runtime_config_path: str | Path = DEFAULT_RUNTIME_CONFIG,
) -> FineSUBHardwareAdapter:
    """Use the same final-throttle force feedback and RPM diagnostics as AUTO."""

    return build_runtime_hardware_adapter(load_runtime_config(runtime_config_path))


def to_finesub_command(output, *, armed: bool, adapter=None):
    """Convert the fused translation force and direct yaw moment."""

    hardware = adapter or build_hardware_adapter()
    if hasattr(output, "mpc"):
        force = output.mpc.force
        moment = output.mpc.yaw_moment
    else:
        force = output.force
        moment = output.yaw_moment
    return hardware.convert(force, moment, armed=armed, yaw_direct=True)


def build_tracker(
    runtime_config_path: str | Path = DEFAULT_RUNTIME_CONFIG,
) -> RotationAwareMPCTracker:
    """Build the rotation tracker without changing the active AUTO selection."""

    source = load_runtime_config(runtime_config_path)
    runtime = build_experimental_runtime_config(source)
    # The guarded experiment currently selects translation-only MPC.  This
    # read-only helper remains an explicit rotation-model construction path for
    # offline comparisons and its dedicated regression tests.
    runtime["auto_runtime"]["required_model"] = "dual-yaw"
    runtime["auto_runtime"]["active_yaw_parameters"][
        "enabled_for_control"
    ] = True
    return build_auto_tracker(runtime)


def one_control_update(
    tracker: RotationAwareMPCTracker,
    position_camera_xyz,
    imu_yaw_rad: float,
    imu_yaw_rate_rad_s: float,
    last_achieved_force_body,
    last_achieved_yaw_moment: float,
    roll_pitch_control=(0.0, 0.0),
    rotation_body_from_camera=None,
    camera_origin_in_body=None,
):
    """Call once for each timestamp-aligned 3-D camera measurement."""

    rotation = (
        getattr(tracker, "rotation_body_from_camera", ALIGNED_OPENCV_TO_BODY)
        if rotation_body_from_camera is None
        else np.asarray(rotation_body_from_camera, dtype=float)
    )
    origin = (
        getattr(tracker, "camera_origin_in_body", np.zeros(3))
        if camera_origin_in_body is None
        else np.asarray(camera_origin_in_body, dtype=float)
    )
    position_body = camera_to_body_position(
        position_camera_xyz,
        rotation_body_from_camera=rotation,
        camera_origin_in_body=origin,
    )
    rotation_visibility_from_body = ALIGNED_OPENCV_TO_BODY @ rotation.T
    return tracker.update(
        position_body=position_body,
        yaw_rad=imu_yaw_rad,
        yaw_rate_rad_s=imu_yaw_rate_rad_s,
        force_achieved_previous=last_achieved_force_body,
        yaw_moment_achieved_previous=last_achieved_yaw_moment,
        roll_pitch_control=roll_pitch_control,
        rotation_visibility_from_body=rotation_visibility_from_body,
        camera_origin_in_body=origin,
    )
