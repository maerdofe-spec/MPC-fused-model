"""Camera-to-FineSUB body coordinate conversion."""

from __future__ import annotations

import numpy as np


ALIGNED_OPENCV_TO_BODY = np.array(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
)

# The PID hardware path must use the same camera/body calibration as the
# current real-vehicle MPC experiment.  The ROS vision topic is OpenCV
# ``[right, down, forward]``; the camera is not axis-aligned with the vehicle
# and its optical origin is offset from the body origin.  Keeping this
# calibration here (inside PID_controller) avoids importing or executing any
# MPC code while preventing the old identity-axis assumption from reaching
# the motors.
PID_CONTROL_ROTATION_BODY_FROM_CAMERA = np.array(
    [
        [0.105484, -0.006125, 0.994402],
        [0.994329, -0.012919, -0.105555],
        [0.013493, 0.999898, 0.004727],
    ],
    dtype=float,
)
PID_CONTROL_CAMERA_ORIGIN_IN_BODY = np.array(
    [0.260993, 0.007788, -0.123651],
    dtype=float,
)

# The calibrated image-centre standoff is the PID hover reference.  It is
# exactly the transformed point [camera-right, camera-down, camera-forward] =
# [0, 0, 0.60] m, rather than the old uncalibrated (0.80, 0, 0) point.
PID_CONTROL_REFERENCE_POSITION_BODY = np.array(
    [0.857634, -0.055545, -0.120815],
    dtype=float,
)


def camera_to_body_position(
    position_camera: object,
    rotation_body_from_camera: object = ALIGNED_OPENCV_TO_BODY,
    camera_origin_in_body: object = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Convert OpenCV [right, down, forward] to body FRD."""
    position = np.asarray(position_camera, dtype=float).reshape(-1)
    rotation = np.asarray(rotation_body_from_camera, dtype=float)
    origin = np.asarray(camera_origin_in_body, dtype=float).reshape(-1)
    if position.shape != (3,) or rotation.shape != (3, 3) or origin.shape != (3,):
        raise ValueError("position/origin must be 3-vectors and rotation must be 3x3")
    if not all(np.all(np.isfinite(item)) for item in (position, rotation, origin)):
        raise ValueError("camera transform contains NaN or infinity")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("rotation_body_from_camera must be orthonormal")
    return rotation @ position + origin


def camera_to_pid_body_position(position_camera: object) -> np.ndarray:
    """Convert a live vision sample using the PID hardware calibration.

    ``camera_to_body_position`` intentionally keeps its historical aligned
    default for callers that need the mathematical baseline transform.  All
    PID live/hardware entry points call this explicit calibrated helper.
    """
    return camera_to_body_position(
        position_camera,
        rotation_body_from_camera=PID_CONTROL_ROTATION_BODY_FROM_CAMERA,
        camera_origin_in_body=PID_CONTROL_CAMERA_ORIGIN_IN_BODY,
    )
