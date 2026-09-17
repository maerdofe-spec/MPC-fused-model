"""Camera-to-body position transforms used before the MPC/Kalman filter."""

from __future__ import annotations

import numpy as np


# OpenCV camera [right, down, forward] -> aligned ROV body [forward, right, down].
ALIGNED_OPENCV_TO_BODY = np.array(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
)


def camera_to_body_position(
    position_camera,
    rotation_body_from_camera=ALIGNED_OPENCV_TO_BODY,
    camera_origin_in_body=(0.0, 0.0, 0.0),
) -> np.ndarray:
    """Transform target position from camera coordinates to ROV body FRD."""
    position = np.asarray(position_camera, dtype=float).reshape(-1)
    rotation = np.asarray(rotation_body_from_camera, dtype=float)
    translation = np.asarray(camera_origin_in_body, dtype=float).reshape(-1)
    if position.shape != (3,) or translation.shape != (3,) or rotation.shape != (3, 3):
        raise ValueError("position/translation must be 3-vectors and rotation must be 3x3")
    if not all(np.all(np.isfinite(value)) for value in (position, rotation, translation)):
        raise ValueError("camera transform contains NaN or infinity")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("rotation_body_from_camera must be orthonormal")
    return rotation @ position + translation


