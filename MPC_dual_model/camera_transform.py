"""Camera-to-body position transforms used before the MPC/Kalman filter."""

from __future__ import annotations

import numpy as np


# OpenCV camera [right, down, forward] -> aligned ROV body [forward, right, down].
ALIGNED_OPENCV_TO_BODY = np.array(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
)

# OpenCV [right, down, forward] -> visibility [forward, right, down].
VISIBILITY_FROM_OPENCV = np.array(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
)


def wrap_angle(angle_rad: float) -> float:
    """Wrap one finite angle to ``[-pi, pi)``."""
    angle = float(angle_rad)
    if not np.isfinite(angle):
        raise ValueError("angle_rad must be finite")
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def rotation_body_from_previous(delta_yaw_rad: float) -> np.ndarray:
    """Map previous-body FRD vector coordinates into the new body frame."""
    delta = float(delta_yaw_rad)
    if not np.isfinite(delta):
        raise ValueError("delta_yaw_rad must be finite")
    cosine = np.cos(delta)
    sine = np.sin(delta)
    return np.array(
        [[cosine, sine, 0.0], [-sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )


def rotation_state_body_from_previous(delta_yaw_rad: float) -> np.ndarray:
    """Return the 6x6 position/velocity coordinate rotation."""
    rotation = rotation_body_from_previous(delta_yaw_rad)
    result = np.zeros((6, 6))
    result[:3, :3] = rotation
    result[3:, 3:] = rotation
    return result


def camera_visibility_geometry(
    rotation_body_from_camera=ALIGNED_OPENCV_TO_BODY,
    camera_origin_in_body=(0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Return body->camera-visibility rotation and camera origin in body FRD.

    The returned frame is ordered ``[camera forward, camera right,
    camera down]`` so the MPC's existing forward/horizontal/vertical axis
    indices remain ``0/1/2``.
    """
    rotation = np.asarray(rotation_body_from_camera, dtype=float)
    origin = np.asarray(camera_origin_in_body, dtype=float).reshape(-1)
    if rotation.shape != (3, 3) or origin.shape != (3,):
        raise ValueError("camera rotation must be 3x3 and origin must be a 3-vector")
    if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(origin)):
        raise ValueError("camera geometry contains NaN or infinity")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-5):
        raise ValueError("rotation_body_from_camera must be orthonormal")
    return VISIBILITY_FROM_OPENCV @ rotation.T, origin.copy()


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
