"""Fossen yaw dynamics and body-frame rotation for the dual translation model.

Coordinate convention:
    body x: forward, body y: right, body z: down (FRD)
    positive yaw/omega: nose turns to the right
    p_rel = p_target - p_vehicle, expressed in the current body frame

The translation model is first propagated in the old body frame and is then
expressed in the new body frame with Rz(-delta_yaw).  This implements

    p_dot_body = v_rel_body - omega_body x p_body

without using a small-angle approximation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from MPC_dual_model.fossen_fixed_dl_model import (
    FixedLinearDampingRelativeModel,
    vector3,
)


Array = np.ndarray


def finite_scalar(value, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def wrap_angle(angle_rad: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    angle = finite_scalar(angle_rad, "angle_rad")
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def rotation_body_from_previous(delta_yaw_rad: float) -> Array:
    """Map vector coordinates from the previous body frame to the new one.

    For a positive vehicle yaw increment, the coordinate representation of an
    unchanged inertial vector rotates by the negative increment.
    """
    delta = finite_scalar(delta_yaw_rad, "delta_yaw_rad")
    cosine = np.cos(delta)
    sine = np.sin(delta)
    return np.array(
        [[cosine, sine, 0.0], [-sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )


def rotation_state_matrix(delta_yaw_rad: float) -> Array:
    rotation = rotation_body_from_previous(delta_yaw_rad)
    result = np.zeros((6, 6))
    result[:3, :3] = rotation
    result[3:, 3:] = rotation
    return result


def line_of_sight_angle(position_body, epsilon: float = 1.0e-9) -> float:
    """Horizontal target bearing alpha=atan2(right, forward)."""
    position = vector3(position_body, "position_body")
    if position[0] ** 2 + position[1] ** 2 <= epsilon:
        return 0.0
    return float(np.arctan2(position[1], position[0]))


def visibility_frame_geometry(
    rotation_visibility_from_body=None,
    camera_origin_in_body=(0.0, 0.0, 0.0),
) -> tuple[Array, Array]:
    """Validate the fixed body-to-camera-visibility-frame geometry.

    The visibility frame is ordered ``[camera forward, camera right,
    camera down]``.  Keeping that order lets the existing FOV axis defaults
    remain ``0/1/2`` while supporting an arbitrary calibrated camera pose.
    """
    rotation = (
        np.eye(3)
        if rotation_visibility_from_body is None
        else np.asarray(rotation_visibility_from_body, dtype=float)
    )
    origin = vector3(camera_origin_in_body, "camera_origin_in_body")
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("rotation_visibility_from_body must be a finite 3x3 matrix")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-5):
        raise ValueError("rotation_visibility_from_body must be orthonormal")
    return rotation, origin


def body_to_visibility_position(
    position_body,
    rotation_visibility_from_body=None,
    camera_origin_in_body=(0.0, 0.0, 0.0),
) -> Array:
    """Express a body-origin target position as a camera-origin sight ray."""
    rotation, origin = visibility_frame_geometry(
        rotation_visibility_from_body,
        camera_origin_in_body,
    )
    position = vector3(position_body, "position_body")
    return rotation @ (position - origin)


def rotation_compensated_velocity(
    position_current_body,
    position_previous_body,
    delta_yaw_rad: float,
    dt: float,
) -> Array:
    """Finite-difference translational relative velocity with yaw removed."""
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive")
    current = vector3(position_current_body, "position_current_body")
    previous = vector3(position_previous_body, "position_previous_body")
    previous_in_current = rotation_body_from_previous(delta_yaw_rad) @ previous
    return (current - previous_in_current) / float(dt)


@dataclass
class LinearYawDynamics:
    """Reduced Fossen yaw model with optional quadratic damping.

    The implemented continuous model is

        m_omega * omega_dot
            + d_omega * omega
            + d_omega2 * |omega| * omega = N.

    ``effective_inertia`` is I_z-N_rdot and must include added inertia.
    The linear part is discretized exactly.  During one sample the quadratic
    term is frozen at the measured/predicted angular rate; this is appropriate
    here because yaw is predicted outside the translation QP.
    """

    effective_inertia: float
    linear_damping: float
    dt: float
    yaw_moment_base: float = 0.0
    quadratic_damping: float = 0.0

    def __post_init__(self) -> None:
        self.effective_inertia = finite_scalar(
            self.effective_inertia, "effective_inertia"
        )
        self.linear_damping = finite_scalar(self.linear_damping, "linear_damping")
        self.dt = finite_scalar(self.dt, "dt")
        self.yaw_moment_base = finite_scalar(
            self.yaw_moment_base, "yaw_moment_base"
        )
        self.quadratic_damping = finite_scalar(
            self.quadratic_damping, "quadratic_damping"
        )
        if self.effective_inertia <= 0.0:
            raise ValueError("effective_inertia must be positive")
        if self.linear_damping < 0.0:
            raise ValueError("linear_damping must be nonnegative")
        if self.quadratic_damping < 0.0:
            raise ValueError("quadratic_damping must be nonnegative")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")

        decay_rate = self.linear_damping / self.effective_inertia
        if decay_rate == 0.0:
            self.a_omega = 1.0
            self.b_omega = self.dt / self.effective_inertia
        else:
            self.a_omega = float(np.exp(-decay_rate * self.dt))
            self.b_omega = (1.0 - self.a_omega) / self.linear_damping

        # Compatibility aliases for callers created before omega notation was
        # adopted.  New code uses a_omega/b_omega exclusively.
        self.a_r = self.a_omega
        self.b_r = self.b_omega

    def set_yaw_moment_base(self, yaw_moment: float) -> None:
        self.yaw_moment_base = finite_scalar(yaw_moment, "yaw_moment")

    def predict_rate(self, yaw_rate: float, yaw_moment: float) -> float:
        """Predict omega[k+1] with the quadratic term frozen over the step."""
        rate = finite_scalar(yaw_rate, "yaw_rate")
        moment = finite_scalar(yaw_moment, "yaw_moment")
        nonlinear_drag = self.quadratic_damping * abs(rate) * rate
        return self.a_omega * rate + self.b_omega * (moment - nonlinear_drag)

    def predict_yaw_step(
        self,
        yaw_angle: float,
        yaw_rate: float,
        yaw_moment: float,
    ) -> tuple[float, float, float]:
        """Return (psi_next, omega_next, delta_psi) using trapezoidal yaw."""
        psi = finite_scalar(yaw_angle, "yaw_angle")
        omega = finite_scalar(yaw_rate, "yaw_rate")
        omega_next = self.predict_rate(omega, yaw_moment)
        delta_psi = 0.5 * self.dt * (omega + omega_next)
        return wrap_angle(psi + delta_psi), omega_next, float(delta_psi)


class RotationAwareRelativeModel:
    """Combine the fixed-D_L translation model with reduced yaw dynamics."""

    def __init__(
        self,
        translation: FixedLinearDampingRelativeModel,
        yaw: LinearYawDynamics,
    ) -> None:
        if not np.isclose(translation.dt, yaw.dt, rtol=0.0, atol=1.0e-12):
            raise ValueError("translation and yaw models must use the same dt")
        self.translation = translation
        self.yaw = yaw
        self.dt = translation.dt

    def rotation_aware_matrices(self, delta_yaw_rad: float) -> tuple[Array, Array]:
        """Return the PDF discrete model in the next body frame.

        The velocity is propagated first and position uses the resulting
        end-of-step velocity:

            v+ = R (F v + G f_eff)
            p+ = R p + dt v+.

        This deliberately differs from the translation package's exact-ZOH
        position integral and matches the multi-step position recursion in the
        supplied mathematical structure.
        """
        rotation = rotation_body_from_previous(delta_yaw_rad)
        rotated_F = rotation @ self.translation.F
        rotated_G = rotation @ self.translation.G
        transition = np.block(
            [
                [rotation, self.dt * rotated_F],
                [np.zeros((3, 3)), rotated_F],
            ]
        )
        input_matrix = np.vstack((self.dt * rotated_G, rotated_G))
        return transition, input_matrix

    def rotate_state(self, state_old_body, delta_yaw_rad: float) -> Array:
        state = np.asarray(state_old_body, dtype=float).reshape(-1)
        if state.shape != (6,) or not np.all(np.isfinite(state)):
            raise ValueError("state_old_body must be finite with shape (6,)")
        return rotation_state_matrix(delta_yaw_rad) @ state

    def predict_translation(
        self,
        state_current_body,
        effective_force,
        delta_yaw_rad: float,
    ) -> Array:
        """Propagate in the old frame, then express the result in the new frame."""
        state = np.asarray(state_current_body, dtype=float).reshape(-1)
        if state.shape != (6,) or not np.all(np.isfinite(state)):
            raise ValueError("state_current_body must be finite with shape (6,)")
        force = vector3(effective_force, "effective_force")
        A_rot, B_rot = self.rotation_aware_matrices(delta_yaw_rad)
        return A_rot @ state + B_rot @ force

    def predict_model1(
        self,
        state_current_body,
        force_current,
        tau_base,
        delta_yaw_rad: float,
    ) -> Array:
        """Predict the baseline-relative branch used by translation model 1."""
        return self.predict_translation(
            state_current_body,
            vector3(force_current, "force_current")
            - vector3(tau_base, "tau_base"),
            delta_yaw_rad,
        )

    def predict_model2(
        self,
        state_current_body,
        force_current,
        delta_yaw_rad: float,
    ) -> Array:
        """Predict the target-stationary Fossen branch using ``tau-tau_h``."""
        return self.predict_translation(
            state_current_body,
            vector3(force_current, "force_current")
            - self.translation.restoring_force,
            delta_yaw_rad,
        )
