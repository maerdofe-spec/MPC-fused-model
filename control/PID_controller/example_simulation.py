"""Deterministic closed-loop smoke test for the pure PID controller."""

from __future__ import annotations

import numpy as np

try:
    from .live_integration_example import build_tracker
    from .yaw_pid_controller import wrap_angle
except ImportError:
    from live_integration_example import build_tracker
    from yaw_pid_controller import wrap_angle


def main() -> None:
    tracker = build_tracker()
    dt = tracker.controller.config.dt
    position = np.array([1.40, 0.25, -0.18])
    relative_velocity = np.zeros(3)
    force = np.zeros(3)
    yaw = 0.0
    yaw_rate = 0.0
    yaw_moment = 0.0
    target_world_yaw = np.deg2rad(25.0)
    tracker.latch_baseline(force, yaw_moment, yaw)

    # This plant exists only for demonstration; it is not used by PID.
    effective_mass = np.array([26.1, 26.8, 26.1])
    linear_damping = np.array([93.9, 143.7, 280.9])
    print("step | relative position FRD       | force FRD        | yaw/goal deg | N_yaw")
    for step in range(240):
        # Express the same target in the rotating body frame so yaw PID uses
        # its horizontal camera bearing, just as it does in live operation.
        bearing = wrap_angle(target_world_yaw - yaw)
        planar_range = np.hypot(position[0], position[1])
        position[:2] = planar_range * np.array([np.cos(bearing), np.sin(bearing)])
        output = tracker.update(
            position,
            force,
            yaw_rad=yaw,
            yaw_rate_rad_s=yaw_rate,
            achieved_yaw_moment_previous=yaw_moment,
        )
        force = output.pid.force.copy()
        yaw_moment = output.yaw_pid.yaw_moment
        acceleration = (-linear_damping * relative_velocity - force) / effective_mass
        relative_velocity += acceleration * dt
        position += relative_velocity * dt
        yaw_rate += (yaw_moment - 0.8 * yaw_rate) / 0.8 * dt
        yaw = wrap_angle(yaw + yaw_rate * dt)
        if 80 <= step < 105:
            position[1] += 0.003  # moving target disturbance
        if step % 40 == 0 or step == 239:
            print(
                f"{step:4d} | {position!s:28s} | {force!s:16s} | "
                f"{np.rad2deg(yaw):6.2f}/{np.rad2deg(target_world_yaw):5.1f} | "
                f"{yaw_moment:6.3f}"
            )
    print("reference:", tracker.controller.config.reference_position)
    print("final error:", position - tracker.controller.config.reference_position)
    print("final yaw error deg:", np.rad2deg(wrap_angle(target_world_yaw - yaw)))


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    main()
