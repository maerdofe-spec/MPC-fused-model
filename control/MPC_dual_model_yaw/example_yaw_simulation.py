"""Hardware-free demonstration of yaw PID plus frozen-rotation translation MPC."""

from __future__ import annotations

import numpy as np

from .live_integration_example import build_tracker


def main() -> None:
    tracker = build_tracker()
    model = tracker.model
    controller = tracker.controller
    yaw_controller = tracker.yaw_controller

    # Target starts forward-right in the body frame.  The plant below uses the
    # absolute-force translation candidate and the linear Fossen yaw model.
    state = np.array([1.10, 0.55, -0.08, 0.0, 0.0, 0.0])
    yaw = 0.0
    yaw_rate = 0.0
    force_previous = np.zeros(3)
    yaw_moment_previous = 0.0

    print("step | p_body [forward right down] | alpha(deg) | omega(deg/s) | N(Nm) | mode")
    for step in range(120):
        alpha = float(np.arctan2(state[1], state[0]))
        yaw_control = yaw_controller.update(
            yaw_angle=yaw,
            yaw_rate=yaw_rate,
            alpha=alpha,
            previous_achieved_moment=yaw_moment_previous,
            horizon=controller.config.horizon,
        )
        result = controller.solve(
            state=state,
            force_previous=force_previous,
            yaw_prediction=yaw_control.prediction,
            # The demonstration plant below is exactly candidate model 2, so
            # use its known weight here.  The real tracker estimates this
            # weight online from completed position predictions.
            model1_weight=(0.0, 0.0, 0.0),
        )

        # The actual yaw increment must come from the IMU in real use.  This
        # simulation integrates the model because no IMU exists here.
        yaw, yaw_rate, delta_yaw = model.yaw.predict_yaw_step(
            yaw, yaw_rate, result.yaw_moment
        )
        state = model.predict_model2(state, result.force, delta_yaw)
        force_previous = result.force.copy()
        yaw_moment_previous = result.yaw_moment

        if step % 10 == 0 or step == 119:
            alpha_deg = np.rad2deg(np.arctan2(state[1], state[0]))
            print(
                f"{step:4d} | {state[:3]} | {alpha_deg:9.3f} | "
                f"{np.rad2deg(yaw_rate):8.3f} | {result.yaw_moment:7.3f} "
                f"| {yaw_control.mode.value}"
            )

    print("final yaw (deg):", np.rad2deg(yaw))
    print("final state:", state)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    main()
