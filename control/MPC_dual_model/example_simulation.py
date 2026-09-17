"""Closed-loop, hardware-free demonstration of the fixed-D_L MPC."""

from __future__ import annotations

import numpy as np

from fossen_fixed_dl_model import FixedLinearDampingRelativeModel
from mpc_controller import MPCConfig, RelativeMPCController


def main() -> None:
    model = FixedLinearDampingRelativeModel(
        # PLACEHOLDERS: replace with identified effective masses and damping.
        M_t=np.diag([20.0, 25.0, 30.0]),
        D_L=np.diag([8.0, 10.0, 12.0]),
        dt=0.10,
    )
    config = MPCConfig(
        horizon=8,
        reference_position=(0.60, 0.0, 0.0),
        force_min=(-20.0, -15.0, -15.0),
        force_max=(20.0, 15.0, 15.0),
        delta_force_min=(-4.0, -3.0, -3.0),
        delta_force_max=(4.0, 3.0, 3.0),
    )
    controller = RelativeMPCController(model, config)

    # [forward, right, down, forward_speed, right_speed, down_speed]
    state = np.array([1.20, 0.20, -0.15, 0.0, 0.0, 0.0])
    tau_previous = np.zeros(3)
    model1_weight = np.array([0.8, 0.8, 0.8])

    print("step | p_rel forward/right/down | tau forward/right/down | status")
    for step in range(60):
        result = controller.solve(
            state=state,
            tau_previous=tau_previous,
            model1_weight=model1_weight,
        )
        # Demonstration plant: full-force model plus a short target acceleration.
        state = model.A_d @ state + model.B_d @ result.force
        if 15 <= step < 25:
            state[3] += 0.025
        tau_previous = result.force.copy()

        if step % 10 == 0 or step == 59:
            print(
                f"{step:4d} | {state[:3]} | {result.force} | {result.status}"
            )

    print("reference position:", config.reference_position)
    print("final state:", state)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    main()
