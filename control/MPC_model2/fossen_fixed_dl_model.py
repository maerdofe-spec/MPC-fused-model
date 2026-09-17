"""Fixed-linear-damping Fossen model with model-2 force semantics."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from MPC_dual_model.fossen_fixed_dl_model import (
    FixedLinearDampingRelativeModel as _SharedFixedLinearDampingRelativeModel,
    matrix3,
    matrix_exponential,
    vector3,
    zero_order_hold,
)


class FixedLinearDampingRelativeModel(_SharedFixedLinearDampingRelativeModel):
    """Three-axis model whose input is always the absolute body force.

    The shared parent retains ``tau_base`` for package compatibility, but this
    model never reads it in state prediction or normal control.
    """

    def predict(self, p_rel, v_rel, tau) -> tuple[np.ndarray, np.ndarray]:
        state = np.concatenate(
            (vector3(p_rel, "p_rel"), vector3(v_rel, "v_rel"))
        )
        state_next = self.A_d @ state + self.B_d @ vector3(tau, "tau")
        return state_next[:3], state_next[3:]

    def predict_delta(self, p_rel, v_rel, tau) -> tuple[np.ndarray, np.ndarray]:
        """Compatibility alias; model 2 interprets the input as absolute force."""
        return self.predict(p_rel, v_rel, tau)

    def rollout(self, p_rel, v_rel, tau_sequence: Iterable) -> np.ndarray:
        p = vector3(p_rel, "p_rel")
        v = vector3(v_rel, "v_rel")
        states = [np.concatenate((p, v))]
        for tau in tau_sequence:
            p, v = self.predict(p, v, tau)
            states.append(np.concatenate((p, v)))
        return np.asarray(states)

    def absolute_force_affine_model(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return x_next=A_d*x+B_d*tau with zero affine offset."""
        return self.A_d.copy(), self.B_d.copy(), np.zeros(6)


__all__ = [
    "FixedLinearDampingRelativeModel",
    "matrix3",
    "matrix_exponential",
    "vector3",
    "zero_order_hold",
]
