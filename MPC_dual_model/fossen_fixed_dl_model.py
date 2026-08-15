"""Low-speed Fossen relative-motion model with fixed linear damping.

Coordinate convention used by the MPC package:
    body x: forward, body y: right, body z: down
    p_rel = p_target - p_vehicle
    v_rel = v_target - v_vehicle

The zero-speed reduced vehicle model is

    M_t u_dot + D_L u + tau_h = tau

where ``tau_h=h(0)`` is the fixed restoring/environment balance force.  The
relative-motion candidates from the design PDF are

    model 1: v_rel+ = F v_rel + G (tau - tau_base,k)
    model 2: v_rel+ = F v_rel + G (tau - tau_h)

The translation tracker updates ``tau_base,k`` with a per-axis gated EMA of
the achieved force, then holds it fixed over that solve's complete horizon.
Consecutive planned forces and the first actual force change remain separate
force-rate cost/constraint quantities. ``tau_h`` is never learned by that EMA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


Array = np.ndarray


def vector3(value, name: str) -> Array:
    result = np.asarray(value, dtype=float).reshape(-1)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector with shape (3,)")
    return result


def matrix3(value, name: str) -> Array:
    result = np.asarray(value, dtype=float)
    if result.shape != (3, 3) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite matrix with shape (3, 3)")
    return result


def matrix_exponential(matrix) -> Array:
    """Dense matrix exponential using Pade-13 scaling and squaring."""
    A = np.asarray(matrix, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("matrix must be square")
    if not np.all(np.isfinite(A)):
        raise ValueError("matrix contains NaN or infinity")

    identity = np.eye(A.shape[0])
    norm_1 = np.linalg.norm(A, 1)
    if norm_1 == 0.0:
        return identity

    theta_13 = 5.371920351148152
    scale = max(0, int(np.ceil(np.log2(norm_1 / theta_13))))
    A = A / (2.0**scale)
    b = np.array(
        [
            64764752532480000.0,
            32382376266240000.0,
            7771770303897600.0,
            1187353796428800.0,
            129060195264000.0,
            10559470521600.0,
            670442572800.0,
            33522128640.0,
            1323241920.0,
            40840800.0,
            960960.0,
            16380.0,
            182.0,
            1.0,
        ]
    )

    A2 = A @ A
    A4 = A2 @ A2
    A6 = A4 @ A2
    U = A @ (
        A6 @ (b[13] * A6 + b[11] * A4 + b[9] * A2)
        + b[7] * A6
        + b[5] * A4
        + b[3] * A2
        + b[1] * identity
    )
    V = (
        A6 @ (b[12] * A6 + b[10] * A4 + b[8] * A2)
        + b[6] * A6
        + b[4] * A4
        + b[2] * A2
        + b[0] * identity
    )
    result = np.linalg.solve(V - U, V + U)
    for _ in range(scale):
        result = result @ result
    return result


def zero_order_hold(A, B, dt: float) -> tuple[Array, Array]:
    """Exactly discretize x_dot=A*x+B*u for piecewise-constant input."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    if B.ndim != 2 or B.shape[0] != A.shape[0]:
        raise ValueError("B has incompatible dimensions")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive")

    n, m = A.shape[0], B.shape[1]
    augmented = np.zeros((n + m, n + m))
    augmented[:n, :n] = A
    augmented[:n, n:] = B
    discrete = matrix_exponential(augmented * dt)
    return discrete[:n, :n], discrete[:n, n:]


@dataclass
class FixedLinearDampingRelativeModel:
    """Fixed-D_L model with a separate, immutable Fossen balance force."""

    M_t: object
    D_L: object
    dt: float
    restoring_force: object = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        self.M_t = matrix3(self.M_t, "M_t")
        self.D_L = matrix3(self.D_L, "D_L")
        self.restoring_force = vector3(
            self.restoring_force, "restoring_force"
        )
        self.restoring_force.setflags(write=False)
        # Compatibility storage for the inactive legacy yaw/model-2 packages.
        # The maintained dual tracker/controller never reads this value.
        self._legacy_tau_base = np.zeros(3)
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if not np.allclose(self.M_t, self.M_t.T, atol=1e-10):
            raise ValueError("M_t must be symmetric")
        if np.min(np.linalg.eigvalsh(self.M_t)) <= 0.0:
            raise ValueError("M_t must be positive definite")
        damping_symmetric = 0.5 * (self.D_L + self.D_L.T)
        if np.min(np.linalg.eigvalsh(damping_symmetric)) < -1e-10:
            raise ValueError("the symmetric part of D_L must be positive semidefinite")
        self._build()

    def _build(self) -> None:
        I3 = np.eye(3)
        self.A_v = -np.linalg.solve(self.M_t, self.D_L)
        self.B_v = -np.linalg.solve(self.M_t, I3)
        self.F, self.G = zero_order_hold(self.A_v, self.B_v, self.dt)

        self.A_c = np.block(
            [[np.zeros((3, 3)), I3], [np.zeros((3, 3)), self.A_v]]
        )
        self.B_c = np.vstack((np.zeros((3, 3)), self.B_v))

        # Match the supplied PDF exactly: velocity is propagated by exact ZOH
        # and position then uses the end-of-step relative velocity.
        #
        #   v+ = F v + G f_eff
        #   p+ = p + dt v+
        self.A_d = np.block(
            [[I3, self.dt * self.F], [np.zeros((3, 3)), self.F]]
        )
        self.B_d = np.vstack((self.dt * self.G, self.G))
        if np.max(np.abs(np.linalg.eigvals(self.F))) > 1.0 + 1e-9:
            raise ValueError("M_t and D_L produce an unstable velocity model")

    @property
    def tau_h(self) -> Array:
        """Fixed ``h(0)`` force in body FRD coordinates."""
        return self.restoring_force.copy()

    @property
    def tau_base(self) -> Array:
        """Legacy-only storage; maintained dual code must not use this."""
        return self._legacy_tau_base.copy()

    @tau_base.setter
    def tau_base(self, value) -> None:
        self._legacy_tau_base = vector3(value, "legacy_tau_base")

    def set_tau_base(self, tau_achieved) -> None:
        """Legacy compatibility for inactive packages.

        The active dual model uses the caller-supplied preceding force and
        never consults this value.
        """
        self.tau_base = tau_achieved

    def predict_delta(self, p_rel, v_rel, delta_tau) -> tuple[Array, Array]:
        state = np.concatenate(
            (vector3(p_rel, "p_rel"), vector3(v_rel, "v_rel"))
        )
        state_next = self.A_d @ state + self.B_d @ vector3(delta_tau, "delta_tau")
        return state_next[:3], state_next[3:]

    def predict_stationary(self, p_rel, v_rel, tau) -> tuple[Array, Array]:
        """Predict model 2 using total force minus fixed ``h(0)``."""
        effective = vector3(tau, "tau") - self.restoring_force
        return self.predict_delta(p_rel, v_rel, effective)

    def predict_matched(
        self,
        p_rel,
        v_rel,
        tau,
        tau_base,
    ) -> tuple[Array, Array]:
        """Predict model 1 about one solve's fixed matched-force point."""
        delta = vector3(tau, "tau") - vector3(tau_base, "tau_base")
        return self.predict_delta(p_rel, v_rel, delta)

    def predict(self, p_rel, v_rel, tau) -> tuple[Array, Array]:
        """Predict the target-stationary branch from total physical force."""
        return self.predict_stationary(p_rel, v_rel, tau)

    def rollout(self, p_rel, v_rel, tau_sequence: Iterable) -> Array:
        """Roll out the target-stationary branch."""
        p = vector3(p_rel, "p_rel")
        v = vector3(v_rel, "v_rel")
        states = [np.concatenate((p, v))]
        for tau in tau_sequence:
            p, v = self.predict_stationary(p, v, tau)
            states.append(np.concatenate((p, v)))
        return np.asarray(states)

    def absolute_force_affine_model(self) -> tuple[Array, Array, Array]:
        """Return x_next=A_d*x+B_d*tau+c for absolute-force optimizers."""
        return (
            self.A_d.copy(),
            self.B_d.copy(),
            -self.B_d @ self.restoring_force,
        )
