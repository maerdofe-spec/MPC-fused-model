"""Low-speed Fossen relative-motion model with fixed linear damping.

Coordinate convention used by the MPC package:
    body x: forward, body y: right, body z: down
    p_rel = p_target - p_vehicle
    v_rel = v_target - v_vehicle

The reduced model is

    M_t v_rel_dot + D_L v_rel = -(tau - tau_base)
    p_rel_dot = v_rel

where M_t includes translational added mass and D_L is held fixed.
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
    """Fixed-D_L three-axis model and its exact discrete state matrices."""

    M_t: object
    D_L: object
    dt: float
    tau_base: object = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        self.M_t = matrix3(self.M_t, "M_t")
        self.D_L = matrix3(self.D_L, "D_L")
        self.tau_base = vector3(self.tau_base, "tau_base")
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
        self.A_d, self.B_d = zero_order_hold(self.A_c, self.B_c, self.dt)
        if np.max(np.abs(np.linalg.eigvals(self.F))) > 1.0 + 1e-9:
            raise ValueError("M_t and D_L produce an unstable velocity model")

    def set_tau_base(self, tau_achieved) -> None:
        self.tau_base = vector3(tau_achieved, "tau_achieved")

    def predict_delta(self, p_rel, v_rel, delta_tau) -> tuple[Array, Array]:
        state = np.concatenate(
            (vector3(p_rel, "p_rel"), vector3(v_rel, "v_rel"))
        )
        state_next = self.A_d @ state + self.B_d @ vector3(delta_tau, "delta_tau")
        return state_next[:3], state_next[3:]

    def predict(self, p_rel, v_rel, tau, tau_base=None) -> tuple[Array, Array]:
        base = self.tau_base if tau_base is None else vector3(tau_base, "tau_base")
        return self.predict_delta(p_rel, v_rel, vector3(tau, "tau") - base)

    def rollout(self, p_rel, v_rel, tau_sequence: Iterable, tau_base=None) -> Array:
        p = vector3(p_rel, "p_rel")
        v = vector3(v_rel, "v_rel")
        base = self.tau_base if tau_base is None else vector3(tau_base, "tau_base")
        states = [np.concatenate((p, v))]
        for tau in tau_sequence:
            p, v = self.predict(p, v, tau, base)
            states.append(np.concatenate((p, v)))
        return np.asarray(states)

    def absolute_force_affine_model(self) -> tuple[Array, Array, Array]:
        """Return x_next=A_d*x+B_d*tau+c for absolute-force optimizers."""
        return self.A_d.copy(), self.B_d.copy(), -self.B_d @ self.tau_base


