"""Convert body forces in newtons to the existing ROV command scales."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from .fossen_fixed_dl_model import vector3
except ImportError:
    from fossen_fixed_dl_model import vector3


@dataclass(frozen=True)
class DeviceCommand:
    planar_forward: int
    planar_right: int
    depth_force: int


@dataclass
class ForceCommandAdapter:
    """Linear force-to-command calibration.

    positive_force_at_limit gives the body force corresponding to the positive
    device command limit. signs adapts body-axis signs to lower-controller
    signs. Replace these calibration values before enabling real hardware.
    """

    positive_force_at_limit: object = (20.0, 15.0, 15.0)
    signs: object = (1.0, 1.0, 1.0)
    command_limits: object = (99.0, 99.0, 45.0)

    def __post_init__(self) -> None:
        self.positive_force_at_limit = vector3(
            self.positive_force_at_limit, "positive_force_at_limit"
        )
        self.signs = vector3(self.signs, "signs")
        self.command_limits = vector3(self.command_limits, "command_limits")
        if np.any(self.positive_force_at_limit <= 0.0):
            raise ValueError("positive_force_at_limit must be positive")
        if np.any(self.command_limits <= 0.0):
            raise ValueError("command_limits must be positive")
        if np.any(np.abs(self.signs) != 1.0):
            raise ValueError("each sign must be +1 or -1")

    def convert(self, force_body) -> DeviceCommand:
        force = vector3(force_body, "force_body")
        raw = (
            self.signs
            * force
            / self.positive_force_at_limit
            * self.command_limits
        )
        clipped = np.clip(raw, -self.command_limits, self.command_limits)
        command = np.rint(clipped).astype(int)
        return DeviceCommand(
            planar_forward=int(command[0]),
            planar_right=int(command[1]),
            depth_force=int(command[2]),
        )


@dataclass(frozen=True)
class ThrusterAllocation:
    """FineSUB motor throttles in TaskSUB.cpp construction order."""

    throttles: np.ndarray
    translation_channels: np.ndarray
    attitude_channels: np.ndarray


@dataclass
class FineSUBThrusterAllocator:
    """Reproduce the checked FineSUB upper/lower manual control distribution.

    Motor order:
      0 LFLower, 1 LFUpper, 2 LBUpper, 3 LBLower,
      4 RBLower, 5 RBUpper, 6 RFUpper, 7 RFLower.

    ``attitude_control`` is the normalized [roll, pitch, yaw] PID output used
    by FineSUB. Translation is supplied as MPC body force [forward,right,down]
    in newtons and calibrated into the firmware's normalized mixer channels.
    """

    positive_force_at_limit: object = (20.0, 15.0, 15.0)
    translation_channel_limits: object = (0.35, 0.35, 0.50)
    attitude_channel_limits: object = (0.20, 0.20, 0.20)
    deadband: float = 0.01
    enable_depth: bool = True

    def __post_init__(self) -> None:
        self.positive_force_at_limit = vector3(
            self.positive_force_at_limit, "positive_force_at_limit"
        )
        self.translation_channel_limits = vector3(
            self.translation_channel_limits, "translation_channel_limits"
        )
        self.attitude_channel_limits = vector3(
            self.attitude_channel_limits, "attitude_channel_limits"
        )
        if np.any(self.positive_force_at_limit <= 0.0):
            raise ValueError("positive_force_at_limit must be positive")
        if np.any(self.translation_channel_limits <= 0.0):
            raise ValueError("translation_channel_limits must be positive")
        if np.any(self.attitude_channel_limits <= 0.0):
            raise ValueError("attitude_channel_limits must be positive")
        if self.deadband < 0.0:
            raise ValueError("deadband must be nonnegative")

        # Raw matrices copied from V5_SUB.hpp. The row order is the order in
        # which those motor groups are written, before the per-motor sign.
        self.upper_matrix = np.array(
            [[1.0, -1.0, -1.0],
             [1.0,  1.0, -1.0],
             [-1.0, 1.0, -1.0],
             [-1.0, -1.0, -1.0]]
        )
        self.lower_matrix = np.array(
            [[1.0, -1.0, 1.0],
             [-1.0, 1.0, 1.0],
             [1.0, 1.0, -1.0],
             [-1.0, -1.0, -1.0]]
        )
        self.upper_indices = np.array([1, 2, 5, 6])
        self.lower_indices = np.array([0, 3, 4, 7])
        self.upper_signs = np.array([1.0, -1.0, 1.0, 1.0])
        self.lower_signs = np.array([1.0, -1.0, 1.0, -1.0])

    def allocate(self, force_body, attitude_control=(0.0, 0.0, 0.0)) -> ThrusterAllocation:
        force = vector3(force_body, "force_body")
        attitude = vector3(attitude_control, "attitude_control")
        translation = np.clip(
            force / self.positive_force_at_limit * self.translation_channel_limits,
            -self.translation_channel_limits,
            self.translation_channel_limits,
        )
        if not self.enable_depth:
            translation[2] = 0.0
        attitude = np.clip(
            attitude, -self.attitude_channel_limits, self.attitude_channel_limits
        )
        roll, pitch, yaw = attitude
        forward, right, down = translation
        upper_raw = self.upper_matrix @ np.array([roll, pitch, down])
        lower_raw = self.lower_matrix @ np.array([yaw, forward, right])
        upper_raw[np.abs(upper_raw) < self.deadband] = 0.0
        lower_raw[np.abs(lower_raw) < self.deadband] = 0.0

        throttles = np.zeros(8)
        throttles[self.upper_indices] = self.upper_signs * upper_raw
        throttles[self.lower_indices] = self.lower_signs * lower_raw
        # FineSUB ultimately saturates each DSHOT channel independently.
        throttles = np.clip(throttles, -1.0, 1.0)
        return ThrusterAllocation(
            throttles=throttles,
            translation_channels=translation.copy(),
            attitude_channels=attitude.copy(),
        )

