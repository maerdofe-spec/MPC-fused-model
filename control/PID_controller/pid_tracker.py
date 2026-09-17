"""High-level position measurement -> pure PID -> hardware command bridge."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from .device_adapter import (
        DeviceCommand,
        FineSUBThrusterAllocator,
        ForceCommandAdapter,
        ThrusterAllocation,
        YawMomentChannelAdapter,
    )
    from .pid_controller import PIDResult, RelativePIDController, vector3
    from .yaw_pid_controller import (
        YawPIDController,
        YawPIDResult,
        finite_scalar,
        wrap_angle,
    )
except ImportError:
    from device_adapter import (
        DeviceCommand,
        FineSUBThrusterAllocator,
        ForceCommandAdapter,
        ThrusterAllocation,
        YawMomentChannelAdapter,
    )
    from pid_controller import PIDResult, RelativePIDController, vector3
    from yaw_pid_controller import (
        YawPIDController,
        YawPIDResult,
        finite_scalar,
        wrap_angle,
    )


@dataclass(frozen=True)
class PIDTrackerOutput:
    measured_state: np.ndarray
    pid: PIDResult
    device_command: DeviceCommand
    thruster_allocation: ThrusterAllocation | None
    yaw_pid: YawPIDResult | None = None
    yaw_channel: float = 0.0

    @property
    def force(self) -> np.ndarray:
        return self.pid.force


@dataclass(frozen=True)
class SafeControlOutput:
    force: np.ndarray
    device_command: DeviceCommand
    thruster_allocation: ThrusterAllocation | None
    status: str
    yaw_moment: float = 0.0
    yaw_channel: float = 0.0


class PIDTracker:
    """Pure PID tracking entry point with no model-based state estimation."""

    def __init__(
        self,
        controller: RelativePIDController,
        adapter: ForceCommandAdapter,
        thruster_allocator: FineSUBThrusterAllocator | None = None,
        yaw_controller: YawPIDController | None = None,
        yaw_adapter: YawMomentChannelAdapter | None = None,
        track_target_bearing: bool = True,
        yaw_enabled: bool = True,
    ) -> None:
        self.controller = controller
        self.adapter = adapter
        self.thruster_allocator = thruster_allocator
        self.yaw_controller = yaw_controller
        self.yaw_adapter = yaw_adapter
        self.track_target_bearing = bool(track_target_bearing)
        self.yaw_enabled = bool(yaw_enabled)
        if (yaw_controller is None) != (yaw_adapter is None):
            raise ValueError("yaw_controller and yaw_adapter must be supplied together")

    def freeze_yaw(self) -> None:
        """Disable yaw output while retaining the yaw PID object for diagnostics."""
        self.yaw_enabled = False

    def enable_yaw(self) -> None:
        """Re-enable yaw output for an explicitly authorised offline/bench run."""
        self.yaw_enabled = True

    def latch_baseline(
        self,
        achieved_force: object,
        achieved_yaw_moment: float = 0.0,
        yaw_rad: float | None = None,
    ) -> None:
        self.controller.latch_baseline(achieved_force)
        if self.yaw_controller is not None:
            self.yaw_controller.latch_baseline(achieved_yaw_moment, yaw_rad)

    def update(
        self,
        position_body: object,
        achieved_force_previous: object,
        reference_position: object | None = None,
        yaw_rad: float | None = None,
        yaw_rate_rad_s: float | None = None,
        achieved_yaw_moment_previous: float = 0.0,
        reference_yaw_rad: float | None = None,
    ) -> PIDTrackerOutput:
        position = vector3(position_body, "position_body")
        result = self.controller.update(
            position,
            achieved_force_previous,
            reference_position,
        )
        yaw_result: YawPIDResult | None = None
        yaw_channel = 0.0
        if self.yaw_enabled and yaw_rad is not None and self.yaw_controller is not None:
            yaw = wrap_angle(yaw_rad)
            yaw_reference = reference_yaw_rad
            if yaw_reference is None and self.track_target_bearing:
                yaw_reference = wrap_angle(
                    yaw + np.arctan2(position[1], position[0])
                )
            yaw_result = self.yaw_controller.update(
                yaw_angle=yaw,
                previous_yaw_moment=finite_scalar(
                    achieved_yaw_moment_previous,
                    "achieved_yaw_moment_previous",
                ),
                reference_yaw=yaw_reference,
                yaw_rate=yaw_rate_rad_s,
            )
            yaw_channel = self.yaw_adapter.convert(yaw_result.yaw_moment)
        # The second half is PID's filtered measured error derivative. It is
        # diagnostic state only and is never produced by a model/observer.
        measured_state = np.concatenate((position, result.error_derivative))
        allocation = (
            None
            if self.thruster_allocator is None
            else self.thruster_allocator.allocate(
                result.force, attitude_control=(0.0, 0.0, yaw_channel)
            )
        )
        return PIDTrackerOutput(
            measured_state=measured_state,
            pid=result,
            device_command=self.adapter.convert(result.force),
            thruster_allocation=allocation,
            yaw_pid=yaw_result,
            yaw_channel=yaw_channel,
        )

    def target_lost(
        self,
        achieved_force_previous: object,
        achieved_yaw_moment_previous: float = 0.0,
    ) -> SafeControlOutput:
        force = self.controller.safe_force(achieved_force_previous)
        if not self.yaw_enabled or self.yaw_controller is None:
            yaw_moment = 0.0
            yaw_channel = 0.0
        else:
            yaw_moment = self.yaw_controller.safe_moment(achieved_yaw_moment_previous)
            yaw_channel = (
                0.0
                if self.yaw_adapter is None
                else self.yaw_adapter.convert(yaw_moment)
            )
        allocation = (
            None
            if self.thruster_allocator is None
            else self.thruster_allocator.allocate(
                force, attitude_control=(0.0, 0.0, yaw_channel)
            )
        )
        return SafeControlOutput(
            force=force,
            device_command=self.adapter.convert(force),
            thruster_allocation=allocation,
            status="target_lost:return_to_baseline",
            yaw_moment=yaw_moment,
            yaw_channel=yaw_channel,
        )
