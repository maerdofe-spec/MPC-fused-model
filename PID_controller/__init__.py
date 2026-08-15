"""Pure PID relative-position tracking for FineSUB."""

from .pid_controller import PIDConfig, PIDResult, RelativePIDController
from .pid_tracker import PIDTracker, PIDTrackerOutput, SafeControlOutput
from .yaw_pid_controller import YawPIDConfig, YawPIDController, YawPIDResult
from .hardware_session import (
    PIDHardwareSession,
    build_runtime_hardware_session,
    build_serial_hardware_session,
)
from .vision_gate import PIDVisionGate, VisionGateConfig, VisionGateResult
from .camera_transform import (
    PID_CONTROL_CAMERA_ORIGIN_IN_BODY,
    PID_CONTROL_REFERENCE_POSITION_BODY,
    PID_CONTROL_ROTATION_BODY_FROM_CAMERA,
    camera_to_body_position,
    camera_to_pid_body_position,
)

__all__ = [
    "PIDConfig",
    "PIDResult",
    "RelativePIDController",
    "PIDTracker",
    "PIDTrackerOutput",
    "SafeControlOutput",
    "YawPIDConfig",
    "YawPIDController",
    "YawPIDResult",
    "PIDHardwareSession",
    "build_runtime_hardware_session",
    "build_serial_hardware_session",
    "PIDVisionGate",
    "VisionGateConfig",
    "VisionGateResult",
    "PID_CONTROL_CAMERA_ORIGIN_IN_BODY",
    "PID_CONTROL_REFERENCE_POSITION_BODY",
    "PID_CONTROL_ROTATION_BODY_FROM_CAMERA",
    "camera_to_body_position",
    "camera_to_pid_body_position",
]
