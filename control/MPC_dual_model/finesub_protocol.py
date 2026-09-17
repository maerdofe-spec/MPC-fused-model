"""FineSUB MPC v5 wire protocol and achieved-wrench reconstruction.

The four MPC variants work in SI body-frame wrench units (FRD: forward,
right, down, with positive yaw turning the bow right).  Firmware commands use
bounded mixer channels, while telemetry reports the command decision, the
eight physical motor set-points, and the eight DSHOT RPM measurements.

When the configured same-vehicle RPM/force curves and the required DSHOT-valid
bits are available, achieved force is reconstructed from measured motor RPM.
An affected axis falls back explicitly to the applied motor set-points only
when its RPM group is invalid.  This keeps estimator input tied to the actual
rotor response while preserving deterministic operation during a missing RPM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import struct
import time

import numpy as np

try:
    from .fossen_fixed_dl_model import vector3
except ImportError:  # Support direct execution from this model directory.
    from fossen_fixed_dl_model import vector3


PROTOCOL_VERSION = 5

COMMAND_SYNC = b"\xAA\x00"
COMMAND_TAIL = 0xBB
TELEMETRY_SYNC = b"\x55\x54"
TELEMETRY_MESSAGE_STATE_EXECUTION = 0x01

COMMAND_FLAG_ARMED = 0x01
COMMAND_FLAG_MPC_DIRECT = 0x02
COMMAND_FLAG_YAW_DIRECT = 0x04
COMMAND_FLAG_CALIBRATION_MOTOR = 0x08
COMMAND_FLAG_CALIBRATION_CHANNEL = 0x10
COMMAND_FLAG_CALIBRATION_ROLL_ONLY = 0x20
COMMAND_FLAG_CALIBRATION_PITCH_ONLY = 0x40
COMMAND_FLAG_CALIBRATION_YAW_ONLY = 0x80

TELEMETRY_FLAG_ARMED = 0x01
TELEMETRY_FLAG_MPC_DIRECT = 0x02
TELEMETRY_FLAG_FAILSAFE = 0x04
TELEMETRY_FLAG_YAW_DIRECT = 0x08
TELEMETRY_FLAG_EXECUTION_FEEDBACK = 0x10
TELEMETRY_FLAG_RPM_AVAILABLE = 0x20

COMMAND_STATUS_NONE = 0
COMMAND_STATUS_ACCEPTED = 1
COMMAND_STATUS_REJECTED = 2

COMMAND_REJECT_CRC = 1 << 0
COMMAND_REJECT_VERSION = 1 << 1
COMMAND_REJECT_FLAGS = 1 << 2
COMMAND_REJECT_NONFINITE = 1 << 3
COMMAND_REJECT_STALE_SEQUENCE = 1 << 4
COMMAND_REJECT_SESSION_REQUIRES_DISARM = 1 << 5
COMMAND_REJECT_FORMAT = 1 << 6
COMMAND_REJECT_CALIBRATION = 1 << 7

# Native 37-byte NX command: sync, version, flags, sequence, session,
# sender monotonic ms, forward/right/down/yaw, four reserved zero bytes,
# CRC16, tail.  The zero second sync byte also leaves old v1 firmware disarmed.
_COMMAND_BODY = struct.Struct("<2sBBHIIffff4x")

# telemetry header: sync, message type, version, payload length, sequence,
#                   MCU tick, payload, CRC16
_TELEMETRY_HEADER = struct.Struct("<2sBBHHI")

# telemetry payload integer section followed by 33 float32 values:
# flags, state, command status, RPM-valid mask,
# reject flags, command session, command sequence, command CRC,
# sender monotonic ms, command count, rejected count,
# quat[4], angular velocity[3], acceleration[3], yaw, depth, pressure,
# received mixer channels[4], applied physical motor throttle[8], RPM[8].
_TELEMETRY_PAYLOAD = struct.Struct("<BBBBIIHHIII33f")
_CRC = struct.Struct("<H")

COMMAND_FRAME_SIZE = _COMMAND_BODY.size + _CRC.size + 1
TELEMETRY_PAYLOAD_SIZE = _TELEMETRY_PAYLOAD.size
TELEMETRY_FRAME_SIZE = _TELEMETRY_HEADER.size + TELEMETRY_PAYLOAD_SIZE + _CRC.size

MOTOR_COUNT = 8

# These are the exact matrices and final motor signs used by V5_SUB.hpp.
# Physical motor order is M1..M8 as built in TaskSUB.cpp.
_LOWER_MIXER = np.asarray(
    [
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ],
    dtype=float,
)  # columns: yaw, forward, right
_UPPER_MIXER = np.asarray(
    [
        [-1.0, 1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
    ],
    dtype=float,
)  # columns: roll, pitch, down


def _finite_scalar(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _finite_tuple(value, length: int, name: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != length:
        raise ValueError(f"{name} must contain {length} values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return tuple(float(item) for item in array)


def crc16_modbus(data: bytes | bytearray | memoryview) -> int:
    """Return CRC-16/MODBUS, matching ``CRC16Calc`` in FineSUB."""

    crc = 0xFFFF
    for value in data:
        crc ^= int(value)
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc & 0xFFFF


def is_newer_u16_sequence(current: int, previous: int) -> bool:
    """Return whether ``current`` is newer using modulo-16-bit ordering."""

    delta = (int(current) - int(previous)) & 0xFFFF
    return 0 < delta < 0x8000


@dataclass(frozen=True)
class FineSUBControlCommand:
    """Normalized mixer channels requested from the lower controller."""

    forward: float
    right: float
    down: float
    yaw: float
    armed: bool
    yaw_direct: bool = True
    calibration_motor_index: int | None = None
    calibration_motor_throttle: float = 0.0
    calibration_channel: bool = False
    calibration_attitude_axis: str | None = None

    def __post_init__(self) -> None:
        _finite_tuple(
            (self.forward, self.right, self.down, self.yaw),
            4,
            "FineSUB control channels",
        )
        throttle = _finite_scalar(
            self.calibration_motor_throttle,
            "calibration_motor_throttle",
        )
        if self.calibration_attitude_axis not in (None, "roll", "pitch", "yaw"):
            raise ValueError(
                "calibration_attitude_axis must be roll, pitch, yaw, or None"
            )
        if self.calibration_channel:
            if self.calibration_motor_index is not None:
                raise ValueError(
                    "calibration-channel and calibration-motor modes are mutually exclusive"
                )
            if any(
                abs(value) > 0.100001
                for value in (self.forward, self.right, self.down, self.yaw)
            ):
                raise ValueError("calibration channels are hard limited to +/-0.10")
        selected_modes = sum(
            (
                bool(self.calibration_channel),
                self.calibration_motor_index is not None,
                self.calibration_attitude_axis is not None,
            )
        )
        if selected_modes > 1:
            raise ValueError("calibration modes are mutually exclusive")
        if self.calibration_attitude_axis is not None:
            expected_yaw_direct = self.calibration_attitude_axis != "yaw"
            if self.yaw_direct != expected_yaw_direct:
                raise ValueError(
                    "roll/pitch calibration requires yaw_direct; yaw calibration "
                    "requires local yaw feedback"
                )
            if any(
                abs(value) > 1.0e-12
                for value in (self.forward, self.right, self.down, self.yaw)
            ):
                raise ValueError(
                    "mixer channels must be zero in single-axis attitude calibration"
                )
        if self.calibration_motor_index is None:
            if abs(throttle) > 1.0e-12:
                raise ValueError(
                    "calibration_motor_throttle requires calibration_motor_index"
                )
            return
        index = int(self.calibration_motor_index)
        if index != self.calibration_motor_index or not 1 <= index <= MOTOR_COUNT:
            raise ValueError(f"calibration_motor_index must be in [1, {MOTOR_COUNT}]")
        if any(abs(value) > 1.0e-12 for value in (
            self.forward,
            self.right,
            self.down,
            self.yaw,
        )):
            raise ValueError("mixer channels must be zero in calibration-motor mode")
        if abs(throttle) > 0.100001:
            raise ValueError("calibration motor throttle is hard limited to +/-0.10")


@dataclass(frozen=True)
class FineSUBCommandEnvelope:
    command: FineSUBControlCommand
    sequence: int
    session_id: int
    sender_time_ms: int
    crc: int


@dataclass(frozen=True)
class FineSUBTelemetry:
    sequence: int
    tick_ms: int
    state: int
    armed: bool
    mpc_direct: bool
    yaw_direct: bool
    failsafe: bool
    yaw_rad: float
    yaw_rate_rad_s: float
    depth_m: float
    forward: float
    right: float
    down: float
    yaw: float
    command_status: int = COMMAND_STATUS_NONE
    reject_flags: int = 0
    last_command_session: int = 0
    last_command_sequence: int = 0
    last_command_crc: int = 0
    last_command_sender_time_ms: int = 0
    command_count: int = 0
    rejected_command_count: int = 0
    quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    angular_velocity_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    linear_acceleration_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pressure_pa: float = 0.0
    applied_motor_throttle: tuple[float, ...] = (0.0,) * MOTOR_COUNT
    motor_rpm: tuple[float, ...] = (0.0,) * MOTOR_COUNT
    execution_feedback_valid: bool = False
    rpm_available: bool = False
    rpm_valid_mask: int = 0
    received_monotonic: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        _finite_tuple(
            (
                self.yaw_rad,
                self.yaw_rate_rad_s,
                self.depth_m,
                self.forward,
                self.right,
                self.down,
                self.yaw,
                self.pressure_pa,
                self.received_monotonic,
            ),
            9,
            "FineSUB telemetry scalars",
        )
        object.__setattr__(
            self,
            "quat_wxyz",
            _finite_tuple(self.quat_wxyz, 4, "quat_wxyz"),
        )
        object.__setattr__(
            self,
            "angular_velocity_xyz",
            _finite_tuple(
                self.angular_velocity_xyz,
                3,
                "angular_velocity_xyz",
            ),
        )
        object.__setattr__(
            self,
            "linear_acceleration_xyz",
            _finite_tuple(
                self.linear_acceleration_xyz,
                3,
                "linear_acceleration_xyz",
            ),
        )
        object.__setattr__(
            self,
            "applied_motor_throttle",
            _finite_tuple(
                self.applied_motor_throttle,
                MOTOR_COUNT,
                "applied_motor_throttle",
            ),
        )
        object.__setattr__(
            self,
            "motor_rpm",
            _finite_tuple(self.motor_rpm, MOTOR_COUNT, "motor_rpm"),
        )

    @property
    def last_command_accepted(self) -> bool:
        return self.command_status == COMMAND_STATUS_ACCEPTED

    @property
    def last_command_rejected(self) -> bool:
        return self.command_status == COMMAND_STATUS_REJECTED

    @property
    def body_frd_yaw_rate_rad_s(self) -> float:
        """Return yaw rate in the body-FRD convention used by V5_SUB.

        Protocol v5 currently carries the raw H30 angular-rate vector, while
        the lower attitude controller negates that vector before using it as
        body FRD. Keep the wire value intact and expose the converted value
        explicitly for host prediction.
        """
        return -float(self.angular_velocity_xyz[2])


def is_newer_telemetry(
    current: FineSUBTelemetry,
    previous: FineSUBTelemetry,
) -> bool:
    """Accept forward progress or a clearly disarmed controller reboot."""

    if is_newer_u16_sequence(current.sequence, previous.sequence):
        return True
    return (
        not current.armed
        and current.sequence < 8
        and current.tick_ms < previous.tick_ms
    )


def motor_throttles_to_channels(
    motor_throttle,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Invert the exact FineSUB mixer from physical M1..M8 set-points.

    Returns ``(translation_channels, yaw_channel, roll_pitch_channels)``.
    The mixer columns are orthogonal, so the inverse is the transpose divided
    by four. Rows correspond directly to measured horizontal
    ``[M1,M2,M6,M7]`` and vertical ``[M3,M4,M5,M8]`` groups.
    """

    physical = np.asarray(
        _finite_tuple(motor_throttle, MOTOR_COUNT, "motor_throttle"),
        dtype=float,
    )
    lower = physical[[0, 1, 5, 6]]
    upper = physical[[2, 3, 4, 7]]
    yaw_forward_right = _LOWER_MIXER.T @ lower / 4.0
    roll_pitch_down = _UPPER_MIXER.T @ upper / 4.0
    translation = np.asarray(
        [yaw_forward_right[1], yaw_forward_right[2], roll_pitch_down[2]],
        dtype=float,
    )
    return translation, float(yaw_forward_right[0]), roll_pitch_down[:2]


@dataclass
class FineSUBHardwareAdapter:
    """Convert between MPC wrench units and bounded firmware mixer channels."""

    positive_force_at_limit: object = (20.0, 15.0, 15.0)
    negative_force_at_limit: object | None = None
    translation_channel_limits: object = (0.35, 0.35, 0.50)
    translation_signs: object = (1.0, 1.0, 1.0)
    positive_yaw_moment_at_limit: float = 4.0
    negative_yaw_moment_at_limit: float | None = None
    yaw_channel_limit: float = 0.20
    yaw_sign: float = 1.0
    use_rpm_for_force_estimate: bool = False
    log_rpm_force_estimate: bool = False
    rpm_c1_positive: object | None = None
    rpm_c1_negative: object | None = None
    rpm_force_directions_frd: object | None = None
    rpm_yaw_moment_arms: object | None = None
    rpm_positive_force_limit: object | None = None
    rpm_negative_force_limit: object | None = None

    def __post_init__(self) -> None:
        self.positive_force_at_limit = vector3(
            self.positive_force_at_limit, "positive_force_at_limit"
        )
        self.negative_force_at_limit = (
            self.positive_force_at_limit.copy()
            if self.negative_force_at_limit is None
            else vector3(self.negative_force_at_limit, "negative_force_at_limit")
        )
        self.translation_channel_limits = vector3(
            self.translation_channel_limits, "translation_channel_limits"
        )
        self.translation_signs = vector3(
            self.translation_signs, "translation_signs"
        )
        self.positive_yaw_moment_at_limit = _finite_scalar(
            self.positive_yaw_moment_at_limit,
            "positive_yaw_moment_at_limit",
        )
        self.negative_yaw_moment_at_limit = (
            self.positive_yaw_moment_at_limit
            if self.negative_yaw_moment_at_limit is None
            else _finite_scalar(
                self.negative_yaw_moment_at_limit,
                "negative_yaw_moment_at_limit",
            )
        )
        self.yaw_channel_limit = _finite_scalar(
            self.yaw_channel_limit, "yaw_channel_limit"
        )
        self.yaw_sign = _finite_scalar(self.yaw_sign, "yaw_sign")
        if np.any(self.positive_force_at_limit <= 0.0):
            raise ValueError("positive_force_at_limit must be positive")
        if np.any(self.negative_force_at_limit <= 0.0):
            raise ValueError("negative_force_at_limit must be positive")
        if np.any(self.translation_channel_limits <= 0.0):
            raise ValueError("translation_channel_limits must be positive")
        if np.any(np.abs(self.translation_signs) != 1.0):
            raise ValueError("translation_signs entries must be +1 or -1")
        if self.positive_yaw_moment_at_limit <= 0.0:
            raise ValueError("positive_yaw_moment_at_limit must be positive")
        if self.negative_yaw_moment_at_limit <= 0.0:
            raise ValueError("negative_yaw_moment_at_limit must be positive")
        if self.yaw_channel_limit <= 0.0:
            raise ValueError("yaw_channel_limit must be positive")
        if abs(self.yaw_sign) != 1.0:
            raise ValueError("yaw_sign must be +1 or -1")
        self.use_rpm_for_force_estimate = bool(
            self.use_rpm_for_force_estimate
        )
        self.log_rpm_force_estimate = bool(self.log_rpm_force_estimate)
        rpm_estimation_configured = (
            self.use_rpm_for_force_estimate or self.log_rpm_force_estimate
        )
        rpm_values = (
            self.rpm_c1_positive,
            self.rpm_c1_negative,
            self.rpm_force_directions_frd,
            self.rpm_yaw_moment_arms,
            self.rpm_positive_force_limit,
            self.rpm_negative_force_limit,
        )
        if rpm_estimation_configured and any(
            value is None for value in rpm_values
        ):
            raise ValueError(
                "RPM force estimation requires coefficients, geometry, and force limits"
            )
        if rpm_estimation_configured:
            self.rpm_c1_positive = self._rpm_vector(
                self.rpm_c1_positive, "rpm_c1_positive", positive=True
            )
            self.rpm_c1_negative = self._rpm_vector(
                self.rpm_c1_negative, "rpm_c1_negative", positive=True
            )
            directions = np.asarray(
                self.rpm_force_directions_frd, dtype=float
            )
            if directions.shape != (MOTOR_COUNT, 3) or not np.all(
                np.isfinite(directions)
            ):
                raise ValueError(
                    "rpm_force_directions_frd must have shape (8,3)"
                )
            norms = np.linalg.norm(directions, axis=1)
            if not np.allclose(norms, 1.0, atol=1.0e-5):
                raise ValueError("each RPM force direction must be unit length")
            self.rpm_force_directions_frd = directions
            self.rpm_yaw_moment_arms = self._rpm_vector(
                self.rpm_yaw_moment_arms, "rpm_yaw_moment_arms"
            )
            self.rpm_positive_force_limit = self._rpm_vector(
                self.rpm_positive_force_limit,
                "rpm_positive_force_limit",
                positive=True,
            )
            self.rpm_negative_force_limit = self._rpm_vector(
                self.rpm_negative_force_limit,
                "rpm_negative_force_limit",
                positive=True,
            )
        self.last_achieved_wrench_diagnostics = {
            "force_axis_sources": ["uninitialized"] * 3,
            "yaw_source": "uninitialized",
            "rpm_valid_mask": 0,
            "rpm_thruster_force_n": [0.0] * MOTOR_COUNT,
            "rpm_force_frd_n": [0.0] * 3,
            "rpm_yaw_moment_n_m": 0.0,
        }

    @staticmethod
    def _rpm_vector(value, name: str, *, positive: bool = False) -> np.ndarray:
        result = np.asarray(value, dtype=float).reshape(-1)
        if result.shape != (MOTOR_COUNT,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must contain eight finite values")
        if positive and np.any(result <= 0.0):
            raise ValueError(f"{name} must be positive")
        return result

    def convert(
        self,
        force_body,
        yaw_moment: float,
        *,
        armed: bool,
        yaw_direct: bool = True,
    ) -> FineSUBControlCommand:
        force = vector3(force_body, "force_body")
        moment = _finite_scalar(yaw_moment, "yaw_moment")
        signed_force = self.translation_signs * force
        force_at_limit = np.where(
            signed_force >= 0.0,
            self.positive_force_at_limit,
            self.negative_force_at_limit,
        )
        translation = np.clip(
            signed_force / force_at_limit * self.translation_channel_limits,
            -self.translation_channel_limits,
            self.translation_channel_limits,
        )
        signed_moment = self.yaw_sign * moment
        moment_at_limit = (
            self.positive_yaw_moment_at_limit
            if signed_moment >= 0.0
            else self.negative_yaw_moment_at_limit
        )
        yaw = float(
            np.clip(
                signed_moment / moment_at_limit * self.yaw_channel_limit,
                -self.yaw_channel_limit,
                self.yaw_channel_limit,
            )
        )
        return FineSUBControlCommand(
            forward=float(translation[0]),
            right=float(translation[1]),
            down=float(translation[2]),
            yaw=yaw,
            armed=bool(armed),
            yaw_direct=bool(yaw_direct),
        )

    def _channels_to_wrench(
        self,
        translation_channels,
        yaw_channel: float,
    ) -> tuple[np.ndarray, float]:
        channels = vector3(translation_channels, "translation_channels")
        force_at_limit = np.where(
            channels >= 0.0,
            self.positive_force_at_limit,
            self.negative_force_at_limit,
        )
        force = (
            channels / self.translation_channel_limits * force_at_limit
            / self.translation_signs
        )
        moment_at_limit = (
            self.positive_yaw_moment_at_limit
            if yaw_channel >= 0.0
            else self.negative_yaw_moment_at_limit
        )
        moment = float(yaw_channel) / self.yaw_channel_limit * moment_at_limit
        moment /= self.yaw_sign
        return force, float(moment)

    def achieved_wrench(
        self, telemetry: FineSUBTelemetry
    ) -> tuple[np.ndarray, float]:
        """Return command-derived force and optionally monitor RPM force.

        The control path defaults to the final applied motor set-points. RPM
        curves may still be evaluated for trace diagnostics without replacing
        that force. If RPM control is explicitly enabled, horizontal bits
        ``M1/M2/M6/M7`` and vertical bits ``M3/M4/M5/M8`` are gated
        independently before their corresponding axes replace the command
        estimate.
        """

        if telemetry.failsafe or not telemetry.armed or not telemetry.mpc_direct:
            self.last_achieved_wrench_diagnostics = {
                "force_axis_sources": ["zero", "zero", "zero"],
                "yaw_source": "zero",
                "rpm_valid_mask": int(telemetry.rpm_valid_mask),
                "rpm_thruster_force_n": [0.0] * MOTOR_COUNT,
                "rpm_force_frd_n": [0.0] * 3,
                "rpm_yaw_moment_n_m": 0.0,
            }
            return np.zeros(3, dtype=float), 0.0
        if telemetry.execution_feedback_valid:
            translation, yaw, _roll_pitch = motor_throttles_to_channels(
                telemetry.applied_motor_throttle
            )
            fallback_source = "applied_motor_throttle"
        else:
            translation = np.asarray(
                [telemetry.forward, telemetry.right, telemetry.down],
                dtype=float,
            )
            yaw = telemetry.yaw
            fallback_source = "command_echo"
        command_force, command_moment = self._channels_to_wrench(
            translation, yaw
        )
        force = command_force.copy()
        moment = float(command_moment)
        sources = [fallback_source] * 3
        yaw_source = fallback_source
        rpm_thruster_force = np.zeros(MOTOR_COUNT)
        rpm_force = np.zeros(3)
        rpm_moment = 0.0
        rpm_estimation_configured = (
            self.use_rpm_for_force_estimate or self.log_rpm_force_estimate
        )
        if rpm_estimation_configured and telemetry.rpm_available:
            rpm = np.asarray(telemetry.motor_rpm, dtype=float)
            omega = rpm * (2.0 * np.pi / 60.0)
            coefficient = np.where(
                rpm >= 0.0,
                self.rpm_c1_positive,
                self.rpm_c1_negative,
            )
            rpm_thruster_force = np.sign(rpm) * coefficient * omega * omega
            rpm_thruster_force = np.clip(
                rpm_thruster_force,
                -self.rpm_negative_force_limit,
                self.rpm_positive_force_limit,
            )
            rpm_force = rpm_thruster_force @ self.rpm_force_directions_frd
            rpm_moment = float(rpm_thruster_force @ self.rpm_yaw_moment_arms)
            mask = int(telemetry.rpm_valid_mask) & 0xFF
            horizontal_mask = sum(1 << index for index in (0, 1, 5, 6))
            vertical_mask = sum(1 << index for index in (2, 3, 4, 7))
            if (
                self.use_rpm_for_force_estimate
                and (mask & horizontal_mask) == horizontal_mask
            ):
                force[:2] = rpm_force[:2]
                moment = rpm_moment
                sources[:2] = ["rpm", "rpm"]
                yaw_source = "rpm"
            if (
                self.use_rpm_for_force_estimate
                and (mask & vertical_mask) == vertical_mask
            ):
                force[2] = rpm_force[2]
                sources[2] = "rpm"
        self.last_achieved_wrench_diagnostics = {
            "force_axis_sources": sources,
            "yaw_source": yaw_source,
            "rpm_valid_mask": int(telemetry.rpm_valid_mask),
            "rpm_thruster_force_n": rpm_thruster_force.tolist(),
            "rpm_force_frd_n": rpm_force.tolist(),
            "rpm_yaw_moment_n_m": float(rpm_moment),
        }
        return force, moment


def build_runtime_hardware_adapter(config: dict) -> FineSUBHardwareAdapter:
    """Build the AUTO adapter, including optional RPM-force diagnostics."""
    adapter = dict(config["hardware_adapter"])
    feedback = config["thruster_feedback"]
    prior = feedback["rpm_force_prior"]
    geometry = config["thruster_geometry"]
    enabled = bool(feedback.get("use_rpm_for_force_estimate", False))
    diagnostics_enabled = bool(feedback.get("log_rpm_force_estimate", False))
    adapter.update(
        use_rpm_for_force_estimate=enabled,
        log_rpm_force_estimate=diagnostics_enabled,
        rpm_c1_positive=prior.get("c1_positive_n_per_rad_s_sq_m1_m8"),
        rpm_c1_negative=prior.get("c1_negative_abs_n_per_rad_s_sq_m1_m8"),
        rpm_force_directions_frd=geometry.get(
            "positive_throttle_force_directions_frd_m1_m8"
        ),
        rpm_yaw_moment_arms=geometry.get(
            "yaw_moment_arm_about_cad_origin_m_per_positive_force_m1_m8"
        ),
        rpm_positive_force_limit=prior.get(
            "positive_force_limit_prior_n_m1_m8"
        ),
        rpm_negative_force_limit=prior.get(
            "negative_force_limit_abs_prior_n_m1_m8"
        ),
    )
    return FineSUBHardwareAdapter(**adapter)


def build_default_hardware_adapter() -> FineSUBHardwareAdapter:
    """Return conservative initial calibration for all four MPC variants."""

    return FineSUBHardwareAdapter(
        # Replace these wrench limits after a vehicle-specific bollard test.
        positive_force_at_limit=(20.0, 15.0, 15.0),
        translation_channel_limits=(0.35, 0.35, 0.50),
        translation_signs=(1.0, 1.0, 1.0),
        positive_yaw_moment_at_limit=4.0,
        yaw_channel_limit=0.20,
        yaw_sign=1.0,
    )


def pack_command(
    command: FineSUBControlCommand,
    sequence: int,
    *,
    session_id: int = 0,
    sender_time_ms: int = 0,
) -> bytes:
    flags = COMMAND_FLAG_MPC_DIRECT
    if command.armed:
        flags |= COMMAND_FLAG_ARMED
    if command.yaw_direct:
        flags |= COMMAND_FLAG_YAW_DIRECT
    if command.calibration_channel:
        flags |= COMMAND_FLAG_CALIBRATION_CHANNEL
    if command.calibration_attitude_axis == "roll":
        flags |= COMMAND_FLAG_CALIBRATION_ROLL_ONLY
    elif command.calibration_attitude_axis == "pitch":
        flags |= COMMAND_FLAG_CALIBRATION_PITCH_ONLY
    elif command.calibration_attitude_axis == "yaw":
        flags |= COMMAND_FLAG_CALIBRATION_YAW_ONLY
    wire_channels = (
        command.forward,
        command.right,
        command.down,
        command.yaw,
    )
    if command.calibration_motor_index is not None:
        flags |= COMMAND_FLAG_CALIBRATION_MOTOR
        wire_channels = (
            float(command.calibration_motor_index),
            float(command.calibration_motor_throttle),
            0.0,
            0.0,
        )
    body = _COMMAND_BODY.pack(
        COMMAND_SYNC,
        PROTOCOL_VERSION,
        flags,
        int(sequence) & 0xFFFF,
        int(session_id) & 0xFFFFFFFF,
        int(sender_time_ms) & 0xFFFFFFFF,
        *wire_channels,
    )
    return body + _CRC.pack(crc16_modbus(body)) + bytes((COMMAND_TAIL,))


def unpack_command_frame(frame: bytes) -> FineSUBCommandEnvelope:
    if len(frame) != COMMAND_FRAME_SIZE:
        raise ValueError(f"command frame must be {COMMAND_FRAME_SIZE} bytes")
    if frame[-1] != COMMAND_TAIL:
        raise ValueError("command tail mismatch")
    body = frame[: _COMMAND_BODY.size]
    received_crc = _CRC.unpack_from(frame, len(body))[0]
    if crc16_modbus(body) != received_crc:
        raise ValueError("command CRC mismatch")
    (
        sync,
        version,
        flags,
        sequence,
        session_id,
        sender_time_ms,
        forward,
        right,
        down,
        yaw,
    ) = _COMMAND_BODY.unpack(body)
    if sync != COMMAND_SYNC or version != PROTOCOL_VERSION:
        raise ValueError("unsupported command frame")
    if not flags & COMMAND_FLAG_MPC_DIRECT:
        raise ValueError("command is not an MPC-direct frame")
    known_flags = (
        COMMAND_FLAG_ARMED
        | COMMAND_FLAG_MPC_DIRECT
        | COMMAND_FLAG_YAW_DIRECT
        | COMMAND_FLAG_CALIBRATION_MOTOR
        | COMMAND_FLAG_CALIBRATION_CHANNEL
        | COMMAND_FLAG_CALIBRATION_ROLL_ONLY
        | COMMAND_FLAG_CALIBRATION_PITCH_ONLY
        | COMMAND_FLAG_CALIBRATION_YAW_ONLY
    )
    if flags & ~known_flags:
        raise ValueError("command contains unsupported flags")
    calibration_motor = bool(flags & COMMAND_FLAG_CALIBRATION_MOTOR)
    calibration_channel = bool(flags & COMMAND_FLAG_CALIBRATION_CHANNEL)
    calibration_roll = bool(flags & COMMAND_FLAG_CALIBRATION_ROLL_ONLY)
    calibration_pitch = bool(flags & COMMAND_FLAG_CALIBRATION_PITCH_ONLY)
    calibration_yaw = bool(flags & COMMAND_FLAG_CALIBRATION_YAW_ONLY)
    if sum(
        (
            calibration_motor,
            calibration_channel,
            calibration_roll,
            calibration_pitch,
            calibration_yaw,
        )
    ) > 1:
        raise ValueError("command selects multiple calibration modes")
    if (calibration_roll or calibration_pitch or calibration_yaw) and (
        ((calibration_yaw and flags & COMMAND_FLAG_YAW_DIRECT) or
         ((calibration_roll or calibration_pitch) and
          not flags & COMMAND_FLAG_YAW_DIRECT))
        or any(abs(value) > 1.0e-6 for value in (forward, right, down, yaw))
    ):
        raise ValueError("invalid single-axis attitude calibration command")
    if calibration_channel and any(
        abs(value) > 0.100001 for value in (forward, right, down, yaw)
    ):
        raise ValueError("invalid channel calibration command")
    if calibration_motor:
        rounded_motor = round(forward)
        if (
            not 1 <= rounded_motor <= MOTOR_COUNT
            or abs(forward - rounded_motor) > 1.0e-4
            or abs(right) > 0.100001
            or abs(down) > 1.0e-6
            or abs(yaw) > 1.0e-6
        ):
            raise ValueError("invalid physical-motor calibration command")
    command = FineSUBControlCommand(
        forward=0.0 if calibration_motor else forward,
        right=0.0 if calibration_motor else right,
        down=0.0 if calibration_motor else down,
        yaw=0.0 if calibration_motor else yaw,
        armed=bool(flags & COMMAND_FLAG_ARMED),
        yaw_direct=bool(flags & COMMAND_FLAG_YAW_DIRECT),
        calibration_motor_index=int(rounded_motor) if calibration_motor else None,
        calibration_motor_throttle=right if calibration_motor else 0.0,
        calibration_channel=calibration_channel,
        calibration_attitude_axis=(
            "roll"
            if calibration_roll
            else "pitch"
            if calibration_pitch
            else "yaw"
            if calibration_yaw
            else None
        ),
    )
    return FineSUBCommandEnvelope(
        command=command,
        sequence=sequence,
        session_id=session_id,
        sender_time_ms=sender_time_ms,
        crc=received_crc,
    )


def unpack_command(frame: bytes) -> tuple[FineSUBControlCommand, int]:
    """Compatibility helper returning the command and sequence only."""

    decoded = unpack_command_frame(frame)
    return decoded.command, decoded.sequence


def pack_telemetry(telemetry: FineSUBTelemetry) -> bytes:
    """Reference encoder used by tests and hardware-loop recordings."""

    flags = 0
    if telemetry.armed:
        flags |= TELEMETRY_FLAG_ARMED
    if telemetry.mpc_direct:
        flags |= TELEMETRY_FLAG_MPC_DIRECT
    if telemetry.yaw_direct:
        flags |= TELEMETRY_FLAG_YAW_DIRECT
    if telemetry.failsafe:
        flags |= TELEMETRY_FLAG_FAILSAFE
    if telemetry.execution_feedback_valid:
        flags |= TELEMETRY_FLAG_EXECUTION_FEEDBACK
    if telemetry.rpm_available:
        flags |= TELEMETRY_FLAG_RPM_AVAILABLE

    angular = list(telemetry.angular_velocity_xyz)
    angular[2] = float(telemetry.yaw_rate_rad_s)
    floats = (
        *telemetry.quat_wxyz,
        *angular,
        *telemetry.linear_acceleration_xyz,
        telemetry.yaw_rad,
        telemetry.depth_m,
        telemetry.pressure_pa,
        telemetry.forward,
        telemetry.right,
        telemetry.down,
        telemetry.yaw,
        *telemetry.applied_motor_throttle,
        *telemetry.motor_rpm,
    )
    payload = _TELEMETRY_PAYLOAD.pack(
        flags,
        int(telemetry.state) & 0xFF,
        int(telemetry.command_status) & 0xFF,
        int(telemetry.rpm_valid_mask) & 0xFF,
        int(telemetry.reject_flags) & 0xFFFFFFFF,
        int(telemetry.last_command_session) & 0xFFFFFFFF,
        int(telemetry.last_command_sequence) & 0xFFFF,
        int(telemetry.last_command_crc) & 0xFFFF,
        int(telemetry.last_command_sender_time_ms) & 0xFFFFFFFF,
        int(telemetry.command_count) & 0xFFFFFFFF,
        int(telemetry.rejected_command_count) & 0xFFFFFFFF,
        *floats,
    )
    header = _TELEMETRY_HEADER.pack(
        TELEMETRY_SYNC,
        TELEMETRY_MESSAGE_STATE_EXECUTION,
        PROTOCOL_VERSION,
        len(payload),
        int(telemetry.sequence) & 0xFFFF,
        int(telemetry.tick_ms) & 0xFFFFFFFF,
    )
    body = header + payload
    return body + _CRC.pack(crc16_modbus(body))


def unpack_telemetry(frame: bytes) -> FineSUBTelemetry:
    if len(frame) < _TELEMETRY_HEADER.size + _CRC.size:
        raise ValueError("telemetry frame is too short")
    sync, message_type, version, payload_len, sequence, tick_ms = (
        _TELEMETRY_HEADER.unpack_from(frame)
    )
    if sync != TELEMETRY_SYNC:
        raise ValueError("invalid telemetry sync")
    if message_type != TELEMETRY_MESSAGE_STATE_EXECUTION:
        raise ValueError(f"unsupported telemetry type {message_type}")
    if version != PROTOCOL_VERSION:
        raise ValueError(f"unsupported telemetry version {version}")
    expected_size = _TELEMETRY_HEADER.size + payload_len + _CRC.size
    if len(frame) != expected_size:
        raise ValueError(f"unexpected telemetry frame size {len(frame)}")
    if payload_len != TELEMETRY_PAYLOAD_SIZE:
        raise ValueError(f"unexpected telemetry payload size {payload_len}")
    body = frame[:-_CRC.size]
    if crc16_modbus(body) != _CRC.unpack_from(frame, len(body))[0]:
        raise ValueError("telemetry CRC mismatch")

    unpacked = _TELEMETRY_PAYLOAD.unpack_from(frame, _TELEMETRY_HEADER.size)
    flags = int(unpacked[0])
    float_values = unpacked[11:]
    quat = tuple(float(value) for value in float_values[0:4])
    angular = tuple(float(value) for value in float_values[4:7])
    acceleration = tuple(float(value) for value in float_values[7:10])
    received = float_values[13:17]
    applied = tuple(float(value) for value in float_values[17:25])
    rpm = tuple(float(value) for value in float_values[25:33])
    return FineSUBTelemetry(
        sequence=sequence,
        tick_ms=tick_ms,
        state=int(unpacked[1]),
        armed=bool(flags & TELEMETRY_FLAG_ARMED),
        mpc_direct=bool(flags & TELEMETRY_FLAG_MPC_DIRECT),
        yaw_direct=bool(flags & TELEMETRY_FLAG_YAW_DIRECT),
        failsafe=bool(flags & TELEMETRY_FLAG_FAILSAFE),
        yaw_rad=float(float_values[10]),
        yaw_rate_rad_s=float(angular[2]),
        depth_m=float(float_values[11]),
        forward=float(received[0]),
        right=float(received[1]),
        down=float(received[2]),
        yaw=float(received[3]),
        command_status=int(unpacked[2]),
        reject_flags=int(unpacked[4]),
        last_command_session=int(unpacked[5]),
        last_command_sequence=int(unpacked[6]),
        last_command_crc=int(unpacked[7]),
        last_command_sender_time_ms=int(unpacked[8]),
        command_count=int(unpacked[9]),
        rejected_command_count=int(unpacked[10]),
        quat_wxyz=quat,  # type: ignore[arg-type]
        angular_velocity_xyz=angular,  # type: ignore[arg-type]
        linear_acceleration_xyz=acceleration,  # type: ignore[arg-type]
        pressure_pa=float(float_values[12]),
        applied_motor_throttle=applied,
        motor_rpm=rpm,
        execution_feedback_valid=bool(
            flags & TELEMETRY_FLAG_EXECUTION_FEEDBACK
        ),
        rpm_available=bool(flags & TELEMETRY_FLAG_RPM_AVAILABLE),
        rpm_valid_mask=int(unpacked[3]),
    )


class TelemetryStreamDecoder:
    """Recover variable-length telemetry frames from a fragmented byte stream."""

    def __init__(self, *, max_payload_size: int = 512) -> None:
        self._buffer = bytearray()
        self._max_payload_size = int(max_payload_size)
        self.dropped_bytes = 0
        self.crc_errors = 0
        self.decode_errors = 0

    def reset(self) -> None:
        self._buffer.clear()

    def feed(self, data: bytes) -> list[FineSUBTelemetry]:
        if data:
            self._buffer.extend(data)
        decoded: list[FineSUBTelemetry] = []
        while True:
            start = self._buffer.find(TELEMETRY_SYNC)
            if start < 0:
                if self._buffer.endswith(TELEMETRY_SYNC[:1]):
                    self.dropped_bytes += max(0, len(self._buffer) - 1)
                    del self._buffer[:-1]
                else:
                    self.dropped_bytes += len(self._buffer)
                    self._buffer.clear()
                break
            if start:
                self.dropped_bytes += start
                del self._buffer[:start]
            if len(self._buffer) < _TELEMETRY_HEADER.size:
                break
            payload_len = struct.unpack_from("<H", self._buffer, 4)[0]
            if payload_len > self._max_payload_size:
                self.decode_errors += 1
                del self._buffer[0]
                continue
            frame_size = _TELEMETRY_HEADER.size + payload_len + _CRC.size
            if len(self._buffer) < frame_size:
                break
            candidate = bytes(self._buffer[:frame_size])
            try:
                decoded.append(unpack_telemetry(candidate))
                del self._buffer[:frame_size]
            except ValueError as error:
                if "CRC mismatch" in str(error):
                    self.crc_errors += 1
                else:
                    self.decode_errors += 1
                del self._buffer[0]
        return decoded
