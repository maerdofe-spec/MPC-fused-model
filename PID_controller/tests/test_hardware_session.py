from __future__ import annotations

import numpy as np

from finesub_protocol import (
    COMMAND_STATUS_ACCEPTED,
    COMMAND_STATUS_REJECTED,
    FineSUBHardwareAdapter,
    FineSUBTelemetry,
    pack_telemetry,
    unpack_command_frame,
)
from finesub_transport import DryRunTransport, FineSUBConnection, UdpTransport
from hardware_session import PIDHardwareSession, build_runtime_hardware_session
from live_integration_example import build_tracker
from vision_gate import PIDVisionGate, VisionGateConfig


def telemetry_for(
    envelope,
    *,
    accepted=True,
    sequence=None,
    pid_direct=True,
    yaw_direct=True,
) -> FineSUBTelemetry:
    return FineSUBTelemetry(
        sequence=envelope.sequence if sequence is None else sequence,
        tick_ms=100 + envelope.sequence,
        state=1,
        armed=envelope.command.armed,
        pid_direct=pid_direct,
        yaw_direct=yaw_direct,
        failsafe=False,
        yaw_rad=0.1,
        yaw_rate_rad_s=0.0,
        depth_m=0.2,
        forward=envelope.command.forward,
        right=envelope.command.right,
        down=envelope.command.down,
        yaw=envelope.command.yaw,
        command_status=(
            COMMAND_STATUS_ACCEPTED if accepted else COMMAND_STATUS_REJECTED
        ),
        reject_flags=0 if accepted else 1,
        last_command_session=envelope.session_id,
        last_command_sequence=envelope.sequence,
        last_command_crc=envelope.crc,
        last_command_sender_time_ms=envelope.sender_time_ms,
        command_count=envelope.sequence + 1,
        rejected_command_count=0 if accepted else 1,
        execution_feedback_valid=True,
        applied_motor_throttle=(0.0,) * 8,
    )


def build_dry_session(vision_gate=None):
    tracker = build_tracker()
    transport = DryRunTransport()
    connection = FineSUBConnection(
        transport,
        session_id=0x12345678,
        logger=None,
    )
    config = tracker.controller.config
    yaw_config = tracker.yaw_controller.config
    adapter = FineSUBHardwareAdapter(
        positive_force_at_limit=config.force_max,
        negative_force_at_limit=-config.force_min,
        positive_yaw_moment_at_limit=yaw_config.yaw_moment_max,
        negative_yaw_moment_at_limit=-yaw_config.yaw_moment_min,
    )
    return PIDHardwareSession(tracker, connection, adapter, vision_gate), transport


def test_new_session_cannot_arm_before_confirmed_disarm() -> None:
    session, transport = build_dry_session()
    assert session.connect()
    handshake = unpack_command_frame(transport.writes[-1])
    assert not handshake.command.armed

    first = session.step((0.2, 0.0, 1.0), arm_requested=True)
    assert not first.transmitted_arm
    assert first.status == "disarmed:no_fresh_telemetry"

    transport.inject_read(pack_telemetry(telemetry_for(handshake)))
    pending = session.step((0.2, 0.0, 1.0), arm_requested=True)
    assert pending.status == "disarmed:vision_pending"
    pending = session.step((0.2, 0.0, 1.0), arm_requested=True)
    assert pending.status == "disarmed:vision_pending"
    armed = session.step((0.2, 0.0, 1.0), arm_requested=True)
    assert armed.command_sent
    assert armed.transmitted_arm
    assert armed.controller_output is not None
    wire = unpack_command_frame(transport.writes[-1])
    assert wire.command.armed
    assert abs(wire.command.yaw) <= 0.20
    assert abs(wire.command.forward) <= 0.35


def test_rejected_telemetry_forces_zero_disarm() -> None:
    session, transport = build_dry_session()
    session.connect()
    handshake = unpack_command_frame(transport.writes[-1])
    transport.inject_read(pack_telemetry(telemetry_for(handshake, accepted=False)))
    result = session.step((0.2, 0.0, 1.0), arm_requested=True)
    assert result.status == "disarmed:unsafe_telemetry"
    assert not result.transmitted_arm
    disarm = unpack_command_frame(transport.writes[-1])
    assert not disarm.command.armed
    np.testing.assert_allclose(
        (disarm.command.forward, disarm.command.right, disarm.command.down, disarm.command.yaw),
        0.0,
    )


def test_non_direct_lower_mode_forces_zero_disarm() -> None:
    session, transport = build_dry_session()
    session.connect()
    handshake = unpack_command_frame(transport.writes[-1])
    transport.inject_read(pack_telemetry(telemetry_for(handshake, yaw_direct=False)))
    result = session.step((0.2, 0.0, 1.0), arm_requested=True)
    assert result.status == "disarmed:unsafe_telemetry"
    assert not result.transmitted_arm


def test_disarmed_step_never_transmits_nonzero_pid_output() -> None:
    session, transport = build_dry_session()
    assert session.connect()
    handshake = unpack_command_frame(transport.writes[-1])
    transport.inject_read(pack_telemetry(telemetry_for(handshake)))
    result = session.step((0.2, 0.0, 1.0), arm_requested=False)
    assert result.status == "disarmed:vision_pending"
    command = unpack_command_frame(transport.writes[-1]).command
    assert not command.armed
    np.testing.assert_allclose(
        (command.forward, command.right, command.down, command.yaw), 0.0
    )


def test_vision_jump_holds_last_position_and_keeps_active_session() -> None:
    gate = PIDVisionGate(
        VisionGateConfig(startup_confirmation_samples=1, reacquire_confirmation_samples=2)
    )
    session, transport = build_dry_session(gate)
    assert session.connect()
    handshake = unpack_command_frame(transport.writes[-1])
    transport.inject_read(pack_telemetry(telemetry_for(handshake)))
    first = session.step((0.0, 0.0, 1.0), arm_requested=True)
    armed_frame = unpack_command_frame(transport.writes[-1])
    assert first.transmitted_arm
    transport.inject_read(pack_telemetry(telemetry_for(armed_frame)))
    active = session.step((0.0, 0.0, 1.0), arm_requested=True)
    assert active.status == "active:confirmed"

    jumped = session.step((0.5, 0.0, 1.0), arm_requested=True)
    assert jumped.status == "active:confirmed"
    assert not session.safety_latched
    assert unpack_command_frame(transport.writes[-1]).command.armed
    # The outlier is not fed to PID: its held camera position produces the
    # same controller output as another update at the last valid sample.
    held = session.step((0.0, 0.0, 1.0), arm_requested=True)
    assert held.status == "active:confirmed"
    assert unpack_command_frame(transport.writes[-1]).command.armed


def test_mpc_style_runtime_config_builds_udp_without_opening_it(tmp_path) -> None:
    config = {
        "transport": {
            "type": "udp",
            "bind_host": "127.0.0.1",
            "bind_port": 0,
            "remote_host": "127.0.0.1",
            "remote_port": 5000,
        },
        "control": {"telemetry_max_age_sec": 0.2},
    }
    import json

    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    session = build_runtime_hardware_session(str(path), logger=None)
    assert isinstance(session.connection.transport, UdpTransport)
    assert session.connection._opened is False
    assert session.connection.transport._socket is None
