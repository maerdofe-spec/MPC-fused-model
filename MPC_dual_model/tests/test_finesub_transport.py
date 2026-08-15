import unittest
from unittest.mock import patch

from MPC_dual_model.finesub_protocol import (
    COMMAND_REJECT_STALE_SEQUENCE,
    COMMAND_STATUS_ACCEPTED,
    COMMAND_STATUS_REJECTED,
    FineSUBControlCommand,
    FineSUBTelemetry,
    pack_telemetry,
    pack_command,
    unpack_command_frame,
)
from MPC_dual_model.finesub_transport import (
    DryRunTransport,
    FineSUBConnection,
    UdpTransport,
)


def telemetry_for_command(envelope, *, accepted=True):
    return FineSUBTelemetry(
        sequence=envelope.sequence,
        tick_ms=100,
        state=1,
        armed=False,
        mpc_direct=True,
        yaw_direct=envelope.command.yaw_direct,
        failsafe=False,
        yaw_rad=0.0,
        yaw_rate_rad_s=0.0,
        depth_m=0.0,
        forward=envelope.command.forward,
        right=envelope.command.right,
        down=envelope.command.down,
        yaw=envelope.command.yaw,
        command_status=(
            COMMAND_STATUS_ACCEPTED if accepted else COMMAND_STATUS_REJECTED
        ),
        reject_flags=0 if accepted else COMMAND_REJECT_STALE_SEQUENCE,
        last_command_session=envelope.session_id,
        last_command_sequence=envelope.sequence,
        last_command_crc=envelope.crc,
        last_command_sender_time_ms=envelope.sender_time_ms,
        command_count=1,
        rejected_command_count=0 if accepted else 1,
        execution_feedback_valid=True,
        rpm_available=True,
    )


class FineSUBTransportTest(unittest.TestCase):
    def test_udp_sends_native_37_byte_nx_command_unchanged(self) -> None:
        class FakeSocket:
            def __init__(self) -> None:
                self.datagram = b""

            def sendto(self, data, _address):
                self.datagram = bytes(data)
                return len(data)

        transport = UdpTransport(
            "192.168.0.10",
            54321,
            "192.168.0.2",
            58766,
            command_datagram_size=0,
        )
        fake = FakeSocket()
        transport._socket = fake
        frame = pack_command(
            FineSUBControlCommand(0.0, 0.0, 0.0, 0.0, False),
            1,
        )
        transport.write(frame)
        self.assertEqual(len(fake.datagram), 37)
        self.assertEqual(fake.datagram, frame)
        self.assertEqual(fake.datagram[:2], b"\xAA\x00")
        self.assertEqual(fake.datagram[-1], 0xBB)

    def test_new_session_requires_confirmed_disarmed_frame(self) -> None:
        transport = DryRunTransport()
        link = FineSUBConnection(
            transport,
            session_id=0x12345678,
            logger=None,
        )
        requested = FineSUBControlCommand(0.1, 0.0, 0.0, 0.0, True)

        self.assertTrue(link.send(requested))
        handshake = unpack_command_frame(transport.writes[-1])
        self.assertFalse(handshake.command.armed)
        self.assertEqual(handshake.session_id, 0x12345678)

        transport.inject_read(pack_telemetry(telemetry_for_command(handshake)))
        link.poll_telemetry()
        self.assertTrue(link.session_confirmed)
        self.assertTrue(link.confirmation_fresh())
        self.assertFalse(link.armed_confirmation_fresh())

        self.assertTrue(link.send(requested))
        armed = unpack_command_frame(transport.writes[-1])
        self.assertTrue(armed.command.armed)
        transport.inject_read(pack_telemetry(telemetry_for_command(armed)))
        link.poll_telemetry()
        self.assertTrue(link.armed_confirmation_fresh())

    def test_rejected_echo_invalidates_confirmation(self) -> None:
        transport = DryRunTransport()
        link = FineSUBConnection(transport, session_id=9, logger=None)
        link.send(FineSUBControlCommand(0.0, 0.0, 0.0, 0.0, False))
        envelope = unpack_command_frame(transport.writes[-1])
        transport.inject_read(pack_telemetry(telemetry_for_command(envelope)))
        link.poll_telemetry()
        self.assertTrue(link.confirmation_fresh())

        link.send(FineSUBControlCommand(0.1, 0.0, 0.0, 0.0, True))
        rejected = unpack_command_frame(transport.writes[-1])
        transport.inject_read(
            pack_telemetry(telemetry_for_command(rejected, accepted=False))
        )
        link.poll_telemetry()
        self.assertFalse(link.session_confirmed)
        self.assertFalse(link.confirmation_fresh())

    def test_reconnect_cannot_be_confirmed_by_an_old_ack(self) -> None:
        transport = DryRunTransport()
        link = FineSUBConnection(
            transport,
            session_id=11,
            reconnect_interval_s=0.0,
            logger=None,
        )
        link.send(FineSUBControlCommand(0.0, 0.0, 0.0, 0.0, False))
        old_command = unpack_command_frame(transport.writes[-1])

        link.close()
        self.assertTrue(link.connect())
        transport.inject_read(pack_telemetry(telemetry_for_command(old_command)))
        link.poll_telemetry()

        self.assertFalse(link.session_confirmed)
        self.assertFalse(link.confirmation_fresh())

    def test_ack_older_than_confirmation_limit_is_ignored(self) -> None:
        transport = DryRunTransport()
        link = FineSUBConnection(
            transport,
            session_id=12,
            confirmation_max_age_s=0.25,
            logger=None,
        )
        with patch(
            "MPC_dual_model.finesub_transport.time.monotonic",
            return_value=100.0,
        ):
            link.send(FineSUBControlCommand(0.0, 0.0, 0.0, 0.0, False))
        delayed_command = unpack_command_frame(transport.writes[-1])
        transport.inject_read(
            pack_telemetry(telemetry_for_command(delayed_command))
        )

        with patch(
            "MPC_dual_model.finesub_transport.time.monotonic",
            return_value=100.3,
        ):
            link.poll_telemetry()

        self.assertFalse(link.session_confirmed)
        self.assertFalse(link.confirmation_fresh())


if __name__ == "__main__":
    unittest.main()
