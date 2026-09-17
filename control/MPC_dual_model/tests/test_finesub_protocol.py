import unittest
from dataclasses import replace

import numpy as np

from MPC_dual_model.finesub_protocol import (
    COMMAND_SYNC,
    COMMAND_TAIL,
    COMMAND_STATUS_ACCEPTED,
    COMMAND_FRAME_SIZE,
    TELEMETRY_FRAME_SIZE,
    FineSUBControlCommand,
    FineSUBHardwareAdapter,
    FineSUBTelemetry,
    TelemetryStreamDecoder,
    crc16_modbus,
    is_newer_telemetry,
    is_newer_u16_sequence,
    motor_throttles_to_channels,
    pack_command,
    pack_telemetry,
    unpack_command,
    unpack_command_frame,
    unpack_telemetry,
)


class FineSUBProtocolTest(unittest.TestCase):
    @staticmethod
    def _rpm_adapter(force_limit=10.0) -> FineSUBHardwareAdapter:
        return FineSUBHardwareAdapter(
            use_rpm_for_force_estimate=True,
            rpm_c1_positive=[1.0] * 8,
            rpm_c1_negative=[1.0] * 8,
            rpm_force_directions_frd=[
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            rpm_yaw_moment_arms=[0.2, -0.2, 0.0, 0.0, 0.0, 0.2, -0.2, 0.0],
            rpm_positive_force_limit=[force_limit] * 8,
            rpm_negative_force_limit=[force_limit] * 8,
        )

    def test_crc_matches_modbus_reference_vector(self) -> None:
        self.assertEqual(crc16_modbus(b"123456789"), 0x4B37)

    def test_physical_motor_calibration_command_round_trips_and_is_limited(self) -> None:
        command = FineSUBControlCommand(
            0.0,
            0.0,
            0.0,
            0.0,
            armed=True,
            calibration_motor_index=4,
            calibration_motor_throttle=0.10,
        )
        decoded = unpack_command_frame(
            pack_command(command, 7, session_id=9, sender_time_ms=11)
        )
        self.assertTrue(decoded.command.armed)
        self.assertEqual(decoded.command.calibration_motor_index, 4)
        self.assertAlmostEqual(decoded.command.calibration_motor_throttle, 0.10)
        with self.assertRaises(ValueError):
            FineSUBControlCommand(
                0.0,
                0.0,
                0.0,
                0.0,
                armed=True,
                calibration_motor_index=4,
                calibration_motor_throttle=0.1001,
            )

    def test_calibration_channel_command_round_trips_and_is_limited(self) -> None:
        command = FineSUBControlCommand(
            0.05,
            -0.075,
            0.10,
            -0.05,
            armed=True,
            calibration_channel=True,
        )
        decoded = unpack_command_frame(
            pack_command(command, 8, session_id=10, sender_time_ms=12)
        )
        self.assertTrue(decoded.command.armed)
        self.assertTrue(decoded.command.calibration_channel)
        np.testing.assert_allclose(
            (
                decoded.command.forward,
                decoded.command.right,
                decoded.command.down,
                decoded.command.yaw,
            ),
            (0.05, -0.075, 0.10, -0.05),
            atol=1.0e-7,
        )
        with self.assertRaises(ValueError):
            FineSUBControlCommand(
                0.1001,
                0.0,
                0.0,
                0.0,
                armed=True,
                calibration_channel=True,
            )

    def test_single_attitude_axis_calibration_round_trips(self) -> None:
        for axis in ("roll", "pitch", "yaw"):
            with self.subTest(axis=axis):
                command = FineSUBControlCommand(
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    armed=True,
                    yaw_direct=axis != "yaw",
                    calibration_attitude_axis=axis,
                )
                decoded = unpack_command_frame(
                    pack_command(command, 9, session_id=11, sender_time_ms=13)
                )
                self.assertEqual(decoded.command.calibration_attitude_axis, axis)
        with self.assertRaises(ValueError):
            FineSUBControlCommand(
                0.0,
                0.0,
                0.0,
                0.0,
                armed=True,
                yaw_direct=False,
                calibration_attitude_axis="pitch",
            )
        with self.assertRaises(ValueError):
            FineSUBControlCommand(
                0.01,
                0.0,
                0.0,
                0.0,
                armed=True,
                calibration_attitude_axis="roll",
            )
        with self.assertRaises(ValueError):
            FineSUBControlCommand(
                0.0,
                0.0,
                0.0,
                0.0,
                armed=True,
                calibration_motor_index=1,
                calibration_channel=True,
            )

    def test_mpc_wrench_round_trips_through_normalized_channels(self) -> None:
        adapter = FineSUBHardwareAdapter()
        command = adapter.convert((10.0, -7.5, 3.0), 2.0, armed=True)
        frame = pack_command(
            command,
            0x1234,
            session_id=0x10203040,
            sender_time_ms=123456,
        )
        self.assertEqual(len(frame), COMMAND_FRAME_SIZE)
        self.assertEqual(COMMAND_FRAME_SIZE, 37)
        self.assertEqual(frame[:2], COMMAND_SYNC)
        self.assertEqual(frame[1], 0)
        self.assertEqual(frame[-1], COMMAND_TAIL)
        decoded, sequence = unpack_command(frame)
        self.assertEqual(sequence, 0x1234)
        self.assertTrue(decoded.armed)
        self.assertTrue(decoded.yaw_direct)
        envelope = unpack_command_frame(frame)
        self.assertEqual(envelope.session_id, 0x10203040)
        self.assertEqual(envelope.sender_time_ms, 123456)

        telemetry = FineSUBTelemetry(
            sequence=sequence,
            tick_ms=100,
            state=2,
            armed=True,
            mpc_direct=True,
            yaw_direct=True,
            failsafe=False,
            yaw_rad=0.20,
            yaw_rate_rad_s=-0.05,
            depth_m=0.8,
            forward=decoded.forward,
            right=decoded.right,
            down=decoded.down,
            yaw=decoded.yaw,
        )
        force, moment = adapter.achieved_wrench(telemetry)
        np.testing.assert_allclose(force, (10.0, -7.5, 3.0), atol=1.0e-6)
        self.assertAlmostEqual(moment, 2.0, places=6)

    def test_asymmetric_wrench_scales_round_trip(self) -> None:
        adapter = FineSUBHardwareAdapter(
            positive_force_at_limit=(2.0, 3.0, 4.0),
            negative_force_at_limit=(2.5, 3.5, 4.5),
            translation_channel_limits=(0.10, 0.10, 0.10),
            positive_yaw_moment_at_limit=0.8,
            negative_yaw_moment_at_limit=1.0,
            yaw_channel_limit=0.10,
        )
        command = adapter.convert((1.0, -1.75, 4.0), -0.5, armed=True)
        np.testing.assert_allclose(
            (command.forward, command.right, command.down, command.yaw),
            (0.05, -0.05, 0.10, -0.05),
        )
        telemetry = FineSUBTelemetry(
            sequence=1,
            tick_ms=100,
            state=2,
            armed=True,
            mpc_direct=True,
            yaw_direct=True,
            failsafe=False,
            yaw_rad=0.0,
            yaw_rate_rad_s=0.0,
            depth_m=0.0,
            forward=command.forward,
            right=command.right,
            down=command.down,
            yaw=command.yaw,
        )
        force, moment = adapter.achieved_wrench(telemetry)
        np.testing.assert_allclose(force, (1.0, -1.75, 4.0), atol=1.0e-6)
        self.assertAlmostEqual(moment, -0.5, places=6)

    def test_stream_decoder_handles_noise_fragmentation_and_bad_crc(self) -> None:
        telemetry = FineSUBTelemetry(
            sequence=7,
            tick_ms=1234,
            state=2,
            armed=True,
            mpc_direct=True,
            yaw_direct=False,
            failsafe=False,
            yaw_rad=0.35,
            yaw_rate_rad_s=0.07,
            depth_m=1.2,
            forward=0.1,
            right=-0.2,
            down=0.3,
            yaw=-0.05,
        )
        frame = pack_telemetry(telemetry)
        self.assertEqual(len(frame), TELEMETRY_FRAME_SIZE)
        bad = bytearray(frame)
        bad[-1] ^= 0xFF

        decoder = TelemetryStreamDecoder()
        self.assertEqual(decoder.feed(b"noise" + bytes(bad) + frame[:9]), [])
        output = decoder.feed(frame[9:] + frame)
        self.assertEqual([item.sequence for item in output], [7, 7])
        self.assertFalse(output[0].yaw_direct)
        self.assertAlmostEqual(output[0].depth_m, 1.2, places=6)
        # The resynchronizer counts both the leading noise and bytes skipped
        # while scanning past the deliberately corrupted frame.
        self.assertGreaterEqual(decoder.dropped_bytes, 5)
        self.assertEqual(decoder.crc_errors, 1)

    def test_full_v4_telemetry_round_trip(self) -> None:
        telemetry = FineSUBTelemetry(
            sequence=17,
            tick_ms=9988,
            state=2,
            armed=True,
            mpc_direct=True,
            yaw_direct=True,
            failsafe=False,
            yaw_rad=0.4,
            yaw_rate_rad_s=-0.2,
            depth_m=1.3,
            forward=0.1,
            right=-0.2,
            down=0.3,
            yaw=-0.04,
            command_status=COMMAND_STATUS_ACCEPTED,
            last_command_session=0xAABBCCDD,
            last_command_sequence=9,
            last_command_crc=0x1234,
            last_command_sender_time_ms=55,
            command_count=19,
            rejected_command_count=2,
            quat_wxyz=(0.9, 0.1, 0.2, 0.3),
            angular_velocity_xyz=(0.01, 0.02, -0.2),
            linear_acceleration_xyz=(1.0, 2.0, 3.0),
            pressure_pa=12345.0,
            applied_motor_throttle=tuple(i / 20.0 for i in range(8)),
            motor_rpm=tuple(100.0 * i for i in range(8)),
            execution_feedback_valid=True,
            rpm_available=True,
            rpm_valid_mask=0xA5,
        )
        decoded = unpack_telemetry(pack_telemetry(telemetry))
        self.assertTrue(decoded.last_command_accepted)
        self.assertEqual(decoded.last_command_session, 0xAABBCCDD)
        self.assertEqual(decoded.last_command_crc, 0x1234)
        self.assertTrue(decoded.execution_feedback_valid)
        self.assertTrue(decoded.rpm_available)
        self.assertEqual(decoded.rpm_valid_mask, 0xA5)
        self.assertAlmostEqual(decoded.body_frd_yaw_rate_rad_s, 0.2)
        np.testing.assert_allclose(decoded.quat_wxyz, telemetry.quat_wxyz)
        np.testing.assert_allclose(decoded.motor_rpm, telemetry.motor_rpm)

    def test_applied_motor_feedback_is_inverted_into_achieved_wrench(self) -> None:
        translation = np.array([0.12, -0.08, 0.25])
        yaw = 0.06
        roll_pitch = np.array([0.03, -0.04])
        lower_mixer = np.array(
            [
                [-1.0, -1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0],
                [-1.0, 1.0, 1.0],
            ]
        )
        upper_mixer = np.array(
            [
                [-1.0, 1.0, 1.0],
                [1.0, 1.0, -1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )
        lower = lower_mixer @ np.array([yaw, translation[0], translation[1]])
        upper = upper_mixer @ np.array([roll_pitch[0], roll_pitch[1], translation[2]])
        physical = np.array(
            [
                lower[0],
                lower[1],
                upper[0],
                upper[1],
                upper[2],
                lower[2],
                lower[3],
                upper[3],
            ]
        )
        reconstructed, reconstructed_yaw, reconstructed_roll_pitch = (
            motor_throttles_to_channels(physical)
        )
        np.testing.assert_allclose(reconstructed, translation, atol=1e-12)
        self.assertAlmostEqual(reconstructed_yaw, yaw)
        np.testing.assert_allclose(reconstructed_roll_pitch, roll_pitch, atol=1e-12)

        adapter = FineSUBHardwareAdapter()
        telemetry = FineSUBTelemetry(
            sequence=1,
            tick_ms=2,
            state=2,
            armed=True,
            mpc_direct=True,
            yaw_direct=True,
            failsafe=False,
            yaw_rad=0.0,
            yaw_rate_rad_s=0.0,
            depth_m=0.0,
            forward=0.0,
            right=0.0,
            down=0.0,
            yaw=0.0,
            applied_motor_throttle=tuple(physical),
            execution_feedback_valid=True,
        )
        force, moment = adapter.achieved_wrench(telemetry)
        expected_force, expected_moment = adapter._channels_to_wrench(
            translation, yaw
        )
        np.testing.assert_allclose(force, expected_force, atol=1e-6)
        self.assertAlmostEqual(moment, expected_moment, places=6)

    def test_valid_rpm_reconstructs_achieved_force_and_yaw(self) -> None:
        adapter = self._rpm_adapter()
        one_rad_s_rpm = 60.0 / (2.0 * np.pi)
        rpm = [0.0] * 8
        rpm[0] = one_rad_s_rpm
        rpm[2] = one_rad_s_rpm
        telemetry = FineSUBTelemetry(
            sequence=1,
            tick_ms=2,
            state=2,
            armed=True,
            mpc_direct=True,
            yaw_direct=False,
            failsafe=False,
            yaw_rad=0.0,
            yaw_rate_rad_s=0.0,
            depth_m=0.0,
            forward=0.2,
            right=0.2,
            down=0.2,
            yaw=0.1,
            motor_rpm=tuple(rpm),
            rpm_available=True,
            rpm_valid_mask=0xFF,
        )
        force, moment = adapter.achieved_wrench(telemetry)
        np.testing.assert_allclose(force, (1.0, 0.0, 1.0), atol=1.0e-12)
        self.assertAlmostEqual(moment, 0.2, places=12)
        self.assertEqual(
            adapter.last_achieved_wrench_diagnostics["force_axis_sources"],
            ["rpm", "rpm", "rpm"],
        )
        self.assertEqual(
            adapter.last_achieved_wrench_diagnostics["yaw_source"], "rpm"
        )

    def test_invalid_horizontal_rpm_group_falls_back_without_discarding_vertical(self) -> None:
        adapter = self._rpm_adapter()
        one_rad_s_rpm = 60.0 / (2.0 * np.pi)
        rpm = [0.0] * 8
        rpm[2] = one_rad_s_rpm
        telemetry = FineSUBTelemetry(
            sequence=1,
            tick_ms=2,
            state=2,
            armed=True,
            mpc_direct=True,
            yaw_direct=False,
            failsafe=False,
            yaw_rad=0.0,
            yaw_rate_rad_s=0.0,
            depth_m=0.0,
            forward=0.1,
            right=-0.1,
            down=0.2,
            yaw=0.05,
            motor_rpm=tuple(rpm),
            rpm_available=True,
            rpm_valid_mask=sum(1 << index for index in (2, 3, 4, 7)),
        )
        force, moment = adapter.achieved_wrench(telemetry)
        command_force, command_moment = adapter._channels_to_wrench(
            (0.1, -0.1, 0.2), 0.05
        )
        np.testing.assert_allclose(force[:2], command_force[:2])
        self.assertAlmostEqual(force[2], 1.0, places=12)
        self.assertAlmostEqual(moment, command_moment)
        self.assertEqual(
            adapter.last_achieved_wrench_diagnostics["force_axis_sources"],
            ["command_echo", "command_echo", "rpm"],
        )

    def test_diagnostic_only_rpm_does_not_replace_applied_motor_force(self) -> None:
        control_adapter = self._rpm_adapter()
        control_adapter.use_rpm_for_force_estimate = False
        control_adapter.log_rpm_force_estimate = True
        one_rad_s_rpm = 60.0 / (2.0 * np.pi)
        translation = np.array([0.12, -0.08, 0.25])
        yaw = 0.06
        lower_mixer = np.array(
            [
                [-1.0, -1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0],
                [-1.0, 1.0, 1.0],
            ]
        )
        upper_mixer = np.array(
            [
                [-1.0, 1.0, 1.0],
                [1.0, 1.0, -1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )
        physical = np.zeros(8)
        physical[[0, 1, 5, 6]] = lower_mixer @ np.array(
            [yaw, translation[0], translation[1]]
        )
        physical[[2, 3, 4, 7]] = upper_mixer @ np.array(
            [0.0, 0.0, translation[2]]
        )
        rpm = [0.0] * 8
        rpm[0] = one_rad_s_rpm
        rpm[2] = one_rad_s_rpm
        telemetry = FineSUBTelemetry(
            sequence=1,
            tick_ms=2,
            state=2,
            armed=True,
            mpc_direct=True,
            yaw_direct=False,
            failsafe=False,
            yaw_rad=0.0,
            yaw_rate_rad_s=0.0,
            depth_m=0.0,
            forward=0.0,
            right=0.0,
            down=0.0,
            yaw=0.0,
            applied_motor_throttle=tuple(physical),
            execution_feedback_valid=True,
            motor_rpm=tuple(rpm),
            rpm_available=True,
            rpm_valid_mask=0xFF,
        )
        force, moment = control_adapter.achieved_wrench(telemetry)
        expected_force, expected_moment = control_adapter._channels_to_wrench(
            translation, yaw
        )
        np.testing.assert_allclose(force, expected_force, atol=1.0e-12)
        self.assertAlmostEqual(moment, expected_moment, places=12)
        self.assertEqual(
            control_adapter.last_achieved_wrench_diagnostics["force_axis_sources"],
            ["applied_motor_throttle"] * 3,
        )
        np.testing.assert_allclose(
            control_adapter.last_achieved_wrench_diagnostics["rpm_force_frd_n"],
            (1.0, 0.0, 1.0),
            atol=1.0e-12,
        )

    def test_rpm_force_estimate_is_clamped_to_calibrated_curve_limit(self) -> None:
        adapter = self._rpm_adapter(force_limit=2.0)
        rpm = [0.0] * 8
        rpm[0] = 1000.0
        telemetry = FineSUBTelemetry(
            sequence=1,
            tick_ms=2,
            state=2,
            armed=True,
            mpc_direct=True,
            yaw_direct=False,
            failsafe=False,
            yaw_rad=0.0,
            yaw_rate_rad_s=0.0,
            depth_m=0.0,
            forward=0.0,
            right=0.0,
            down=0.0,
            yaw=0.0,
            motor_rpm=tuple(rpm),
            rpm_available=True,
            rpm_valid_mask=0xFF,
        )
        force, moment = adapter.achieved_wrench(telemetry)
        np.testing.assert_allclose(force, (2.0, 0.0, 0.0))
        self.assertAlmostEqual(moment, 0.4)

    def test_sequence_order_handles_duplicates_old_frames_and_wrap(self) -> None:
        self.assertTrue(is_newer_u16_sequence(11, 10))
        self.assertFalse(is_newer_u16_sequence(10, 10))
        self.assertFalse(is_newer_u16_sequence(9, 10))
        self.assertTrue(is_newer_u16_sequence(0, 0xFFFF))
        self.assertFalse(is_newer_u16_sequence(0xFFFF, 0))

        previous = FineSUBTelemetry(
            sequence=100,
            tick_ms=5000,
            state=2,
            armed=True,
            mpc_direct=True,
            yaw_direct=True,
            failsafe=False,
            yaw_rad=0.0,
            yaw_rate_rad_s=0.0,
            depth_m=0.5,
            forward=0.0,
            right=0.0,
            down=0.0,
            yaw=0.0,
        )
        self.assertFalse(is_newer_telemetry(replace(previous), previous))
        self.assertFalse(
            is_newer_telemetry(replace(previous, sequence=99), previous)
        )
        self.assertTrue(
            is_newer_telemetry(
                replace(
                    previous,
                    sequence=0,
                    tick_ms=20,
                    state=0,
                    armed=False,
                ),
                previous,
            )
        )

    def test_disarmed_or_failsafe_telemetry_maps_to_zero_wrench(self) -> None:
        adapter = FineSUBHardwareAdapter()
        active = FineSUBTelemetry(
            sequence=1,
            tick_ms=50,
            state=2,
            armed=True,
            mpc_direct=True,
            yaw_direct=True,
            failsafe=False,
            yaw_rad=0.0,
            yaw_rate_rad_s=0.0,
            depth_m=0.5,
            forward=0.2,
            right=-0.1,
            down=0.3,
            yaw=0.1,
        )
        for telemetry in (
            replace(active, armed=False),
            replace(active, failsafe=True),
            replace(active, mpc_direct=False),
        ):
            with self.subTest(telemetry=telemetry):
                force, moment = adapter.achieved_wrench(telemetry)
                np.testing.assert_array_equal(force, np.zeros(3))
                self.assertEqual(moment, 0.0)

    def test_nonfinite_telemetry_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            FineSUBTelemetry(
                sequence=1,
                tick_ms=50,
                state=2,
                armed=True,
                mpc_direct=True,
                yaw_direct=True,
                failsafe=False,
                yaw_rad=float("nan"),
                yaw_rate_rad_s=0.0,
                depth_m=0.5,
                forward=0.0,
                right=0.0,
                down=0.0,
                yaw=0.0,
            )


if __name__ == "__main__":
    unittest.main()
