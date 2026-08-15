from pathlib import Path
import json
import tempfile
import time
import unittest
from unittest.mock import patch

from MPC_dual_model.auto_only_runtime import run_auto_only
from MPC_dual_model.auto_readiness import evaluate_auto_readiness
from MPC_dual_model.finesub_protocol import (
    COMMAND_STATUS_ACCEPTED,
    FineSUBTelemetry,
)

from MPC_dual_model.tests.test_auto_readiness import valid_config


def vision_record(frame: int, *, valid: bool = True) -> dict:
    acquired = time.time() - 0.02
    return {
        "frame_idx": frame,
        "frame_ts_mean": acquired,
        "result_time": acquired + 0.01,
        "tracks": []
        if not valid
        else [
            {
                "position": [0.0, 0.0, 0.50],
                "confidence": 0.9,
                "depth_valid": True,
                "depth_confidence": 0.9,
                "depth_nis": 1.0,
                "depth_filter_mode": "update",
            }
        ],
    }


class FakeTail:
    """Emit five valid frames, then a target-loss record."""

    def __init__(self, *_args, **_kwargs):
        self.poll_count = 0

    def poll(self):
        self.poll_count += 1
        if self.poll_count == 1:  # Runtime's explicit establish-EOF call.
            return []
        frame = self.poll_count - 1
        return [vision_record(frame, valid=frame <= 5)]


class FakeAlternatingTail:
    """Alternate finite tracks and empty processed frames."""

    def __init__(self, *_args, **_kwargs):
        self.poll_count = 0

    def poll(self):
        self.poll_count += 1
        if self.poll_count == 1:
            return []
        frame = self.poll_count - 1
        return [vision_record(frame, valid=frame % 2 == 1)]


class FakeConnection:
    """Immediate positive echoes without opening a real transport."""

    instances = []

    def __init__(self, _transport, **_kwargs):
        self.session_id = 123
        self.session_confirmed = False
        self.last_confirmed_armed = None
        self.armed = False
        self.yaw_direct = True
        self.commands = []
        self.closed = False
        self.__class__.instances.append(self)

    def send(self, command):
        self.commands.append(command)
        self.session_confirmed = True
        self.last_confirmed_armed = command.armed
        self.armed = command.armed
        self.yaw_direct = command.yaw_direct
        return True

    def fresh_telemetry(self):
        return FineSUBTelemetry(
            sequence=len(self.commands),
            tick_ms=len(self.commands) * 50,
            state=1,
            armed=self.armed,
            mpc_direct=True,
            yaw_direct=self.yaw_direct,
            failsafe=False,
            yaw_rad=0.0,
            yaw_rate_rad_s=0.0,
            depth_m=0.0,
            forward=0.0,
            right=0.0,
            down=0.0,
            yaw=0.0,
            command_status=COMMAND_STATUS_ACCEPTED,
            last_command_session=self.session_id,
            execution_feedback_valid=True,
            applied_motor_throttle=(0.0,) * 8,
        )

    def confirmation_fresh(self):
        return self.session_confirmed

    def armed_confirmation_fresh(self):
        return self.session_confirmed and self.last_confirmed_armed is True

    def poll_telemetry(self):
        return None

    def close(self):
        self.closed = True


class AutoOnlyRuntimeTests(unittest.TestCase):
    def test_translation_model_uses_lower_controller_local_yaw_hold(self):
        with tempfile.TemporaryDirectory() as directory:
            config, config_path = valid_config(Path(directory))
            config["auto_runtime"]["required_model"] = "dual"
            config["auto_runtime"]["active_yaw_parameters"][
                "enabled_for_control"
            ] = False
            report = evaluate_auto_readiness(
                config, config_path=config_path, selected_model="dual"
            )
            self.assertTrue(report.ready, report.blockers)
            FakeConnection.instances.clear()
            trace_path = Path(directory) / "translation.jsonl"
            with (
                patch(
                    "MPC_dual_model.auto_only_runtime.make_transport",
                    return_value=object(),
                ),
                patch(
                    "MPC_dual_model.auto_only_runtime.FineSUBConnection",
                    FakeConnection,
                ),
                patch(
                    "MPC_dual_model.auto_only_runtime.PipelineJsonlTail",
                    FakeTail,
                ),
            ):
                exit_code = run_auto_only(
                    config,
                    report,
                    execute=True,
                    max_runtime_s=1.0,
                    logger=lambda _message: None,
                    trace_jsonl_path=trace_path,
                )

            self.assertEqual(exit_code, 3)
            link = FakeConnection.instances[-1]
            self.assertTrue(link.commands)
            self.assertTrue(all(not command.yaw_direct for command in link.commands))
            self.assertTrue(all(command.yaw == 0.0 for command in link.commands))
            trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
            update = next(item for item in trace if item["event"] == "control_update")
            self.assertEqual(update["requested_yaw_moment_n_m"], 0.0)
            self.assertEqual(update["requested_yaw_channel"], 0.0)
            self.assertEqual(update["yaw_control_mode"], "lower_local_hold")

    def test_valid_sequence_reaches_mpc_then_target_loss_latches_disarm(self):
        with tempfile.TemporaryDirectory() as directory:
            config, config_path = valid_config(Path(directory))
            report = evaluate_auto_readiness(config, config_path=config_path)
            self.assertTrue(report.ready, report.blockers)
            FakeConnection.instances.clear()
            messages = []
            trace_path = Path(directory) / "experiment.jsonl"
            with (
                patch(
                    "MPC_dual_model.auto_only_runtime.make_transport",
                    return_value=object(),
                ),
                patch(
                    "MPC_dual_model.auto_only_runtime.FineSUBConnection",
                    FakeConnection,
                ),
                patch(
                    "MPC_dual_model.auto_only_runtime.PipelineJsonlTail",
                    FakeTail,
                ),
            ):
                exit_code = run_auto_only(
                    config,
                    report,
                    execute=True,
                    max_runtime_s=1.0,
                    logger=messages.append,
                    trace_jsonl_path=trace_path,
                )

            self.assertEqual(exit_code, 3)
            link = FakeConnection.instances[-1]
            self.assertTrue(link.closed)
            self.assertTrue(any(command.armed for command in link.commands))
            self.assertTrue(all(command.yaw_direct for command in link.commands))
            self.assertTrue(
                any(
                    command.armed
                    and max(abs(command.forward), abs(command.right), abs(command.down))
                    > 0.0
                    for command in link.commands
                )
            )
            self.assertTrue(all(not command.armed for command in link.commands[-3:]))
            self.assertTrue(any("vision gate rejected" in item for item in messages))
            trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
            self.assertTrue(any(item["event"] == "control_update" for item in trace))
            update = next(item for item in trace if item["event"] == "control_update")
            self.assertIn("requested_yaw_moment_n_m", update)
            self.assertIn("mpc_predicted_yaw_moment_n_m", update)
            self.assertIn("achieved_yaw_moment_previous_n_m", update)
            self.assertEqual(trace[-1]["event"], "stop")
            self.assertIn("vision gate rejected", trace[-1]["fatal_reason"])

    def test_experimental_vision_loss_disarms_and_waits_for_reacquisition(self):
        with tempfile.TemporaryDirectory() as directory:
            config, config_path = valid_config(Path(directory))
            report = evaluate_auto_readiness(config, config_path=config_path)
            self.assertTrue(report.ready, report.blockers)
            FakeConnection.instances.clear()
            trace_path = Path(directory) / "reacquire.jsonl"
            with (
                patch(
                    "MPC_dual_model.auto_only_runtime.make_transport",
                    return_value=object(),
                ),
                patch(
                    "MPC_dual_model.auto_only_runtime.FineSUBConnection",
                    FakeConnection,
                ),
                patch(
                    "MPC_dual_model.auto_only_runtime.PipelineJsonlTail",
                    FakeTail,
                ),
            ):
                exit_code = run_auto_only(
                    config,
                    report,
                    execute=True,
                    max_runtime_s=0.4,
                    logger=lambda _message: None,
                    trace_jsonl_path=trace_path,
                    reacquire_on_vision_loss=True,
                )

            self.assertEqual(exit_code, 0)
            link = FakeConnection.instances[-1]
            self.assertTrue(any(command.armed for command in link.commands))
            self.assertTrue(any(not command.armed for command in link.commands))
            trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
            self.assertTrue(any(item["event"] == "vision_reacquire" for item in trace))
            self.assertEqual(trace[-1]["event"], "stop")
            self.assertIsNone(trace[-1]["fatal_reason"])

    def test_accept_any_track_ignores_interleaved_empty_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            config, config_path = valid_config(Path(directory))
            report = evaluate_auto_readiness(config, config_path=config_path)
            self.assertTrue(report.ready, report.blockers)
            FakeConnection.instances.clear()
            trace_path = Path(directory) / "empty-frames.jsonl"
            with (
                patch(
                    "MPC_dual_model.auto_only_runtime.make_transport",
                    return_value=object(),
                ),
                patch(
                    "MPC_dual_model.auto_only_runtime.FineSUBConnection",
                    FakeConnection,
                ),
                patch(
                    "MPC_dual_model.auto_only_runtime.PipelineJsonlTail",
                    FakeAlternatingTail,
                ),
            ):
                exit_code = run_auto_only(
                    config,
                    report,
                    execute=True,
                    max_runtime_s=0.3,
                    logger=lambda _message: None,
                    trace_jsonl_path=trace_path,
                    reacquire_on_vision_loss=True,
                    accept_any_vision_track=True,
                )

            self.assertEqual(exit_code, 0)
            trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
            self.assertTrue(
                any(item["event"] == "vision_empty_ignored" for item in trace)
            )
            self.assertTrue(any(item["event"] == "control_update" for item in trace))

    def test_experimental_holds_fossen_force_and_locks_first_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            config, config_path = valid_config(Path(directory))
            report = evaluate_auto_readiness(config, config_path=config_path)
            self.assertTrue(report.ready, report.blockers)
            FakeConnection.instances.clear()
            trace_path = Path(directory) / "armed-hold.jsonl"
            with (
                patch(
                    "MPC_dual_model.auto_only_runtime.make_transport",
                    return_value=object(),
                ),
                patch(
                    "MPC_dual_model.auto_only_runtime.FineSUBConnection",
                    FakeConnection,
                ),
                patch(
                    "MPC_dual_model.auto_only_runtime.PipelineJsonlTail",
                    FakeTail,
                ),
            ):
                exit_code = run_auto_only(
                    config,
                    report,
                    execute=True,
                    max_runtime_s=0.6,
                    logger=lambda _message: None,
                    trace_jsonl_path=trace_path,
                    accept_any_vision_track=True,
                    hold_armed_on_vision_loss=True,
                    lock_reference_to_first_measurement=True,
                )

            self.assertEqual(exit_code, 0)
            link = FakeConnection.instances[-1]
            first_armed = next(
                index for index, command in enumerate(link.commands) if command.armed
            )
            self.assertTrue(
                all(command.armed for command in link.commands[first_armed:-3])
            )
            trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
            locked = next(item for item in trace if item["event"] == "reference_locked")
            update = next(item for item in trace if item["event"] == "control_update")
            self.assertEqual(
                locked["reference_position_body_frd_m"],
                update["position_body_frd_m"],
            )
            self.assertEqual(
                update["reference_position_body_frd_m"],
                update["position_body_frd_m"],
            )
            self.assertTrue(any(item["event"] == "vision_hold_started" for item in trace))
            self.assertTrue(
                any(command.armed and command.down > 0.0 for command in link.commands)
            )
            self.assertIn("model1_fixed_base_force_frd_n", update)
            self.assertEqual(update["fossen_restoring_force_frd_n"], [0.0, 0.0, 0.3])
            self.assertIn("mpc_planned_delta_force_first3_frd_n", update)


if __name__ == "__main__":
    unittest.main()
