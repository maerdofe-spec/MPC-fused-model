import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from MPC_dual_model.vision_measurement import (
    PipelineJsonlTail,
    VisionGateConfig,
    VisionMeasurementGate,
)


def record(
    frame: int,
    time_s: float,
    position=(0.0, 0.0, 0.5),
    *,
    confidence=0.8,
    depth_confidence=0.8,
    depth_nis=1.0,
):
    return {
        "frame_idx": frame,
        "frame_ts_mean": time_s,
        "result_time": time_s + 0.02,
        "tracker_state": "YOLO_ONLY",
        "tracks": [
            {
                "position": list(position),
                "confidence": confidence,
                "depth_valid": True,
                "depth_confidence": depth_confidence,
                "depth_nis": depth_nis,
                "depth_filter_mode": "update",
            }
        ],
    }


class PipelineJsonlTailTests(unittest.TestCase):
    def test_reads_only_complete_new_json_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vision.jsonl"
            path.write_text(json.dumps({"frame_idx": 1}) + "\n", encoding="utf-8")
            tail = PipelineJsonlTail(path, start_at_end=False)
            self.assertEqual(tail.poll(), [{"frame_idx": 1}])

            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"frame_idx": 2}))
            self.assertEqual(tail.poll(), [])
            with path.open("a", encoding="utf-8") as handle:
                handle.write("\n")
            self.assertEqual(tail.poll(), [{"frame_idx": 2}])

    def test_defaults_to_ignoring_historical_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vision.jsonl"
            path.write_text(json.dumps({"frame_idx": 1}) + "\n", encoding="utf-8")
            tail = PipelineJsonlTail(path)
            self.assertEqual(tail.poll(), [])
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"frame_idx": 2}) + "\n")
            self.assertEqual(tail.poll(), [{"frame_idx": 2}])


class VisionMeasurementGateTests(unittest.TestCase):
    def setUp(self):
        self.config = VisionGateConfig(
            startup_confirmation_samples=3,
            reacquire_confirmation_samples=5,
        )
        self.gate = VisionMeasurementGate(self.config)

    def evaluate(self, value):
        return self.gate.evaluate(value, now_s=value["result_time"] + 0.01)

    def lock(self):
        decisions = [
            self.evaluate(record(index, 10.0 + index * 0.1, (0.01 * index, 0.0, 0.5)))
            for index in range(1, 4)
        ]
        self.assertEqual([item.control_ready for item in decisions], [False, False, True])

    def test_startup_requires_consecutive_measurements(self):
        self.lock()

    def test_rejects_quality_timestamp_and_depth_failures(self):
        cases = (
            (record(1, 10.0, confidence=0.2), "low_confidence"),
            (record(2, 10.1, depth_confidence=0.1), "low_depth_confidence"),
            (record(3, 10.2, depth_nis=1859.0), "depth_nis_outlier"),
        )
        for value, expected in cases:
            with self.subTest(expected=expected):
                decision = self.evaluate(value)
                self.assertFalse(decision.control_ready)
                self.assertEqual(decision.reason, expected)

        stale = record(4, 10.3)
        decision = self.gate.evaluate(stale, now_s=stale["result_time"] + 1.0)
        self.assertEqual(decision.reason, "stale_result")

        too_far = record(5, 10.4, position=(0.0, 0.0, 1.51))
        self.assertEqual(self.evaluate(too_far).reason, "forward_range")

    def test_accept_any_finite_track_output_bypasses_quality_and_motion_gates(self):
        gate = VisionMeasurementGate(
            self.config,
            accept_any_finite_track_output=True,
        )
        first = record(
            1,
            10.0,
            position=(5.0, -4.0, 0.01),
            confidence=0.0,
            depth_confidence=0.0,
            depth_nis=9999.0,
        )
        first["tracks"][0]["depth_valid"] = False
        first["tracks"][0]["depth_filter_mode"] = "weak_update"
        decision = gate.evaluate(first, now_s=first["result_time"] + 0.01)
        self.assertTrue(decision.control_ready)
        self.assertEqual(decision.reason, "track_output")

        jumped = record(2, 10.01, position=(-6.0, 7.0, 8.0))
        decision = gate.evaluate(jumped, now_s=jumped["result_time"] + 0.01)
        self.assertTrue(decision.control_ready)
        self.assertEqual(decision.confirmation_count, 1)

    def test_weak_update_can_be_explicitly_accepted(self):
        config = VisionGateConfig(
            startup_confirmation_samples=1,
            reacquire_confirmation_samples=1,
            accepted_depth_filter_modes=("update", "weak_update"),
        )
        gate = VisionMeasurementGate(config)
        value = record(1, 10.0)
        value["tracks"][0]["depth_filter_mode"] = "weak_update"
        decision = gate.evaluate(value, now_s=10.02)
        self.assertTrue(decision.control_ready, decision.reason)
        self.assertEqual(decision.reason, "startup_confirmed")

    def test_implausible_step_can_be_clamped_instead_of_dropped(self):
        config = VisionGateConfig(
            max_forward_m=3.0,
            jump_margin_m=0.05,
            max_speed_m_s=1.0,
            startup_confirmation_samples=1,
            reacquire_confirmation_samples=1,
            clamp_implausible_steps=True,
        )
        gate = VisionMeasurementGate(config)
        first = record(1, 10.0, position=(0.0, 0.0, 0.5))
        self.assertTrue(gate.evaluate(first, now_s=10.02).control_ready)
        jumped = record(2, 10.1, position=(0.0, 0.0, 2.0))
        decision = gate.evaluate(jumped, now_s=10.12)
        self.assertTrue(decision.control_ready, decision.reason)
        self.assertEqual(decision.reason, "tracking_clamped")
        self.assertAlmostEqual(decision.measurement.position_camera_xyz_m[2], 0.65)

    def test_rejects_four_frame_depth_excursion_and_recovers(self):
        self.lock()
        frame = 4
        time_s = 10.4
        for index in range(4):
            bad = record(
                frame + index,
                time_s + index * 0.1,
                (0.55, -0.28, 1.84),
                confidence=0.7,
                depth_nis=1.0,
            )
            decision = self.evaluate(bad)
            self.assertFalse(decision.control_ready)

        return_jump = record(
            frame + 4,
            time_s + 0.4,
            (0.04, 0.0, 0.48),
            depth_nis=1757.0,
        )
        self.assertEqual(self.evaluate(return_jump).reason, "depth_nis_outlier")

        recovered = []
        for index in range(5):
            value = record(
                frame + 5 + index,
                time_s + 0.5 + index * 0.1,
                (0.04 + 0.005 * index, 0.0, 0.48),
            )
            recovered.append(self.evaluate(value))
        self.assertEqual(
            [item.control_ready for item in recovered],
            [False, False, False, False, True],
        )
        self.assertEqual(recovered[-1].reason, "reacquired")
        np.testing.assert_allclose(
            recovered[-1].measurement.position_camera_xyz_m,
            (0.06, 0.0, 0.48),
        )

    def test_duplicate_does_not_destroy_pending_confirmation(self):
        first = record(1, 10.1)
        self.assertEqual(self.evaluate(first).confirmation_count, 1)
        duplicate = self.evaluate(first)
        self.assertEqual(duplicate.reason, "duplicate_or_out_of_order")
        second = self.evaluate(record(2, 10.2))
        self.assertEqual(second.confirmation_count, 2)

    def test_malformed_track_confidence_fails_closed_without_crashing(self):
        value = record(1, 10.1)
        value["tracks"].insert(0, {"confidence": None})
        decision = self.evaluate(value)
        self.assertFalse(decision.control_ready)
        self.assertEqual(decision.reason, "startup_pending")


if __name__ == "__main__":
    unittest.main()
