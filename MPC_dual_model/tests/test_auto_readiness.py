import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from MPC_dual_model.auto_only_runtime import main as auto_main
from MPC_dual_model.auto_readiness import evaluate_auto_readiness


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_config(directory: Path) -> tuple[dict, Path]:
    vision = directory / "pipeline_results.jsonl"
    frozen = directory / "frozen.jsonl"
    analysis = directory / "analysis.json"
    vision.write_text("", encoding="utf-8")
    frozen.write_text('{"record": 1}\n', encoding="utf-8")
    analysis.write_text('{"accepted": true}\n', encoding="utf-8")
    config_path = directory / "runtime.json"
    gate = {
        "max_result_age_s": 0.25,
        "max_pipeline_delay_s": 0.15,
        "min_confidence": 0.50,
        "min_depth_confidence": 0.20,
        "max_depth_nis": 25.0,
        "forward_range_m": [0.15, 1.50],
        "max_speed_m_s": 1.0,
        "jump_margin_m": 0.10,
        "max_inter_sample_gap_s": 0.50,
        "startup_confirmation_samples": 3,
        "reacquire_confirmation_samples": 5,
    }
    active = {
        "enabled_for_control": True,
        "model_family": "dual",
        "model": {
            "effective_mass_matrix_kg": [
                [10.0, 0.0, 0.0],
                [0.0, 11.0, 0.0],
                [0.0, 0.0, 12.0],
            ],
            "linear_damping_matrix_n_s_per_m": [
                [5.0, 0.0, 0.0],
                [0.0, 6.0, 0.0],
                [0.0, 0.0, 7.0],
            ],
            "sample_period_s": 0.05,
            "restoring_force_frd_n": [0.0, 0.0, 0.3],
        },
        "kalman": {
            "position_std": [0.02, 0.02, 0.03],
            "acceleration_std": [0.3, 0.3, 0.4],
            "initial_position_std": [0.04, 0.04, 0.06],
            "initial_velocity_std": [0.3, 0.3, 0.4],
        },
        "baseline_adaptation": {
            "enabled": True,
            "update_mode": "gated_ema",
            "axis_enabled": [True, True, True],
            "adaptation_rate": 0.02,
            "transient_adaptation_rate": 0.08,
            "steady_position_error_tolerance": 0.03,
            "steady_velocity_tolerance": 0.02,
            "position_error_tolerance": 0.20,
            "velocity_tolerance": 0.20,
            "matched_motion_axis_enabled": [False, False, False],
            "matched_motion_min_velocity": 0.05,
            "matched_motion_max_velocity": 0.20,
            "matched_motion_acceleration_tolerance": 0.20,
            "matched_motion_confirmation_updates": 3,
        },
        "controller": {
            "horizon": 2,
            "reference_position": [0.8, 0.0, 0.0],
            "position_weights": [10.0, 10.0, 10.0],
            "velocity_weights": [1.0, 1.0, 1.0],
            "terminal_weight_scale": 2.0,
            "force_weights": [0.1, 0.1, 0.1],
            "delta_force_weights": [0.2, 0.2, 0.2],
            "force_min": [-2.0, -2.0, -2.0],
            "force_max": [2.0, 2.0, 2.0],
            "delta_force_min": [-0.5, -0.5, -0.5],
            "delta_force_max": [0.5, 0.5, 0.5],
            "thruster_command_matrix": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "thruster_command_limit": 2.0,
            "thruster_command_min": [-2.0, -2.0, -2.0],
            "thruster_command_max": [2.0, 2.0, 2.0],
            "forward_distance_min": 0.25,
            "forward_distance_max": 1.50,
            "horizontal_half_fov_deg": 42.0,
            "vertical_half_fov_deg": 30.0,
            "fov_margin_deg": 5.0,
            "slack_quadratic_weight": 20000.0,
            "slack_linear_weight": 50.0,
            "slack_max": 5.0,
            "forward_axis": 0,
            "horizontal_axis": 1,
            "vertical_axis": 2,
            "solver_settings": {
                "backend": "osqp",
                "rho": 10.0,
                "sigma": 1e-8,
                "max_iterations": 1500,
                "absolute_tolerance": 2e-5,
                "relative_tolerance": 3e-4,
                "adaptive_rho": True,
                "adaptive_interval": 25,
                "adaptive_tolerance": 5.0,
                "rho_min": 1e-4,
                "rho_max": 1e5,
                "time_limit_seconds": 0.035,
                "check_termination_interval": 10,
            },
        },
        "fusion": {
            "window": 2,
            "prediction_horizon": 1,
            "forgetting_factor": 0.8,
            "horizon_weight_decay": 0.95,
            "weight_update_rate": 0.35,
            "epsilon": 1e-10,
            "indistinguishable_score_threshold": [1e-7, 1e-7, 1e-7],
            "prediction_horizon_weights": [1.0],
            "staircase_horizon_caps": [1, 1],
            "initial_model1_weight": [0.8, 0.8, 0.8],
            "minimum_weight": 0.01,
            "position_error_clip": [0.5, 0.5, 0.5],
        },
    }
    active_yaw = {
        "enabled_for_control": True,
        "dynamics": {
            "effective_inertia_kg_m2": 0.35,
            "linear_damping_n_m_per_rad_s": 0.30,
            "quadratic_damping_n_m_per_rad_s2": 0.0,
        },
        "controller": {
            "alpha_on": 0.05235987755982989,
            "alpha_off": 0.020943951023931952,
            "alpha_emergency": 0.13962634015954636,
            "trigger_frames": 3,
            "settle_frames": 5,
            "require_outward_motion_to_trigger": False,
            "yaw_tolerance": 0.03490658503988659,
            "omega_tolerance": 0.03490658503988659,
            "outer_kp": 1.5,
            "outer_ki": 0.0,
            "outer_kd": 0.2,
            "outer_integral_limit": 0.5235987755982988,
            "inner_kp": 0.6,
            "inner_ki": 0.02,
            "inner_kd": 0.0,
            "inner_integral_limit": 1.0471975511965976,
            "omega_command_max": 0.5235987755982988,
            "omega_command_acceleration_max": 1.0471975511965976,
            "yaw_moment_min": -1.0,
            "yaw_moment_max": 1.0,
            "delta_yaw_moment_min": -0.25,
            "delta_yaw_moment_max": 0.25,
            "use_dynamics_feedforward": False,
        },
        "actuator_model": {
            "enabled": True,
            "thruster_force_limit_scale": 0.1,
        },
    }
    config = {
        "auto_runtime": {
            "mode": "AUTO_ONLY",
            "enabled": True,
            "required_model": "dual-yaw",
            "require_execute_flag": True,
            "legacy_joystick_csrt_entry_allowed_for_auto": False,
            "vision_jsonl": str(vision),
            "active_camera_transform": {
                "enabled_for_control": True,
                "rotation_body_from_camera": [
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                "camera_origin_in_body_frd_m": [0.2, 0.0, -0.12],
            },
            "active_mpc_parameters": active,
            "active_yaw_parameters": active_yaw,
        },
        "control": {
            "protocol_version": 5,
            "period_sec": 0.05,
            "telemetry_max_age_sec": 0.25,
            "confirmation_max_age_sec": 0.25,
            "calibration_max_channel_abs": 0.10,
        },
        "transport": {
            "type": "udp",
            "bind_host": "127.0.0.1",
            "bind_port": 54321,
            "remote_host": "127.0.0.2",
            "remote_port": 58766,
            "reconnect_interval_sec": 0.0,
        },
        "hardware_adapter": {
            "positive_force_at_limit": [2.0, 2.0, 2.0],
            "negative_force_at_limit": [2.0, 2.0, 2.0],
            "translation_channel_limits": [0.1, 0.1, 0.1],
            "translation_signs": [1.0, 1.0, 1.0],
            "positive_yaw_moment_at_limit": 1.0,
            "negative_yaw_moment_at_limit": 1.0,
            "yaw_channel_limit": 0.1,
            "yaw_sign": 1.0,
        },
        "thruster_feedback": {
            "use_rpm_for_force_estimate": False,
            "log_rpm_force_estimate": True,
            "rpm_force_prior": {
                "enabled_for_control": False,
                "enabled_for_diagnostics": True,
                "same_physical_vehicle_confirmed_by_operator_20260812": True,
                "c1_positive_n_per_rad_s_sq_m1_m8": [1e-4] * 8,
                "c1_negative_abs_n_per_rad_s_sq_m1_m8": [1e-4] * 8,
                "positive_force_limit_prior_n_m1_m8": [3.0] * 8,
                "negative_force_limit_abs_prior_n_m1_m8": [3.0] * 8,
            }
        },
        "thruster_geometry": {
            "positive_throttle_force_directions_frd_m1_m8": [
                [-0.707106781, -0.707106781, 0.0],
                [-0.707106781, 0.707106781, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0],
                [-0.707106781, 0.707106781, 0.0],
                [0.707106781, 0.707106781, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "yaw_moment_arm_about_cad_origin_m_per_positive_force_m1_m8": [
                -0.2,
                -0.2,
                0.0,
                0.0,
                0.0,
                0.2,
                -0.2,
                0.0,
            ],
        },
        "camera_calibration_prior": {
            "enabled_for_mpc_state_correction": True,
            "onboard_stereo_udp5600_historical_real_calibration": {
                "enabled_for_control": True,
                "external_red_fish_pipeline_20260812": {
                    "enabled_for_control": True,
                    "mpc_input_gate": gate,
                    "rigid_target_cross_check_20260813": {
                        "acceptance_gates_passed": True,
                        "enabled_for_control": True,
                        "frozen_front_jsonl": frozen.name,
                        "frozen_front_jsonl_sha256": _digest(frozen),
                        "analysis_file": analysis.name,
                        "analysis_file_sha256": _digest(analysis),
                    },
                },
            },
        },
        "real_vehicle_calibration_candidates": {
            "enabled_for_control": True,
            "unresolved_gates": [],
        },
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config, config_path


class AutoReadinessTests(unittest.TestCase):
    def test_complete_approved_config_is_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            config, path = valid_config(Path(directory))
            report = evaluate_auto_readiness(config, config_path=path)
            self.assertTrue(report.ready, report.blockers)
            self.assertEqual(report.blockers, ())

    def test_each_control_authority_flag_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            config, path = valid_config(Path(directory))
            cases = (
                (("auto_runtime", "enabled"), "auto_runtime.enabled"),
                (
                    ("camera_calibration_prior", "enabled_for_mpc_state_correction"),
                    "camera calibration",
                ),
                (
                    (
                        "real_vehicle_calibration_candidates",
                        "enabled_for_control",
                    ),
                    "real-vehicle calibration set",
                ),
            )
            for keys, expected in cases:
                candidate = copy.deepcopy(config)
                target = candidate
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = False
                with self.subTest(keys=keys):
                    report = evaluate_auto_readiness(candidate, config_path=path)
                    self.assertFalse(report.ready)
                    self.assertTrue(any(expected in item for item in report.blockers))

    def test_weaker_vision_range_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config, path = valid_config(Path(directory))
            pipeline = config["camera_calibration_prior"][
                "onboard_stereo_udp5600_historical_real_calibration"
            ]["external_red_fish_pipeline_20260812"]
            pipeline["mpc_input_gate"]["forward_range_m"][1] = 2.5
            report = evaluate_auto_readiness(config, config_path=path)
            self.assertFalse(report.ready)
            self.assertTrue(any("forward range" in item for item in report.blockers))

    def test_missing_yaw_authority_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            config, path = valid_config(Path(directory))
            config["auto_runtime"]["active_yaw_parameters"][
                "enabled_for_control"
            ] = False
            report = evaluate_auto_readiness(config, config_path=path)
            self.assertFalse(report.ready)
            self.assertTrue(
                any("active_yaw_parameters" in item for item in report.blockers)
            )

    def test_weaker_runtime_timeouts_and_non_udp_transport_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config, path = valid_config(Path(directory))
            config["control"]["telemetry_max_age_sec"] = 0.5
            config["transport"]["type"] = "tcp"
            report = evaluate_auto_readiness(config, config_path=path)
            self.assertFalse(report.ready)
            self.assertTrue(any("telemetry maximum age" in item for item in report.blockers))
            self.assertTrue(any("UDP transport" in item for item in report.blockers))

    def test_evidence_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config, path = valid_config(Path(directory))
            rigid = config["camera_calibration_prior"][
                "onboard_stereo_udp5600_historical_real_calibration"
            ]["external_red_fish_pipeline_20260812"]["rigid_target_cross_check_20260813"]
            rigid["analysis_file_sha256"] = "0" * 64
            report = evaluate_auto_readiness(config, config_path=path)
            self.assertTrue(any("SHA-256 mismatch" in item for item in report.blockers))

    def test_repository_config_refuses_before_transport_creation(self):
        config_path = Path(__file__).parents[1] / "finesub_v4pro1_mpc.json"
        with patch(
            "MPC_dual_model.auto_only_runtime.make_transport",
            side_effect=AssertionError("transport must not be created"),
        ):
            exit_code = auto_main(["--config", str(config_path), "--execute"])
        self.assertEqual(exit_code, 2)

    def test_default_valid_invocation_is_preflight_without_transport(self):
        with tempfile.TemporaryDirectory() as directory:
            _config, config_path = valid_config(Path(directory))
            with patch(
                "MPC_dual_model.auto_only_runtime.make_transport",
                side_effect=AssertionError("preflight must not create transport"),
            ):
                exit_code = auto_main(["--config", str(config_path)])
            self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
