"""Build the formal AUTO-only tracker exclusively from approved runtime data.

This module deliberately has no simulation/default parameter fallback.  A
missing or malformed physical parameter is a hard error so real hardware can
never inherit the values in ``live_integration_example.py`` by accident.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import numpy as np

from .dense_qp import QPSolverSettings
from .camera_transform import camera_visibility_geometry
from .device_adapter import ForceCommandAdapter
from .fossen_fixed_dl_model import FixedLinearDampingRelativeModel
from .model_fusion import FusionConfig, OnlineModelFusion
from .mpc_controller import MPCConfig, RelativeMPCController
from .mpc_tracker import BaselineAdaptationConfig, MPCTracker
from .relative_kalman import KalmanConfig, RelativePositionKalmanFilter


class AutoParameterError(ValueError):
    """Raised when the approved AUTO parameter block is incomplete."""


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AutoParameterError(f"{name} must be an object")
    return value


def _required(mapping: dict[str, Any], name: str, keys: set[str]) -> None:
    missing = sorted(key for key in keys if mapping.get(key) is None)
    if missing:
        raise AutoParameterError(f"{name} is missing: {', '.join(missing)}")


def _present(mapping: dict[str, Any], name: str, keys: set[str]) -> None:
    missing = sorted(key for key in keys if key not in mapping)
    if missing:
        raise AutoParameterError(f"{name} is missing: {', '.join(missing)}")


def _dataclass_kwargs(
    mapping: dict[str, Any],
    dataclass_type: type,
    name: str,
) -> dict[str, Any]:
    allowed = {item.name for item in fields(dataclass_type)}
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise AutoParameterError(f"{name} has unknown fields: {', '.join(unknown)}")
    return dict(mapping)


def build_auto_tracker(runtime_config: dict[str, Any]) -> MPCTracker:
    """Construct the dual-model translation tracker from the active block."""

    auto = _mapping(runtime_config.get("auto_runtime"), "auto_runtime")
    active = _mapping(
        auto.get("active_mpc_parameters"),
        "auto_runtime.active_mpc_parameters",
    )
    if active.get("enabled_for_control") is not True:
        raise AutoParameterError("active_mpc_parameters is not enabled for control")
    if active.get("model_family") != "dual":
        raise AutoParameterError("active_mpc_parameters.model_family must be dual")

    model_data = _mapping(active.get("model"), "active_mpc_parameters.model")
    _required(
        model_data,
        "active_mpc_parameters.model",
        {
            "effective_mass_matrix_kg",
            "restoring_force_frd_n",
            "linear_damping_matrix_n_s_per_m",
            "sample_period_s",
        },
    )
    model = FixedLinearDampingRelativeModel(
        M_t=model_data["effective_mass_matrix_kg"],
        D_L=model_data["linear_damping_matrix_n_s_per_m"],
        dt=float(model_data["sample_period_s"]),
        # Fixed h(0) from the Fossen equation. It is independent of the
        # model-1 operating force latched from the preceding achieved command.
        restoring_force=model_data["restoring_force_frd_n"],
    )

    kalman_data = _mapping(active.get("kalman"), "active_mpc_parameters.kalman")
    _required(
        kalman_data,
        "active_mpc_parameters.kalman",
        {
            "position_std",
            "acceleration_std",
            "initial_position_std",
            "initial_velocity_std",
        },
    )
    kalman_config = KalmanConfig(
        **_dataclass_kwargs(kalman_data, KalmanConfig, "active_mpc_parameters.kalman")
    )

    controller_data = _mapping(
        active.get("controller"),
        "active_mpc_parameters.controller",
    )
    _required(
        controller_data,
        "active_mpc_parameters.controller",
        {
            "horizon",
            "reference_position",
            "position_weights",
            "velocity_weights",
            "terminal_weight_scale",
            "force_weights",
            "delta_force_weights",
            "force_min",
            "force_max",
            "delta_force_min",
            "delta_force_max",
            "thruster_command_matrix",
            "thruster_command_limit",
            "thruster_command_min",
            "thruster_command_max",
            "forward_distance_min",
            "forward_distance_max",
            "horizontal_half_fov_deg",
            "vertical_half_fov_deg",
            "fov_margin_deg",
            "slack_quadratic_weight",
            "slack_linear_weight",
            "slack_max",
            "forward_axis",
            "horizontal_axis",
            "vertical_axis",
            "solver_settings",
        },
    )
    solver_data = _mapping(
        controller_data.get("solver_settings"),
        "active_mpc_parameters.controller.solver_settings",
    )
    _required(
        solver_data,
        "active_mpc_parameters.controller.solver_settings",
        {item.name for item in fields(QPSolverSettings)},
    )
    solver = QPSolverSettings(
        **_dataclass_kwargs(
            solver_data,
            QPSolverSettings,
            "active_mpc_parameters.controller.solver_settings",
        )
    )
    camera_transform = _mapping(
        auto.get("active_camera_transform"),
        "auto_runtime.active_camera_transform",
    )
    _required(
        camera_transform,
        "auto_runtime.active_camera_transform",
        {"rotation_body_from_camera", "camera_origin_in_body_frd_m"},
    )
    visibility_rotation, camera_origin = camera_visibility_geometry(
        camera_transform["rotation_body_from_camera"],
        camera_transform["camera_origin_in_body_frd_m"],
    )

    mpc_data = dict(controller_data)
    mpc_data["solver_settings"] = solver
    mpc_data["rotation_visibility_from_body"] = visibility_rotation
    mpc_data["camera_origin_in_body"] = camera_origin
    mpc_config = MPCConfig(
        **_dataclass_kwargs(
            mpc_data,
            MPCConfig,
            "active_mpc_parameters.controller",
        )
    )

    fusion_data = _mapping(active.get("fusion"), "active_mpc_parameters.fusion")
    _required(
        fusion_data,
        "active_mpc_parameters.fusion",
        {
            "window",
            "prediction_horizon",
            "forgetting_factor",
            "horizon_weight_decay",
            "weight_update_rate",
            "epsilon",
            "indistinguishable_score_threshold",
            "prediction_horizon_weights",
            "staircase_horizon_caps",
            "initial_model1_weight",
            "minimum_weight",
            "position_error_clip",
        },
    )
    fusion = OnlineModelFusion(
        FusionConfig(
            **_dataclass_kwargs(
                fusion_data,
                FusionConfig,
                "active_mpc_parameters.fusion",
            )
        )
    )

    baseline_data = _mapping(
        active.get("baseline_adaptation"),
        "active_mpc_parameters.baseline_adaptation",
    )
    _required(
        baseline_data,
        "active_mpc_parameters.baseline_adaptation",
        {item.name for item in fields(BaselineAdaptationConfig)},
    )
    baseline_adaptation = BaselineAdaptationConfig(
        **_dataclass_kwargs(
            baseline_data,
            BaselineAdaptationConfig,
            "active_mpc_parameters.baseline_adaptation",
        )
    )

    estimator = RelativePositionKalmanFilter(model, kalman_config)
    controller = RelativeMPCController(model, mpc_config)
    # MPCTracker still exposes a legacy integer DeviceCommand in its output.
    # The AUTO runtime ignores it and performs the final conversion through
    # FineSUBHardwareAdapter.  This identity adapter therefore has no physical
    # authority and merely satisfies the tracker interface.
    unused_adapter = ForceCommandAdapter(
        positive_force_at_limit=(1.0, 1.0, 1.0),
        signs=(1.0, 1.0, 1.0),
        command_limits=(1.0, 1.0, 1.0),
    )
    return MPCTracker(
        model,
        estimator,
        controller,
        unused_adapter,
        fusion=fusion,
        thruster_allocator=None,
        baseline_adaptation=baseline_adaptation,
    )
