"""High-level bridge: position measurement -> state estimate -> MPC -> command."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

try:
    from .camera_transform import rotation_state_body_from_previous
    from .device_adapter import (
        DeviceCommand,
        FineSUBThrusterAllocator,
        ForceCommandAdapter,
        ThrusterAllocation,
    )
    from .fossen_fixed_dl_model import FixedLinearDampingRelativeModel, vector3
    from .model_fusion import FusionConfig, OnlineModelFusion
    from .mpc_controller import MPCResult, RelativeMPCController
    from .relative_kalman import RelativePositionKalmanFilter
except ImportError:
    from camera_transform import rotation_state_body_from_previous
    from device_adapter import (
        DeviceCommand,
        FineSUBThrusterAllocator,
        ForceCommandAdapter,
        ThrusterAllocation,
    )
    from fossen_fixed_dl_model import FixedLinearDampingRelativeModel, vector3
    from model_fusion import FusionConfig, OnlineModelFusion
    from mpc_controller import MPCResult, RelativeMPCController
    from relative_kalman import RelativePositionKalmanFilter


DEFAULT_STAIRCASE_HORIZON_CAPS = (3, 3, 2, 2, 1, 1)
DEFAULT_PREDICTION_HORIZON_WEIGHTS = (0.5, 0.3, 0.2)


def build_default_staircase_fusion() -> OnlineModelFusion:
    """Single source of truth for the maintained translation history shape."""
    return OnlineModelFusion(
        FusionConfig(
            window=len(DEFAULT_STAIRCASE_HORIZON_CAPS),
            prediction_horizon=len(DEFAULT_PREDICTION_HORIZON_WEIGHTS),
            forgetting_factor=0.8,
            weight_update_rate=0.35,
            prediction_horizon_weights=DEFAULT_PREDICTION_HORIZON_WEIGHTS,
            staircase_horizon_caps=DEFAULT_STAIRCASE_HORIZON_CAPS,
            initial_model1_weight=(0.80, 0.80, 0.80),
            minimum_weight=0.01,
        )
    )


@dataclass
class TrackerOutput:
    estimated_state: np.ndarray
    mpc: MPCResult
    device_command: DeviceCommand
    thruster_allocation: ThrusterAllocation | None


@dataclass
class SafeControlOutput:
    force: np.ndarray
    device_command: DeviceCommand
    status: str


@dataclass
class BaselineAdaptationConfig:
    """Per-axis gated EMA for the model-1 matched-motion force."""

    enabled: bool = False
    update_mode: str = "gated_ema"
    axis_enabled: object = (True, True, True)
    adaptation_rate: float = 0.02
    transient_adaptation_rate: float = 0.08
    steady_position_error_tolerance: float = 0.03
    steady_velocity_tolerance: float = 0.02
    position_error_tolerance: float = 0.20
    velocity_tolerance: float = 0.20
    # A moving target can leave a persistent relative velocity while the
    # vehicle is catching up.  Permit model-1 to learn that operating force
    # only on the forward axis after several same-sign, low-acceleration
    # updates.  The ordinary static gate above remains unchanged for all
    # axes; these fields are deliberately a separate, narrower gate.
    matched_motion_axis_enabled: object = (False, False, False)
    matched_motion_min_velocity: float = 0.05
    matched_motion_max_velocity: float = 0.20
    matched_motion_acceleration_tolerance: float = 0.20
    matched_motion_confirmation_updates: int = 3

    def __post_init__(self) -> None:
        if str(self.update_mode).strip().lower() != "gated_ema":
            raise ValueError("baseline update_mode must be gated_ema")
        axes = np.asarray(self.axis_enabled, dtype=bool).reshape(-1)
        if axes.shape != (3,):
            raise ValueError("baseline axis_enabled must contain three values")
        self.axis_enabled = axes
        matched_axes = np.asarray(
            self.matched_motion_axis_enabled, dtype=bool
        ).reshape(-1)
        if matched_axes.shape != (3,):
            raise ValueError(
                "baseline matched_motion_axis_enabled must contain three values"
            )
        self.matched_motion_axis_enabled = matched_axes
        for name in ("adaptation_rate", "transient_adaptation_rate"):
            value = float(getattr(self, name))
            if not 0.0 < value <= 1.0:
                raise ValueError(f"baseline {name} must be in (0,1]")
            setattr(self, name, value)
        for name in (
            "matched_motion_min_velocity",
            "matched_motion_max_velocity",
            "matched_motion_acceleration_tolerance",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"baseline {name} must be finite and nonnegative")
            setattr(self, name, value)
        confirmation_updates = int(self.matched_motion_confirmation_updates)
        if confirmation_updates < 1:
            raise ValueError(
                "baseline matched_motion_confirmation_updates must be positive"
            )
        self.matched_motion_confirmation_updates = confirmation_updates
        if (
            self.matched_motion_min_velocity
            > self.matched_motion_max_velocity
        ):
            raise ValueError(
                "matched-motion minimum velocity exceeds its maximum"
            )
        for name in (
            "steady_position_error_tolerance",
            "steady_velocity_tolerance",
            "position_error_tolerance",
            "velocity_tolerance",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"baseline {name} must be finite and nonnegative")
            setattr(self, name, value)
        if self.steady_position_error_tolerance > self.position_error_tolerance:
            raise ValueError("steady position tolerance exceeds the eligibility gate")
        if self.steady_velocity_tolerance > self.velocity_tolerance:
            raise ValueError("steady velocity tolerance exceeds the eligibility gate")


@dataclass
class _PendingPositionPrediction:
    """A historical model comparison waiting for future position measurements."""

    origin_index: int
    state1: np.ndarray
    state2: np.ndarray
    tau_base: np.ndarray
    steps: int = 0


class MPCTracker:
    """Combine estimator, controller, and lower-controller command scaling."""

    def __init__(
        self,
        model: FixedLinearDampingRelativeModel,
        estimator: RelativePositionKalmanFilter,
        controller: RelativeMPCController,
        adapter: ForceCommandAdapter,
        fusion: OnlineModelFusion | None = None,
        thruster_allocator: FineSUBThrusterAllocator | None = None,
        baseline_adaptation: BaselineAdaptationConfig | None = None,
    ) -> None:
        if estimator.model is not model or controller.model is not model:
            raise ValueError("model, estimator.model, and controller.model must be identical")
        self.model = model
        self.estimator = estimator
        self.controller = controller
        self.adapter = adapter
        self.fusion = fusion or build_default_staircase_fusion()
        self.thruster_allocator = thruster_allocator
        self.baseline_adaptation = baseline_adaptation or BaselineAdaptationConfig()
        self._pending_position_predictions: deque[_PendingPositionPrediction] = deque()
        self._frame_index = -1
        self._previous_achieved_force: np.ndarray | None = None
        self._model1_base_force: np.ndarray | None = None
        self._previous_baseline_velocity: np.ndarray | None = None
        self._matched_motion_streak = np.zeros(3, dtype=int)

    def latch_baseline(self, tau_achieved) -> None:
        """Start a track with the preceding achieved force as model-1 base."""
        achieved = vector3(tau_achieved, "tau_achieved")
        self.reset_history()
        self._previous_achieved_force = achieved.copy()
        self._model1_base_force = achieved.copy()
        self._previous_baseline_velocity = None
        self._matched_motion_streak.fill(0)

    def reset_history(self) -> None:
        """Reset estimator-independent control and force history."""
        self.fusion.reset()
        self.controller.reset()
        self._pending_position_predictions.clear()
        self._frame_index = -1
        self._previous_achieved_force = None
        self._model1_base_force = None
        self._previous_baseline_velocity = None
        self._matched_motion_streak.fill(0)

    def _adapt_model1_base(
        self,
        state: np.ndarray,
        tau_achieved: np.ndarray,
        reference_position,
    ) -> np.ndarray:
        """Update the matched-motion force only on observable, eligible axes."""

        if self._model1_base_force is None:
            self._model1_base_force = tau_achieved.copy()
        cfg = self.baseline_adaptation
        if not cfg.enabled:
            return self._model1_base_force.copy()

        reference = (
            self.controller.config.reference_position
            if reference_position is None
            else vector3(reference_position, "reference_position")
        )
        position_error = np.abs(state[:3] - reference)
        signed_velocity = np.asarray(state[3:], dtype=float)
        velocity = np.abs(signed_velocity)
        previous_velocity = self._previous_baseline_velocity
        self._previous_baseline_velocity = signed_velocity.copy()
        if previous_velocity is None:
            acceleration = np.full(3, np.inf)
            same_direction = np.zeros(3, dtype=bool)
        else:
            acceleration = np.abs(signed_velocity - previous_velocity) / self.model.dt
            same_direction = (
                (signed_velocity * previous_velocity > 0.0)
                & (velocity > 0.0)
            )
        matched_candidate = (
            cfg.axis_enabled
            & cfg.matched_motion_axis_enabled
            & (position_error <= cfg.position_error_tolerance)
            & (velocity >= cfg.matched_motion_min_velocity)
            & (velocity <= cfg.matched_motion_max_velocity)
            & (acceleration <= cfg.matched_motion_acceleration_tolerance)
            & same_direction
        )
        self._matched_motion_streak = np.where(
            matched_candidate,
            self._matched_motion_streak + 1,
            0,
        )
        matched_motion = matched_candidate & (
            self._matched_motion_streak
            >= cfg.matched_motion_confirmation_updates
        )
        static_eligible = (
            cfg.axis_enabled
            & (position_error <= cfg.position_error_tolerance)
            & (velocity <= cfg.velocity_tolerance)
        )
        eligible = static_eligible | matched_motion
        steady = (
            (position_error <= cfg.steady_position_error_tolerance)
            & (velocity <= cfg.steady_velocity_tolerance)
        )
        alpha = np.where(
            steady,
            cfg.adaptation_rate,
            cfg.transient_adaptation_rate,
        )
        candidate = (
            (1.0 - alpha) * self._model1_base_force
            + alpha * tau_achieved
        )
        candidate = np.clip(
            candidate,
            self.controller.config.force_min,
            self.controller.config.force_max,
        )
        self._model1_base_force = np.where(
            eligible,
            candidate,
            self._model1_base_force,
        )
        return self._model1_base_force.copy()

    def target_lost(self, tau_achieved_previous) -> SafeControlOutput:
        """Stop target maneuvers and return toward fixed Fossen balance force."""
        self.controller.reset()
        self._pending_position_predictions.clear()
        force = self.controller.safe_force(tau_achieved_previous)
        return SafeControlOutput(
            force=force,
            device_command=self.adapter.convert(force),
            status="target_lost:return_to_restoring_force",
        )

    def _score_completed_position_predictions(
        self,
        filtered_position: np.ndarray,
        tau_achieved: np.ndarray,
        yaw_delta_rad: float,
    ) -> None:
        """Advance all historical candidates with actual force, then score p.

        A record started at frame i is advanced by every actually applied force
        until frame i+h arrives.  At that point its model-1 and model-2
        predicted positions are compared against the filtered camera position.
        """
        retained: deque[_PendingPositionPrediction] = deque()
        rotation = rotation_state_body_from_previous(yaw_delta_rad)
        for record in self._pending_position_predictions:
            record.state1 = rotation @ (
                self.model.A_d @ record.state1
                + self.model.B_d @ (tau_achieved - record.tau_base)
            )
            record.state2 = rotation @ (
                self.model.A_d @ record.state2
                + self.model.B_d
                @ (tau_achieved - self.model.restoring_force)
            )
            record.steps += 1
            self.fusion.observe_position(
                actual_position=filtered_position,
                prediction1=record.state1[:3],
                prediction2=record.state2[:3],
                horizon_step=record.steps,
                origin_index=record.origin_index,
                target_index=self._frame_index,
            )
            if record.steps < self.fusion.config.prediction_horizon:
                retained.append(record)
        self._pending_position_predictions = retained

    def _start_position_prediction(self, state: np.ndarray, tau_base: np.ndarray) -> None:
        """Save the current posterior as a new multi-step comparison origin."""
        self._pending_position_predictions.append(
            _PendingPositionPrediction(
                origin_index=self._frame_index,
                state1=state.copy(),
                state2=state.copy(),
                tau_base=tau_base.copy(),
            )
        )

    def update(
        self,
        position_body,
        tau_achieved_previous,
        reference_position=None,
        yaw_delta_rad: float = 0.0,
        yaw_rate_rad_s: float = 0.0,
    ) -> TrackerOutput:
        """Process one new 3-D position measurement.

        tau_achieved_previous is the force that acted during the interval since
        the preceding image. When no force observer is available, pass the
        prior saturated command as an approximation.
        """
        tau_achieved = vector3(tau_achieved_previous, "tau_achieved_previous")
        yaw_delta = float(yaw_delta_rad)
        yaw_rate = float(yaw_rate_rad_s)
        if not np.isfinite(yaw_delta) or not np.isfinite(yaw_rate):
            raise ValueError("yaw delta/rate must be finite")
        rotation = rotation_state_body_from_previous(yaw_delta)
        self._frame_index += 1
        if not self.estimator.initialized:
            state = self.estimator.initialize(position_body)
            self.fusion.advance_time(self._frame_index)
        else:
            old_state = self.estimator.x.copy()
            tau_base_for_interval = (
                tau_achieved
                if self._model1_base_force is None
                else self._model1_base_force
            )
            prediction1 = rotation @ (
                self.model.A_d @ old_state
                + self.model.B_d @ (tau_achieved - tau_base_for_interval)
            )
            prediction2 = rotation @ (
                self.model.A_d @ old_state
                + self.model.B_d
                @ (tau_achieved - self.model.restoring_force)
            )
            weight6 = np.diag(
                np.concatenate(
                    (self.fusion.model1_weight, self.fusion.model1_weight)
                )
            )
            fused_prediction = (
                weight6 @ prediction1 + (np.eye(6) - weight6) @ prediction2
            )
            self.estimator.predict_mean(
                fused_prediction,
                yaw_delta_rad=yaw_delta,
            )
            state = self.estimator.update(position_body)
            self._score_completed_position_predictions(
                state[:3],
                tau_achieved,
                yaw_delta,
            )
        self._previous_achieved_force = tau_achieved.copy()
        tau_base_for_solve = self._adapt_model1_base(
            state,
            tau_achieved,
            reference_position,
        )
        self._start_position_prediction(state, tau_base_for_solve)
        result = self.controller.solve(
            state=state,
            tau_previous=tau_achieved,
            tau_base=tau_base_for_solve,
            reference_position=reference_position,
            model1_weight=self.fusion.model1_weight,
            yaw_rate_rad_s=yaw_rate,
        )
        allocation = (
            None
            if self.thruster_allocator is None
            else self.thruster_allocator.allocate(result.force)
        )
        return TrackerOutput(
            estimated_state=state,
            mpc=result,
            device_command=self.adapter.convert(result.force),
            thruster_allocation=allocation,
        )
