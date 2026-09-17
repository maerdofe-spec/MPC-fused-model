"""Position measurement -> model-1 estimate -> MPC -> FineSUB command."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from .device_adapter import (
        DeviceCommand,
        FineSUBThrusterAllocator,
        ForceCommandAdapter,
        ThrusterAllocation,
    )
    from .fossen_fixed_dl_model import FixedLinearDampingRelativeModel, vector3
    from .mpc_controller import MPCResult, RelativeMPCController
    from .relative_kalman import RelativePositionKalmanFilter
except ImportError:
    from device_adapter import (
        DeviceCommand,
        FineSUBThrusterAllocator,
        ForceCommandAdapter,
        ThrusterAllocation,
    )
    from fossen_fixed_dl_model import FixedLinearDampingRelativeModel, vector3
    from mpc_controller import MPCResult, RelativeMPCController
    from relative_kalman import RelativePositionKalmanFilter


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
    """Optional slow hold-force learning when position and speed are settled."""

    enabled: bool = False
    adaptation_rate: float = 0.02
    position_error_tolerance: float = 0.06
    velocity_tolerance: float = 0.04


class MPCTracker:
    """High-level model-1-only tracker.

    ``tau_achieved_previous`` is the force that acted since the preceding
    image. With no force observer, the previous saturated command is the usual
    approximation.
    """

    def __init__(
        self,
        model: FixedLinearDampingRelativeModel,
        estimator: RelativePositionKalmanFilter,
        controller: RelativeMPCController,
        adapter: ForceCommandAdapter,
        thruster_allocator: FineSUBThrusterAllocator | None = None,
        baseline_adaptation: BaselineAdaptationConfig | None = None,
    ) -> None:
        if estimator.model is not model or controller.model is not model:
            raise ValueError("model, estimator.model, and controller.model must be identical")
        self.model = model
        self.estimator = estimator
        self.controller = controller
        self.adapter = adapter
        self.thruster_allocator = thruster_allocator
        self.baseline_adaptation = baseline_adaptation or BaselineAdaptationConfig()
        self._tau_before_last = np.zeros(3)

    def latch_baseline(self, tau_achieved) -> None:
        """Capture force history when automatic tracking is enabled."""
        achieved = vector3(tau_achieved, "tau_achieved")
        self.model.set_tau_base(achieved)
        self._tau_before_last = achieved.copy()
        self.controller.reset()

    def target_lost(self, tau_achieved_previous) -> SafeControlOutput:
        """Rate-limit the command back toward the latched hold force."""
        self.controller.reset()
        force = self.controller.safe_force(tau_achieved_previous)
        return SafeControlOutput(
            force=force,
            device_command=self.adapter.convert(force),
            status="target_lost:return_to_baseline",
        )

    def _adapt_baseline_if_matched(
        self,
        state: np.ndarray,
        tau_achieved: np.ndarray,
        reference_position,
    ) -> None:
        cfg = self.baseline_adaptation
        if not cfg.enabled:
            return
        reference = (
            self.controller.config.reference_position
            if reference_position is None
            else vector3(reference_position, "reference_position")
        )
        if (
            np.linalg.norm(state[:3] - reference) <= cfg.position_error_tolerance
            and np.linalg.norm(state[3:]) <= cfg.velocity_tolerance
        ):
            if not 0.0 < cfg.adaptation_rate <= 1.0:
                raise ValueError("baseline adaptation rate must be in (0, 1]")
            self.model.tau_base = (
                (1.0 - cfg.adaptation_rate) * self.model.tau_base
                + cfg.adaptation_rate * tau_achieved
            )

    def update(
        self,
        position_body,
        tau_achieved_previous,
        reference_position=None,
    ) -> TrackerOutput:
        tau_achieved = vector3(tau_achieved_previous, "tau_achieved_previous")
        if not self.estimator.initialized:
            state = self.estimator.initialize(position_body)
        else:
            old_state = self.estimator.x.copy()
            delta_tau = tau_achieved - self._tau_before_last
            model1_prediction = (
                self.model.A_d @ old_state + self.model.B_d @ delta_tau
            )
            self.estimator.predict_mean(model1_prediction)
            state = self.estimator.update(position_body)

        self._tau_before_last = tau_achieved.copy()
        self._adapt_baseline_if_matched(state, tau_achieved, reference_position)
        result = self.controller.solve(
            state=state,
            tau_previous=tau_achieved,
            reference_position=reference_position,
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
