"""Position measurement -> model-2 estimate -> MPC -> FineSUB command."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from MPC_dual_model.device_adapter import (
    DeviceCommand,
    FineSUBThrusterAllocator,
    ForceCommandAdapter,
    ThrusterAllocation,
)
from .fossen_fixed_dl_model import FixedLinearDampingRelativeModel, vector3
from .mpc_controller import MPCResult, RelativeMPCController
from .relative_kalman import RelativePositionKalmanFilter


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


class MPCTracker:
    """High-level model-2-only tracker.

    ``tau_achieved_previous`` is the absolute body force that acted since the
    preceding image. With no force observer, the previous saturated command is
    the usual approximation.
    """

    def __init__(
        self,
        model: FixedLinearDampingRelativeModel,
        estimator: RelativePositionKalmanFilter,
        controller: RelativeMPCController,
        adapter: ForceCommandAdapter,
        thruster_allocator: FineSUBThrusterAllocator | None = None,
    ) -> None:
        if estimator.model is not model or controller.model is not model:
            raise ValueError("model, estimator.model, and controller.model must be identical")
        self.model = model
        self.estimator = estimator
        self.controller = controller
        self.adapter = adapter
        self.thruster_allocator = thruster_allocator

    def latch_baseline(self, tau_achieved) -> None:
        """Reset optimizer history when automatic tracking is enabled.

        Model 2 has no matched-force baseline to latch.  The argument remains
        for compatibility with the other tracker API and is intentionally
        ignored.
        """
        self.controller.reset()

    def target_lost(self, tau_achieved_previous) -> SafeControlOutput:
        """Rate-limit the command back toward the latched safe hold force."""
        self.controller.reset()
        force = self.controller.safe_force(tau_achieved_previous)
        return SafeControlOutput(
            force=force,
            device_command=self.adapter.convert(force),
            status="target_lost:return_to_baseline",
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
            model2_prediction = (
                self.model.A_d @ self.estimator.x
                + self.model.B_d @ tau_achieved
            )
            self.estimator.predict_mean(model2_prediction)
            state = self.estimator.update(position_body)

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
