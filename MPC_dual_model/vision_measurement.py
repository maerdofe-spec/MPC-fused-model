"""Read and safety-gate external stereo-vision measurements for the MPC.

The vision process remains an independent, read-only producer.  This module
tails its JSONL output and rejects measurements that are stale, low quality or
kinematically implausible before they reach the Kalman filter/controller.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VisionGateConfig:
    """Conservative real-vehicle gates for the current red-fish pipeline."""

    max_result_age_s: float = 0.25
    max_pipeline_delay_s: float = 0.15
    min_confidence: float = 0.50
    min_depth_confidence: float = 0.20
    max_depth_nis: float = 25.0
    min_forward_m: float = 0.15
    max_forward_m: float = 1.50
    max_speed_m_s: float = 1.0
    jump_margin_m: float = 0.10
    max_step_m: float = 0.30
    max_inter_sample_gap_s: float = 0.50
    startup_confirmation_samples: int = 3
    reacquire_confirmation_samples: int = 5
    accepted_depth_filter_modes: tuple[str, ...] = ("update",)
    clamp_implausible_steps: bool = False

    def __post_init__(self) -> None:
        positive = (
            self.max_result_age_s,
            self.max_pipeline_delay_s,
            self.max_depth_nis,
            self.min_forward_m,
            self.max_forward_m,
            self.max_speed_m_s,
            self.jump_margin_m,
            self.max_step_m,
            self.max_inter_sample_gap_s,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("vision gate limits must be finite and positive")
        if self.max_forward_m <= self.min_forward_m:
            raise ValueError("max_forward_m must exceed min_forward_m")
        if self.max_step_m < self.jump_margin_m:
            raise ValueError("max_step_m must be at least jump_margin_m")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        if not 0.0 <= self.min_depth_confidence <= 1.0:
            raise ValueError("min_depth_confidence must be in [0, 1]")
        if self.startup_confirmation_samples < 1 or self.reacquire_confirmation_samples < 1:
            raise ValueError("vision confirmation counts must be positive")
        modes = tuple(
            str(value).strip().lower() for value in self.accepted_depth_filter_modes
        )
        if not modes or any(not value for value in modes):
            raise ValueError("accepted depth filter modes must be nonempty")
        object.__setattr__(self, "accepted_depth_filter_modes", modes)

    def step_limit(self, dt: float) -> float:
        """Return the bounded spatial step allowed for one visual sample."""

        return min(
            self.jump_margin_m + self.max_speed_m_s * float(dt),
            self.max_step_m,
        )


@dataclass(frozen=True)
class VisionMeasurement:
    """One validated measurement in OpenCV camera coordinates."""

    frame_index: int
    acquisition_time_s: float
    result_time_s: float
    position_camera_xyz_m: np.ndarray
    confidence: float
    depth_confidence: float
    depth_nis: float


@dataclass(frozen=True)
class VisionGateDecision:
    """Result of evaluating one external pipeline record."""

    control_ready: bool
    reason: str
    measurement: VisionMeasurement | None = None
    confirmation_count: int = 0


class PipelineJsonlTail:
    """Non-blocking tail reader that never writes to the vision output file."""

    def __init__(self, path: str | Path, *, start_at_end: bool = True) -> None:
        self.path = Path(path)
        self.start_at_end = bool(start_at_end)
        self._offset = 0
        self._identity: tuple[int, int] | None = None
        self._initialized = False

    def poll(self) -> list[dict[str, Any]]:
        """Return all newly completed JSON objects; ignore malformed lines."""

        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return []
        identity = (int(stat.st_dev), int(stat.st_ino))
        replaced = self._identity is not None and identity != self._identity
        truncated = int(stat.st_size) < self._offset
        if not self._initialized or replaced or truncated:
            self._identity = identity
            self._offset = int(stat.st_size) if self.start_at_end and not self._initialized else 0
            self._initialized = True
        if int(stat.st_size) <= self._offset:
            return []

        records: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            handle.seek(self._offset)
            while True:
                line_start = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.endswith("\n"):
                    handle.seek(line_start)
                    break
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    records.append(value)
            self._offset = handle.tell()
        return records


class VisionMeasurementGate:
    """Stateful fail-closed gate for the external stereo JSONL stream."""

    def __init__(
        self,
        config: VisionGateConfig | None = None,
        *,
        accept_any_finite_track_output: bool = False,
    ) -> None:
        self.config = config or VisionGateConfig()
        self.accept_any_finite_track_output = bool(accept_any_finite_track_output)
        self.reset()

    def reset(self) -> None:
        self._last_frame_index: int | None = None
        self._last_accepted: VisionMeasurement | None = None
        self._pending: VisionMeasurement | None = None
        self._pending_count = 0
        self._has_ever_locked = False

    @staticmethod
    def _finite_float(value: Any) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    def _parse(self, record: dict[str, Any]) -> tuple[VisionMeasurement | None, str]:
        frame_index_value = record.get("frame_idx")
        try:
            frame_index = int(frame_index_value)
        except (TypeError, ValueError):
            return None, "missing_frame_index"
        if self._last_frame_index is not None and frame_index <= self._last_frame_index:
            return None, "duplicate_or_out_of_order"
        self._last_frame_index = frame_index

        tracks = record.get("tracks")
        if not isinstance(tracks, list) or not tracks:
            return None, "no_target"
        tracks = [track for track in tracks if isinstance(track, dict)]
        if not tracks:
            return None, "no_target"
        def confidence_key(item: dict[str, Any]) -> float:
            value = self._finite_float(item.get("confidence"))
            return -1.0 if value is None else value

        track = max(tracks, key=confidence_key)

        position_value = track.get("position", track.get("position_xyz"))
        try:
            position = np.asarray(position_value, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return None, "invalid_position"
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            return None, "invalid_position"

        acquisition_time = self._finite_float(record.get("frame_ts_mean"))
        result_time = self._finite_float(record.get("result_time"))
        confidence = self._finite_float(track.get("confidence"))
        depth_confidence = self._finite_float(track.get("depth_confidence"))
        depth_nis = self._finite_float(track.get("depth_nis"))
        if acquisition_time is None or result_time is None:
            return None, "missing_timestamp"
        if self.accept_any_finite_track_output:
            return (
                VisionMeasurement(
                    frame_index=frame_index,
                    acquisition_time_s=acquisition_time,
                    result_time_s=result_time,
                    position_camera_xyz_m=position.copy(),
                    confidence=0.0 if confidence is None else confidence,
                    depth_confidence=(
                        0.0 if depth_confidence is None else depth_confidence
                    ),
                    depth_nis=0.0 if depth_nis is None else depth_nis,
                ),
                "finite_track_output",
            )
        if confidence is None or confidence < self.config.min_confidence:
            return None, "low_confidence"
        if track.get("depth_valid") is not True:
            return None, "invalid_depth"
        if depth_confidence is None or depth_confidence < self.config.min_depth_confidence:
            return None, "low_depth_confidence"
        if depth_nis is None:
            return None, "missing_depth_nis"
        if depth_nis > self.config.max_depth_nis:
            return None, "depth_nis_outlier"
        mode = str(track.get("depth_filter_mode", "")).strip().lower()
        if mode not in self.config.accepted_depth_filter_modes:
            return None, f"depth_mode:{mode or 'missing'}"
        if not self.config.min_forward_m <= position[2] <= self.config.max_forward_m:
            return None, "forward_range"
        if result_time < acquisition_time:
            return None, "negative_pipeline_delay"
        if result_time - acquisition_time > self.config.max_pipeline_delay_s:
            return None, "pipeline_delay"
        return (
            VisionMeasurement(
                frame_index=frame_index,
                acquisition_time_s=acquisition_time,
                result_time_s=result_time,
                position_camera_xyz_m=position.copy(),
                confidence=confidence,
                depth_confidence=depth_confidence,
                depth_nis=depth_nis,
            ),
            "valid",
        )

    def _plausible_step(
        self,
        previous: VisionMeasurement,
        current: VisionMeasurement,
    ) -> bool:
        dt = current.acquisition_time_s - previous.acquisition_time_s
        if dt <= 0.0 or dt > self.config.max_inter_sample_gap_s:
            return False
        distance = float(
            np.linalg.norm(
                current.position_camera_xyz_m - previous.position_camera_xyz_m
            )
        )
        limit = self.config.step_limit(dt)
        return distance <= limit

    def _clamp_step(
        self,
        previous: VisionMeasurement,
        current: VisionMeasurement,
    ) -> VisionMeasurement:
        """Limit one accepted position increment without dropping the update."""

        dt = current.acquisition_time_s - previous.acquisition_time_s
        delta = current.position_camera_xyz_m - previous.position_camera_xyz_m
        distance = float(np.linalg.norm(delta))
        limit = self.config.step_limit(dt)
        if distance <= limit or distance <= np.finfo(float).eps:
            return current
        return VisionMeasurement(
            frame_index=current.frame_index,
            acquisition_time_s=current.acquisition_time_s,
            result_time_s=current.result_time_s,
            position_camera_xyz_m=(
                previous.position_camera_xyz_m + delta * (limit / distance)
            ),
            confidence=current.confidence,
            depth_confidence=current.depth_confidence,
            depth_nis=current.depth_nis,
        )

    def _reject(self, reason: str) -> VisionGateDecision:
        self._pending = None
        self._pending_count = 0
        return VisionGateDecision(False, reason)

    def evaluate(
        self,
        record: dict[str, Any],
        *,
        now_s: float | None = None,
    ) -> VisionGateDecision:
        """Validate one record and require stable samples before control use."""

        measurement, reason = self._parse(record)
        if measurement is None:
            if reason == "duplicate_or_out_of_order":
                return VisionGateDecision(False, reason)
            return self._reject(reason)

        now = time.time() if now_s is None else float(now_s)
        age = now - measurement.result_time_s
        if not math.isfinite(age) or age < -0.05 or age > self.config.max_result_age_s:
            return self._reject("stale_result")

        if self.accept_any_finite_track_output:
            self._last_accepted = measurement
            self._pending = None
            self._pending_count = 0
            self._has_ever_locked = True
            return VisionGateDecision(True, "track_output", measurement, 1)

        previous = self._last_accepted
        if previous is not None:
            gap = measurement.acquisition_time_s - previous.acquisition_time_s
            if gap <= self.config.max_inter_sample_gap_s and self._plausible_step(
                previous, measurement
            ):
                self._last_accepted = measurement
                self._pending = None
                self._pending_count = 0
                return VisionGateDecision(True, "tracking", measurement, 1)
            if (
                self.config.clamp_implausible_steps
                and 0.0 < gap <= self.config.max_inter_sample_gap_s
            ):
                clamped = self._clamp_step(previous, measurement)
                self._last_accepted = clamped
                self._pending = None
                self._pending_count = 0
                return VisionGateDecision(True, "tracking_clamped", clamped, 1)
            # When clamping is disabled, retain the last trusted sample and
            # reject the entire implausible step.  Do not pass the first bad
            # sample into the reacquisition counter: with a one-sample
            # reacquire setting, that would accept a sustained false-depth
            # track on the very next frame.
            if 0.0 < gap <= self.config.max_inter_sample_gap_s:
                return self._reject("implausible_step")
            if gap > self.config.max_inter_sample_gap_s:
                self._last_accepted = None

        if self._pending is None or not self._plausible_step(self._pending, measurement):
            self._pending = measurement
            self._pending_count = 1
        else:
            self._pending = measurement
            self._pending_count += 1

        required = (
            self.config.reacquire_confirmation_samples
            if self._has_ever_locked
            else self.config.startup_confirmation_samples
        )
        if self._pending_count < required:
            return VisionGateDecision(
                False,
                "reacquire_pending" if self._has_ever_locked else "startup_pending",
                measurement,
                self._pending_count,
            )

        self._last_accepted = measurement
        self._pending = None
        self._pending_count = 0
        reason = "reacquired" if self._has_ever_locked else "startup_confirmed"
        self._has_ever_locked = True
        return VisionGateDecision(True, reason, measurement, required)
