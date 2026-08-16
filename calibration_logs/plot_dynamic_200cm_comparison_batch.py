#!/usr/bin/env python3
"""Aggregate SMC/MPC 200 cm experiments at approximately 20 s and 10 s.

The pool-top camera supplies the motion segmentation and the Tag17 speed used
to classify each complete motion segment. The control traces are cut to the
same moving interval, so the stop/hold time after each 200 cm run is excluded.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from MPC_dual_model.fusion_identifiability_validation import _load_overhead


REFERENCE = np.asarray([0.857634, -0.055545, -0.120815], dtype=float)
LOG_ROOT = ROOT / "calibration_logs"

OUTPUT_LOW = LOG_ROOT / "smc_mpc_200cm_20s_comparison_20260816.png"
OUTPUT_HIGH = LOG_ROOT / "smc_mpc_200cm_10s_comparison_20260816.png"
OUTPUT_LOW_CORRECTED = LOG_ROOT / "smc_mpc_200cm_20s_corrected_20260816.png"
OUTPUT_HIGH_CORRECTED = LOG_ROOT / "smc_mpc_200cm_10s_corrected_20260816.png"
OUTPUT_FIVE = LOG_ROOT / "smc_mpc_200cm_5s_corrected_20260816.png"
SUMMARY_OUTPUT = LOG_ROOT / "smc_mpc_200cm_comparison_summary_20260816.json"

# Tag17 is the hand-held target; tags 15/16 are combined into the ROV.
# The measured Tag17-speed gap is the most repeatable way to separate the two
# commanded speed groups in this batch of experiments.
DEFAULT_TAG17_SPEED_THRESHOLD_M_S = 0.085
MOVING_SPEED_THRESHOLD_M_S = 0.015
MIN_MOTION_DISPLACEMENT_M = 0.8
MIN_MOTION_DURATION_S = 4.0

COLORS = {"MPC": "#1f77b4", "SMC": "#d62728"}
LINESTYLES = {"MPC": "--", "SMC": "-"}


@dataclass(frozen=True)
class ExperimentSpec:
    method: str
    experiment_id: str
    control_path: Path
    overhead_path: Path
    smc: bool


@dataclass
class OverheadTrace:
    time: np.ndarray
    vehicle: np.ndarray
    target: np.ndarray
    vehicle_velocity: np.ndarray
    target_velocity: np.ndarray
    axis: np.ndarray


@dataclass
class MotionSegment:
    start: float
    end: float
    displacement_m: float
    duration_s: float
    median_speed_m_s: float
    tag17_median_speed_m_s: float
    tag17_p90_speed_m_s: float


@dataclass
class Trial:
    spec: ExperimentSpec
    segment: MotionSegment
    control: dict[str, np.ndarray]
    speed: dict[str, np.ndarray]
    metrics: dict[str, float]
    group: str


EXPERIMENTS = (
    ExperimentSpec(
        "MPC", "experimental_auto_20260816_175230",
        LOG_ROOT / "experimental_auto_20260816_175230.jsonl",
        LOG_ROOT / "top_camera_experiment_20260816_175208" / "top_camera_experiment_20260816_175208_0.db3",
        False,
    ),
    ExperimentSpec(
        "MPC", "experimental_auto_20260816_180128",
        LOG_ROOT / "experimental_auto_20260816_180128.jsonl",
        LOG_ROOT / "top_camera_experiment_20260816_180058" / "top_camera_experiment_20260816_180058_0.db3",
        False,
    ),
    ExperimentSpec(
        "MPC", "experimental_auto_20260816_180343",
        LOG_ROOT / "experimental_auto_20260816_180343.jsonl",
        LOG_ROOT / "top_camera_experiment_20260816_180313" / "top_camera_experiment_20260816_180313_0.db3",
        False,
    ),
    ExperimentSpec(
        "MPC", "experimental_auto_20260816_180538",
        LOG_ROOT / "experimental_auto_20260816_180538.jsonl",
        LOG_ROOT / "top_camera_experiment_20260816_180507" / "top_camera_experiment_20260816_180507_0.db3",
        False,
    ),
    ExperimentSpec(
        "MPC", "experimental_auto_20260816_180744",
        LOG_ROOT / "experimental_auto_20260816_180744.jsonl",
        LOG_ROOT / "top_camera_experiment_20260816_180705" / "top_camera_experiment_20260816_180705_0.db3",
        False,
    ),
    ExperimentSpec(
        "MPC", "experimental_auto_20260816_181255",
        LOG_ROOT / "experimental_auto_20260816_181255.jsonl",
        LOG_ROOT / "top_camera_experiment_20260816_181229" / "top_camera_experiment_20260816_181229_0.db3",
        False,
    ),
    ExperimentSpec(
        "SMC", "smc_real_20260816_175419",
        LOG_ROOT / "smc_real_20260816_175419.jsonl",
        LOG_ROOT / "smc_synced_overhead_20260816_175419.csv",
        True,
    ),
    ExperimentSpec(
        "SMC", "smc_real_20260816_175715",
        LOG_ROOT / "smc_real_20260816_175715.jsonl",
        LOG_ROOT / "smc_synced_overhead_20260816_175715.csv",
        True,
    ),
    ExperimentSpec(
        "SMC", "smc_real_20260816_181743",
        LOG_ROOT / "smc_real_20260816_181743.jsonl",
        LOG_ROOT / "smc_synced_overhead_20260816_181743.csv",
        True,
    ),
)

FIVE_SECOND_EXPERIMENTS = (
    ExperimentSpec(
        "MPC", "experimental_auto_20260816_172732",
        LOG_ROOT / "experimental_auto_20260816_172732.jsonl",
        LOG_ROOT / "top_camera_experiment_20260816_172625" / "top_camera_experiment_20260816_172625_0.db3",
        False,
    ),
    ExperimentSpec(
        "SMC", "smc_real_20260816_174133",
        LOG_ROOT / "smc_real_20260816_174133.jsonl",
        LOG_ROOT / "smc_synced_overhead_20260816_174133.csv",
        True,
    ),
)


def local_linear_velocity(times: np.ndarray, positions: np.ndarray, window_s: float = 0.5) -> np.ndarray:
    velocity = np.full_like(positions, np.nan, dtype=float)
    left = 0
    right = 0
    for index, centre in enumerate(times):
        while left < len(times) and times[left] < centre - window_s:
            left += 1
        while right < len(times) and times[right] <= centre + window_s:
            right += 1
        if right - left < 5:
            continue
        local_time = times[left:right] - centre
        denominator = float(local_time @ local_time)
        if denominator > 1.0e-9:
            centred = positions[left:right] - np.mean(positions[left:right], axis=0)
            velocity[index] = local_time @ centred / denominator
    return velocity


def load_smc_overhead(path: Path) -> OverheadTrace:
    rows: list[tuple[float, np.ndarray, np.ndarray]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                details = json.loads(row["vision_detections_detail"])
                detections = {
                    int(item["tag_id"]): np.asarray(item["world_xy_yaw"], dtype=float)
                    for item in details
                    if "tag_id" in item and "world_xy_yaw" in item
                }
                # Keep the same synchronized two-tag vehicle definition as
                # the earlier comparison. A single visible tag can jump when
                # the other marker is temporarily occluded.
                if 15 not in detections or 16 not in detections:
                    continue
                vehicle = 0.5 * (detections[15][:2] + detections[16][:2])
                target = detections.get(17, np.full(3, np.nan))[:2]
                rows.append((float(row["host_monotonic_s"]), vehicle, target))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
    if len(rows) < 20:
        raise ValueError(f"too few overhead samples in {path}")
    times = np.asarray([item[0] for item in rows], dtype=float)
    vehicle = np.vstack([item[1] for item in rows])
    target = np.vstack([item[2] for item in rows])
    axis = np.linalg.svd(vehicle - vehicle.mean(axis=0), full_matrices=False)[2][0]
    vehicle_velocity = local_linear_velocity(times, vehicle)
    target_velocity = np.full_like(target, np.nan)
    target_valid = np.all(np.isfinite(target), axis=1)
    if np.count_nonzero(target_valid) >= 20:
        target_velocity[target_valid] = local_linear_velocity(times[target_valid], target[target_valid])
    return OverheadTrace(times, vehicle, target, vehicle_velocity, target_velocity, axis)


def load_mpc_overhead(path: Path) -> OverheadTrace:
    data = _load_overhead(path, read_only_snapshot=True)
    vehicle = np.asarray(data["vehicle"], dtype=float)
    axis = np.linalg.svd(vehicle - vehicle.mean(axis=0), full_matrices=False)[2][0]
    return OverheadTrace(
        np.asarray(data["time"], dtype=float), vehicle,
        np.asarray(data["target"], dtype=float),
        np.asarray(data["vehicle_velocity"], dtype=float),
        np.asarray(data["target_velocity"], dtype=float), axis,
    )


def load_control_log(path: Path, *, smc: bool) -> dict[str, np.ndarray]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") == "control_update":
                rows.append(row)
    if not rows:
        raise ValueError(f"no control updates in {path}")
    time_key = "host_monotonic_s" if smc else "host_time_s"
    times = np.asarray([row[time_key] for row in rows], dtype=float)
    estimated = np.asarray([row["estimated_state"][:3] for row in rows], dtype=float)
    valid = np.isfinite(times) & np.all(np.isfinite(estimated), axis=1)
    return {"time": times[valid], "error": estimated[valid] - REFERENCE}


def _fill_short_gaps(active: np.ndarray, max_gap_samples: int) -> np.ndarray:
    active = active.copy()
    index = 0
    while index < len(active):
        end = index + 1
        while end < len(active) and active[end] == active[index]:
            end += 1
        if not active[index] and index > 0 and end < len(active) and end - index <= max_gap_samples:
            active[index:end] = True
        index = end
    return active


def _speed_statistics(trace: OverheadTrace, start: float, end: float) -> tuple[float, float]:
    mask = (trace.time >= start) & (trace.time <= end)
    speed = np.linalg.norm(trace.target_velocity[mask], axis=1)
    speed = speed[np.isfinite(speed)]
    if not len(speed):
        return float("nan"), float("nan")
    return float(np.median(speed)), float(np.percentile(speed, 90))


def find_motion_segments(
    trace: OverheadTrace,
    *,
    moving_speed_threshold_m_s: float = MOVING_SPEED_THRESHOLD_M_S,
    min_displacement_m: float = MIN_MOTION_DISPLACEMENT_M,
) -> list[MotionSegment]:
    """Find complete moving intervals while ignoring post-run holds."""
    q = (trace.vehicle - trace.vehicle.mean(axis=0)) @ trace.axis
    velocity = local_linear_velocity(trace.time, q[:, None])[:, 0]
    valid = np.isfinite(trace.time) & np.isfinite(q) & np.isfinite(velocity)
    times = trace.time[valid]
    q = q[valid]
    velocity = velocity[valid]
    if len(times) < 10:
        return []
    grid = np.arange(times[0], times[-1] + 1.0e-9, 0.1)
    q_grid = np.interp(grid, times, q)
    v_grid = np.interp(grid, times, velocity)
    v_grid = np.convolve(np.pad(v_grid, (5, 5), mode="edge"), np.ones(11) / 11.0, mode="valid")
    active = _fill_short_gaps(np.abs(v_grid) > moving_speed_threshold_m_s, 7)

    segments: list[MotionSegment] = []
    index = 0
    while index < len(active):
        end = index + 1
        while end < len(active) and active[end] == active[index]:
            end += 1
        if active[index] and end - index >= 50:
            start_time = float(grid[index])
            end_time = float(grid[end - 1])
            q_start = float(np.interp(start_time, times, q))
            q_end = float(np.interp(end_time, times, q))
            displacement = abs(q_end - q_start)
            duration = end_time - start_time
            median_speed = float(np.median(np.abs(v_grid[index:end])))
            tag17_median, tag17_p90 = _speed_statistics(trace, start_time, end_time)
            if displacement >= min_displacement_m and duration >= MIN_MOTION_DURATION_S:
                segments.append(MotionSegment(
                    start_time, end_time, displacement, duration, median_speed,
                    tag17_median, tag17_p90,
                ))
        index = end
    return segments


def cut_control(control: dict[str, np.ndarray], segment: MotionSegment) -> dict[str, np.ndarray]:
    mask = (control["time"] >= segment.start) & (control["time"] <= segment.end)
    if np.count_nonzero(mask) < 10:
        raise ValueError(f"too few control samples in {segment}")
    time = control["time"][mask]
    return {"time": time - time[0], "error": control["error"][mask]}


def cut_speed(overhead: OverheadTrace, segment: MotionSegment) -> dict[str, np.ndarray]:
    mask = (overhead.time >= segment.start) & (overhead.time <= segment.end)
    time = overhead.time[mask]
    if len(time) < 5:
        raise ValueError(f"too few camera samples in {segment}")
    return {
        "time": time - time[0],
        "target_speed": np.linalg.norm(overhead.target_velocity[mask], axis=1),
        "vehicle_speed": np.linalg.norm(overhead.vehicle_velocity[mask], axis=1),
    }


def rolling_mean(values: np.ndarray, window: int = 21) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) < 3:
        return values.copy()
    finite = np.isfinite(values)
    if not np.any(finite):
        return values.copy()
    filled = np.interp(np.arange(len(values)), np.flatnonzero(finite), values[finite])
    window = max(3, min(window, len(values) // 2 * 2 - 1))
    kernel = np.ones(window) / window
    padded = np.pad(filled, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def metrics(error: np.ndarray) -> dict[str, float]:
    forward_cm = error[:, 0] * 100.0
    norm_cm = np.linalg.norm(error, axis=1) * 100.0
    return {
        "forward_rmse_cm": float(np.sqrt(np.mean(forward_cm**2))),
        "forward_mae_cm": float(np.mean(np.abs(forward_cm))),
        "forward_p95_cm": float(np.percentile(np.abs(forward_cm), 95)),
        "forward_peak_cm": float(np.max(np.abs(forward_cm))),
        "norm_rmse_cm": float(np.sqrt(np.mean(norm_cm**2))),
        "norm_p95_cm": float(np.percentile(norm_cm, 95)),
        "norm_peak_cm": float(np.max(norm_cm)),
    }


def classify_segment(segment: MotionSegment, threshold_m_s: float) -> str:
    speed = segment.tag17_median_speed_m_s
    if not np.isfinite(speed):
        speed = segment.median_speed_m_s
    return "20s" if speed < threshold_m_s else "10s"


def _effective_segment(segment: MotionSegment, control: dict[str, np.ndarray]) -> MotionSegment | None:
    start = max(segment.start, float(control["time"][0]))
    end = min(segment.end, float(control["time"][-1]))
    if end - start < max(4.0, 0.7 * segment.duration_s):
        return None
    return MotionSegment(
        start, end, segment.displacement_m, end - start, segment.median_speed_m_s,
        segment.tag17_median_speed_m_s, segment.tag17_p90_speed_m_s,
    )


def load_trials(
    specs: tuple[ExperimentSpec, ...] = EXPERIMENTS,
    *,
    tag17_speed_threshold_m_s: float = DEFAULT_TAG17_SPEED_THRESHOLD_M_S,
    group_override: str | None = None,
    group_min_tag17_speed_m_s: float | None = None,
) -> tuple[list[Trial], list[dict]]:
    trials: list[Trial] = []
    source_summary: list[dict] = []
    for spec in specs:
        source = {
            "method": spec.method,
            "experiment_id": spec.experiment_id,
            "control_path": str(spec.control_path),
            "overhead_path": str(spec.overhead_path),
            "segments": [],
        }
        try:
            overhead = load_smc_overhead(spec.overhead_path) if spec.smc else load_mpc_overhead(spec.overhead_path)
            control = load_control_log(spec.control_path, smc=spec.smc)
            segments = find_motion_segments(overhead)
        except (OSError, ValueError, KeyError, TypeError) as error:
            source["status"] = "load_error"
            source["error"] = str(error)
            source_summary.append(source)
            continue

        for index, segment in enumerate(segments, start=1):
            record = {
                "segment_index": index,
                "start_s": segment.start,
                "end_s": segment.end,
                "duration_s": segment.duration_s,
                "displacement_m": segment.displacement_m,
                "vehicle_median_speed_m_s": segment.median_speed_m_s,
                "tag17_median_speed_m_s": segment.tag17_median_speed_m_s,
                "tag17_p90_speed_m_s": segment.tag17_p90_speed_m_s,
            }
            effective = _effective_segment(segment, control)
            if effective is None:
                record["status"] = "excluded_control_overlap"
                source["segments"].append(record)
                continue
            try:
                control_cut = cut_control(control, effective)
                speed_cut = cut_speed(overhead, effective)
            except ValueError as error:
                record["status"] = "excluded_insufficient_samples"
                record["error"] = str(error)
                source["segments"].append(record)
                continue
            if group_override is not None:
                tag17_speed = segment.tag17_median_speed_m_s
                if (
                    group_min_tag17_speed_m_s is not None
                    and (
                        not np.isfinite(tag17_speed)
                        or tag17_speed < group_min_tag17_speed_m_s
                    )
                ):
                    record["status"] = "excluded_speed_group"
                    source["segments"].append(record)
                    continue
                group = group_override
            else:
                group = classify_segment(segment, tag17_speed_threshold_m_s)
            trial_metrics = metrics(control_cut["error"])
            trial_metrics.update({
                "duration_s": effective.duration_s,
                "displacement_m": segment.displacement_m,
                "vehicle_median_speed_m_s": segment.median_speed_m_s,
                "tag17_median_speed_m_s": segment.tag17_median_speed_m_s,
                "tag17_p90_speed_m_s": segment.tag17_p90_speed_m_s,
            })
            trials.append(Trial(spec, effective, control_cut, speed_cut, trial_metrics, group))
            record["status"] = "selected"
            record["group"] = group
            record["metrics"] = trial_metrics
            source["segments"].append(record)
        source["status"] = "ok" if segments else "no_complete_segment"
        source_summary.append(source)
    return trials, source_summary


def _resample(
    data: list[dict[str, np.ndarray]], key: str, grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stack = np.full((len(data), len(grid)), np.nan, dtype=float)
    for row_index, row in enumerate(data):
        time = np.asarray(row["time"], dtype=float)
        values = np.asarray(row[key], dtype=float)
        valid = np.isfinite(time) & np.isfinite(values)
        if np.count_nonzero(valid) < 2:
            continue
        time = time[valid]
        values = rolling_mean(values[valid], window=11)
        inside = (grid >= time[0]) & (grid <= time[-1])
        stack[row_index, inside] = np.interp(grid[inside], time, values)
    count = np.sum(np.isfinite(stack), axis=0)
    mean = np.full(len(grid), np.nan)
    p25 = np.full(len(grid), np.nan)
    p75 = np.full(len(grid), np.nan)
    for index in np.flatnonzero(count):
        values = stack[:, index]
        values = values[np.isfinite(values)]
        mean[index] = np.mean(values)
        p25[index] = np.percentile(values, 25)
        p75[index] = np.percentile(values, 75)
    return grid, mean, p25, p75, count


def _plot_method_curves(
    axis: plt.Axes,
    trials: list[Trial],
    method: str,
    data_key: str,
    color: str,
    linestyle: str,
    *,
    scale: float = 1.0,
    label: str,
) -> None:
    selected = [trial for trial in trials if trial.spec.method == method]
    for trial in selected:
        if data_key == "forward_abs":
            time = trial.control["time"]
            values = np.abs(trial.control["error"][:, 0]) * scale
        else:
            time = trial.speed["time"]
            values = trial.speed[data_key] * scale
        axis.plot(time, values, color=color, alpha=0.16, linewidth=0.7)

    if not selected:
        return
    source = []
    for trial in selected:
        if data_key == "forward_abs":
            source.append({"time": trial.control["time"], "forward_abs": np.abs(trial.control["error"][:, 0]) * scale})
        else:
            source.append({"time": trial.speed["time"], data_key: trial.speed[data_key] * scale})
    max_time = max(float(row["time"][-1]) for row in source)
    grid = np.arange(0.0, max_time + 0.0501, 0.05)
    _, mean, p25, p75, _ = _resample(source, data_key, grid)
    axis.fill_between(grid, p25, p75, color=color, alpha=0.10, linewidth=0)
    axis.plot(grid, mean, color=color, linestyle=linestyle, linewidth=2.4, label=f"{label} mean (n={len(selected)})")


def _group_metrics(trials: list[Trial]) -> dict[str, object]:
    result: dict[str, object] = {}
    for method in ("SMC", "MPC"):
        selected = [trial for trial in trials if trial.spec.method == method]
        if not selected:
            result[method] = {"n": 0}
            continue
        keys = tuple(selected[0].metrics.keys())
        result[method] = {
            "n": len(selected),
            "median_across_trials": {
                key: float(np.median([trial.metrics[key] for trial in selected])) for key in keys
            },
            "mean_across_trials": {
                key: float(np.mean([trial.metrics[key] for trial in selected])) for key in keys
            },
            "trial_metrics": [
                {
                    "experiment_id": trial.spec.experiment_id,
                    "segment_start_s": trial.segment.start,
                    "segment_end_s": trial.segment.end,
                    **trial.metrics,
                }
                for trial in selected
            ],
        }
    return result


def plot_group(trials: list[Trial], group: str, output: Path) -> dict[str, object]:
    group_trials = [trial for trial in trials if trial.group == group]
    if not group_trials:
        raise ValueError(f"no trials in {group}")
    if group == "20s":
        label = "Low speed: nominal 200 cm / 20 s"
    elif group == "10s":
        label = "High speed: nominal 200 cm / 10 s"
    else:
        label = "Very high speed: nominal 200 cm / 5 s"
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), constrained_layout=False)
    forward_ax, norm_ax, speed_ax = axes

    for method in ("SMC", "MPC"):
        method_trials = [trial for trial in group_trials if trial.spec.method == method]
        if not method_trials:
            continue
        color = COLORS[method]
        linestyle = LINESTYLES[method]
        _plot_method_curves(forward_ax, method_trials, method, "forward_abs", color, linestyle, scale=100.0, label=method)

        norm_trials: list[dict[str, np.ndarray]] = []
        for trial in method_trials:
            norm_trials.append({
                "time": trial.control["time"],
                "norm_error": np.linalg.norm(trial.control["error"], axis=1) * 100.0,
            })
        for row in norm_trials:
            norm_ax.plot(row["time"], row["norm_error"], color=color, alpha=0.16, linewidth=0.7)
        max_time = max(float(row["time"][-1]) for row in norm_trials)
        grid = np.arange(0.0, max_time + 0.0501, 0.05)
        _, mean, p25, p75, _ = _resample(norm_trials, "norm_error", grid)
        norm_ax.fill_between(grid, p25, p75, color=color, alpha=0.10, linewidth=0)
        norm_ax.plot(grid, mean, color=color, linestyle=linestyle, linewidth=2.4, label=f"{method} mean (n={len(method_trials)})")

        _plot_method_curves(speed_ax, method_trials, method, "target_speed", color, linestyle, label=f"{method} Tag17")

    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_xlim(left=0.0)

    forward_ax.axhline(5.0, color="#f2a541", linestyle=":", linewidth=0.9)
    forward_ax.set_title("Forward position-error magnitude — moving interval only")
    forward_ax.set_ylabel("|Forward error| (cm)")
    forward_ax.legend(loc="upper right")

    norm_ax.axhline(5.0, color="#f2a541", linestyle=":", linewidth=0.9)
    norm_ax.set_title("3D position-error norm — moving interval only")
    norm_ax.set_ylabel("‖error‖ (cm)")
    norm_ax.legend(loc="upper right")

    speed_ax.set_title("Pool-top camera Tag17 speed — grouping / motion check")
    speed_ax.set_xlabel("Elapsed time after motion-segment start (s)")
    speed_ax.set_ylabel("Tag17 speed (m/s)")
    speed_ax.legend(loc="upper right")

    durations = [trial.segment.duration_s for trial in group_trials]
    displacements = [trial.segment.displacement_m for trial in group_trials]
    tag_speeds = [trial.segment.tag17_median_speed_m_s for trial in group_trials if np.isfinite(trial.segment.tag17_median_speed_m_s)]
    count_text = (
        f"SMC n={sum(trial.spec.method == 'SMC' for trial in group_trials)} | "
        f"MPC n={sum(trial.spec.method == 'MPC' for trial in group_trials)}"
    )
    note_text = (
        f"camera duration {min(durations):.1f}–{max(durations):.1f} s | "
        f"displacement {min(displacements) * 100:.0f}–{max(displacements) * 100:.0f} cm | "
        f"Tag17 median speed {min(tag_speeds):.3f}–{max(tag_speeds):.3f} m/s\n"
        "mean curve; shaded 25–75 percentile; stops removed; forward error shown as magnitude"
    )
    fig.suptitle(f"{label} — SMC vs MPC\n{count_text}", fontsize=14, fontweight="bold", y=0.985)
    fig.text(0.5, 0.945, note_text, ha="center", va="top", fontsize=9)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.08, hspace=0.42)
    fig.savefig(output, dpi=220, facecolor="white")
    plt.close(fig)
    return {
        "label": label,
        "output": str(output),
        "n_trials": len(group_trials),
        "metrics": _group_metrics(group_trials),
        "trial_ids": [f"{trial.spec.method}:{trial.spec.experiment_id}" for trial in group_trials],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=SUMMARY_OUTPUT)
    parser.add_argument("--output-low", type=Path, default=OUTPUT_LOW_CORRECTED)
    parser.add_argument("--output-high", type=Path, default=OUTPUT_HIGH_CORRECTED)
    parser.add_argument("--output-five", type=Path, default=OUTPUT_FIVE)
    parser.add_argument("--tag17-speed-threshold", type=float, default=DEFAULT_TAG17_SPEED_THRESHOLD_M_S)
    args = parser.parse_args()

    trials, sources = load_trials(tag17_speed_threshold_m_s=args.tag17_speed_threshold)
    if not trials:
        raise SystemExit("no complete motion trials were found")
    low = [trial for trial in trials if trial.group == "20s"]
    high = [trial for trial in trials if trial.group == "10s"]
    if not low or not high:
        raise SystemExit(f"speed grouping failed: low={len(low)}, high={len(high)}")

    print("Selected trials:")
    for trial in trials:
        print(
            f"  {trial.group:>3} {trial.spec.method:<3} {trial.spec.experiment_id} "
            f"duration={trial.segment.duration_s:.1f}s displacement={trial.segment.displacement_m:.2f}m "
            f"Tag17={trial.segment.tag17_median_speed_m_s:.3f}m/s"
        )

    five_trials, five_sources = load_trials(
        FIVE_SECOND_EXPERIMENTS,
        group_override="5s",
        group_min_tag17_speed_m_s=0.12,
    )
    if not five_trials:
        raise SystemExit("no complete 5 s motion trials were found")
    print("Selected 5 s trials:")
    for trial in five_trials:
        print(
            f"  {trial.spec.method:<3} {trial.spec.experiment_id} "
            f"duration={trial.segment.duration_s:.1f}s displacement={trial.segment.displacement_m:.2f}m "
            f"Tag17={trial.segment.tag17_median_speed_m_s:.3f}m/s"
        )

    summary = {
        "reference_position_body_frd_m": REFERENCE.tolist(),
        "source_date": "2026-08-16",
        "selection": {
            "moving_speed_threshold_m_s": MOVING_SPEED_THRESHOLD_M_S,
            "minimum_camera_displacement_m": MIN_MOTION_DISPLACEMENT_M,
            "minimum_motion_duration_s": MIN_MOTION_DURATION_S,
            "tag17_median_speed_threshold_m_s": args.tag17_speed_threshold,
            "low_speed_group": "Tag17 median speed below threshold; nominal 200 cm / 20 s",
            "high_speed_group": "Tag17 median speed at or above threshold; nominal 200 cm / 10 s",
            "five_second_group": "Tag17 median speed at or above 0.12 m/s; nominal 200 cm / 5 s",
            "forward_error_plot": "absolute forward error magnitude; sign differences from bidirectional runs are removed",
            "aggregate_curve": "mean across trials with 25-75 percentile band",
            "stop_handling": "only camera-detected moving intervals are retained; post-run holds are excluded",
        },
        "groups": {
            "20s": plot_group(low, "20s", args.output_low),
            "10s": plot_group(high, "10s", args.output_high),
            "5s": plot_group(five_trials, "5s", args.output_five),
        },
        "sources": sources + five_sources,
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {args.output_low}")
    print(f"saved: {args.output_high}")
    print(f"saved: {args.output_five}")
    print(f"saved: {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
