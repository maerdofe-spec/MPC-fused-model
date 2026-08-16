#!/usr/bin/env python3
"""Compare one low-speed and one high-speed 200 cm motion segment."""

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
MPC_LOG = ROOT / "calibration_logs" / "experimental_auto_20260816_172732.jsonl"
MPC_BAG = (
    ROOT
    / "calibration_logs"
    / "top_camera_experiment_20260816_172625"
    / "top_camera_experiment_20260816_172625_0.db3"
)
SMC_LOG = ROOT / "calibration_logs" / "smc_real_20260816_174133.jsonl"
SMC_CSV = ROOT / "calibration_logs" / "smc_synced_overhead_20260816_174133.csv"
OUTPUT_LOW = ROOT / "calibration_logs" / "smc_mpc_200cm_low_speed_20260816.png"
OUTPUT_HIGH = ROOT / "calibration_logs" / "smc_mpc_200cm_high_speed_20260816.png"
SUMMARY_OUTPUT = ROOT / "calibration_logs" / "smc_mpc_200cm_dynamic_summary_20260816.json"

COLORS = {"MPC": "#1f77b4", "SMC-new": "#d62728"}


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
        target_velocity[target_valid] = local_linear_velocity(
            times[target_valid], target[target_valid]
        )
    return OverheadTrace(times, vehicle, target, vehicle_velocity, target_velocity, axis)


def load_mpc_overhead(path: Path) -> OverheadTrace:
    data = _load_overhead(path, read_only_snapshot=True)
    vehicle = np.asarray(data["vehicle"], dtype=float)
    axis = np.linalg.svd(vehicle - vehicle.mean(axis=0), full_matrices=False)[2][0]
    return OverheadTrace(
        np.asarray(data["time"], dtype=float),
        vehicle,
        np.asarray(data["target"], dtype=float),
        np.asarray(data["vehicle_velocity"], dtype=float),
        np.asarray(data["target_velocity"], dtype=float),
        axis,
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


def find_motion_segments(trace: OverheadTrace) -> list[MotionSegment]:
    q = (trace.vehicle - trace.vehicle.mean(axis=0)) @ trace.axis
    velocity = local_linear_velocity(trace.time, q[:, None])[:, 0]
    valid = np.isfinite(trace.time) & np.isfinite(q) & np.isfinite(velocity)
    times = trace.time[valid]
    q = q[valid]
    velocity = velocity[valid]
    grid = np.arange(times[0], times[-1] + 1.0e-9, 0.1)
    q_grid = np.interp(grid, times, q)
    v_grid = np.interp(grid, times, velocity)
    v_grid = np.convolve(np.pad(v_grid, (5, 5), mode="edge"), np.ones(11) / 11.0, mode="valid")
    active = np.abs(v_grid) > 0.015
    active = _fill_short_gaps(active, 7)

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
            if displacement >= 0.8 and duration >= 4.0:
                segments.append(
                    MotionSegment(start_time, end_time, displacement, duration, median_speed)
                )
        index = end
    return segments


def select_speed_segments(trace: OverheadTrace) -> tuple[MotionSegment, MotionSegment, list[MotionSegment]]:
    segments = find_motion_segments(trace)
    if len(segments) < 2:
        raise ValueError(f"could not find two complete motion segments; found {segments}")
    low = max(segments, key=lambda segment: segment.duration_s)
    high = max(segments, key=lambda segment: segment.median_speed_m_s)
    if low.start == high.start and len(segments) > 1:
        high = min(segments, key=lambda segment: segment.duration_s)
    return low, high, segments


def cut_control(control: dict[str, np.ndarray], segment: MotionSegment) -> dict[str, np.ndarray]:
    mask = (control["time"] >= segment.start) & (control["time"] <= segment.end)
    if np.count_nonzero(mask) < 10:
        raise ValueError(f"too few control samples in {segment}")
    time = control["time"][mask]
    return {"time": time - time[0], "error": control["error"][mask]}


def cut_speed(overhead: OverheadTrace, segment: MotionSegment) -> dict[str, np.ndarray]:
    mask = (overhead.time >= segment.start) & (overhead.time <= segment.end)
    time = overhead.time[mask]
    target_speed = np.linalg.norm(overhead.target_velocity[mask], axis=1)
    vehicle_speed = np.linalg.norm(overhead.vehicle_velocity[mask], axis=1)
    return {
        "time": time - time[0],
        "target_speed": target_speed,
        "vehicle_speed": vehicle_speed,
    }


def rolling_mean(values: np.ndarray, window: int = 21) -> np.ndarray:
    if len(values) < 3:
        return values.copy()
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return values
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


def plot_one(
    label: str,
    smc: dict[str, np.ndarray],
    mpc: dict[str, np.ndarray],
    smc_speed: dict[str, np.ndarray],
    mpc_speed: dict[str, np.ndarray],
    smc_segment: MotionSegment,
    mpc_segment: MotionSegment,
    output: Path,
) -> dict[str, dict[str, float]]:
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), constrained_layout=True)
    error_ax, norm_ax, speed_ax = axes
    result: dict[str, dict[str, float]] = {}
    for name, data, segment, color, linestyle in (
        ("SMC-new", smc, smc_segment, COLORS["SMC-new"], "-"),
        ("MPC", mpc, mpc_segment, COLORS["MPC"], "--"),
    ):
        error = data["error"]
        result[name] = metrics(error)
        forward_cm = error[:, 0] * 100.0
        norm_cm = np.linalg.norm(error, axis=1) * 100.0
        error_ax.plot(data["time"], forward_cm, color=color, alpha=0.18, linewidth=0.6)
        error_ax.plot(
            data["time"], rolling_mean(forward_cm), color=color, linestyle=linestyle,
            linewidth=2.0, label=name,
        )
        norm_ax.plot(data["time"], norm_cm, color=color, alpha=0.18, linewidth=0.6)
        norm_ax.plot(
            data["time"], rolling_mean(norm_cm), color=color, linestyle=linestyle,
            linewidth=2.0, label=name,
        )
        speed = smc_speed if name == "SMC-new" else mpc_speed
        speed_ax.plot(
            speed["time"], speed["target_speed"], color=color, linestyle=linestyle,
            linewidth=1.6, alpha=0.85, label=f"{name} Tag17",
        )
        speed_ax.plot(
            speed["time"], speed["vehicle_speed"], color=color, linestyle=":",
            linewidth=1.0, alpha=0.65, label=f"{name} ROV",
        )
        result[name]["segment_duration_s"] = segment.duration_s
        result[name]["segment_displacement_m"] = segment.displacement_m
        result[name]["median_vehicle_speed_m_s"] = segment.median_speed_m_s

    for axis in (error_ax, norm_ax, speed_ax):
        axis.grid(True, alpha=0.25)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    error_ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    error_ax.axhline(5.0, color="#f2a541", linestyle=":", linewidth=0.9)
    error_ax.axhline(-5.0, color="#f2a541", linestyle=":", linewidth=0.9)
    error_ax.set_title("Forward position error — stops removed")
    error_ax.set_ylabel("Error (cm)")
    error_ax.legend(loc="upper right")
    norm_ax.axhline(5.0, color="#f2a541", linestyle=":", linewidth=0.9, label="5 cm")
    norm_ax.set_title("3D position-error norm — stops removed")
    norm_ax.set_ylabel("‖error‖ (cm)")
    norm_ax.legend(loc="upper right")
    speed_ax.set_title("Motion-speed check from pool-top camera")
    speed_ax.set_xlabel("Time since selected 200 cm segment start (s)")
    speed_ax.set_ylabel("Speed (m/s)")
    speed_ax.legend(ncol=2, loc="upper right")
    fig.suptitle(
        f"{label} 200 cm segment — SMC-new vs MPC\n"
        f"SMC duration {smc_segment.duration_s:.1f}s / {smc_segment.displacement_m*100:.0f}cm | "
        f"MPC duration {mpc_segment.duration_s:.1f}s / {mpc_segment.displacement_m*100:.0f}cm",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(output, dpi=220, facecolor="white")
    plt.close(fig)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mpc-log", type=Path, default=MPC_LOG)
    parser.add_argument("--mpc-bag", type=Path, default=MPC_BAG)
    parser.add_argument("--smc-log", type=Path, default=SMC_LOG)
    parser.add_argument("--smc-csv", type=Path, default=SMC_CSV)
    parser.add_argument("--summary", type=Path, default=SUMMARY_OUTPUT)
    args = parser.parse_args()

    mpc_overhead = load_mpc_overhead(args.mpc_bag)
    smc_overhead = load_smc_overhead(args.smc_csv)
    mpc_control = load_control_log(args.mpc_log, smc=False)
    smc_control = load_control_log(args.smc_log, smc=True)
    mpc_low, mpc_high, mpc_segments = select_speed_segments(mpc_overhead)
    smc_low, smc_high, smc_segments = select_speed_segments(smc_overhead)

    print("MPC motion segments:")
    for segment in mpc_segments:
        print(segment)
    print("SMC-new motion segments:")
    for segment in smc_segments:
        print(segment)
    print("selected low:", smc_low, mpc_low)
    print("selected high:", smc_high, mpc_high)

    smc_low_data = cut_control(smc_control, smc_low)
    mpc_low_data = cut_control(mpc_control, mpc_low)
    smc_high_data = cut_control(smc_control, smc_high)
    mpc_high_data = cut_control(mpc_control, mpc_high)
    smc_low_speed = cut_speed(smc_overhead, smc_low)
    mpc_low_speed = cut_speed(mpc_overhead, mpc_low)
    smc_high_speed = cut_speed(smc_overhead, smc_high)
    mpc_high_speed = cut_speed(mpc_overhead, mpc_high)

    summary = {
        "reference_position_body_frd_m": REFERENCE.tolist(),
        "selection": {"low": "longest complete moving segment", "high": "highest median moving speed segment"},
        "low": plot_one(
            "Low-speed", smc_low_data, mpc_low_data, smc_low_speed, mpc_low_speed,
            smc_low, mpc_low, OUTPUT_LOW,
        ),
        "high": plot_one(
            "High-speed", smc_high_data, mpc_high_data, smc_high_speed, mpc_high_speed,
            smc_high, mpc_high, OUTPUT_HIGH,
        ),
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {OUTPUT_LOW}")
    print(f"saved: {OUTPUT_HIGH}")
    print(f"saved: {args.summary}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
