#!/usr/bin/env python3
"""Plot Forward/3D errors and pool-top Tag17 target speed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from MPC_dual_model.fusion_identifiability_validation import _load_overhead


DEFAULT_SMC = ROOT / "calibration_logs" / "smc_real_20260816_164712.jsonl"
DEFAULT_MPC = ROOT / "calibration_logs" / "experimental_auto_20260816_165314.jsonl"
DEFAULT_OVERHEAD = (
    ROOT
    / "calibration_logs"
    / "top_camera_experiment_20260816_163841"
    / "top_camera_experiment_20260816_163841_0.db3"
)
DEFAULT_CONFIG = ROOT / "MPC_dual_model" / "finesub_v4pro1_mpc.json"
DEFAULT_OUTPUT = ROOT / "calibration_logs" / "smc_mpc_comparison_20260816.png"

CONTROLLER_COLORS = {"SMC": "#d62728", "MPC": "#1f77b4"}


def load_reference(config_path: Path) -> np.ndarray:
    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    reference = np.asarray(
        config["experimental_auto"]["active_mpc_parameters"]["controller"][
            "reference_position"
        ],
        dtype=float,
    )
    if reference.shape != (3,) or not np.all(np.isfinite(reference)):
        raise ValueError("reference_position must be a finite 3-vector")
    return reference


def load_trace(path: Path, reference: np.ndarray) -> dict[str, np.ndarray]:
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
        raise ValueError(f"no control_update records found in {path}")

    times = np.asarray(
        [row.get("host_monotonic_s", row.get("host_time_s")) for row in rows],
        dtype=float,
    )
    estimated = np.asarray([row["estimated_state"][:3] for row in rows], dtype=float)
    valid = np.isfinite(times) & np.all(np.isfinite(estimated), axis=1)
    times = times[valid]
    estimated = estimated[valid]
    if len(times) == 0:
        raise ValueError(f"no finite samples found in {path}")

    error = estimated - reference
    return {
        "time_s": times - times[0],
        "error_m": error,
        "error_norm_m": np.linalg.norm(error, axis=1),
    }


def load_tag17_speed(path: Path) -> dict[str, np.ndarray]:
    """Load absolute velocity of Tag17 from a pool-top ROS2 bag."""
    overhead = _load_overhead(path, read_only_snapshot=True)
    time_s = np.asarray(overhead["time"], dtype=float)
    target_velocity = np.asarray(overhead["target_velocity"], dtype=float)
    speed_m_s = np.linalg.norm(target_velocity, axis=1)
    valid_time = np.isfinite(time_s)
    if not np.any(valid_time & np.isfinite(speed_m_s)):
        raise ValueError(f"no finite Tag17 target velocity found in {path}")
    time_s = time_s[valid_time]
    speed_m_s = speed_m_s[valid_time]
    return {"time_s": time_s - time_s[0], "speed_m_s": speed_m_s}


def rolling_mean(values: np.ndarray, window: int = 31) -> np.ndarray:
    if len(values) < 3:
        return values.copy()
    window = max(3, min(window, len(values) // 2 * 2 - 1))
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def plot_comparison(
    smc: dict[str, np.ndarray],
    mpc: dict[str, np.ndarray],
    tag17: dict[str, np.ndarray],
    output: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 120,
        }
    )
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), constrained_layout=True)
    error_ax, norm_ax, speed_ax = axes

    for name, data, linestyle in (("SMC", smc, "-"), ("MPC", mpc, "--")):
        color = CONTROLLER_COLORS[name]
        forward_cm = data["error_m"][:, 0] * 100.0
        error_ax.plot(
            data["time_s"], forward_cm, color=color, linewidth=0.65,
            alpha=0.20, linestyle=linestyle,
        )
        error_ax.plot(
            data["time_s"], rolling_mean(forward_cm), color=color,
            linewidth=2.0, linestyle=linestyle, label=name,
        )
    error_ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    error_ax.axhline(5.0, color="#f2a541", linestyle=":", linewidth=0.9, alpha=0.8)
    error_ax.axhline(-5.0, color="#f2a541", linestyle=":", linewidth=0.9, alpha=0.8)
    error_ax.set_title("Forward position error")
    error_ax.set_xlabel("Elapsed time (s)")
    error_ax.set_ylabel("Error (cm)")
    error_ax.legend(loc="upper right")

    for name, data, color in (
        ("SMC", smc, CONTROLLER_COLORS["SMC"]),
        ("MPC", mpc, CONTROLLER_COLORS["MPC"]),
    ):
        norm_cm = data["error_norm_m"] * 100.0
        norm_ax.plot(
            data["time_s"], norm_cm, color=color, linewidth=0.65, alpha=0.20
        )
        norm_ax.plot(
            data["time_s"], rolling_mean(norm_cm), color=color, linewidth=2.0,
            label=name,
        )
    for level, color in ((3.0, "#54a24b"), (5.0, "#f2a541"), (10.0, "#b279a2")):
        norm_ax.axhline(level, color=color, linestyle=":", linewidth=0.9, alpha=0.8)
    norm_ax.set_title("3D position-error norm")
    norm_ax.set_xlabel("Elapsed time (s)")
    norm_ax.set_ylabel("‖error‖ (cm)")
    norm_ax.legend(loc="upper right")

    speed_m_s = tag17["speed_m_s"]
    speed_valid = np.isfinite(speed_m_s)
    speed_ax.plot(
        tag17["time_s"][speed_valid], speed_m_s[speed_valid], color="#f28e2b",
        linewidth=0.65, alpha=0.20,
    )
    speed_ax.plot(
        tag17["time_s"][speed_valid], rolling_mean(speed_m_s[speed_valid]),
        color="#f28e2b", linewidth=2.0,
        label="Tag17 target speed",
    )
    speed_ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    speed_ax.set_title("Pool-top camera target speed — AprilTag 17")
    speed_ax.set_xlabel("Elapsed time in Tag17 bag (s)")
    speed_ax.set_ylabel("Absolute speed (m/s)")
    speed_ax.set_xlim(0.0, float(tag17["time_s"][-1]))
    speed_ax.legend(loc="upper right")

    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    smc_rms = np.sqrt(np.mean(np.sum(smc["error_m"] ** 2, axis=1))) * 100.0
    mpc_rms = np.sqrt(np.mean(np.sum(mpc["error_m"] ** 2, axis=1))) * 100.0
    fig.suptitle(
        "SMC vs MPC — Forward/3D error and Tag17 target speed\n"
        f"RMS 3D error: SMC {smc_rms:.2f} cm | MPC {mpc_rms:.2f} cm    "
        f"Tag17 valid samples: {int(np.count_nonzero(speed_valid))}",
        fontsize=15,
        fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, facecolor="white")
    plt.close(fig)

    print(f"saved: {output}")
    print(
        f"SMC samples={len(smc['time_s'])}, duration={smc['time_s'][-1]:.2f}s, "
        f"rms_error={smc_rms:.3f}cm"
    )
    print(
        f"MPC samples={len(mpc['time_s'])}, duration={mpc['time_s'][-1]:.2f}s, "
        f"rms_error={mpc_rms:.3f}cm"
    )
    print(
        f"Tag17 valid samples={int(np.count_nonzero(speed_valid))}, "
        f"bag duration={tag17['time_s'][-1]:.2f}s, "
        f"mean_speed={np.nanmean(speed_m_s):.4f}m/s"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smc", type=Path, default=DEFAULT_SMC)
    parser.add_argument("--mpc", type=Path, default=DEFAULT_MPC)
    parser.add_argument("--overhead-bag", type=Path, default=DEFAULT_OVERHEAD)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    reference = load_reference(args.config)
    smc = load_trace(args.smc, reference)
    mpc = load_trace(args.mpc, reference)
    tag17 = load_tag17_speed(args.overhead_bag)
    plot_comparison(smc, mpc, tag17, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
