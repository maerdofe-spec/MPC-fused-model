"""Run the reproducible OpenAUV two-DOF saturated-SMC simulation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from openauv_smc import (
    OpenAUVState,
    build_default_controller,
    build_openauv_model,
    wrap_angle,
)


def reference_at(time_s: float) -> tuple[float, float]:
    """Piecewise depth [m] and yaw [rad] commands used by the demo."""

    if time_s < 2.0:
        depth = 0.0
    elif time_s < 24.0:
        depth = 1.0
    else:
        depth = 1.5

    if time_s < 12.0:
        yaw = 0.0
    elif time_s < 36.0:
        yaw = math.radians(60.0)
    else:
        yaw = math.radians(-30.0)
    return depth, yaw


def disturbance_at(time_s: float) -> tuple[float, float]:
    """Repeatable waves plus finite force/moment pulses."""

    heave = 3.0 * math.sin(0.55 * time_s)
    yaw = 0.20 * math.sin(0.80 * time_s + 0.3)
    if 18.0 <= time_s < 22.0:
        heave += 16.0
    if 29.0 <= time_s < 33.0:
        yaw -= 2.2
    return heave, yaw


def run_simulation(
    duration: float = 55.0,
    dt: float = 0.01,
    *,
    with_disturbance: bool = True,
) -> dict[str, np.ndarray]:
    if duration <= 0.0 or dt <= 0.0:
        raise ValueError("duration and dt must be positive")

    model = build_openauv_model()
    controller = build_default_controller(model)
    state = OpenAUVState()
    steps = int(round(duration / dt)) + 1

    columns = {
        "time_s": [],
        "depth_m": [],
        "depth_ref_m": [],
        "heave_velocity_m_s": [],
        "heave_rate_ref_m_s": [],
        "heave_force_n": [],
        "heave_disturbance_n": [],
        "yaw_rad": [],
        "yaw_ref_rad": [],
        "yaw_rate_rad_s": [],
        "yaw_rate_ref_rad_s": [],
        "yaw_moment_nm": [],
        "yaw_disturbance_nm": [],
        "heave_sliding": [],
        "yaw_sliding": [],
    }

    for step in range(steps):
        time_s = step * dt
        depth_ref, yaw_ref = reference_at(time_s)
        if with_disturbance:
            heave_disturbance, yaw_disturbance = disturbance_at(time_s)
        else:
            heave_disturbance, yaw_disturbance = 0.0, 0.0

        output = controller.compute(state, depth_ref, yaw_ref, dt)

        columns["time_s"].append(time_s)
        columns["depth_m"].append(state.depth)
        columns["depth_ref_m"].append(depth_ref)
        columns["heave_velocity_m_s"].append(state.heave_velocity)
        columns["heave_rate_ref_m_s"].append(output.heave.rate_reference)
        columns["heave_force_n"].append(output.heave_force)
        columns["heave_disturbance_n"].append(heave_disturbance)
        columns["yaw_rad"].append(state.yaw)
        columns["yaw_ref_rad"].append(yaw_ref)
        columns["yaw_rate_rad_s"].append(state.yaw_rate)
        columns["yaw_rate_ref_rad_s"].append(output.yaw.rate_reference)
        columns["yaw_moment_nm"].append(output.yaw_moment)
        columns["yaw_disturbance_nm"].append(yaw_disturbance)
        columns["heave_sliding"].append(output.heave.sliding_variable)
        columns["yaw_sliding"].append(output.yaw.sliding_variable)

        if step < steps - 1:
            state = model.step_rk4(
                state,
                output.heave_force,
                output.yaw_moment,
                dt,
                heave_disturbance,
                yaw_disturbance,
            )

    return {name: np.asarray(values, dtype=float) for name, values in columns.items()}


def calculate_metrics(history: dict[str, np.ndarray]) -> dict[str, float]:
    depth_error = history["depth_ref_m"] - history["depth_m"]
    yaw_error = np.array(
        [
            wrap_angle(reference - actual)
            for reference, actual in zip(
                history["yaw_ref_rad"], history["yaw_rad"], strict=True
            )
        ]
    )

    active = history["time_s"] >= 2.0
    final_window = history["time_s"] >= history["time_s"][-1] - 5.0
    return {
        "depth_rmse_m": float(np.sqrt(np.mean(depth_error[active] ** 2))),
        "depth_max_abs_error_m": float(np.max(np.abs(depth_error[active]))),
        "depth_final_mean_abs_error_m": float(
            np.mean(np.abs(depth_error[final_window]))
        ),
        "yaw_rmse_deg": float(np.degrees(np.sqrt(np.mean(yaw_error[active] ** 2)))),
        "yaw_max_abs_error_deg": float(np.degrees(np.max(np.abs(yaw_error[active])))),
        "yaw_final_mean_abs_error_deg": float(
            np.degrees(np.mean(np.abs(yaw_error[final_window])))
        ),
        "max_abs_heave_force_n": float(np.max(np.abs(history["heave_force_n"]))),
        "max_abs_yaw_moment_nm": float(np.max(np.abs(history["yaw_moment_nm"]))),
    }


def save_csv(history: dict[str, np.ndarray], path: Path) -> None:
    names = list(history)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(names)
        writer.writerows(zip(*(history[name] for name in names), strict=True))


def save_plot(history: dict[str, np.ndarray], path: Path) -> None:
    time_s = history["time_s"]
    yaw_deg = np.degrees(history["yaw_rad"])
    yaw_ref_deg = np.degrees(history["yaw_ref_rad"])

    figure, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)

    axes[0, 0].plot(time_s, history["depth_ref_m"], "k--", label="reference")
    axes[0, 0].plot(time_s, history["depth_m"], color="#0068b5", label="depth")
    axes[0, 0].set_ylabel("Depth z [m], down +")
    axes[0, 0].set_title("Depth tracking")
    axes[0, 0].legend(loc="best")

    axes[1, 0].plot(time_s, history["heave_force_n"], color="#d1495b", label="Fz")
    axes[1, 0].plot(
        time_s,
        history["heave_disturbance_n"],
        color="#777777",
        alpha=0.75,
        label="disturbance",
    )
    axes[1, 0].set_ylabel("Force [N]")
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 0].set_title("Heave input and disturbance")
    axes[1, 0].legend(loc="best")

    axes[0, 1].plot(time_s, yaw_ref_deg, "k--", label="reference")
    axes[0, 1].plot(time_s, yaw_deg, color="#00876c", label="yaw")
    axes[0, 1].set_ylabel("Yaw psi [deg], right +")
    axes[0, 1].set_title("Yaw tracking")
    axes[0, 1].legend(loc="best")

    axes[1, 1].plot(time_s, history["yaw_moment_nm"], color="#d1495b", label="N")
    axes[1, 1].plot(
        time_s,
        history["yaw_disturbance_nm"],
        color="#777777",
        alpha=0.75,
        label="disturbance",
    )
    axes[1, 1].set_ylabel("Moment [N m]")
    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].set_title("Yaw input and disturbance")
    axes[1, 1].legend(loc="best")

    for axis in axes.flat:
        axis.grid(True, alpha=0.25)

    figure.suptitle("Sliding Mode Control - OpenAUV 2-DOF simulation")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=55.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--no-disturbance",
        action="store_true",
        help="disable the deterministic disturbance scenario",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    history = run_simulation(
        duration=args.duration,
        dt=args.dt,
        with_disturbance=not args.no_disturbance,
    )
    metrics = calculate_metrics(history)

    save_csv(history, output_dir / "simulation_history.csv")
    save_plot(history, output_dir / "tracking_results.png")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved results to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
