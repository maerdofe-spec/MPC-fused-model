"""Live position-error plot for FineSUB experimental AUTO JSONL traces."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from .auto_readiness import DEFAULT_CONFIG_PATH


AXIS_NAMES = ("Forward", "Right", "Down")
AXIS_COLORS = ("tab:red", "tab:green", "tab:blue")
ERROR_REFERENCE_LEVELS_CM = (5.0, 10.0)


@dataclass(frozen=True)
class PositionErrorSample:
    time_s: float
    measured_error_m: np.ndarray
    estimated_error_m: np.ndarray
    estimated_velocity_m_s: np.ndarray
    model1_weight: np.ndarray


def load_default_reference(config_path: str | Path) -> np.ndarray:
    with Path(config_path).open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    reference = config["experimental_auto"]["active_mpc_parameters"][
        "controller"
    ]["reference_position"]
    result = np.asarray(reference, dtype=float).reshape(-1)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError("configured reference_position must be a finite 3-vector")
    return result


def extract_position_error_sample(
    record: dict,
    default_reference: np.ndarray,
) -> PositionErrorSample | None:
    if record.get("event") != "control_update":
        return None
    reference_value = record.get("reference_position_body_frd_m")
    reference = (
        default_reference
        if reference_value is None
        else np.asarray(reference_value, dtype=float).reshape(-1)
    )
    measured = np.asarray(record.get("position_body_frd_m"), dtype=float).reshape(-1)
    estimated_state = np.asarray(record.get("estimated_state"), dtype=float).reshape(-1)
    if reference.shape != (3,) or measured.shape != (3,) or estimated_state.size < 3:
        return None
    estimated = estimated_state[:3]
    values = np.concatenate((reference, measured, estimated))
    if not np.all(np.isfinite(values)):
        return None
    timestamp = float(record.get("host_monotonic_s", record.get("host_time_s")))
    if not np.isfinite(timestamp):
        return None
    velocity = np.full(3, np.nan)
    if estimated_state.size >= 6 and np.all(np.isfinite(estimated_state[3:6])):
        velocity = estimated_state[3:6].copy()
    weight_value = record.get("model1_weight")
    model1_weight = np.full(3, np.nan)
    if weight_value is not None:
        candidate = np.asarray(weight_value, dtype=float).reshape(-1)
        if candidate.shape == (3,) and np.all(np.isfinite(candidate)):
            model1_weight = candidate
    return PositionErrorSample(
        time_s=timestamp,
        measured_error_m=measured - reference,
        estimated_error_m=estimated - reference,
        estimated_velocity_m_s=velocity,
        model1_weight=model1_weight,
    )


class JsonlTraceFollower:
    """Incrementally follow one trace, or automatically select the latest trace."""

    def __init__(
        self,
        trace_path: str | Path | None,
        log_directory: str | Path,
        default_reference: np.ndarray,
    ) -> None:
        self.explicit_path = None if trace_path is None else Path(trace_path).resolve()
        self.log_directory = Path(log_directory).resolve()
        self.default_reference = np.asarray(default_reference, dtype=float)
        self.path: Path | None = None
        self.offset = 0
        self.partial = ""

    def _latest_trace(self) -> Path | None:
        candidates = list(self.log_directory.glob("experimental_auto_*.jsonl"))
        if not candidates:
            return None
        return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))

    def _selected_path(self) -> Path | None:
        return self.explicit_path or self._latest_trace()

    def poll(self) -> tuple[bool, list[PositionErrorSample]]:
        selected = self._selected_path()
        switched = selected is not None and selected != self.path
        if switched:
            self.path = selected
            self.offset = 0
            self.partial = ""
        if self.path is None or not self.path.exists():
            return switched, []
        size = self.path.stat().st_size
        if size < self.offset:
            self.offset = 0
            self.partial = ""
            switched = True
        with self.path.open("r", encoding="utf-8") as stream:
            stream.seek(self.offset)
            chunk = stream.read()
            self.offset = stream.tell()
        if not chunk:
            return switched, []
        text = self.partial + chunk
        lines = text.splitlines(keepends=True)
        if lines and not lines[-1].endswith(("\n", "\r")):
            self.partial = lines.pop()
        else:
            self.partial = ""
        samples: list[PositionErrorSample] = []
        for line in lines:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample = extract_position_error_sample(record, self.default_reference)
            if sample is not None:
                samples.append(sample)
        return switched, samples


class LivePositionErrorPlot:
    def __init__(
        self,
        follower: JsonlTraceFollower,
        *,
        window_s: float = 60.0,
        maximum_samples: int = 12000,
    ) -> None:
        self.follower = follower
        self.window_s = float(window_s)
        self.samples: deque[PositionErrorSample] = deque(maxlen=maximum_samples)
        self.figure, axes = plt.subplots(3, 2, figsize=(11, 8.5), sharex=True)
        self.axes = tuple(axes.ravel())
        self.measured_lines = []
        self.estimated_lines = []
        for axis_index, axis in enumerate(self.axes[:3]):
            measured, = axis.plot([], [], "--", color="0.65", linewidth=1.0,
                                  label="Measured")
            estimated, = axis.plot([], [], color=AXIS_COLORS[axis_index],
                                   linewidth=1.8, label="Kalman estimate")
            axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
            for level_cm in ERROR_REFERENCE_LEVELS_CM:
                label = f"±{level_cm:g} cm" if axis_index == 0 else None
                axis.axhline(
                    level_cm,
                    color="tab:orange" if level_cm == 5.0 else "tab:purple",
                    linestyle=":",
                    linewidth=0.9,
                    alpha=0.75,
                    label=label,
                )
                axis.axhline(
                    -level_cm,
                    color="tab:orange" if level_cm == 5.0 else "tab:purple",
                    linestyle=":",
                    linewidth=0.9,
                    alpha=0.75,
                )
            axis.set_title(f"{AXIS_NAMES[axis_index]} error")
            axis.set_ylabel("Error (cm)")
            axis.grid(True, alpha=0.25)
            axis.legend(loc="upper right")
            self.measured_lines.append(measured)
            self.estimated_lines.append(estimated)
        norm_axis = self.axes[3]
        self.norm_line, = norm_axis.plot([], [], color="tab:purple", linewidth=2.0)
        for level_cm in ERROR_REFERENCE_LEVELS_CM:
            norm_axis.axhline(
                level_cm,
                color="tab:orange" if level_cm == 5.0 else "tab:purple",
                linestyle=":",
                linewidth=0.9,
                alpha=0.75,
                label=f"{level_cm:g} cm",
            )
        norm_axis.set_title("Estimated 3D position-error norm")
        norm_axis.set_ylabel("Error norm (cm)")
        norm_axis.grid(True, alpha=0.25)
        norm_axis.legend(loc="upper right")
        weight_axis = self.axes[4]
        self.model1_forward_weight_line, = weight_axis.plot(
            [], [], color=AXIS_COLORS[0], linewidth=2.0, label="Forward"
        )
        weight_axis.axhline(1.0, color="black", linewidth=0.7, alpha=0.5)
        weight_axis.axhline(0.5, color="0.5", linewidth=0.7, alpha=0.4)
        weight_axis.set_ylim(-0.03, 1.03)
        weight_axis.set_title("Forward model-1 fusion weight")
        weight_axis.set_ylabel("Weight")
        weight_axis.grid(True, alpha=0.25)
        weight_axis.legend(loc="lower right")

        velocity_axis = self.axes[5]
        self.velocity_lines = []
        for axis_index in range(3):
            line, = velocity_axis.plot(
                [], [], color=AXIS_COLORS[axis_index], linewidth=1.6,
                label=AXIS_NAMES[axis_index],
            )
            self.velocity_lines.append(line)
        velocity_axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
        velocity_axis.set_title("Estimated relative velocity")
        velocity_axis.set_ylabel("Velocity (m/s)")
        velocity_axis.grid(True, alpha=0.25)
        velocity_axis.legend(loc="upper right", ncols=3)

        for axis in self.axes[4:]:
            axis.set_xlabel("Time in current trace (s)")
        self.status_text = self.figure.suptitle("Waiting for MPC control updates")
        self.figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.84))

    def _visible_samples(self) -> list[PositionErrorSample]:
        if not self.samples:
            return []
        latest = self.samples[-1].time_s
        minimum = latest - self.window_s
        return [sample for sample in self.samples if sample.time_s >= minimum]

    def refresh(self) -> None:
        switched, new_samples = self.follower.poll()
        if switched:
            self.samples.clear()
        self.samples.extend(new_samples)
        visible = self._visible_samples()
        if not visible:
            selected = self.follower.path
            name = "no trace" if selected is None else selected.name
            self.status_text.set_text(f"Waiting for MPC control updates — {name}")
            return

        time0 = self.samples[0].time_s
        times = np.asarray([sample.time_s - time0 for sample in visible])
        measured_cm = 100.0 * np.vstack(
            [sample.measured_error_m for sample in visible]
        )
        estimated_cm = 100.0 * np.vstack(
            [sample.estimated_error_m for sample in visible]
        )
        estimated_velocity = np.vstack(
            [sample.estimated_velocity_m_s for sample in visible]
        )
        model1_weight = np.vstack([sample.model1_weight for sample in visible])
        for axis_index, axis in enumerate(self.axes[:3]):
            self.measured_lines[axis_index].set_data(times, measured_cm[:, axis_index])
            self.estimated_lines[axis_index].set_data(times, estimated_cm[:, axis_index])
            values = np.concatenate(
                (
                    measured_cm[:, axis_index],
                    estimated_cm[:, axis_index],
                    [-10.0, 10.0],
                )
            )
            limit = max(10.0, float(np.nanpercentile(np.abs(values), 99.0)) * 1.2)
            axis.set_ylim(-limit, limit)

        norms_cm = np.linalg.norm(estimated_cm, axis=1)
        self.norm_line.set_data(times, norms_cm)
        self.axes[3].set_ylim(
            0.0,
            max(10.0, float(np.nanpercentile(norms_cm, 99.0)) * 1.2),
        )
        self.model1_forward_weight_line.set_data(times, model1_weight[:, 0])
        for axis_index in range(3):
            self.velocity_lines[axis_index].set_data(
                times, estimated_velocity[:, axis_index]
            )
        finite_velocity = estimated_velocity[np.isfinite(estimated_velocity)]
        velocity_limit = (
            max(0.02, float(np.nanpercentile(np.abs(finite_velocity), 99.0)) * 1.2)
            if finite_velocity.size
            else 0.02
        )
        self.axes[5].set_ylim(-velocity_limit, velocity_limit)
        left = max(float(times[-1]) - self.window_s, float(times[0]))
        right = max(float(times[-1]), left + 1.0)
        for axis in self.axes:
            axis.set_xlim(left, right)

        current = estimated_cm[-1]
        rms = np.sqrt(np.mean(estimated_cm**2, axis=0))
        p95_norm = float(np.percentile(norms_cm, 95.0))
        recent = times >= times[-1] - 5.0
        forward_mean_5s = float(np.mean(estimated_cm[recent, 0]))
        forward_rms_5s = float(np.sqrt(np.mean(estimated_cm[recent, 0] ** 2)))
        forward_weights = model1_weight[recent, 0]
        finite_forward_weights = forward_weights[np.isfinite(forward_weights)]
        forward_weight_5s = (
            float(np.mean(finite_forward_weights))
            if finite_forward_weights.size
            else float("nan")
        )
        sample_age_s = max(0.0, time.monotonic() - self.samples[-1].time_s)
        live_status = "LIVE" if sample_age_s <= 1.0 else f"NO NEW MPC DATA {sample_age_s:.1f}s"
        trace_name = self.follower.path.name if self.follower.path else "unknown"
        short_trace_name = trace_name.removeprefix("experimental_auto_").removesuffix(
            ".jsonl"
        )
        self.status_text.set_text(
            f"{live_status} — Run {short_trace_name}\nCurrent FRD error="
            f"[{current[0]:+.1f}, {current[1]:+.1f}, {current[2]:+.1f}] cm   "
            f"RMS=[{rms[0]:.1f}, {rms[1]:.1f}, {rms[2]:.1f}] cm   "
            f"3D P95={p95_norm:.1f} cm\n"
            f"Forward 5s mean/RMS={forward_mean_5s:+.1f}/{forward_rms_5s:.1f} cm   "
            f"a1-forward 5s mean={forward_weight_5s:.3f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-jsonl", help="specific trace; default follows latest")
    parser.add_argument(
        "--log-directory",
        default=str(Path(__file__).resolve().parents[1] / "calibration_logs"),
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--window-sec", type=float, default=60.0)
    parser.add_argument("--refresh-ms", type=int, default=200)
    parser.add_argument("--save-png", help="optional snapshot path")
    parser.add_argument("--snapshot-only", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.window_sec <= 0.0 or args.refresh_ms <= 0:
        raise SystemExit("window-sec and refresh-ms must be positive")
    reference = load_default_reference(args.config)
    follower = JsonlTraceFollower(args.trace_jsonl, args.log_directory, reference)
    plot = LivePositionErrorPlot(follower, window_s=args.window_sec)
    plot.refresh()
    if args.snapshot_only:
        if not args.save_png:
            raise SystemExit("--snapshot-only requires --save-png")
        plot.figure.savefig(args.save_png, dpi=150)
        return 0

    def update(_frame):
        plot.refresh()
        if args.save_png:
            plot.figure.savefig(args.save_png, dpi=120)
        return ()

    animation = FuncAnimation(
        plot.figure,
        update,
        interval=args.refresh_ms,
        cache_frame_data=False,
    )
    plot.figure._finesub_animation = animation  # Keep the timer alive.
    manager = plt.get_current_fig_manager()
    manager.set_window_title("FineSUB MPC Position Error — Live")
    plt.show(block=False)
    window = getattr(manager, "window", None)
    if window is not None:
        def place_window() -> None:
            screen = window.screen()
            available = screen.availableGeometry()
            width = min(1200, max(800, available.width() - 80))
            height = min(800, max(600, available.height() - 80))
            window.setGeometry(
                available.x() + 40,
                available.y() + 40,
                width,
                height,
            )
            window.raise_()
            window.activateWindow()

        from matplotlib.backends.qt_compat import QtCore

        QtCore.QTimer.singleShot(500, place_window)
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
