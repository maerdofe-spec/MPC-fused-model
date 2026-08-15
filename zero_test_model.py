from __future__ import annotations

from dataclasses import dataclass
import math
import time

from geometry_msgs.msg import PoseWithCovarianceStamped
from finsrov_teleop.thrust_allocator import (
    DEFAULT_ALLOCATION_MATRIX,
    DEFAULT_UNITY_CENTER_OF_MASS_BODY_M,
    DEFAULT_UNITY_THRUSTER_DIRECTIONS_BODY,
    DEFAULT_UNITY_THRUSTER_POSITIONS_BODY_M,
    GeometryThrustAllocator,
    ThrustAllocator,
)
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String


@dataclass(frozen=True)
class MotionSegment:
    name: str
    duration_sec: float
    command: list[float]


@dataclass(frozen=True)
class PoseSnapshot:
    x: float
    y: float
    z: float
    yaw_rad: float


class ZeroTestModelNode(Node):
    def __init__(self) -> None:
        super().__init__("zero_test_model")

        self._topic_prefix = str(self.declare_parameter("topic_prefix", "").value).strip()
        self._command_topic = str(
            self.declare_parameter(
                "command_topic", "/sim/finsrov/direct_world_wrench"
            ).value
        ).strip()
        self._publish_hz = float(self.declare_parameter("publish_hz", 20.0).value)
        self._thruster_count = int(self.declare_parameter("thruster_count", 8).value)
        self._settle_duration_sec = float(
            self.declare_parameter("settle_duration_sec", 2.0).value
        )
        self._segment_duration_sec = float(
            self.declare_parameter("segment_duration_sec", 6.0).value
        )
        self._surge_segment_duration_sec = float(
            self.declare_parameter("surge_segment_duration_sec", -1.0).value
        )
        self._sway_segment_duration_sec = float(
            self.declare_parameter("sway_segment_duration_sec", -1.0).value
        )
        self._diagonal_segment_duration_sec = float(
            self.declare_parameter("diagonal_segment_duration_sec", -1.0).value
        )
        self._pause_duration_sec = float(
            self.declare_parameter("pause_duration_sec", 2.0).value
        )
        self._pose_topic = str(
            self.declare_parameter("pose_topic", "/sim/finsrov/pose").value
        ).strip()
        self._pose_yaw_axis = str(
            self.declare_parameter("pose_yaw_axis", "auto").value
        ).strip().lower()
        self._stream_connected_topic = str(
            self.declare_parameter(
                "stream_connected_topic", "/grpc_ros_adapter/remote_control/connected"
            ).value
        ).strip()
        self._test_mode = str(
            self.declare_parameter("test_mode", "translation_sequence").value
        ).strip().lower()
        self._output_mode = str(
            self.declare_parameter("output_mode", "direct_world_wrench").value
        ).strip().lower()
        self._allocation_mode = str(
            self.declare_parameter("allocation_mode", "unity_geometry_6d").value
        ).strip().lower()
        self._loop_sequence = bool(self.declare_parameter("loop_sequence", True).value)
        self._restart_on_pose_resume = bool(
            self.declare_parameter("restart_on_pose_resume", True).value
        )
        self._pose_resume_gap_sec = float(
            self.declare_parameter("pose_resume_gap_sec", 0.75).value
        )
        self._report_segment_motion = bool(
            self.declare_parameter("report_segment_motion", True).value
        )
        self._x_force_n = float(
            self.declare_parameter("x_force_n", 18.0).value
        )
        self._z_force_n = float(
            self.declare_parameter("z_force_n", 18.0).value
        )
        self._diagonal_x_force_n = float(
            self.declare_parameter("diagonal_x_force_n", 13.0).value
        )
        self._diagonal_z_force_n = float(
            self.declare_parameter("diagonal_z_force_n", 13.0).value
        )
        allocation_matrix = list(
            self.declare_parameter(
                "allocation_matrix",
                DEFAULT_ALLOCATION_MATRIX.reshape(-1).tolist(),
            ).value
        )
        force_limits_positive_n = list(
            self.declare_parameter("force_limits_positive_n", [7.0] * 8).value
        )
        force_limits_negative_n = list(
            self.declare_parameter("force_limits_negative_n", [7.0] * 8).value
        )
        thruster_positions_body_m = list(
            self.declare_parameter(
                "thruster_positions_body_m",
                DEFAULT_UNITY_THRUSTER_POSITIONS_BODY_M.reshape(-1).tolist(),
            ).value
        )
        thruster_directions_body = list(
            self.declare_parameter(
                "thruster_directions_body",
                DEFAULT_UNITY_THRUSTER_DIRECTIONS_BODY.reshape(-1).tolist(),
            ).value
        )
        center_of_mass_body_m = list(
            self.declare_parameter(
                "center_of_mass_body_m",
                DEFAULT_UNITY_CENTER_OF_MASS_BODY_M.tolist(),
            ).value
        )

        if self._output_mode not in {"direct_world_wrench", "thruster_force"}:
            self.get_logger().warning(
                f"unknown output_mode `{self._output_mode}`, falling back to direct_world_wrench"
            )
            self._output_mode = "direct_world_wrench"
        if self._allocation_mode not in {"unity_geometry_6d", "fixed_matrix_4d"}:
            self.get_logger().warning(
                f"unknown allocation_mode `{self._allocation_mode}`, falling back to unity_geometry_6d"
            )
            self._allocation_mode = "unity_geometry_6d"
        self._allocator = ThrustAllocator(
            allocation_matrix=allocation_matrix,
            force_limits_positive_n=force_limits_positive_n,
            force_limits_negative_n=force_limits_negative_n,
        )
        self._geometry_allocator = GeometryThrustAllocator(
            thruster_positions_body_m=thruster_positions_body_m,
            thruster_directions_body=thruster_directions_body,
            center_of_mass_body_m=center_of_mass_body_m,
            force_limits_positive_n=force_limits_positive_n,
            force_limits_negative_n=force_limits_negative_n,
        )

        if self._thruster_count < 8:
            self.get_logger().warning(
                "zero_test_model requires 8 thruster channels; padding thruster_count to 8"
            )
            self._thruster_count = 8
        self._publish_hz = max(self._publish_hz, 1.0)
        self._settle_duration_sec = max(self._settle_duration_sec, 0.0)
        self._segment_duration_sec = max(self._segment_duration_sec, 0.0)
        self._surge_segment_duration_sec = self._resolve_segment_duration(
            self._surge_segment_duration_sec
        )
        self._sway_segment_duration_sec = self._resolve_segment_duration(
            self._sway_segment_duration_sec
        )
        self._diagonal_segment_duration_sec = self._resolve_segment_duration(
            self._diagonal_segment_duration_sec
        )
        self._pause_duration_sec = max(self._pause_duration_sec, 0.0)
        self._pose_resume_gap_sec = max(self._pose_resume_gap_sec, 0.1)

        self._publisher = self.create_publisher(Float32MultiArray, self._command_topic, 10)
        self._message = Float32MultiArray()
        self._zero_command = [0.0] * self._thruster_count
        self._segments = self._build_segments()
        self._cycle_duration_sec = sum(segment.duration_sec for segment in self._segments)
        self._start_time = time.monotonic()
        self._active_segment_name: str | None = None
        self._last_pose_monotonic: float | None = None
        self._latest_pose_snapshot: PoseSnapshot | None = None
        self._segment_start_pose_snapshot: PoseSnapshot | None = None
        self._stream_connected_topics = self._build_stream_connected_topics()

        self._message.data = self._zero_command
        if self._pose_topic:
            self.create_subscription(
                PoseWithCovarianceStamped,
                self._pose_topic,
                self._pose_callback,
                10,
            )
        for stream_connected_topic in self._stream_connected_topics:
            self.create_subscription(
                String,
                stream_connected_topic,
                self._stream_connected_callback,
                10,
            )

        self._timer = self.create_timer(1.0 / self._publish_hz, self._publish_test_command)

        self.get_logger().info(
            "zero test model started: "
            f"command_topic={self._command_topic}, "
            f"output_mode={self._output_mode}, "
            f"allocation_mode={self._allocation_mode}, "
            f"command_layout={self._command_layout()}, "
            f"test_mode={self._test_mode}, "
            f"publish_hz={self._publish_hz:.1f}, "
            f"thruster_count={self._thruster_count}, "
            f"settle_duration_sec={self._settle_duration_sec:.1f}, "
            f"segment_duration_sec={self._segment_duration_sec:.1f}, "
            f"surge_segment_duration_sec={self._surge_segment_duration_sec:.1f}, "
            f"sway_segment_duration_sec={self._sway_segment_duration_sec:.1f}, "
            f"diagonal_segment_duration_sec={self._diagonal_segment_duration_sec:.1f}, "
            f"pause_duration_sec={self._pause_duration_sec:.1f}, "
            f"loop_sequence={self._loop_sequence}, "
            f"pose_topic={self._pose_topic or '<disabled>'}, "
            f"pose_yaw_axis={self._resolve_pose_yaw_axis()}, "
            f"stream_connected_topics={self._stream_connected_topics or ['<disabled>']}, "
            f"restart_on_pose_resume={self._restart_on_pose_resume}"
        )

    def _resolve_segment_duration(self, configured_duration_sec: float) -> float:
        if configured_duration_sec < 0.0:
            return self._segment_duration_sec
        return max(configured_duration_sec, 0.0)

    @staticmethod
    def _normalize_topic(topic: str) -> str:
        normalized = str(topic or "").strip()
        if not normalized:
            return ""
        if not normalized.startswith("/"):
            normalized = "/" + normalized
        while "//" in normalized:
            normalized = normalized.replace("//", "/")
        return normalized

    def _build_stream_connected_topics(self) -> list[str]:
        configured_topic = self._normalize_topic(self._stream_connected_topic)
        topics: list[str] = []
        if configured_topic:
            topics.append(configured_topic)

        normalized_prefix = self._normalize_topic(self._topic_prefix)
        if normalized_prefix and configured_topic:
            prefixed_topic = self._normalize_topic(
                f"{normalized_prefix.rstrip('/')}/{configured_topic.lstrip('/')}"
            )
            if prefixed_topic not in topics:
                topics.append(prefixed_topic)
        return topics

    def _direct_world_wrench_command(
        self,
        force_x_n: float,
        force_y_n: float,
        force_z_n: float,
        torque_x_nm: float = 0.0,
        torque_y_nm: float = 0.0,
        torque_z_nm: float = 0.0,
    ) -> list[float]:
        command = [0.0] * self._thruster_count
        command[0] = float(force_x_n)
        command[1] = float(force_y_n)
        command[2] = float(force_z_n)
        command[3] = float(torque_x_nm)
        command[4] = float(torque_y_nm)
        command[5] = float(torque_z_nm)
        return command

    def _command_layout(self) -> str:
        if self._output_mode == "thruster_force":
            return "[V_LF,V_LB,V_RB,V_RF,H_LF,H_LB,H_RB,H_RF]_force_N"
        return "[Fx_N,Fy_N,Fz_N,Tx_Nm,Ty_Nm,Tz_Nm,0,0]"

    def _motion_command(
        self,
        force_x_n: float,
        force_y_n: float,
        force_z_n: float,
        torque_x_nm: float = 0.0,
        torque_y_nm: float = 0.0,
        torque_z_nm: float = 0.0,
    ) -> list[float]:
        if self._output_mode == "thruster_force":
            if self._allocation_mode == "unity_geometry_6d":
                result = self._geometry_allocator.allocate(
                    force_x_n,
                    force_y_n,
                    force_z_n,
                    torque_x_nm,
                    torque_y_nm,
                    torque_z_nm,
                )
                if result.clamped:
                    self.get_logger().info(
                        "geometry allocation uniformly scaled to preserve zero moments: "
                        f"scale={result.scale:.4f}, achieved_wrench="
                        f"{[round(value, 6) for value in result.achieved_wrench]}"
                    )
                return result.values

            if abs(torque_x_nm) > 1e-9 or abs(torque_z_nm) > 1e-9:
                self.get_logger().warning(
                    "fixed_matrix_4d supports Unity x/z translation, y heave, and y-axis yaw only"
                )
            return self._allocator.allocate(
                force_x_n,
                force_z_n,
                force_y_n,
                torque_y_nm,
            ).values
        return self._direct_world_wrench_command(
            force_x_n,
            force_y_n,
            force_z_n,
            torque_x_nm,
            torque_y_nm,
            torque_z_nm,
        )

    def _build_segments(self) -> list[MotionSegment]:
        if self._test_mode != "translation_sequence":
            self.get_logger().warning(
                f"unknown test_mode `{self._test_mode}`, falling back to translation_sequence"
            )

        unity_x_pos = self._motion_command(
            self._x_force_n, 0.0, 0.0
        )
        unity_x_neg = self._motion_command(
            -self._x_force_n, 0.0, 0.0
        )
        unity_z_pos = self._motion_command(
            0.0, 0.0, self._z_force_n
        )
        unity_z_neg = self._motion_command(
            0.0, 0.0, -self._z_force_n
        )
        unity_xz_pos = self._motion_command(
            self._diagonal_x_force_n,
            0.0,
            self._diagonal_z_force_n,
        )
        unity_xz_neg = self._motion_command(
            -self._diagonal_x_force_n,
            0.0,
            -self._diagonal_z_force_n,
        )

        segments: list[MotionSegment] = []
        if self._settle_duration_sec > 0.0:
            segments.append(
                MotionSegment("hold_start", self._settle_duration_sec, self._zero_command)
            )
        if self._sway_segment_duration_sec > 0.0:
            segments.append(
                MotionSegment("unity_z_pos", self._sway_segment_duration_sec, unity_z_pos)
            )
        if self._pause_duration_sec > 0.0:
            segments.append(
                MotionSegment("hold_after_unity_z_pos", self._pause_duration_sec, self._zero_command)
            )
        if self._sway_segment_duration_sec > 0.0:
            segments.append(
                MotionSegment("unity_z_neg", self._sway_segment_duration_sec, unity_z_neg)
            )
        if self._pause_duration_sec > 0.0:
            segments.append(
                MotionSegment("hold_after_unity_z_neg", self._pause_duration_sec, self._zero_command)
            )
        if self._surge_segment_duration_sec > 0.0:
            segments.append(
                MotionSegment("unity_x_pos", self._surge_segment_duration_sec, unity_x_pos)
            )
        if self._pause_duration_sec > 0.0:
            segments.append(
                MotionSegment("hold_after_unity_x_pos", self._pause_duration_sec, self._zero_command)
            )
        if self._surge_segment_duration_sec > 0.0:
            segments.append(
                MotionSegment("unity_x_neg", self._surge_segment_duration_sec, unity_x_neg)
            )
        if self._pause_duration_sec > 0.0:
            segments.append(
                MotionSegment("hold_after_unity_x_neg", self._pause_duration_sec, self._zero_command)
            )
        if self._diagonal_segment_duration_sec > 0.0:
            segments.append(
                MotionSegment("unity_xz_pos", self._diagonal_segment_duration_sec, unity_xz_pos)
            )
        if self._pause_duration_sec > 0.0:
            segments.append(
                MotionSegment("hold_after_unity_xz_pos", self._pause_duration_sec, self._zero_command)
            )
        if self._diagonal_segment_duration_sec > 0.0:
            segments.append(
                MotionSegment("unity_xz_neg", self._diagonal_segment_duration_sec, unity_xz_neg)
            )
        if self._pause_duration_sec > 0.0:
            segments.append(MotionSegment("hold_end", self._pause_duration_sec, self._zero_command))
        if not segments:
            segments.append(MotionSegment("hold_only", 0.0, self._zero_command))
        return segments

    def _restart_sequence(self, reason: str) -> None:
        self._start_time = time.monotonic()
        self._active_segment_name = None
        self._segment_start_pose_snapshot = None
        self.get_logger().info(f"sequence restarted: reason={reason}")

    def _resolve_pose_yaw_axis(self) -> str:
        if self._pose_yaw_axis in {"x", "y", "z"}:
            return self._pose_yaw_axis
        return "y" if "/controller/" in self._pose_topic else "z"

    def _yaw_from_quaternion(self, x: float, y: float, z: float, w: float) -> float:
        yaw_axis = self._resolve_pose_yaw_axis()
        if yaw_axis == "y":
            siny_cosp = 2.0 * (w * y + x * z)
            cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
            return math.atan2(siny_cosp, cosy_cosp)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _wrap_angle_rad(angle_rad: float) -> float:
        return math.atan2(math.sin(angle_rad), math.cos(angle_rad))

    def _pose_callback(self, _msg: PoseWithCovarianceStamped) -> None:
        now = time.monotonic()
        pose = _msg.pose.pose
        self._latest_pose_snapshot = PoseSnapshot(
            x=float(pose.position.x),
            y=float(pose.position.y),
            z=float(pose.position.z),
            yaw_rad=self._yaw_from_quaternion(
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                float(pose.orientation.w),
            ),
        )
        if self._segment_start_pose_snapshot is None:
            self._segment_start_pose_snapshot = self._latest_pose_snapshot

        if self._restart_on_pose_resume:
            if self._last_pose_monotonic is None:
                self._restart_sequence("pose_first_seen")
            elif now - self._last_pose_monotonic > self._pose_resume_gap_sec:
                self._restart_sequence("pose_resumed_after_gap")
        self._last_pose_monotonic = now

    def _stream_connected_callback(self, msg: String) -> None:
        connected_topic = str(msg.data).strip()
        if not connected_topic:
            return
        if connected_topic != self._command_topic:
            return
        self._restart_sequence("command_stream_connected")

    def _log_segment_motion_summary(self, segment_name: str) -> None:
        if not self._report_segment_motion:
            return
        if self._segment_start_pose_snapshot is None or self._latest_pose_snapshot is None:
            return
        if segment_name.startswith("hold"):
            return

        start = self._segment_start_pose_snapshot
        end = self._latest_pose_snapshot
        delta_yaw_deg = math.degrees(self._wrap_angle_rad(end.yaw_rad - start.yaw_rad))
        self.get_logger().info(
            "segment_result="
            f"{segment_name}, "
            f"delta_position=[{end.x - start.x:.4f}, {end.y - start.y:.4f}, {end.z - start.z:.4f}], "
            f"delta_yaw_deg={delta_yaw_deg:.3f}"
        )

    def _segment_for_elapsed(self, elapsed_sec: float) -> MotionSegment:
        if self._cycle_duration_sec <= 0.0:
            return MotionSegment("hold_only", 0.0, self._zero_command)

        if self._loop_sequence:
            elapsed_sec %= self._cycle_duration_sec
        elif elapsed_sec >= self._cycle_duration_sec:
            return MotionSegment("hold_complete", 0.0, self._zero_command)

        remaining = elapsed_sec
        for segment in self._segments:
            if remaining <= segment.duration_sec:
                return segment
            remaining -= segment.duration_sec
        return self._segments[-1]

    def _publish_test_command(self) -> None:
        elapsed_sec = time.monotonic() - self._start_time
        active_segment = self._segment_for_elapsed(elapsed_sec)

        if active_segment.name != self._active_segment_name:
            if self._active_segment_name is not None:
                self._log_segment_motion_summary(self._active_segment_name)
            self._active_segment_name = active_segment.name
            self._segment_start_pose_snapshot = self._latest_pose_snapshot
            self.get_logger().info(
                f"segment={active_segment.name}, command={active_segment.command}"
            )

        self._message.data = active_segment.command
        self._publisher.publish(self._message)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ZeroTestModelNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
