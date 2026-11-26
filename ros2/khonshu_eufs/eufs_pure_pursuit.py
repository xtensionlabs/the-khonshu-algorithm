"""Low-speed EUFS follower for validating the Stage A centerline in simulation."""

from __future__ import annotations

import math
import sys
import time

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from eufs_msgs.msg import CanState
from eufs_msgs.srv import SetCanState
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from std_msgs.msg import String


def _yaw_from_quaternion(rotation) -> float:
    return math.atan2(
        2.0 * (rotation.w * rotation.z + rotation.x * rotation.y),
        1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z),
    )


def _stamp_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class EufsPurePursuit(Node):
    """Follow the published Stage A path and stop on stale or invalid input."""

    def __init__(self) -> None:
        super().__init__("khonshu_eufs_stage_a_follower")
        self.declare_parameter("path_topic", "/khonshu/centerline")
        self.declare_parameter("state_topic", "/ground_truth/odom")
        self.declare_parameter("command_topic", "/cmd")
        self.declare_parameter("status_topic", "/khonshu/control/status")
        self.declare_parameter("wheelbase_m", 1.55)
        self.declare_parameter("speed_mps", 0.8)
        self.declare_parameter("lookahead_m", 1.6)
        self.declare_parameter("maximum_steering_angle_rad", 0.45)
        self.declare_parameter("path_timeout_s", 0.8)
        self.declare_parameter("state_timeout_s", 0.5)
        self.declare_parameter("lap_detection_enabled", True)
        self.declare_parameter("lap_start_radius_m", 1.5)
        self.declare_parameter("minimum_lap_distance_m", 20.0)
        self.declare_parameter("lap_timeout_s", 120.0)
        self.declare_parameter("control_frequency_hz", 20.0)

        self._wheelbase = float(self.get_parameter("wheelbase_m").value)
        self._speed = float(self.get_parameter("speed_mps").value)
        self._lookahead = float(self.get_parameter("lookahead_m").value)
        self._maximum_steering = float(
            self.get_parameter("maximum_steering_angle_rad").value
        )
        self._path_timeout = float(self.get_parameter("path_timeout_s").value)
        self._state_timeout = float(self.get_parameter("state_timeout_s").value)
        self._lap_enabled = bool(self.get_parameter("lap_detection_enabled").value)
        self._lap_radius = float(self.get_parameter("lap_start_radius_m").value)
        self._minimum_lap_distance = float(
            self.get_parameter("minimum_lap_distance_m").value
        )
        self._lap_timeout = float(self.get_parameter("lap_timeout_s").value)
        frequency = float(self.get_parameter("control_frequency_hz").value)
        for name, value in (
            ("wheelbase_m", self._wheelbase),
            ("speed_mps", self._speed),
            ("lookahead_m", self._lookahead),
            ("maximum_steering_angle_rad", self._maximum_steering),
            ("path_timeout_s", self._path_timeout),
            ("state_timeout_s", self._state_timeout),
            ("lap_start_radius_m", self._lap_radius),
            ("minimum_lap_distance_m", self._minimum_lap_distance),
            ("lap_timeout_s", self._lap_timeout),
            ("control_frequency_hz", frequency),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

        self._path_xy: list[tuple[float, float]] = []
        self._path_stamp_s: float | None = None
        self._state: tuple[float, float, float] | None = None
        self._state_stamp_s: float | None = None
        self._start_xy: tuple[float, float] | None = None
        self._start_yaw: float | None = None
        self._last_xy: tuple[float, float] | None = None
        self._distance_m = 0.0
        self._run_start_s: float | None = None
        self._measured_speed_mps = 0.0
        self._mission_active = False
        self._mission_request_sent = False
        self._mission_request_failed = False
        self._last_stop_reason = ""
        self._last_report_wall_s = 0.0
        self._planner_status = "waiting_for_planner"
        self._finished = False
        self.lap_completed = False

        self._command_pub = self.create_publisher(
            AckermannDriveStamped,
            self.get_parameter("command_topic").value,
            10,
        )
        self._status_pub = self.create_publisher(
            String,
            self.get_parameter("status_topic").value,
            10,
        )
        self.create_subscription(
            Path,
            self.get_parameter("path_topic").value,
            self._on_path,
            10,
        )
        self.create_subscription(
            Odometry,
            self.get_parameter("state_topic").value,
            self._on_state,
            10,
        )
        self.create_subscription(
            CanState,
            "/ros_can/state",
            self._on_can_state,
            10,
        )
        self.create_subscription(
            String,
            "/khonshu/debug/status",
            self._on_planner_status,
            10,
        )
        self._mission_client = self.create_client(SetCanState, "/ros_can/set_mission")
        self.create_timer(1.0 / frequency, self._control_tick)
        self.get_logger().info(
            "Stage A follower armed: low-speed velocity commands, stale input stops, "
            f"lap detection={'on' if self._lap_enabled else 'off'}"
        )

    @property
    def finished(self) -> bool:
        return self._finished

    def _on_path(self, msg: Path) -> None:
        self._path_xy = [
            (float(pose.pose.position.x), float(pose.pose.position.y))
            for pose in msg.poses
        ]
        self._path_stamp_s = _stamp_seconds(msg.header.stamp)

    def _on_state(self, msg: Odometry) -> None:
        stamp_s = _stamp_seconds(msg.header.stamp)
        position = (float(msg.pose.pose.position.x), float(msg.pose.pose.position.y))
        yaw = _yaw_from_quaternion(msg.pose.pose.orientation)
        self._state = (position[0], position[1], yaw)
        self._state_stamp_s = stamp_s
        velocity = msg.twist.twist.linear
        self._measured_speed_mps = math.hypot(float(velocity.x), float(velocity.y))

        if self._start_xy is None:
            self._start_xy = position
            self._start_yaw = yaw
            self._last_xy = position
            self._run_start_s = self.get_clock().now().nanoseconds * 1e-9
            self.get_logger().info(
                f"Lap monitor start x={position[0]:.2f} y={position[1]:.2f} yaw={yaw:.2f}"
            )
        elif self._last_xy is not None:
            self._distance_m += math.hypot(
                position[0] - self._last_xy[0],
                position[1] - self._last_xy[1],
            )
            self._last_xy = position

    def _on_can_state(self, msg: CanState) -> None:
        was_active = self._mission_active
        self._mission_active = msg.as_state == CanState.AS_DRIVING
        if self._mission_active and not was_active:
            self.get_logger().info("EUFS state machine enabled trackdrive control")
        elif msg.as_state in (CanState.AS_EMERGENCY_BRAKE, CanState.AS_FINISHED):
            self._mission_request_failed = True
            self.get_logger().error(f"EUFS state machine stopped the car: AS state={msg.as_state}")

    def _on_planner_status(self, msg: String) -> None:
        self._planner_status = msg.data

    def _request_trackdrive(self) -> bool:
        if self._mission_active:
            return True
        if self._mission_request_failed:
            self._stop("trackdrive_enable_failed")
            return False
        if not self._mission_request_sent:
            if not self._mission_client.service_is_ready():
                self._stop("waiting_for_trackdrive_service")
                return False
            request = SetCanState.Request()
            request.ami_state = CanState.AMI_TRACK_DRIVE
            future = self._mission_client.call_async(request)
            future.add_done_callback(self._on_mission_response)
            self._mission_request_sent = True
            self.get_logger().info("Requested EUFS AMI_TRACK_DRIVE mission")
        self._stop("waiting_for_trackdrive_enable")
        return False

    def _on_mission_response(self, future) -> None:
        try:
            response = future.result()
        except Exception as error:  # service errors keep the car stopped
            self._mission_request_failed = True
            self.get_logger().error(f"EUFS trackdrive service failed: {error}")
            return
        if not response.success:
            self._mission_request_failed = True
            self.get_logger().error("EUFS rejected the trackdrive mission request")
        else:
            self.get_logger().info("EUFS accepted trackdrive; waiting for AS_DRIVING")

    def _publish_command(self, speed_mps: float, steering_rad: float = 0.0) -> None:
        command = AckermannDriveStamped()
        command.header.stamp = self.get_clock().now().to_msg()
        command.header.frame_id = "base_footprint"
        command.drive.speed = float(speed_mps)
        command.drive.steering_angle = float(steering_rad)
        self._command_pub.publish(command)

    def _publish_status(self, text: str) -> None:
        message = String()
        message.data = text
        self._status_pub.publish(message)

    def _stop(self, reason: str) -> None:
        self._publish_command(0.0)
        self._publish_status(f"STOP {reason} distance_m={self._distance_m:.2f}")
        if reason != self._last_stop_reason:
            self.get_logger().warning(f"Follower stopped: {reason}")
            self._last_stop_reason = reason

    def _control_tick(self) -> None:
        now_s = self.get_clock().now().nanoseconds * 1e-9
        if self._finished:
            self._publish_command(0.0)
            return
        if self._state is None or self._state_stamp_s is None:
            self._stop("waiting_for_odometry")
            return
        if now_s - self._state_stamp_s > self._state_timeout:
            self._stop("stale_odometry")
            return
        if not self._path_xy or self._path_stamp_s is None:
            self._stop("waiting_for_valid_path")
            return
        if now_s - self._path_stamp_s > self._path_timeout:
            self._stop("stale_path")
            return

        if self._lap_enabled and self._run_start_s is not None:
            elapsed = now_s - self._run_start_s
            if elapsed > self._lap_timeout:
                self._finish(False, f"LAP_TIMEOUT elapsed_s={elapsed:.1f}")
                return
            if (
                self._distance_m >= self._minimum_lap_distance
                and self._start_xy is not None
                and self._start_yaw is not None
            ):
                distance_to_start = math.hypot(
                    self._state[0] - self._start_xy[0],
                    self._state[1] - self._start_xy[1],
                )
                yaw_error = abs(
                    (self._state[2] - self._start_yaw + math.pi) % (2.0 * math.pi)
                    - math.pi
                )
                if distance_to_start <= self._lap_radius and yaw_error <= 1.0:
                    self._finish(
                        True,
                        f"LAP_COMPLETE distance_m={self._distance_m:.2f} "
                        f"elapsed_s={elapsed:.2f}",
                    )
                    return

        if not self._request_trackdrive():
            return

        x, y, yaw = self._state
        points = self._path_xy
        nearest_index = min(
            range(len(points)),
            key=lambda index: math.hypot(points[index][0] - x, points[index][1] - y),
        )
        nearest_distance = math.hypot(
            points[nearest_index][0] - x,
            points[nearest_index][1] - y,
        )
        if nearest_distance > 5.0:
            self._stop("path_too_far_from_vehicle")
            return

        target = points[nearest_index]
        accumulated = 0.0
        for index in range(nearest_index + 1, len(points)):
            previous = points[index - 1]
            current = points[index]
            accumulated += math.hypot(current[0] - previous[0], current[1] - previous[1])
            target = current
            if accumulated >= self._lookahead:
                break

        dx, dy = target[0] - x, target[1] - y
        target_distance = math.hypot(dx, dy)
        if target_distance < 0.1:
            self._stop("lookahead_target_too_close")
            return
        alpha = (math.atan2(dy, dx) - yaw + math.pi) % (2.0 * math.pi) - math.pi
        if math.cos(alpha) <= 0.0:
            self._stop("lookahead_target_behind_vehicle")
            return
        steering = math.atan2(
            2.0 * self._wheelbase * math.sin(alpha),
            max(target_distance, self._lookahead * 0.5),
        )
        steering = max(-self._maximum_steering, min(self._maximum_steering, steering))
        self._publish_command(self._speed, steering)
        self._last_stop_reason = ""
        self._publish_status(
            f"TRACKING speed_mps={self._speed:.2f} steering_rad={steering:.3f} "
            f"distance_m={self._distance_m:.2f}"
        )
        wall_now = time.monotonic()
        if wall_now - self._last_report_wall_s >= 2.0:
            self.get_logger().info(
                f"TRACKING x={x:.2f} y={y:.2f} yaw={yaw:.2f} "
                f"measured_speed_mps={self._measured_speed_mps:.2f} "
                f"commanded_speed_mps={self._speed:.2f} steering_rad={steering:.3f} "
                f"distance_m={self._distance_m:.2f} points={len(points)} "
                f"planner_status='{self._planner_status}'"
            )
            self._last_report_wall_s = wall_now

    def _finish(self, completed: bool, message: str) -> None:
        self.lap_completed = completed
        self._finished = True
        self._publish_command(0.0)
        self._publish_status(message)
        if completed:
            self.get_logger().info(message)
        else:
            self.get_logger().error(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EufsPurePursuit()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    if not node.lap_completed:
        sys.exit(2)


if __name__ == "__main__":
    main()
