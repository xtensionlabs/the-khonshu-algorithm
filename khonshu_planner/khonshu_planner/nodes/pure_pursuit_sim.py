"""Lightweight Pure Pursuit + speed controller for testing Khonshu in EUFS."""

from __future__ import annotations

import math
import time

import rclpy
from rclpy.node import Node

from ackermann_msgs.msg import AckermannDriveStamped
from eufs_msgs.msg import CarState, Waypoint, WaypointArrayStamped


WHEELBASE_M = 1.58
MAX_STEER_RAD = 0.52
KP_SPEED = 1.5
MAX_ACCEL = 2.5
MAX_BRAKE = -4.5


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


class PurePursuitSimNode(Node):
    def __init__(self) -> None:
        super().__init__("pure_pursuit_sim")
        self._waypoints: list[Waypoint] = []
        self._last_traj_wall_time: float = 0.0
        self._current_speed: float = 0.0
        self._cycle_count: int = 0

        self._start_xy: tuple[float, float] | None = None
        self._prev_xy: tuple[float, float] | None = None
        self._lap_distance: float = 0.0
        self._left_start: bool = False
        self._lap_count: int = 0

        self.create_subscription(
            WaypointArrayStamped,
            "/planning/trajectory",
            self._traj_cb,
            10,
        )
        self.create_subscription(
            CarState,
            "/ground_truth/state",
            self._state_cb,
            10,
        )
        self._cmd_pub = self.create_publisher(AckermannDriveStamped, "/cmd", 10)
        self.create_timer(0.05, self._control_cycle)
        self.get_logger().info("PurePursuitSimNode started (20 Hz -> /cmd)")

    def _traj_cb(self, msg: WaypointArrayStamped) -> None:
        self._waypoints = list(msg.waypoints)
        self._last_traj_wall_time = time.monotonic()

    def _state_cb(self, msg: CarState) -> None:
        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        self._current_speed = math.hypot(vx, vy)

        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        if self._start_xy is None:
            self._start_xy = (x, y)
            self._prev_xy = (x, y)
            return

        assert self._prev_xy is not None
        self._lap_distance += math.hypot(x - self._prev_xy[0], y - self._prev_xy[1])
        self._prev_xy = (x, y)

        dist_from_start = math.hypot(x - self._start_xy[0], y - self._start_xy[1])
        if not self._left_start and dist_from_start > 5.0:
            self._left_start = True
        if self._left_start and self._lap_distance > 40.0 and dist_from_start < 3.5:
            self._lap_count += 1
            self.get_logger().info(
                f"=== LAP {self._lap_count} COMPLETED! (distance={self._lap_distance:.1f} m) ==="
            )
            self._lap_distance = 0.0
            self._left_start = False

    def _control_cycle(self) -> None:
        now_mono = time.monotonic()
        wps = self._waypoints
        stale = (
            self._last_traj_wall_time == 0.0
            or (now_mono - self._last_traj_wall_time) > 1.5
        )
        failsafe = stale or len(wps) < 2 or float(wps[0].speed) <= 0.05

        if failsafe:
            steer = 0.0
            v_target = 0.0
            accel = MAX_BRAKE if self._current_speed > 0.1 else 0.0
        else:
            ld = _clamp(2.2 + 0.35 * self._current_speed, 2.2, 5.0)
            forward_wps = [w for w in wps if float(w.position.x) > 0.2]
            target = None
            for w in forward_wps:
                if math.hypot(float(w.position.x), float(w.position.y)) >= ld:
                    target = w
                    break
            if target is None:
                if forward_wps:
                    target = max(
                        forward_wps,
                        key=lambda w: math.hypot(float(w.position.x), float(w.position.y)),
                    )
                else:
                    target = wps[-1]

            gx = float(target.position.x)
            gy = float(target.position.y)
            dist = max(math.hypot(gx, gy), 1e-3)
            alpha = math.atan2(gy, gx)
            raw_steer = math.atan2(2.0 * WHEELBASE_M * math.sin(alpha), dist)
            steer = _clamp(raw_steer, -MAX_STEER_RAD, MAX_STEER_RAD)

            near_idx = min(2, len(wps) - 1)
            v_target = min(float(wps[near_idx].speed), float(target.speed))
            accel = _clamp(
                KP_SPEED * (v_target - self._current_speed),
                MAX_BRAKE,
                MAX_ACCEL,
            )

        cmd = AckermannDriveStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "base_footprint"
        cmd.drive.steering_angle = float(steer)
        cmd.drive.speed = float(v_target)
        cmd.drive.acceleration = float(accel)
        self._cmd_pub.publish(cmd)

        self._cycle_count += 1
        if self._cycle_count % 40 == 0:
            self.get_logger().info(
                f"v={self._current_speed:.2f}/{v_target:.2f} m/s "
                f"steer={steer:+.3f} rad dist={self._lap_distance:.1f} m "
                f"failsafe={failsafe}"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PurePursuitSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
