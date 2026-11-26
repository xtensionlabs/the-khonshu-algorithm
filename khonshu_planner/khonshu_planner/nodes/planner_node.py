"""ROS 2 node wrapping the pure-Python KhonshuPlanner for live EUFS execution."""

from __future__ import annotations

import math
import time

import rclpy
from rclpy.node import Node

from eufs_msgs.msg import (
    CarState,
    ConeArrayWithCovariance,
    Waypoint as RosWaypoint,
    WaypointArrayStamped,
)
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from khonshu_planner.core.planner import KhonshuPlanner, PlanResult
from khonshu_planner.core.types import Cone, ConeColor


WHEELBASE_M = 1.5
MAX_STEER_RAD = 0.5


def _to_core_cones(msg: ConeArrayWithCovariance) -> list[Cone]:
    """Convert EUFS ConeArrayWithCovariance into ROS-independent Cone objects."""
    cones: list[Cone] = []
    groups = (
        (msg.blue_cones, ConeColor.BLUE),
        (msg.yellow_cones, ConeColor.YELLOW),
        (msg.orange_cones, ConeColor.ORANGE),
        (msg.big_orange_cones, ConeColor.BIG_ORANGE),
        (msg.unknown_color_cones, ConeColor.UNKNOWN),
    )
    for ros_list, color in groups:
        for item in ros_list:
            cov = tuple(float(v) for v in item.covariance[:4])
            if len(cov) != 4:
                cov = (0.0, 0.0, 0.0, 0.0)
            cones.append(Cone(float(item.point.x), float(item.point.y), color, cov))
    return cones


def _speed_color(speed: float, max_speed: float, is_failsafe: bool) -> ColorRGBA:
    """Map target speed to red (slow/stop) -> yellow -> green (fast)."""
    if is_failsafe or max_speed <= 1e-6:
        return ColorRGBA(r=1.0, g=0.1, b=0.1, a=1.0)
    ratio = max(0.0, min(1.0, speed / max_speed))
    return ColorRGBA(r=float(1.0 - ratio), g=float(ratio), b=0.1, a=1.0)


class KhonshuPlannerNode(Node):
    def __init__(self) -> None:
        super().__init__("khonshu_planner_node")
        self.planner = KhonshuPlanner()

        self._latest_cones: list[Cone] = []
        self._latest_frame_id: str = "base_footprint"
        self._last_cone_wall_time: float = 0.0
        self._current_speed: float = 0.0
        self._cycle_count: int = 0

        self.create_subscription(
            ConeArrayWithCovariance,
            "/ground_truth/cones",
            self._cones_cb,
            10,
        )
        self.create_subscription(
            CarState,
            "/ground_truth/state",
            self._state_cb,
            10,
        )

        self._traj_pub = self.create_publisher(
            WaypointArrayStamped, "/planning/trajectory", 10
        )
        self._viz_pub = self.create_publisher(
            MarkerArray, "/planning/viz", 10
        )

        # 20 Hz planning loop (0.05 s period)
        self.create_timer(0.05, self._plan_cycle)
        self.get_logger().info("KhonshuPlannerNode started (20 Hz -> /planning/trajectory)")

    def _cones_cb(self, msg: ConeArrayWithCovariance) -> None:
        self._latest_cones = _to_core_cones(msg)
        if msg.header.frame_id:
            self._latest_frame_id = msg.header.frame_id
        self._last_cone_wall_time = time.monotonic()

    def _state_cb(self, msg: CarState) -> None:
        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        self._current_speed = math.hypot(vx, vy)

    def _plan_cycle(self) -> None:
        now_mono = time.monotonic()
        cones_stale = (
            self._last_cone_wall_time == 0.0
            or (now_mono - self._last_cone_wall_time) > 2.0
        )
        cones = [] if cones_stale else self._latest_cones

        result = self.planner.plan(
            cones,
            current_speed=self._current_speed,
            frame_id=self._latest_frame_id,
        )
        stamp = self.get_clock().now().to_msg()
        self._publish_trajectory(result, stamp)
        self._publish_markers(result, stamp)

        self._cycle_count += 1
        if self._cycle_count % 40 == 0:
            traj = result.trajectory
            self.get_logger().info(
                f"cones={len(cones)} wps={len(traj.waypoints)} "
                f"failsafe={traj.is_failsafe} dt={traj.compute_time_ms:.2f} ms"
            )

    def _publish_trajectory(self, result: PlanResult, stamp) -> None:
        msg = WaypointArrayStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = result.trajectory.frame_id

        for wp in result.trajectory.waypoints:
            rw = RosWaypoint()
            rw.position.x = wp.x
            rw.position.y = wp.y
            rw.position.z = 0.0
            rw.speed = wp.speed
            steer = math.atan(WHEELBASE_M * wp.curvature)
            rw.suggested_steering = max(-MAX_STEER_RAD, min(MAX_STEER_RAD, steer))
            msg.waypoints.append(rw)

        self._traj_pub.publish(msg)

    def _publish_markers(self, result: PlanResult, stamp) -> None:
        frame_id = result.trajectory.frame_id
        markers = MarkerArray()

        # 1. Delaunay triangles (gray wireframe)
        tri_m = Marker()
        tri_m.header.stamp = stamp
        tri_m.header.frame_id = frame_id
        tri_m.ns = "delaunay_triangles"
        tri_m.id = 0
        tri_m.type = Marker.LINE_LIST
        tri_m.scale.x = 0.03
        tri_m.color = ColorRGBA(r=0.7, g=0.7, b=0.7, a=0.35)
        for a, b, c in result.track_estimate.triangles:
            for p1, p2 in ((a, b), (b, c), (c, a)):
                tri_m.points.append(Point(x=p1[0], y=p1[1], z=0.05))
                tri_m.points.append(Point(x=p2[0], y=p2[1], z=0.05))
        tri_m.action = Marker.ADD if tri_m.points else Marker.DELETE
        markers.markers.append(tri_m)

        # 2. Ordered midpoints (white spheres)
        mid_m = Marker()
        mid_m.header.stamp = stamp
        mid_m.header.frame_id = frame_id
        mid_m.ns = "midpoints"
        mid_m.id = 1
        mid_m.type = Marker.SPHERE_LIST
        mid_m.scale.x = 0.18
        mid_m.scale.y = 0.18
        mid_m.scale.z = 0.18
        mid_m.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.9)
        for mx, my in result.track_estimate.midpoints:
            mid_m.points.append(Point(x=mx, y=my, z=0.1))
        mid_m.action = Marker.ADD if mid_m.points else Marker.DELETE
        markers.markers.append(mid_m)

        # 3. Smoothed trajectory coloured by target speed
        path_m = Marker()
        path_m.header.stamp = stamp
        path_m.header.frame_id = frame_id
        path_m.ns = "trajectory"
        path_m.id = 2
        path_m.type = Marker.LINE_STRIP
        path_m.scale.x = 0.14
        max_v = self.planner.config.smoothing.max_speed
        for wp in result.trajectory.waypoints:
            path_m.points.append(Point(x=wp.x, y=wp.y, z=0.15))
            path_m.colors.append(
                _speed_color(wp.speed, max_v, result.trajectory.is_failsafe)
            )
        path_m.action = Marker.ADD if len(path_m.points) >= 2 else Marker.DELETE
        markers.markers.append(path_m)

        self._viz_pub.publish(markers)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = KhonshuPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
