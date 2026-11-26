"""EUFS ROS 2 preview node for testing Khonshu centerline generation.

This node is for visualization and diagnostics only. It does not publish a
control command or the EUFS controller's waypoint message.
"""

from __future__ import annotations

import math
from collections import deque
from threading import Lock
from typing import TypeAlias

import rclpy
from eufs_msgs.msg import CarState, ConeArrayWithCovariance
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from std_msgs.msg import ColorRGBA, Header, String
from visualization_msgs.msg import Marker, MarkerArray

from khonshu import (
    Cone,
    ConeColor,
    LocalPlanner,
    PlannerConfig,
    StageAConfig,
    StageAPlanner,
    VehicleState,
)

VehicleStateMessage: TypeAlias = CarState | Odometry


COLOR_GROUPS = (
    ("blue_cones", ConeColor.BLUE, (0.12, 0.35, 0.95)),
    ("yellow_cones", ConeColor.YELLOW, (0.95, 0.72, 0.08)),
    ("orange_cones", ConeColor.ORANGE_SMALL, (0.98, 0.38, 0.08)),
    ("big_orange_cones", ConeColor.ORANGE_BIG, (0.8, 0.08, 0.08)),
    ("unknown_color_cones", ConeColor.UNKNOWN, (0.45, 0.45, 0.5)),
)


def _rotate_vector(x: float, y: float, z: float, rotation) -> tuple[float, float, float]:
    """Apply a unit quaternion rotation without a tf2_geometry_msgs dependency."""

    qx, qy, qz, qw = rotation.x, rotation.y, rotation.z, rotation.w
    tx = 2.0 * (qy * z - qz * y)
    ty = 2.0 * (qz * x - qx * z)
    tz = 2.0 * (qx * y - qy * x)
    rx = x + qw * tx + (qy * tz - qz * ty)
    ry = y + qw * ty + (qz * tx - qx * tz)
    rz = z + qw * tz + (qx * ty - qy * tx)
    return rx, ry, rz


def _transform_xy_from_pose(x: float, y: float, pose) -> tuple[float, float]:
    rx, ry, _ = _rotate_vector(x, y, 0.0, pose.orientation)
    return rx + pose.position.x, ry + pose.position.y


def _yaw_from_quaternion(rotation) -> float:
    return math.atan2(
        2.0 * (rotation.w * rotation.z + rotation.x * rotation.y),
        1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z),
    )


class CenterlinePreview(Node):
    def __init__(self) -> None:
        super().__init__("khonshu_eufs_centerline_preview")
        self.declare_parameter("cones_topic", "/ground_truth/cones")
        self.declare_parameter("state_topic", "/ground_truth/odom")
        self.declare_parameter("state_type", "nav_msgs/Odometry")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_footprint")
        self.declare_parameter("state_time_tolerance_s", 0.1)
        self.declare_parameter("maximum_cycle_time_ms", 30.0)
        self.declare_parameter("marker_topic", "/khonshu/markers")
        self.declare_parameter("path_topic", "/khonshu/centerline")
        self.declare_parameter("status_topic", "/khonshu/debug/status")
        self.declare_parameter("warm_up", True)
        self.declare_parameter("planner_backend", "stage_a")
        self.declare_parameter("publish_construction_markers", False)
        self._publish_construction_markers = bool(
            self.get_parameter("publish_construction_markers").value
        )

        self._map_frame = self.get_parameter("map_frame").value
        self._base_frame = self.get_parameter("base_frame").value
        self._state_time_tolerance_s = float(
            self.get_parameter("state_time_tolerance_s").value
        )
        if self._state_time_tolerance_s <= 0.0:
            raise ValueError("state_time_tolerance_s must be positive")
        self._state_samples = deque(maxlen=100)
        self._state_lock = Lock()
        self._logged_first_state = False
        self._logged_first_cones = False
        self._logged_first_match = False
        self._logged_first_plan = False
        state_type_name = self.get_parameter("state_type").value
        if state_type_name in {"nav_msgs/Odometry", "odometry"}:
            self._state_message_type = Odometry
        elif state_type_name in {
            "eufs_msgs/CarState",
            "eufs_msgs/msg/CarState",
            "car_state",
        }:
            self._state_message_type = CarState
        else:
            raise ValueError(
                "state_type must be nav_msgs/Odometry or eufs_msgs/CarState"
            )
        maximum_cycle_time_ms = float(
            self.get_parameter("maximum_cycle_time_ms").value
        )
        planner_backend = self.get_parameter("planner_backend").value
        if planner_backend == "stage_a":
            self._planner = StageAPlanner(
                StageAConfig(maximum_cycle_time_ms=maximum_cycle_time_ms)
            )
        elif planner_backend == "fasttube":
            self._planner = LocalPlanner(
                PlannerConfig(maximum_cycle_time_ms=maximum_cycle_time_ms)
            )
        else:
            raise ValueError("planner_backend must be stage_a or fasttube")
        self._markers_pub = self.create_publisher(
            MarkerArray, self.get_parameter("marker_topic").value, 10
        )
        self._path_pub = self.create_publisher(
            Path, self.get_parameter("path_topic").value, 10
        )
        self._status_pub = self.create_publisher(
            String, self.get_parameter("status_topic").value, 10
        )
        self._state_sub = self.create_subscription(
            self._state_message_type,
            self.get_parameter("state_topic").value,
            self._on_state,
            10,
        )
        self._cones_sub = self.create_subscription(
            ConeArrayWithCovariance,
            self.get_parameter("cones_topic").value,
            self._on_cones,
            10,
        )

        if self.get_parameter("warm_up").value and hasattr(self._planner, "warm_up"):
            self.get_logger().info("Warming the path planner; first start can take a while")
            self._planner.warm_up()
            self.get_logger().info("Warm-up complete; waiting for cone observations")

    @staticmethod
    def _stamp_seconds(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _on_state(self, msg: VehicleStateMessage) -> None:
        with self._state_lock:
            self._state_samples.append(msg)
        if not self._logged_first_state:
            self._logged_first_state = True
            self.get_logger().info(
                "Received first vehicle state "
                f"frame={msg.header.frame_id or '<empty>'} "
                f"child={msg.child_frame_id or '<empty>'} "
                f"stamp={self._stamp_seconds(msg.header.stamp):.3f}"
            )

    def _state_for_stamp(self, stamp) -> VehicleStateMessage | None:
        target_s = self._stamp_seconds(stamp)
        with self._state_lock:
            samples = tuple(self._state_samples)
        if not samples:
            return None
        nearest = min(
            samples,
            key=lambda state: abs(self._stamp_seconds(state.header.stamp) - target_s),
        )
        offset_s = abs(self._stamp_seconds(nearest.header.stamp) - target_s)
        return nearest if offset_s <= self._state_time_tolerance_s else None

    def _on_cones(self, msg: ConeArrayWithCovariance) -> None:
        stamp = msg.header.stamp
        cone_frame = msg.header.frame_id or self._map_frame
        if not self._logged_first_cones:
            self._logged_first_cones = True
            cone_count = sum(len(getattr(msg, field_name)) for field_name, _, _ in COLOR_GROUPS)
            self.get_logger().info(
                f"Received first cone batch count={cone_count} "
                f"frame={cone_frame} stamp={self._stamp_seconds(stamp):.3f}"
            )
        state_msg = self._state_for_stamp(stamp)
        if state_msg is None:
            if not self._logged_first_match:
                self._logged_first_match = True
                self.get_logger().warning(
                    "No vehicle state matched the first cone batch within "
                    f"{self._state_time_tolerance_s:.3f} s"
                )
            self._publish_stop("state_timestamp_mismatch", stamp)
            return

        if not self._logged_first_match:
            self._logged_first_match = True
            offset_s = abs(
                self._stamp_seconds(state_msg.header.stamp) - self._stamp_seconds(stamp)
            )
            self.get_logger().info(
                f"Matched cone and state stamps; offset={offset_s * 1000.0:.1f} ms"
            )

        state_frame = state_msg.header.frame_id or self._map_frame
        state_base_frame = state_msg.child_frame_id or self._base_frame
        if state_frame != self._map_frame:
            self._publish_stop("unexpected_state_frame", stamp)
            return
        if cone_frame not in {self._map_frame, self._base_frame, state_base_frame}:
            self._publish_stop("unsupported_cone_frame", stamp)
            return

        pose = state_msg.pose.pose

        cones = []
        for field_name, color, _ in COLOR_GROUPS:
            for cone_msg in getattr(msg, field_name):
                if cone_frame == self._map_frame:
                    x, y = cone_msg.point.x, cone_msg.point.y
                else:
                    x, y = _transform_xy_from_pose(cone_msg.point.x, cone_msg.point.y, pose)
                cones.append(Cone(x, y, color))

        timestamp_s = self._stamp_seconds(stamp)
        linear_velocity = state_msg.twist.twist.linear
        state = VehicleState(
            x=pose.position.x,
            y=pose.position.y,
            yaw_rad=_yaw_from_quaternion(pose.orientation),
            speed_mps=math.hypot(linear_velocity.x, linear_velocity.y),
            timestamp_s=timestamp_s,
            frame_id=self._map_frame,
        )
        log_plan_result = not self._logged_first_plan
        if log_plan_result:
            self._logged_first_plan = True
            self.get_logger().info(f"Planning centerline from {len(cones)} cones")
        trajectory = self._planner.plan(cones, state)
        if log_plan_result:
            diagnostics = getattr(self._planner, "last_diagnostics", {})
            self.get_logger().info(
                f"Centerline plan returned emergency_stop={trajectory.emergency_stop} "
                f"reason={trajectory.reason or '<none>'} "
                f"points={len(trajectory.points)} compute_ms={trajectory.compute_time_ms:.2f} "
                f"diagnostics={diagnostics}"
            )
        if trajectory.emergency_stop:
            self._publish_path_and_markers(msg.header, cones, state, trajectory)
            diagnostics = getattr(self._planner, "last_diagnostics", {})
            self._publish_status(
                f"STOP {trajectory.reason} ({trajectory.compute_time_ms:.2f} ms) "
                f"tri={diagnostics.get('delaunay_edges', 0)} "
                f"vor={diagnostics.get('voronoi_vertices', 0)} "
                f"graph={diagnostics.get('graph_edges', 0)} "
                f"clearance_m={diagnostics.get('minimum_clearance_m', 'na')} "
                f"body_clearance_m={diagnostics.get('minimum_body_clearance_m', 'na')}"
            )
            return

        self._publish_path_and_markers(msg.header, cones, state, trajectory)
        diagnostics = getattr(self._planner, "last_diagnostics", {})
        self._publish_status(
            f"OK points={len(trajectory.points)} "
            f"compute_ms={trajectory.compute_time_ms:.2f} "
            f"tri={diagnostics.get('delaunay_edges', 0)} "
            f"vor={diagnostics.get('voronoi_vertices', 0)} "
            f"graph={diagnostics.get('graph_edges', 0)} "
            f"nodes={diagnostics.get('selected_nodes', 0)} "
            f"clearance_m={diagnostics.get('minimum_clearance_m', 'na')} "
            f"body_clearance_m={diagnostics.get('minimum_body_clearance_m', 'na')}"
        )

    def _publish_status(self, value: str) -> None:
        message = String()
        message.data = value
        self._status_pub.publish(message)

    def _publish_stop(self, reason: str, stamp) -> None:
        clear = Marker()
        clear.header = Header(stamp=stamp, frame_id=self._map_frame)
        clear.action = Marker.DELETEALL
        self._markers_pub.publish(MarkerArray(markers=[clear]))
        empty_path = Path()
        empty_path.header = Header(stamp=stamp, frame_id=self._map_frame)
        self._path_pub.publish(empty_path)
        self._publish_status(f"STOP {reason}")

    def _publish_path_and_markers(self, input_header, cones, state, trajectory) -> None:
        header = Header(stamp=input_header.stamp, frame_id=self._map_frame)
        path_msg = Path()
        path_msg.header = header
        for index, point in enumerate(trajectory.points):
            next_point = trajectory.points[min(index + 1, len(trajectory.points) - 1)]
            previous_point = trajectory.points[max(index - 1, 0)]
            yaw = math.atan2(next_point.y - previous_point.y, next_point.x - previous_point.x)
            pose = PoseStamped()
            pose.header = header
            pose.pose.position.x = point.x
            pose.pose.position.y = point.y
            pose.pose.orientation.z = math.sin(0.5 * yaw)
            pose.pose.orientation.w = math.cos(0.5 * yaw)
            path_msg.poses.append(pose)
        self._path_pub.publish(path_msg)

        markers = []
        for marker_id, (field_name, color, rgb) in enumerate(COLOR_GROUPS):
            points = [Point(x=cone.x, y=cone.y, z=0.05) for cone in cones if cone.color == color]
            marker = Marker()
            marker.header = header
            marker.ns = field_name
            marker.id = marker_id
            marker.type = Marker.SPHERE_LIST
            marker.action = Marker.ADD if points else Marker.DELETE
            marker.scale.x = marker.scale.y = marker.scale.z = 0.28
            marker.color = ColorRGBA(r=rgb[0], g=rgb[1], b=rgb[2], a=1.0)
            marker.points = points
            markers.append(marker)

        if self._publish_construction_markers:
            markers.extend(self._construction_markers(header, state))

        line = Marker()
        line.header = header
        line.ns = "khonshu_centerline"
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD if trajectory.points else Marker.DELETE
        line.scale.x = 0.08
        line.color = ColorRGBA(r=0.1, g=0.85, b=0.25, a=1.0)
        line.points = [Point(x=point.x, y=point.y, z=0.12) for point in trajectory.points]
        markers.append(line)

        car = Marker()
        car.header = header
        car.ns = "vehicle_pose"
        car.id = 0
        car.type = Marker.ARROW
        car.action = Marker.ADD
        car.pose.position.x = state.x
        car.pose.position.y = state.y
        car.pose.orientation.z = math.sin(0.5 * state.yaw_rad)
        car.pose.orientation.w = math.cos(0.5 * state.yaw_rad)
        car.scale.x, car.scale.y, car.scale.z = 0.9, 0.16, 0.16
        car.color = ColorRGBA(r=0.08, g=0.08, b=0.08, a=1.0)
        markers.append(car)
        self._markers_pub.publish(MarkerArray(markers=markers))

    def _construction_markers(self, header, state: VehicleState) -> list[Marker]:
        """Render Stage A's live triangulation, Voronoi, sections, and search graph."""

        geometry = getattr(self._planner, "last_debug_geometry", {})
        cosine, sine = math.cos(state.yaw_rad), math.sin(state.yaw_rad)

        def point(local_xy, z: float) -> Point:
            x, y = float(local_xy[0]), float(local_xy[1])
            return Point(
                x=state.x + cosine * x - sine * y,
                y=state.y + sine * x + cosine * y,
                z=z,
            )

        markers: list[Marker] = []

        def line_list(
            namespace: str,
            marker_id: int,
            segments,
            color: ColorRGBA,
            width: float,
            z: float,
        ) -> None:
            marker = Marker()
            marker.header = header
            marker.ns = namespace
            marker.id = marker_id
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD if segments else Marker.DELETE
            marker.scale.x = width
            marker.color = color
            for first, second in segments:
                marker.points.extend((point(first, z), point(second, z)))
            markers.append(marker)

        def point_list(
            namespace: str,
            marker_id: int,
            locations,
            color: ColorRGBA,
            diameter: float,
            z: float,
        ) -> None:
            marker = Marker()
            marker.header = header
            marker.ns = namespace
            marker.id = marker_id
            marker.type = Marker.SPHERE_LIST
            marker.action = Marker.ADD if locations else Marker.DELETE
            marker.scale.x = marker.scale.y = marker.scale.z = diameter
            marker.color = color
            marker.points = [point(location, z) for location in locations]
            markers.append(marker)

        line_list(
            "stage_a_delaunay",
            10,
            geometry.get("delaunay_edges_local", []),
            ColorRGBA(r=0.2, g=0.65, b=1.0, a=0.48),
            0.025,
            0.04,
        )
        line_list(
            "stage_a_voronoi",
            11,
            geometry.get("voronoi_edges_local", []),
            ColorRGBA(r=0.8, g=0.35, b=1.0, a=0.75),
            0.045,
            0.06,
        )
        line_list(
            "stage_a_cross_sections",
            12,
            [section[:2] for section in geometry.get("cross_sections_local", [])],
            ColorRGBA(r=1.0, g=0.68, b=0.12, a=0.72),
            0.045,
            0.08,
        )
        line_list(
            "stage_a_candidate_graph",
            13,
            geometry.get("candidate_graph_edges_local", []),
            ColorRGBA(r=0.65, g=0.75, b=0.82, a=0.42),
            0.018,
            0.10,
        )
        point_list(
            "stage_a_midpoints",
            14,
            [section[2] for section in geometry.get("cross_sections_local", [])],
            ColorRGBA(r=1.0, g=0.82, b=0.12, a=0.95),
            0.12,
            0.13,
        )
        point_list(
            "stage_a_candidates",
            15,
            geometry.get("center_candidates_local", []),
            ColorRGBA(r=0.18, g=0.88, b=0.92, a=0.8),
            0.075,
            0.14,
        )

        route = geometry.get("selected_route_local", [])
        route_marker = Marker()
        route_marker.header = header
        route_marker.ns = "stage_a_dijkstra_route"
        route_marker.id = 16
        route_marker.type = Marker.LINE_STRIP
        route_marker.action = Marker.ADD if route else Marker.DELETE
        route_marker.scale.x = 0.075
        route_marker.color = ColorRGBA(r=1.0, g=0.22, b=0.72, a=0.95)
        route_marker.points = [point(location, 0.17) for location in route]
        markers.append(route_marker)
        return markers


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CenterlinePreview()
    try:
        rclpy.spin(node)
    except Exception:
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
