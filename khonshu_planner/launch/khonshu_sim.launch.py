"""Launch Khonshu planner_node and optional sim-only Pure Pursuit controller."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    with_controller = LaunchConfiguration("with_controller")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "with_controller",
                default_value="true",
                description="Launch sim-only Pure Pursuit follower alongside planner_node.",
            ),
            Node(
                package="khonshu_planner",
                executable="planner_node",
                name="khonshu_planner_node",
                output="screen",
            ),
            Node(
                package="khonshu_planner",
                executable="pure_pursuit_sim",
                name="pure_pursuit_sim",
                output="screen",
                condition=IfCondition(with_controller),
            ),
        ]
    )
