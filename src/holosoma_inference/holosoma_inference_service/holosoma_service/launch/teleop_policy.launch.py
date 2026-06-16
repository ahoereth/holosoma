"""Live teleop -> WBT policy.

    SmplhCmd -> retargeter_node -> DenseTrackingCmd -> policy_node -> robot

Usage:
    ros2 launch holosoma_service teleop_policy.launch.py \
        urdf_path:=/path/g1_29dof.urdf model_path:=/path/model.onnx preset:=g1-29dof-holosoma-wbt
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    urdf = LaunchConfiguration("urdf_path")
    model = LaunchConfiguration("model_path")
    preset = LaunchConfiguration("preset")
    rl_rate = LaunchConfiguration("rl_rate_hz")

    return LaunchDescription(
        [
            DeclareLaunchArgument("urdf_path"),
            DeclareLaunchArgument("model_path"),
            DeclareLaunchArgument("preset", default_value="g1-29dof-holosoma-wbt"),
            DeclareLaunchArgument("rl_rate_hz", default_value="50.0"),
            Node(
                package="holosoma_service",
                executable="retargeter_node",
                name="retargeter",
                arguments=["--urdf-path", urdf, "--rl-rate-hz", rl_rate],
                output="screen",
            ),
            Node(
                package="holosoma_service",
                executable="policy_node",
                name="wbt_policy",
                arguments=[preset, "--task.model-path", model],
                output="screen",
            ),
        ]
    )
