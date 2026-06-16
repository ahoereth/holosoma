"""Live teleop -> WBT policy.

    SmplhCmd -> retargeter_node -> DenseTrackingCmd -> run_policy (WBT) -> robot

The policy is wbt_wrappers_inference.run_policy with --task.teleop-topic, which
injects a DenseTargetSource (subscribes the dense topic) instead of an NPZ.
Requires both holosoma + wbt_wrappers installed.

Usage:
    ros2 launch holosoma_service teleop_with_holosoma_policy.launch.py \
        urdf_path:=/path/g1_29dof.urdf model_path:=/path/model.onnx preset:=g1-29dof-holosoma-wbt
"""

import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

DENSE_TOPIC = "/holosoma/dense_tracking_command"


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
            ExecuteProcess(
                # FAR-pi's WBT runner; --task.teleop-topic flips it to the live source.
                cmd=[
                    sys.executable,
                    "-m",
                    "wbt_wrappers_inference.run_policy",
                    preset,
                    "--task.model-path",
                    model,
                    "--task.teleop-topic",
                    DENSE_TOPIC,
                ],
                output="screen",
            ),
        ]
    )
