"""Live teleop -> WBT policy.

    (smplh) CmdSMPLH -> retargeter_node -> CmdDense -> holosoma_node (WBT) -> robot
    (dense)                               CmdDense -> holosoma_node (WBT) -> robot

holosoma_node resolves the policy class from ``config.task.policy_type`` via the
``holosoma.policies.by_type`` entry-point group and injects a DenseTargetSource
subscribed to the dense topic. Requires both holosoma + the policy extension
(e.g. wbt_wrappers) installed.

``input_type`` selects how CmdDense is produced:

* ``smplh`` (default): launch the retargeter so a SMPL-H source (``CmdSMPLH``)
  is retargeted into ``CmdDense``. Requires ``urdf_path`` (the retargeter's IK
  model).
* ``dense``: skip the retargeter; an external publisher feeds ``CmdDense``
  directly (e.g. ``scripts/publish_from_npz.py`` or any teleop already in dense
  29-DOF convention). ``urdf_path`` is not needed.

Usage:
    # SMPL-H teleop (retargeter on):
    ros2 launch holosoma_service teleop_with_holosoma_policy.launch.py \
        urdf_path:=/path/g1_29dof.urdf model_path:=/path/model.onnx

    # Dense input (retargeter off):
    ros2 launch holosoma_service teleop_with_holosoma_policy.launch.py \
        input_type:=dense model_path:=/path/model.onnx
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import EqualsSubstitution, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    urdf = LaunchConfiguration("urdf_path")
    model = LaunchConfiguration("model_path")
    preset = LaunchConfiguration("preset")
    rl_rate = LaunchConfiguration("rl_rate_hz")
    input_type = LaunchConfiguration("input_type")

    return LaunchDescription(
        [
            DeclareLaunchArgument("model_path"),
            DeclareLaunchArgument(
                "input_type",
                default_value="smplh",
                choices=["smplh", "dense"],
                description="smplh: run the retargeter (CmdSMPLH->CmdDense). "
                "dense: skip it, an external publisher feeds CmdDense directly.",
            ),
            DeclareLaunchArgument(
                "urdf_path",
                default_value="",
                description="Fixed-base 29-DOF URDF for the retargeter IK. Required for input_type:=smplh.",
            ),
            DeclareLaunchArgument("preset", default_value="g1-29dof-holosoma-wbt"),
            DeclareLaunchArgument("rl_rate_hz", default_value="50.0"),
            # Retargeter: only when consuming SMPL-H input.
            Node(
                package="holosoma_service",
                executable="retargeter_node",
                name="retargeter",
                arguments=["--urdf-path", urdf, "--rl-rate-hz", rl_rate],
                output="screen",
                condition=IfCondition(EqualsSubstitution(input_type, "smplh")),
            ),
            Node(
                package="holosoma_service",
                executable="holosoma_node",
                name="holosoma_policy",
                # tyro CLI: positional preset + --task.* overrides.
                arguments=[preset, "--task.model-path", model],
                output="screen",
            ),
        ]
    )
