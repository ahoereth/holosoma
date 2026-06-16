"""Split-body Unitree controller (arm_sdk + loco)."""

from __future__ import annotations

import numpy as np
import rclpy
import tyro
from holosoma_input_msgs.msg import ExoskeletonCmd
from holosoma_state_msgs.msg import Heartbeat
from loguru import logger
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState

from holosoma_inference.sdk.unitree_high_level import make_mp_arm_client, make_mp_loco_client
from holosoma_inference.utils.rate import RateLimiter

EXOSKELETON_TOPIC = "/holosoma/exoskeleton_command"
CMD_TOPIC = "/holosoma_cmd"
HEARTBEAT_TOPIC = "/holosoma/heartbeat"
CONTROL_RATE_HZ = 50.0
HEARTBEAT_EVERY = 10  # ticks -> 5 Hz at 50 Hz control

# 14-DoF arm command order: [left(7), right(7)], matching ExoskeletonCmd and track_dual_arm.
_ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
ARM_JOINT_NAMES = [f"{side}_{j}" for side in ("left", "right") for j in _ARM_JOINTS]


class UnitreeSplitControllerNode(Node):
    def __init__(self, arm, loco) -> None:
        super().__init__("unitree_split_controller")
        self._arm = arm
        self._loco = loco
        self._latest: ExoskeletonCmd | None = None
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(ExoskeletonCmd, EXOSKELETON_TOPIC, self._cb, qos)
        self._cmd_pub = self.create_publisher(JointState, CMD_TOPIC, 10)
        self._hb_pub = self.create_publisher(Heartbeat, HEARTBEAT_TOPIC, 10)
        self._rate = RateLimiter(CONTROL_RATE_HZ)

    def run(self) -> None:
        logger.info(f"control loop @ {CONTROL_RATE_HZ:.0f} Hz")
        tick = 0
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.0)  # drain pending callbacks (newest cmd)
            cmd = self._latest
            if cmd is not None:
                q_target = np.concatenate([cmd.q_left_arm, cmd.q_right_arm])
                if self._arm is not None:
                    self._arm.track_dual_arm(q_target)
                if self._loco is not None:
                    v = cmd.base_velocity
                    self._loco.set_velocity(v.linear.x, v.linear.y, v.angular.z)
                self._publish_joint_command(q_target)
            if tick % HEARTBEAT_EVERY == 0:
                self._publish_heartbeat("running" if cmd is not None else "waiting_for_cmd")
            tick += 1
            self._rate.sleep()

    def _cb(self, msg: ExoskeletonCmd) -> None:
        self._latest = msg

    def _publish_joint_command(self, q_target: np.ndarray) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name, msg.position = ARM_JOINT_NAMES, q_target.tolist()
        self._cmd_pub.publish(msg)

    def _publish_heartbeat(self, status: str) -> None:
        msg = Heartbeat()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.robot_connected = self._arm is not None or self._loco is not None
        msg.control_mode = 0
        msg.status = status
        self._hb_pub.publish(msg)


def main(iface: str = "eth0", no_arms: bool = False, no_loco: bool = False) -> None:
    arm = loco = None
    if not no_loco:
        logger.info("starting loco client …")
        loco = make_mp_loco_client(iface=iface)
        loco.start()
        loco.set_walk_mode()
    if not no_arms:
        logger.info("starting arm client …")
        arm = make_mp_arm_client(iface=iface, motion_mode=True)
        arm.ctrl_dual_arm_initialization_pose()
        arm.speed_gradual_max()

    rclpy.init()
    controller = UnitreeSplitControllerNode(arm=arm, loco=loco)
    try:
        controller.run()
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("shutting down …")
        controller.destroy_node()
        rclpy.shutdown()
        if loco is not None:
            loco.close()  # type: ignore[attr-defined]
        if arm is not None:
            arm.close()  # type: ignore[attr-defined]


def _cli() -> None:
    tyro.cli(main)


if __name__ == "__main__":
    _cli()
