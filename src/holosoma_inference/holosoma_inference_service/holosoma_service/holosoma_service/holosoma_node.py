"""Holosoma ROS I/O node.

Subscribes to all supported teleop inputs (one-of: whichever publishes, newest
wins) and publishes all service outputs. All rclpy lifecycle lives in
:class:`HolosomaNode` so the rest of holosoma — not a ROS2-first codebase —
never touches rclpy. ``start()`` it, poll ``get_latest()`` from your own control
loop, ``stop()`` in a finally block.

    Inputs  (sub):  ExoskeletonCmd · SmplhCmd · ThreePointCmd
    Outputs (pub):  JointState (commanded joints) · Heartbeat (status)

``get_latest()`` returns the newest input of ANY type (caller switches on it);
the callback only caches it (atomic ref assignment), so the control loop owns
the rate.
"""

from __future__ import annotations

import threading
import time

import rclpy
from holosoma_input_msgs.msg import ExoskeletonCmd, SmplhCmd, ThreePointCmd
from holosoma_state_msgs.msg import Heartbeat
from loguru import logger
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState

# input topics (one per supported type)
EXOSKELETON_TOPIC = "/holosoma/exoskeleton_command"
SMPLH_TOPIC = "/holosoma/smplh_command"
THREE_POINT_TOPIC = "/holosoma/three_point_command"
# output topics
CMD_TOPIC = "/holosoma_cmd"
HEARTBEAT_TOPIC = "/holosoma/heartbeat"

_INPUTS = ((ExoskeletonCmd, EXOSKELETON_TOPIC), (SmplhCmd, SMPLH_TOPIC), (ThreePointCmd, THREE_POINT_TOPIC))


class HolosomaNode(Node):
    """Subscribes all teleop inputs (newest-wins, any type); publishes outputs."""

    def __init__(self) -> None:
        super().__init__("holosoma_node")
        self._latest: ExoskeletonCmd | SmplhCmd | ThreePointCmd | None = None
        self._thread: threading.Thread | None = None
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        for msg_type, topic in _INPUTS:
            self.create_subscription(msg_type, topic, self._cb, qos)
        self._cmd_pub = self.create_publisher(JointState, CMD_TOPIC, 10)
        self._hb_pub = self.create_publisher(Heartbeat, HEARTBEAT_TOPIC, 10)
        logger.info(f"[holosoma-node] sub {[t for _, t in _INPUTS]} | pub {CMD_TOPIC}, {HEARTBEAT_TOPIC}")

    def start(self) -> None:
        """Spin in a daemon thread so the caller's control loop owns the main thread."""
        self._thread = threading.Thread(target=rclpy.spin, args=(self,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.destroy_node()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def _cb(self, msg) -> None:
        self._latest = msg  # newest of any input type wins

    def get_latest(self):
        return self._latest

    def publish_joint_command(self, names, positions, velocities) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name, msg.position, msg.velocity = names, positions, velocities
        self._cmd_pub.publish(msg)

    def publish_heartbeat(self, robot_connected: bool, control_mode: int, status: str) -> None:
        msg = Heartbeat()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.robot_connected, msg.control_mode, msg.status = robot_connected, control_mode, status
        self._hb_pub.publish(msg)


def main(args=None) -> None:
    """Smoke test: spin and log the type of each received input."""
    rclpy.init()
    node = HolosomaNode()
    node.start()
    try:
        while True:
            cmd = node.get_latest()
            if cmd is not None:
                logger.info(f"latest input: {type(cmd).__name__}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
