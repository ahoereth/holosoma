"""ROS2 listener for ``UnitreeTrackerCommand``.

Experimental. Subscribes to the tracker-command topic and exposes the latest
message via a non-blocking ``get_latest()`` poll (and an optional callback).
The node does NOT touch the robot — it only receives. ``run_service.py`` owns
the SDK clients and pulls from here each control tick.
"""

from __future__ import annotations

from collections.abc import Callable

import rclpy
from loguru import logger
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from holosoma_inference.teleop.holosoma_teleop_msgs._ensure_msgs import UnitreeTrackerCommand

DEFAULT_TOPIC = "/holosoma/tracker_command"


class HolosomaTeleopListenerNode(Node):
    """Subscribe to ``UnitreeTrackerCommand`` and cache the latest payload."""

    def __init__(self, topic: str = DEFAULT_TOPIC, on_command: Callable[[UnitreeTrackerCommand], None] | None = None):
        super().__init__("holosoma_teleop_listener")
        self._latest: UnitreeTrackerCommand | None = None
        self._on_command = on_command

        # Best-effort, depth 1: drop stale commands rather than queue them
        # (teleop wants the freshest target, not a backlog).
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._sub = self.create_subscription(UnitreeTrackerCommand, topic, self._cb, qos)
        logger.info(f"[teleop-listener] subscribed to {topic}")

    def _cb(self, msg: UnitreeTrackerCommand) -> None:
        self._latest = msg
        if self._on_command is not None:
            self._on_command(msg)

    def get_latest(self) -> UnitreeTrackerCommand | None:
        """Return the most recent command (or None if nothing received yet)."""
        return self._latest


def main(args=None) -> None:
    """Standalone smoke test: spin the listener and log received commands."""
    rclpy.init(args=args)
    node = HolosomaTeleopListenerNode(
        on_command=lambda m: logger.info(
            f"cmd: L={list(m.q_left_arm)} R={list(m.q_right_arm)} "
            f"v=({m.base_velocity.linear.x:.2f},{m.base_velocity.linear.y:.2f},{m.base_velocity.angular.z:.2f})"
        )
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
