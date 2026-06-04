"""ROS2 listener for ``ExoskeletonCmd``.

Experimental. All rclpy lifecycle (init, spin thread, shutdown) is packaged
inside :class:`TeleopListener` so the rest of holosoma — which is not a
ROS2-first codebase — never touches rclpy. ``start()`` it, poll the latest
command from your own control loop, and ``stop()`` it in a finally block:

    listener = TeleopListener()
    listener.start()
    try:
        while running:
            cmd = listener.get_latest()   # newest ExoskeletonCmd or None
            ...
    finally:
        listener.stop()

``get_latest`` returns the most recent message (newest-wins; no backlog). The
ROS callback only stores it — the caller's loop owns the control rate.
``HolosomaTeleopListenerNode`` is the thin ``Node`` subclass; ``TeleopListener``
owns its lifetime.
"""

from __future__ import annotations

import threading
import time

import rclpy
from loguru import logger
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from holosoma_inference.teleop.holosoma_teleop_msgs._ensure_msgs import ExoskeletonCmd

DEFAULT_TOPIC = "/holosoma/tracker_command"


class HolosomaTeleopListenerNode(Node):
    """Subscribe to ``ExoskeletonCmd`` and cache the latest message."""

    def __init__(self, topic: str = DEFAULT_TOPIC):
        super().__init__("holosoma_teleop_listener")
        # Newest received command. Reference assignment is atomic under the GIL,
        # so the polling loop always reads a whole message, never a torn one.
        self._latest: ExoskeletonCmd | None = None

        # Best-effort, depth 1: drop stale commands rather than queue them
        # (teleop wants the freshest target, not a backlog).
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._sub = self.create_subscription(ExoskeletonCmd, topic, self._cb, qos)
        logger.info(f"[teleop-listener] subscribed to {topic}")

    def _cb(self, msg: ExoskeletonCmd) -> None:
        self._latest = msg

    def get_latest(self) -> ExoskeletonCmd | None:
        return self._latest


class TeleopListener:
    """Owns the full rclpy lifecycle in a background thread.

    Encapsulates ``rclpy.init``/``spin``/``shutdown`` so callers stay
    ROS-agnostic. The spin thread caches each message; poll the newest via
    :meth:`get_latest` from your own control loop.
    """

    def __init__(self, topic: str = DEFAULT_TOPIC):
        self._topic = topic
        self._node: HolosomaTeleopListenerNode | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        rclpy.init()
        self._node = HolosomaTeleopListenerNode(self._topic)
        self._thread = threading.Thread(target=rclpy.spin, args=(self._node,), daemon=True)
        self._thread.start()

    def get_latest(self) -> ExoskeletonCmd | None:
        return self._node.get_latest() if self._node is not None else None

    def stop(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        if rclpy.ok():
            rclpy.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None


def main(args=None) -> None:
    """Standalone smoke test: poll the listener and log received commands."""
    listener = TeleopListener()
    listener.start()
    try:
        while True:
            cmd = listener.get_latest()
            if cmd is not None:
                logger.info(
                    f"cmd: L={list(cmd.q_left_arm)} R={list(cmd.q_right_arm)} "
                    f"v=({cmd.base_velocity.linear.x:.2f},{cmd.base_velocity.linear.y:.2f},"
                    f"{cmd.base_velocity.angular.z:.2f})"
                )
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()


if __name__ == "__main__":
    main()
