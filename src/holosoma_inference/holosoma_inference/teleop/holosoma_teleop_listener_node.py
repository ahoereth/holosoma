"""ROS2 listener for ``UnitreeTrackerCommand``.

Experimental. All rclpy lifecycle (init, spin thread, shutdown) is packaged
inside :class:`TeleopListener` so the rest of holosoma — which is not a
ROS2-first codebase — never touches rclpy. Use it as a context manager:

    with TeleopListener(on_command=ctrl.set_target):
        ctrl.run()              # your own loop; ROS spins in a bg thread

``HolosomaTeleopListenerNode`` is the thin ``Node`` subclass; ``TeleopListener``
owns its lifetime.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Self

import rclpy
from loguru import logger
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from holosoma_inference.teleop.holosoma_teleop_msgs._ensure_msgs import UnitreeTrackerCommand

DEFAULT_TOPIC = "/holosoma/tracker_command"


class HolosomaTeleopListenerNode(Node):
    """Subscribe to ``UnitreeTrackerCommand`` and forward each message to ``on_command``."""

    def __init__(self, topic: str = DEFAULT_TOPIC, on_command: Callable[[UnitreeTrackerCommand], None] | None = None):
        super().__init__("holosoma_teleop_listener")
        self._on_command = on_command

        # Best-effort, depth 1: drop stale commands rather than queue them
        # (teleop wants the freshest target, not a backlog).
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._sub = self.create_subscription(UnitreeTrackerCommand, topic, self._cb, qos)
        logger.info(f"[teleop-listener] subscribed to {topic}")

    def _cb(self, msg: UnitreeTrackerCommand) -> None:
        if self._on_command is not None:
            self._on_command(msg)


class TeleopListener:
    """Owns the full rclpy lifecycle in a background thread.

    Encapsulates ``rclpy.init``/``spin``/``shutdown`` so callers stay
    ROS-agnostic. ``on_command`` fires (in the spin thread) on each message.
    """

    def __init__(self, topic: str = DEFAULT_TOPIC, on_command: Callable[[UnitreeTrackerCommand], None] | None = None):
        self._topic = topic
        self._on_command = on_command
        self._node: HolosomaTeleopListenerNode | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        rclpy.init()
        self._node = HolosomaTeleopListenerNode(self._topic, self._on_command)
        self._thread = threading.Thread(target=rclpy.spin, args=(self._node,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        if rclpy.ok():
            rclpy.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


def main(args=None) -> None:
    """Standalone smoke test: spin the listener and log received commands."""
    listener = TeleopListener(
        on_command=lambda m: logger.info(
            f"cmd: L={list(m.q_left_arm)} R={list(m.q_right_arm)} "
            f"v=({m.base_velocity.linear.x:.2f},{m.base_velocity.linear.y:.2f},{m.base_velocity.angular.z:.2f})"
        )
    )
    listener.start()
    try:
        threading.Event().wait()  # sleep forever; callbacks fire in the spin thread
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()


if __name__ == "__main__":
    main()
