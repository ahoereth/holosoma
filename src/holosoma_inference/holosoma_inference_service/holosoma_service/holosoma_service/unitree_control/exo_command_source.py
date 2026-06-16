"""ExoCommandSource: the split-body target provider.

A thin node that subscribes ``ExoskeletonCmd`` (newest-wins) and serves the
latest frame via ``get_latest()``. Nothing else — publishing and control live in
``ControllerNode``. Spins rclpy in a daemon thread so the controller loop owns
the main thread.
"""

from __future__ import annotations

import threading

import rclpy
from holosoma_input_msgs.msg import ExoskeletonCmd
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

EXOSKELETON_TOPIC = "/holosoma/exoskeleton_command"


class ExoCommandSource(Node):
    def __init__(self) -> None:
        super().__init__("holosoma_exo_command_source")
        self._latest: ExoskeletonCmd | None = None
        self._thread: threading.Thread | None = None
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(ExoskeletonCmd, EXOSKELETON_TOPIC, self._cb, qos)

    def start(self) -> None:
        self._thread = threading.Thread(target=rclpy.spin, args=(self,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.destroy_node()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def _cb(self, msg: ExoskeletonCmd) -> None:
        self._latest = msg

    def get_latest(self) -> ExoskeletonCmd | None:
        return self._latest
