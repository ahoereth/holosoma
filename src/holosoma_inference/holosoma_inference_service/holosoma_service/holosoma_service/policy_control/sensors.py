"""ROS2 sensor implementations for the service node.

The ``Sensor`` ABC lives in ``holosoma_inference.sensors.base``; policies
depend only on that interface. The concrete classes here are a service-layer
detail — they subscribe ROS2 topics on the shared ``ServiceIONode`` and
implement the ``Sensor`` protocol.
"""

from __future__ import annotations

import threading
import time

import numpy as np
from loguru import logger
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from holosoma_inference.sensors.base import Sensor


class Ros2DepthSensor(Sensor):
    """Subscribes a ``sensor_msgs/Image`` topic (encoding ``32FC1``) and
    exposes the latest depth frame as a float32 numpy array.

    Shape: ``(1, 1, H, W)`` — single camera. Multi-camera stacking is a
    follow-up; for now the caller is responsible for combining cameras.

    ``get_latest()`` returns ``None`` if:
    - no message has been received yet, or
    - the most recent message is older than ``timeout`` seconds.

    The publisher is expected to produce ``32FC1`` encoded images. A wrong
    encoding is logged as a warning and the frame is discarded.
    """

    EXPECTED_ENCODING = "32FC1"

    def __init__(self, node: Node, topic: str, timeout: float = 0.5):
        self._timeout = timeout
        self._latest: np.ndarray | None = None
        self._stamp: float = 0.0
        self._lock = threading.Lock()

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        node.create_subscription(Image, topic, self._cb, qos)
        logger.info(f"Ros2DepthSensor subscribed to {topic} (encoding={self.EXPECTED_ENCODING})")

    def start(self) -> None:
        pass  # subscriptions live on the caller's node; nothing to start

    def _cb(self, msg: Image) -> None:
        if msg.encoding != self.EXPECTED_ENCODING:
            logger.warning(
                f"Ros2DepthSensor: expected encoding {self.EXPECTED_ENCODING!r}, got {msg.encoding!r} — frame discarded"
            )
            return
        arr = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        with self._lock:
            self._latest = arr.reshape(1, 1, msg.height, msg.width)
            self._stamp = time.monotonic()

    def get_latest(self) -> np.ndarray | None:
        """Return ``(1, 1, H, W)`` float32 array, or ``None`` if stale/absent."""
        with self._lock:
            if self._latest is None:
                return None
            if self._timeout > 0 and (time.monotonic() - self._stamp) > self._timeout:
                return None
            return self._latest.copy()
