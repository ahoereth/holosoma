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
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from holosoma_inference.sensors.base import Sensor


def _decode_depth(msg: Image, topic: str) -> np.ndarray | None:
    """Decode a 32FC1 depth Image to an (H, W) float32 array, or None on bad encoding."""
    if msg.encoding != Ros2DepthSensor.EXPECTED_ENCODING:
        logger.warning(
            f"Ros2DepthSensor[{topic}]: expected encoding "
            f"{Ros2DepthSensor.EXPECTED_ENCODING!r}, got {msg.encoding!r} — frame discarded"
        )
        return None
    return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)


class Ros2DepthSensor(Sensor):
    """Subscribes one or more ``sensor_msgs/Image`` topics (encoding ``32FC1``)
    and exposes the latest depth frames as a stacked float32 array.

    Output shape: ``(N, 1, H, W)`` where ``N == len(topics)``. Topic order is
    the camera order in the stack (front first, back second — matching the
    image_server convention the policy was trained against). Single-camera is
    just ``N == 1``; the consumer always gets the same stacked layout.

    Multi-camera frames are time-aligned with ``message_filters``'
    ``ApproximateTimeSynchronizer`` so the stack is from (approximately) one
    instant — the independent ROS2 topics are otherwise unsynchronized, unlike
    the old image_server which grabbed both cameras in one call. Single-camera
    needs no sync and uses a plain subscription.

    ``get_latest()`` returns ``None`` until a (synchronized) frame set has
    arrived and while the latest set is older than ``timeout`` seconds, so the
    policy can zero the observation rather than feed a stale/partial stack.

    The publisher must produce ``32FC1`` images; a wrong encoding is logged and
    that frame set is dropped.
    """

    EXPECTED_ENCODING = "32FC1"

    # message_filters slop (s): max timestamp spread within a synced frame set.
    SYNC_SLOP_S = 0.05

    def __init__(self, node: Node, topics: list[str], timeout: float = 0.5):
        if not topics:
            raise ValueError("Ros2DepthSensor requires at least one topic")
        self._topics = list(topics)
        self._timeout = timeout
        self._latest: np.ndarray | None = None  # (N, 1, H, W)
        self._stamp: float = 0.0
        self._lock = threading.Lock()

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        if len(self._topics) == 1:
            node.create_subscription(Image, self._topics[0], self._single_cb, qos)
        else:
            # Sync all camera topics so the stack is from one instant.
            subs = [Subscriber(node, Image, t, qos_profile=qos) for t in self._topics]
            self._sync = ApproximateTimeSynchronizer(subs, queue_size=2, slop=self.SYNC_SLOP_S)
            self._sync.registerCallback(self._synced_cb)

        logger.info(
            f"Ros2DepthSensor subscribed to {len(self._topics)} camera(s): "
            f"{self._topics} (encoding={self.EXPECTED_ENCODING}"
            f"{'' if len(self._topics) == 1 else f', sync slop={self.SYNC_SLOP_S}s'})"
        )

    def start(self) -> None:
        pass  # subscriptions live on the caller's node; nothing to start

    def _store(self, frames: list[np.ndarray]) -> None:
        # (H, W) per camera -> (1, H, W) -> stack to (N, 1, H, W).
        stacked = np.stack([f[np.newaxis, :, :] for f in frames], axis=0)
        with self._lock:
            self._latest = stacked
            self._stamp = time.monotonic()

    def _single_cb(self, msg: Image) -> None:
        frame = _decode_depth(msg, self._topics[0])
        if frame is not None:
            self._store([frame])

    def _synced_cb(self, *msgs: Image) -> None:
        frames = [_decode_depth(m, t) for m, t in zip(msgs, self._topics)]
        if any(f is None for f in frames):
            return  # a bad-encoding frame in the set; drop the whole set
        self._store(frames)

    def get_latest(self) -> np.ndarray | None:
        """Return ``(N, 1, H, W)`` float32 array, or ``None`` if absent/stale."""
        with self._lock:
            if self._latest is None:
                return None
            if self._timeout > 0 and (time.monotonic() - self._stamp) > self._timeout:
                return None
            return self._latest.copy()
