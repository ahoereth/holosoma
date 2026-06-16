"""DenseTargetSource: subscribes to DenseTrackingCmd, serves it as a WBT
``TargetSource``. Newest-wins; holds the last frame between policy ticks."""

from __future__ import annotations

import threading

import numpy as np
import rclpy
from holosoma_input_msgs.msg import DenseTrackingCmd
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

_TOPIC = "/holosoma/dense_tracking_command"


class DenseTargetSource(Node):
    def __init__(self, num_dofs: int, topic: str = _TOPIC):
        super().__init__("holosoma_dense_target")
        self._cmd = np.zeros((1, 2 * num_dofs), dtype=np.float32)  # held until first frame
        self._ref = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # xyzw
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(DenseTrackingCmd, topic, self._cb, qos)
        threading.Thread(target=rclpy.spin, args=(self,), daemon=True).start()

    def _cb(self, msg: DenseTrackingCmd) -> None:
        self._cmd = np.concatenate([msg.q, msg.dq]).astype(np.float32).reshape(1, -1)
        r = msg.root_quat
        self._ref = np.array([r.x, r.y, r.z, r.w], dtype=np.float32)

    def get_target(self, num_dofs: int, rl_rate_hz: float, urdf_path: str | None):
        return self._cmd, self._ref
