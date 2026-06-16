"""Holosoma WBT policy node.

Runs a whole-body-tracking policy driven by a live ``DenseTrackingCmd`` stream:

    DenseTrackingCmd ─▶ DenseTargetSource ─▶ policy.target_source ─▶ policy.run() ─▶ robot

The policy class is resolved by ``config.task.policy_type`` via the
``holosoma.policies.by_type`` entry-point group (registered by the installed
policy extension, e.g. wbt_wrappers' ``HolosomaWBTPolicy``) — so this node never
imports the extension by name. Requires the extension installed in the env.
"""

from __future__ import annotations

import threading

import numpy as np
import rclpy
import tyro
from holosoma_input_msgs.msg import DenseTrackingCmd
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from holosoma_inference.compat import entry_points
from holosoma_inference.config.config_values.inference import get_annotated_inference_config
from holosoma_inference.config.utils import TYRO_CONFIG

DENSE_TOPIC = "/holosoma/dense_tracking_command"
_POLICY_GROUP = "holosoma.policies.by_type"


class DenseTargetSource(Node):
    """Subscribes ``DenseTrackingCmd``; serves it as a WBT ``TargetSource``
    (newest-wins, holds the last frame between policy ticks)."""

    def __init__(self, num_dofs: int, topic: str = DENSE_TOPIC):
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


def _load_policy_class(policy_type: str):
    eps = {ep.name: ep for ep in entry_points(group=_POLICY_GROUP)}
    if policy_type not in eps:
        raise ValueError(f"policy_type {policy_type!r} not in {_POLICY_GROUP}; available: {sorted(eps)}")
    return eps[policy_type].load()


def main() -> None:
    config = tyro.cli(get_annotated_inference_config(), config=TYRO_CONFIG)
    rclpy.init()
    policy_cls = _load_policy_class(config.task.policy_type)
    policy = policy_cls(config=config, target_source=DenseTargetSource(config.robot.num_joints))
    try:
        policy.run()  # owns its RateLimiter + SDK I/O; reads get_target() each step
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
