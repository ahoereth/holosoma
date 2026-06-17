"""Holosoma WBT policy node.

Runs a whole-body-tracking policy driven by a live ``CmdDense`` stream:

    CmdDense ─▶ HolosomaNode ─▶ policy.target_source ─▶ policy.run() ─▶ robot

The policy class is resolved by ``config.task.policy_type`` via the
``holosoma.policies.by_type`` entry-point group (registered by the installed
policy extension) — so this node never imports the extension by name. Requires
a policy extension installed in the env.
"""

from __future__ import annotations

import threading

import numpy as np
import rclpy
import tyro
from holosoma_msgs.msg import CmdDense, Heartbeat
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState

from holosoma_inference.compat import entry_points
from holosoma_inference.config.config_values.inference import get_annotated_inference_config
from holosoma_inference.config.utils import TYRO_CONFIG

DENSE_TOPIC = "/holosoma/dense_tracking_command"
EXECUTED_CMD_TOPIC = "/holosoma/holosoma_executed_cmd"
HEARTBEAT_TOPIC = "/holosoma/heartbeat"
HEARTBEAT_EVERY = 10  # control ticks -> 5 Hz at a 50 Hz control loop
_POLICY_GROUP = "holosoma.policies.by_type"


class HolosomaNode(Node):
    """ROS adapter between the policy and the dense teleop topics.

    * Input: subscribes ``CmdDense`` and serves it as a WBT ``TargetSource``
      (newest-wins; holds the last frame between policy ticks) via
      ``get_target()``, which the policy pulls each control tick.
    * Output: publishes executed-command + heartbeat feedback, driven by the
      policy's per-tick ``_on_command_sent`` hook (so it runs in the policy
      control thread, in every state). Mirrors the split-body controller's
      feedback topics:
        - ``/holosoma/holosoma_executed_cmd`` (``JointState``): the full-body
          joint command actually sent this tick, at the control rate.
        - ``/holosoma/heartbeat`` (``Heartbeat``): liveness + status at ~5 Hz.

    ``dof_names`` is set by :func:`main` after the policy is built (the policy
    can't exist before this node, since it's injected as the target source).
    """

    def __init__(self, num_dofs: int, topic: str = DENSE_TOPIC):
        super().__init__("holosoma_node")
        # Input (target source) state.
        self._cmd = np.zeros((1, 2 * num_dofs), dtype=np.float32)  # held until first frame
        self._ref = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # xyzw
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(CmdDense, topic, self._cb, qos)
        # Output (feedback) state.
        self.dof_names: list[str] = []
        self._cmd_pub = self.create_publisher(JointState, EXECUTED_CMD_TOPIC, 10)
        self._hb_pub = self.create_publisher(Heartbeat, HEARTBEAT_TOPIC, 10)
        self._tick = 0
        threading.Thread(target=rclpy.spin, args=(self,), daemon=True).start()

    # --- Input: target source the policy pulls from each tick ---
    def _cb(self, msg: CmdDense) -> None:
        self._cmd = np.concatenate([msg.q, msg.dq]).astype(np.float32).reshape(1, -1)
        r = msg.root_quat
        self._ref = np.array([r.x, r.y, r.z, r.w], dtype=np.float32)

    def get_target(self, num_dofs: int, rl_rate_hz: float, urdf_path: str | None):
        return self._cmd, self._ref

    # --- Output: feedback, driven by the policy's _on_command_sent hook ---
    def on_command_sent(self, policy, cmd_q) -> None:
        cmd = np.asarray(cmd_q, dtype=np.float64).reshape(-1)

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        if len(self.dof_names) == cmd.shape[0]:
            msg.name = self.dof_names
        msg.position = cmd.tolist()
        self._cmd_pub.publish(msg)

        if self._tick % HEARTBEAT_EVERY == 0:
            hb = Heartbeat()
            hb.header.stamp = msg.header.stamp
            hb.robot_connected = getattr(policy, "interface", None) is not None
            hb.control_mode = 0
            if getattr(policy, "use_policy_action", False):
                hb.status = "running"
            elif getattr(policy, "get_ready_state", False):
                hb.status = "get_ready"
            else:
                hb.status = "stiff_hold"
            self._hb_pub.publish(hb)
        self._tick += 1


def _load_policy_class(policy_type: str):
    eps = {ep.name: ep for ep in entry_points(group=_POLICY_GROUP)}
    if policy_type not in eps:
        raise ValueError(f"policy_type {policy_type!r} not in {_POLICY_GROUP}; available: {sorted(eps)}")
    return eps[policy_type].load()


def main() -> None:
    config = tyro.cli(get_annotated_inference_config(), config=TYRO_CONFIG)
    rclpy.init()
    policy_cls = _load_policy_class(config.task.policy_type)
    node = HolosomaNode(config.robot.num_joints)
    policy = policy_cls(config=config, target_source=node)

    # Now that the policy exists, give the node its joint names and wire the
    # executed-cmd + heartbeat feedback into the policy's per-tick hook.
    node.dof_names = list(getattr(policy, "dof_names", []))
    policy._on_command_sent = lambda cmd_q, state, _p=policy, _n=node: _n.on_command_sent(_p, cmd_q)

    try:
        policy.run()  # owns its RateLimiter + SDK I/O; reads get_target() each step
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
