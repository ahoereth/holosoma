"""RetargeterNode: SmplhCmd in -> SMPLRetargeter -> DenseTrackingCmd out.

Decouples retargeting (variable-latency mink IK) from the control loop: it
solves on its own subscription rate and publishes a dense per-joint target
in holosoma (URDF/Mujoco 29-DOF) convention. Per-policy adapters subscribe to
DenseTrackingCmd and feed their policy.
"""

from __future__ import annotations

import numpy as np
import rclpy
import tyro
from holosoma_input_msgs.msg import DenseTrackingCmd, SmplhCmd
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from holosoma_inference.teleop.retargeting.smpl_retargeter import JOINT_NAMES, NUM_JOINTS, SMPLRetargeter

IN_TOPIC = "/holosoma/smplh_command"
OUT_TOPIC = "/holosoma/dense_tracking_command"
_IDX = {n: i for i, n in enumerate(JOINT_NAMES)}


def _to_transforms(msg: SmplhCmd) -> np.ndarray:
    """SmplhCmd -> (24, 7) [xyz, qxyzw], canonical order, identity for missing."""
    out = np.zeros((NUM_JOINTS, 7))
    out[:, 6] = 1.0
    for name, pose in zip(msg.joint_names, msg.joint_poses):
        if (i := _IDX.get(name)) is not None:
            p, q = pose.position, pose.orientation
            out[i] = [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
    return out


class RetargeterNode(Node):
    def __init__(self, urdf_path: str, rl_rate_hz: float, joint_names: list[str]):
        super().__init__("holosoma_retargeter")
        self._rt = SMPLRetargeter(urdf_path=urdf_path, dt=1.0 / rl_rate_hz)
        self._joint_names = joint_names
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._pub = self.create_publisher(DenseTrackingCmd, OUT_TOPIC, qos)
        self.create_subscription(SmplhCmd, IN_TOPIC, self._cb, qos)

    def _cb(self, msg: SmplhCmd) -> None:
        if not msg.valid or not msg.joint_poses:
            return
        q, dq, wxyz = self._rt.retarget(_to_transforms(msg))
        out = DenseTrackingCmd()
        out.header.stamp = self.get_clock().now().to_msg()
        out.joint_names = self._joint_names
        out.q = np.asarray(q, dtype=np.float32).tolist()
        out.dq = np.asarray(dq, dtype=np.float32).tolist()
        out.root_quat.x, out.root_quat.y, out.root_quat.z, out.root_quat.w = (
            float(wxyz[1]),
            float(wxyz[2]),
            float(wxyz[3]),
            float(wxyz[0]),
        )
        self._pub.publish(out)


def main(args=None) -> None:
    def run(urdf_path: str, rl_rate_hz: float = 50.0):
        rclpy.init(args=args)
        node = RetargeterNode(urdf_path, rl_rate_hz, joint_names=[])
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()

    tyro.cli(run)


if __name__ == "__main__":
    main()
