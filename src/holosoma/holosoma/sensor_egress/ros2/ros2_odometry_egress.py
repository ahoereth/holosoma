"""ROS2 odometry-publishing egress: publishes the robot base pose/velocity as nav_msgs/Odometry.

A self-sourced egress (no camera frames): each control step it reads the base state straight off
``simulator.robot_root_states`` — the sim analog of the robot's onboard sport/odom estimate — and
publishes one ``nav_msgs/Odometry``. It rides the same in-process rclpy transport the image egress
uses (so no CycloneDDS entanglement with the Unitree SDK bridge). A stopgap until a first-class
base-state egress exists.

Like the image egress, the heavy imports (rclpy, nav_msgs) are deferred into ``start()`` so importing
this module never hard-requires a ROS environment. Base velocities in ``robot_root_states`` are
WORLD-frame on every backend (the unified contract); ``nav_msgs/Odometry`` expresses its twist in the
``child_frame_id`` (body) frame, so they are rotated world→body via the same ``quat_rotate_inverse``
helper the SDK bridge uses for the IMU gyro. Timestamps come from sim_time, not wall-clock.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from holosoma.sensor_egress.base import SensorEgress
from holosoma.utils.rotations import quat_rotate_inverse

if TYPE_CHECKING:
    from holosoma.config_types.sensor_egress import ROS2OdometryEgressConfig
    from holosoma.sensor_egress.base import FramePacket, StreamKey
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator


def _sim_time_to_stamp(sim_time: float):
    """Build a builtin_interfaces/Time from sim seconds (deferred import; only after start())."""
    from builtin_interfaces.msg import Time

    sec = int(sim_time)
    nanosec = round((sim_time - sec) * 1e9)
    if nanosec >= 1_000_000_000:  # rounding carry
        sec += 1
        nanosec -= 1_000_000_000
    return Time(sec=sec, nanosec=nanosec)


class ROS2OdometryEgress(SensorEgress):
    """One ROS2 node publishing the robot base pose/velocity as nav_msgs/Odometry."""

    config: ROS2OdometryEgressConfig

    def __init__(self, config: ROS2OdometryEgressConfig, simulator: BaseSimulator) -> None:
        super().__init__(config, simulator)
        # rclpy/nav_msgs objects are typed Any: their stubs are absent in non-ROS envs (e.g. the
        # mujoco venv), and these are only populated in start() under a real ROS environment.
        self._node: Any = None
        self._executor: Any = None
        self._spin_thread: Any = None
        self._publisher: Any = None
        self._Odometry: Any = None

    @property
    def self_sourced(self) -> bool:
        # Reads base state off the simulator every step; the driver ticks it regardless of cameras.
        return True

    def wanted_streams(self) -> set[StreamKey]:
        # No camera snapshot needed — base state comes straight off robot_root_states.
        return set()

    # ----- lifecycle -----

    def start(self) -> None:
        import rclpy
        from nav_msgs.msg import Odometry
        from rclpy.node import Node
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

        if not rclpy.ok():
            rclpy.init()
        self._node = Node(self.config.node_name)
        self._Odometry = Odometry

        reliability = ReliabilityPolicy.RELIABLE if self.config.qos == "reliable" else ReliabilityPolicy.BEST_EFFORT
        qos = QoSProfile(reliability=reliability, history=HistoryPolicy.KEEP_LAST, depth=1)
        self._publisher = self._node.create_publisher(self._Odometry, self.config.topic, qos)

        # Spin in a daemon thread so subscription/QoS handshakes progress without a sim-side spin.
        from rclpy.executors import SingleThreadedExecutor

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        import threading

        self._spin_thread = threading.Thread(
            target=self._executor.spin, name=f"egress-spin:{self.config.node_name}", daemon=True
        )
        self._spin_thread.start()
        logger.info(f"ROS2 odometry egress '{self.config.node_name}' up: publishing {self.config.topic}")

    # ----- per-step publish -----

    def publish(self, frames: dict[StreamKey, FramePacket]) -> None:
        # Self-sourced: ``frames`` is always empty. Read the base state off the sim and publish.
        if self._publisher is None:
            return
        pos, quat_xyzw, lin_vel_body, ang_vel_body, sim_time = self._read_base_state()
        msg = self._Odometry()
        msg.header.stamp = _sim_time_to_stamp(sim_time)
        msg.header.frame_id = self.config.frame_id
        msg.child_frame_id = self.config.child_frame_id

        msg.pose.pose.position.x = pos[0]
        msg.pose.pose.position.y = pos[1]
        msg.pose.pose.position.z = pos[2]
        # robot_root_states quaternion is xyzw; ROS geometry_msgs/Quaternion is also xyzw — direct copy.
        msg.pose.pose.orientation.x = quat_xyzw[0]
        msg.pose.pose.orientation.y = quat_xyzw[1]
        msg.pose.pose.orientation.z = quat_xyzw[2]
        msg.pose.pose.orientation.w = quat_xyzw[3]

        msg.twist.twist.linear.x = lin_vel_body[0]
        msg.twist.twist.linear.y = lin_vel_body[1]
        msg.twist.twist.linear.z = lin_vel_body[2]
        msg.twist.twist.angular.x = ang_vel_body[0]
        msg.twist.twist.angular.y = ang_vel_body[1]
        msg.twist.twist.angular.z = ang_vel_body[2]

        self._publisher.publish(msg)

    def _read_base_state(self) -> tuple[list[float], list[float], list[float], list[float], float]:
        """Read env ``config.env_id`` base state off the sim as plain floats.

        Returns ``(position, quat_xyzw, lin_vel_body, ang_vel_body, sim_time)``. The unified
        ``robot_root_states`` 13-vector is ``[pos(3), quat_xyzw(4), lin_vel_world(3), ang_vel_world(3)]``;
        the world-frame velocities are rotated into the base (body) frame for the Odometry twist.
        """
        env = self.config.env_id
        root = self.simulator.robot_root_states[env]  # [13]
        quat_xyzw = root[3:7]  # xyzw
        lin_vel_world = root[7:10].unsqueeze(0)
        ang_vel_world = root[10:13].unsqueeze(0)
        lin_vel_body = quat_rotate_inverse(quat_xyzw.unsqueeze(0), lin_vel_world, w_last=True).squeeze(0)
        ang_vel_body = quat_rotate_inverse(quat_xyzw.unsqueeze(0), ang_vel_world, w_last=True).squeeze(0)

        pos = root[0:3].detach().cpu().tolist()
        quat = quat_xyzw.detach().cpu().tolist()
        lin = lin_vel_body.detach().cpu().tolist()
        ang = ang_vel_body.detach().cpu().tolist()
        return pos, quat, lin, ang, self.simulator.time()

    # ----- teardown -----

    def stop(self) -> None:
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None
        if self._spin_thread is not None and self._spin_thread.is_alive():
            self._spin_thread.join(timeout=2.0)
        self._spin_thread = None
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        self._publisher = None
