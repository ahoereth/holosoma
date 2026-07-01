"""Sensor egress: publish rendered sim sensor frames to external transports (ROS2).

ROS-free package root. The runtime impl (``ros2/``) is imported only via a config's
``egress_cls`` property, so importing this package pulls in no transport dependency.
"""

from holosoma.sensor_egress.base import CameraIntrinsics, FramePacket, SensorEgress
from holosoma.sensor_egress.driver import SensorEgressDriver

__all__ = ["CameraIntrinsics", "FramePacket", "SensorEgress", "SensorEgressDriver"]
