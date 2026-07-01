"""Unit tests for sensor-egress config validators (pure, no simulator, no ROS).

Construction-time validation checks on the frozen pydantic dataclasses in
``config_types/sensor_egress.py``. No rclpy is imported; ``egress_cls`` is never touched.
Runtime behavior is covered by the driver/ROS tests in ``sensor_egress/tests/``.
"""

from __future__ import annotations

import pytest

from holosoma.config_types.sensor_egress import ROS2ImageEgressConfig, ROS2ImageRoute

pytestmark = pytest.mark.no_sim


def test_inline_mode_is_configurable():
    assert ROS2ImageEgressConfig(async_publish=False, routes={}).async_publish is False


def test_jpeg_quality_validated():
    with pytest.raises(ValueError, match="jpeg_quality"):
        ROS2ImageEgressConfig(jpeg_quality=0, routes={})
    with pytest.raises(ValueError, match="jpeg_quality"):
        ROS2ImageEgressConfig(jpeg_quality=101, routes={})


def test_queue_maxlen_validated():
    with pytest.raises(ValueError, match="queue_maxlen"):
        ROS2ImageEgressConfig(queue_maxlen=0, routes={})


def test_qos_validated():
    with pytest.raises(ValueError, match="qos"):
        ROS2ImageEgressConfig(qos="bogus", routes={})


def test_duplicate_topics_rejected():
    with pytest.raises(ValueError, match="duplicate topics"):
        ROS2ImageEgressConfig(
            routes={
                "a": ROS2ImageRoute(camera="a", topic="/same", modality="rgb", format="jpeg"),
                "b": ROS2ImageRoute(camera="b", topic="/same", modality="rgb", format="jpeg"),
            }
        )


def test_route_format_modality_mismatch_rejected():
    with pytest.raises(ValueError, match="needs a depth format"):
        ROS2ImageRoute(camera="a", topic="/t", modality="depth", format="jpeg")
    with pytest.raises(ValueError, match="needs an rgb format"):
        ROS2ImageRoute(camera="a", topic="/t", modality="rgb", format="32FC1")


def test_route_requires_camera_and_topic():
    with pytest.raises(ValueError, match="non-empty camera"):
        ROS2ImageRoute(camera="", topic="/t", modality="rgb", format="jpeg")
    with pytest.raises(ValueError, match="non-empty topic"):
        ROS2ImageRoute(camera="a", topic="", modality="rgb", format="jpeg")
