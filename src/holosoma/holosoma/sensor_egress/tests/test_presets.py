"""Tests for the core sensor-egress presets and the real ROS2ImageEgress wiring (no ROS env).

The ROS2ImageEgress impl defers rclpy to start(), so we can construct it and exercise the
driver↔egress wiring (egress_cls resolution, wanted_streams, route→sensor validation, async/inline
config) entirely without a ROS environment — only start()/publish() would need rclpy. This pins
the Phase-2/3 contracts and the Phase-4 preset/topic guarantees (incl. the rfmpi teleop topics the
removed holosoma_sim_stereo sidecar used to publish).
"""

from __future__ import annotations

import sys

import pytest

from holosoma.config_types.sensor_egress import ROS2ImageEgressConfig
from holosoma.config_values import sensor_egress as egress_values
from holosoma.config_values import sensors as sensor_values

pytestmark = pytest.mark.no_sim


class _FakeSimEngineCfg:
    fps = 200.0
    control_decimation_steps = 4


class _FakeSimulatorConfig:
    sim = _FakeSimEngineCfg()


class _FakeVideoConfig:
    save_dir = None


class _FakeSimulator:
    """Minimal stand-in exposing what an egress reads off the simulator (no backend)."""

    def __init__(self, sensors_config, num_envs=1):
        self.sensor_config = sensors_config
        self.num_envs = num_envs
        self.headless = True
        self.simulator_config = _FakeSimulatorConfig()
        self.video_config = _FakeVideoConfig()
        self.sensor_manager = None


def test_core_presets_present():
    assert set(egress_values.DEFAULTS) >= {"none", "ros2-image", "ros2-stereo"}
    assert egress_values.DEFAULTS["none"].instances == {}


def test_presets_resolve_egress_cls_without_rclpy():
    # Resolving egress_cls imports the ros2_image_egress MODULE but must NOT import rclpy (deferred
    # to start()), so presets are inspectable in a non-ROS env.
    for name in ("ros2-image", "ros2-stereo"):
        for inst in egress_values.DEFAULTS[name].instances.values():
            cls = inst.egress_cls
            assert cls.__name__ == "ROS2ImageEgress"
    assert "rclpy" not in sys.modules


def test_stereo_preset_publishes_rfmpi_teleop_topics():
    # The rfmpi teleop stack subscribes to /ros_camera/rgb/{left,right}/compressed. Topics are
    # published VERBATIM (no auto-suffix), so the preset spells the full topics out.
    inst = next(iter(egress_values.DEFAULTS["ros2-stereo"].instances.values()))
    assert isinstance(inst, ROS2ImageEgressConfig)
    topics = {r.topic for r in inst.routes.values()}
    assert topics == {"/ros_camera/rgb/left/compressed", "/ros_camera/rgb/right/compressed"}
    cams = {r.camera for r in inst.routes.values()}
    assert cams == {"head_cam_left", "head_cam_right"}


def test_stereo_egress_cameras_exist_in_g1_stereo_sensors():
    # The egress route cameras must exist in a sensors preset you'd pair it with, else the driver
    # fails loud at startup. Pin that ros2-stereo lines up with sensors:g1-stereo.
    stereo_cams = set(sensor_values.DEFAULTS["g1-stereo"].cameras)
    stereo_inst = next(iter(egress_values.DEFAULTS["ros2-stereo"].instances.values()))
    route_cams = {r.camera for r in stereo_inst.routes.values()}
    assert route_cams <= stereo_cams


def test_g1_stereo_wrists_sensor_preset_migrated_to_core():
    # Migrated from the removed holosoma_sim_stereo extension; must now be a core sensors preset
    # with the stereo head pair plus both wrist cameras.
    cams = set(sensor_values.DEFAULTS["g1-stereo-wrists"].cameras)
    assert {"head_cam_left", "head_cam_right", "left_wrist_cam", "right_wrist_cam"} <= cams


def test_real_egress_wanted_streams_and_async_default():
    # Construct the REAL ROS2ImageEgress (no start -> no ROS) and check the wiring the driver relies
    # on: wanted_streams from routes, and async_publish on by default.
    inst = next(iter(egress_values.DEFAULTS["ros2-stereo"].instances.values()))
    egress = inst.egress_cls(inst, _FakeSimulator(sensor_values.DEFAULTS["g1-stereo"]))
    assert egress.wanted_streams() == {("head_cam_left", "rgb", 0), ("head_cam_right", "rgb", 0)}
    assert inst.async_publish is True
    assert "rclpy" not in sys.modules
