"""Tests for the core sensor-egress presets and the real ROS2ImageEgress wiring (no ROS env).

The ROS2ImageEgress impl defers rclpy to start(), so we can construct it and exercise the
driver↔egress wiring (egress_cls resolution, wanted_streams, route→sensor validation, async/inline
config) entirely without a ROS environment — only start()/publish() would need rclpy. This pins
the Phase-2/3 contracts and the Phase-4 preset/topic guarantees (incl. the rfmpi teleop topics the
removed holosoma_sim_stereo sidecar used to publish).
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from holosoma.config_types.sensor_egress import ROS2ImageEgressConfig, ROS2ImageRoute, ROS2OdometryEgressConfig
from holosoma.config_types.sensors import CameraSensorConfig, SensorMountConfig, SensorsConfig
from holosoma.config_values import sensor_egress as egress_values
from holosoma.config_values import sensors as sensor_values
from holosoma.sensor_egress.base import CameraIntrinsics, FramePacket
from holosoma.utils.safe_torch_import import torch

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


def test_waist_depth_color_preset_colorizes_depth_over_rgb_format():
    # The colorized-depth preset publishes DEPTH cameras on an RGB (jpeg) format => colorized to RGB.
    # Cameras must exist in the paired g1-waist sensors preset (else the driver fails loud).
    inst = next(iter(egress_values.DEFAULTS["ros2-waist-depth-color"].instances.values()))
    assert isinstance(inst, ROS2ImageEgressConfig)
    for route in inst.routes.values():
        assert route.modality == "depth" and route.format == "jpeg"  # depth + rgb format = colorize
        assert route.depth_colormap == "turbo"
    waist_cams = set(sensor_values.DEFAULTS["g1-waist"].cameras)
    assert {r.camera for r in inst.routes.values()} <= waist_cams


def test_waist_depth_raw_and_color_preset_shares_one_snapshot_per_camera():
    # The combined preset publishes each waist camera BOTH ways (raw 32FC1 + colorized jpeg) from a
    # single node. The two routes per camera share the same (camera, "depth", env) stream key, so
    # wanted_streams collapses to one triple per camera => the driver does ONE GPU->host copy each,
    # not two. That single-copy guarantee is the reason to prefer one node over two.
    inst = next(iter(egress_values.DEFAULTS["ros2-waist-depth-raw+color"].instances.values()))
    assert isinstance(inst, ROS2ImageEgressConfig)
    # Four distinct topics (2 cameras x 2 formats), all unique (the config validator also enforces).
    topics = {r.topic for r in inst.routes.values()}
    assert len(topics) == len(inst.routes) == 4
    # Both a raw depth-format route and a colorized rgb-format route are present.
    formats = {r.format for r in inst.routes.values()}
    assert "32FC1" in formats and "jpeg" in formats
    # The colorized routes carry the color knobs; the raw ones stay raw metric depth.
    assert any(r.format == "jpeg" and r.depth_colormap == "turbo" for r in inst.routes.values())
    # Cameras exist in the paired sensors preset (else the driver fails loud at startup).
    waist_cams = set(sensor_values.DEFAULTS["g1-waist"].cameras)
    assert {r.camera for r in inst.routes.values()} <= waist_cams
    # The single-copy guarantee: 4 routes but only 2 wanted streams (one per camera).
    egress = inst.egress_cls(inst, _FakeSimulator(sensor_values.DEFAULTS["g1-waist"]))
    assert egress.wanted_streams() == {("waist_front_cam", "depth", 0), ("waist_back_cam", "depth", 0)}
    assert "rclpy" not in sys.modules


def test_real_egress_wanted_streams_and_async_default():
    # Construct the REAL ROS2ImageEgress (no start -> no ROS) and check the wiring the driver relies
    # on: wanted_streams from routes, and async_publish on by default.
    inst = next(iter(egress_values.DEFAULTS["ros2-stereo"].instances.values()))
    egress = inst.egress_cls(inst, _FakeSimulator(sensor_values.DEFAULTS["g1-stereo"]))
    assert egress.wanted_streams() == {("head_cam_left", "rgb", 0), ("head_cam_right", "rgb", 0)}
    assert inst.async_publish is True
    assert "rclpy" not in sys.modules


def _depth_frame(camera, value=2.0, h=6, w=6):
    arr = np.full((h, w, 1), value, np.float32)  # [H,W,1] float meters, as get_camera_data gives
    intr = CameraIntrinsics(width=w, height=h, vertical_fov=45.0, near=0.01, far=100.0)
    return FramePacket(camera=camera, modality="depth", env_id=0, array=arr, sim_time=0.0, intrinsics=intr)


def test_odom_preset_is_self_sourced_and_camera_free():
    # ros2-odom is a camera-free, self-sourced egress: it needs no sensors preset and wants no
    # camera stream, so it can run in an odom-only pipeline.
    inst = next(iter(egress_values.DEFAULTS["ros2-odom"].instances.values()))
    assert isinstance(inst, ROS2OdometryEgressConfig)
    assert inst.egress_cls.__name__ == "ROS2OdometryEgress"
    egress = inst.egress_cls(inst, _FakeSimulator(SensorsConfig(cameras={})))  # no start() -> no rclpy
    assert egress.self_sourced is True
    assert egress.wanted_streams() == set()
    assert inst.topic == "/odom"
    assert "rclpy" not in sys.modules


def test_stereo_and_odom_preset_mixes_image_and_odometry_egress():
    insts = egress_values.DEFAULTS["ros2-stereo+odom"].instances
    kinds = sorted(type(i).__name__ for i in insts.values())
    assert kinds == ["ROS2ImageEgressConfig", "ROS2OdometryEgressConfig"]


class _OdomFakeSim:
    """Minimal sim exposing what ROS2OdometryEgress reads: robot_root_states + time()."""

    def __init__(self, root_states):
        self.robot_root_states = root_states
        self._t = 0.0

    def time(self):
        return self._t


def test_odom_reads_base_state_and_rotates_velocity_to_body_frame():
    # Identity orientation: body-frame == world-frame, so the twist equals the raw world velocities.
    # pos=[1,2,3], quat=xyzw identity, lin_vel_world=[0.5,0,0], ang_vel_world=[0,0,0.2].
    root = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.2]])
    inst = ROS2OdometryEgressConfig()
    egress = inst.egress_cls(inst, _OdomFakeSim(root))
    pos, quat, lin, ang, t = egress._read_base_state()
    assert pos == [1.0, 2.0, 3.0]
    assert quat == [0.0, 0.0, 0.0, 1.0]  # xyzw, copied straight through
    assert lin == pytest.approx([0.5, 0.0, 0.0])
    assert ang == pytest.approx([0.0, 0.0, 0.2])
    assert t == 0.0


def test_odom_velocity_rotation_uses_body_frame_under_yaw():
    # 90deg yaw: a world +x linear velocity is a body +(-y)... verify against quat_rotate_inverse
    # directly so the test pins "twist is in the child/body frame", not a hand-computed constant.
    import math

    from holosoma.utils.rotations import quat_rotate_inverse

    half = math.pi / 4  # 90deg yaw -> quat (x,y,z,w) = (0,0,sin45,cos45)
    qz, qw = math.sin(half), math.cos(half)
    root = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, qz, qw, 1.0, 0.0, 0.0, 0.0, 0.0, 0.3]])
    inst = ROS2OdometryEgressConfig()
    egress = inst.egress_cls(inst, _OdomFakeSim(root))
    _, _, lin, ang, _ = egress._read_base_state()
    quat = root[0, 3:7].unsqueeze(0)
    exp_lin = quat_rotate_inverse(quat, root[0, 7:10].unsqueeze(0), w_last=True).squeeze(0).tolist()
    exp_ang = quat_rotate_inverse(quat, root[0, 10:13].unsqueeze(0), w_last=True).squeeze(0).tolist()
    assert lin == pytest.approx(exp_lin, abs=1e-6)
    assert ang == pytest.approx(exp_ang, abs=1e-6)


def test_encode_threads_route_colormap_and_range_without_ros():
    # ROS2ImageEgress._encode is ROS-free (never touches the node/simulator), so we can verify the
    # route's depth_colormap/depth_range actually reach encode_frame WITHOUT an rclpy env. Each knob
    # is isolated (routes differing in ONLY that field) so the test guards BOTH independently: if
    # _encode dropped one knob to the default, that pair would collide even though the other differs.
    mount = SensorMountConfig(target_kind="robot_link", target="torso_link")
    sensors = SensorsConfig(cameras={"waist": CameraSensorConfig(mount=mount, data_types=["depth"])})

    def _route(topic, colormap, drange):
        return ROS2ImageRoute(
            camera="waist", topic=topic, modality="depth", format="rgb8", depth_colormap=colormap, depth_range=drange
        )

    # Same range, different colormap -> isolates route.depth_colormap threading.
    cmap_a = _route("/t/turbo", "turbo", [0.1, 5.0])
    cmap_b = _route("/t/gray", "gray", [0.1, 5.0])
    # Same colormap, different range -> isolates route.depth_range threading.
    range_a = _route("/t/tight", "gray", [0.1, 3.0])
    range_b = _route("/t/wide", "gray", [0.1, 20.0])
    cfg = ROS2ImageEgressConfig(
        publish_camera_info=False,
        routes={"ca": cmap_a, "cb": cmap_b, "ra": range_a, "rb": range_b},
    )
    egress = cfg.egress_cls(cfg, _FakeSimulator(sensors))  # no start() -> no rclpy
    frame = _depth_frame("waist")
    enc = {k: egress._encode(r, frame).data for k, r in cfg.routes.items()}
    assert enc["ca"] != enc["cb"]  # colormap reaches encode_frame (would collide if defaulted)
    assert enc["ra"] != enc["rb"]  # depth_range reaches encode_frame (would collide if defaulted)
    assert "rclpy" not in sys.modules
