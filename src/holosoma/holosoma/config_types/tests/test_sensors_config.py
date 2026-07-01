"""Unit tests for SensorsConfig / camera-config validators (pure, no simulator)."""

from __future__ import annotations

import pytest

from holosoma.config_types.sensors import (
    CameraSensorConfig,
    MujocoCameraConfig,
    SensorMountConfig,
    SensorsConfig,
)

pytestmark = pytest.mark.no_sim


def _cam(*, target_kind="robot_link", target="pelvis", mujoco=None):
    return CameraSensorConfig(
        mount=SensorMountConfig(target_kind=target_kind, target=target),
        data_types=["rgb"],
        mujoco=mujoco,
    )


def test_actor_mount_named_robot_rejected():
    # The robot is addressed via target_kind="robot_link", never as an actor named "robot".
    with pytest.raises(ValueError, match="use target_kind='robot_link'"):
        SensorsConfig(cameras={"c": _cam(target_kind="actor", target="robot")})


def test_actor_mount_other_name_allowed():
    SensorsConfig(cameras={"c": _cam(target_kind="actor", target="panel")})  # must not raise


def test_conflicting_warp_render_flag_rejected():
    # use_shadows is global to the shared Warp render context; cameras setting it differently are
    # rejected.
    a = _cam(mujoco=MujocoCameraConfig(use_shadows=True))
    b = _cam(mujoco=MujocoCameraConfig(use_shadows=False))
    with pytest.raises(ValueError, match="render flag 'use_shadows'"):
        SensorsConfig(cameras={"a": a, "b": b})


def test_agreeing_warp_render_flag_allowed():
    a = _cam(mujoco=MujocoCameraConfig(use_shadows=True))
    b = _cam(mujoco=MujocoCameraConfig(use_shadows=True))
    SensorsConfig(cameras={"a": a, "b": b})  # agreement is fine


def test_none_warp_render_flag_imposes_no_constraint():
    # One camera sets the flag, the other leaves it None: no conflict.
    a = _cam(mujoco=MujocoCameraConfig(use_textures=False))
    b = _cam()  # mujoco=None
    SensorsConfig(cameras={"a": a, "b": b})  # must not raise


# The camera-frame sink fields (cameras/modalities/record_video/...) live on VizEgressConfig;
# their validation is covered in config_types/tests/test_sensor_egress_config.py.
