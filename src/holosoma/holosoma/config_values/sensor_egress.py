"""Named sensor-egress presets, selectable via the ``sensor_egress:`` subcommand.

Each preset is a :class:`SensorEgressConfig` defining a set of egress sinks. A preset's instance
fields can be overridden at the CLI, keyed by the instance's dict label, e.g.::

    sensor_egress:ros2-image --sensor_egress.instances.head.node_name cams

Core presets are seeded directly into ``DEFAULTS`` below. External packages add presets via the
``holosoma.config.sensor_egress`` entry-point group, folded in at import time with ``setdefault``
(core wins on name collision).
"""

from __future__ import annotations

import sys
from importlib.metadata import entry_points

from loguru import logger

from holosoma.config_types.sensor_egress import (
    ROS2ImageEgressConfig,
    ROS2ImageRoute,
    SensorEgressConfig,
    VizEgressConfig,
)

# No egress (default).
none = SensorEgressConfig(instances={})

# One ROS2 node publishing the G1 head camera as JPEG plus latched CameraInfo. Pair with a sensors
# preset that defines `head_cam` (e.g. `sensors:g1-cameras`).
ros2_image = SensorEgressConfig(
    instances={
        "head": ROS2ImageEgressConfig(
            node_name="sim_cameras_head",
            routes={
                "head": ROS2ImageRoute(
                    camera="head_cam", topic="/sim_cameras/head/image/compressed", modality="rgb", format="jpeg"
                )
            },
        )
    }
)

# Stereo head pair as CompressedImage on the exact topics the rfmpi teleop stack subscribes to:
# /ros_camera/rgb/{left,right}/compressed. Pair with `sensors:g1-stereo` (or
# `sensors:g1-stereo-wrists`). CameraInfo off for this teleop stack.
ros2_stereo = SensorEgressConfig(
    instances={
        "head_stereo": ROS2ImageEgressConfig(
            node_name="sim_cameras_head_stereo",
            publish_camera_info=False,
            routes={
                "left": ROS2ImageRoute(
                    camera="head_cam_left", topic="/ros_camera/rgb/left/compressed", modality="rgb", format="jpeg"
                ),
                "right": ROS2ImageRoute(
                    camera="head_cam_right", topic="/ros_camera/rgb/right/compressed", modality="rgb", format="jpeg"
                ),
            },
        )
    }
)

# One ROS2 node publishing the G1 waist forward/back depth cameras as raw float32-meter Image
# (sensor_msgs/Image, 32FC1) plus latched CameraInfo. Pair with `sensors:g1-waist` (the
# `waist_cameras` preset), which defines `waist_front_cam` / `waist_back_cam` as depth cameras.
ros2_waist_depth = SensorEgressConfig(
    instances={
        "waist": ROS2ImageEgressConfig(
            node_name="sim_cameras_waist",
            routes={
                "front": ROS2ImageRoute(
                    camera="waist_front_cam",
                    topic="/sim_cameras/waist_front/depth",
                    modality="depth",
                    format="32FC1",
                ),
                "back": ROS2ImageRoute(
                    camera="waist_back_cam",
                    topic="/sim_cameras/waist_back/depth",
                    modality="depth",
                    format="32FC1",
                ),
            },
        )
    }
)

# Visualize all configured cameras (every modality): record an mp4 at teardown. Pair with any
# sensors preset; cameras=None watches them all. Add --sensor_egress.instances.viz.live_window True
# for a live cv2 window instead of (or in addition to) the file.
viz = SensorEgressConfig(instances={"viz": VizEgressConfig(live_window=True)})

# A mixed preset: stream the stereo head over ROS2 and record everything to disk in one run.
ros2_stereo_and_viz = SensorEgressConfig(instances={**ros2_stereo.instances, "viz": VizEgressConfig(live_window=True)})

# Core presets, seeded directly (not via entry points).
DEFAULTS: dict[str, SensorEgressConfig] = {
    "none": none,
    "ros2-image": ros2_image,
    "ros2-stereo": ros2_stereo,
    "ros2-waist-depth": ros2_waist_depth,
    "viz": viz,
    "ros2-stereo+viz": ros2_stereo_and_viz,
}


def _register_extension_presets() -> None:
    """Fold ``holosoma.config.sensor_egress`` presets into ``DEFAULTS`` (idempotent, core wins).

    For out-of-tree packages only. An entry-point module is imported at CLI-build time, so it
    should avoid heavy top-level imports.
    """
    if sys.version_info >= (3, 10):
        eps = entry_points(group="holosoma.config.sensor_egress")
    else:
        eps = entry_points().get("holosoma.config.sensor_egress", [])
    for ep in eps:
        if ep.name in DEFAULTS:
            logger.warning(f"Sensor-egress preset '{ep.name}' from {ep.value} ignored: name already in core DEFAULTS.")
            continue
        DEFAULTS.setdefault(ep.name, ep.load())


# Populate external presets at import so the ``sensor_egress:`` subcommand sees every choice
# before RunSimConfig/ExperimentConfig build their Annotated fields.
_register_extension_presets()
