"""Default camera configurations for holosoma_inference.

These describe the camera each perception policy was *trained* with. The
resized dimensions in particular are part of the model's input contract — the
depth backbone's input tensor is exactly
``(resized_height, resized_width)``.
"""

from __future__ import annotations

from holosoma_inference.config.config_types.camera import CameraConfig, CameraPose, CameraProps
from holosoma_inference.utils.config_registry import ConfigRegistry

CAMERA_REGISTRY = ConfigRegistry(CameraConfig, group="holosoma.config.camera")

# Single forward-facing ZED 2i depth camera on the torso, pitched down to see
# the ground ahead — the setup used for the stair/rough-terrain policies.
single_zed2i_depth = CameraConfig(
    poses={
        "cam_front_depth": CameraPose(
            parent_link="robot/torso_link",
            camera_offset=(0.1, 0.0, 0.1),
            camera_rotation=(0.0, 75.0, 0.0),
        ),
    },
    props=CameraProps(
        image_type="depth",
        width=240,
        height=135,
        resized_width=87,
        resized_height=58,
        horizontal_fov=101.41,
        vertical_fov=69.00,
        near_clip=0.1,
        far_clip=2.0,
        frame_rate=10,
        image_show=False,
        depth_delay=0,
    ),
)

CAMERA_REGISTRY.add("single-zed2i-depth", single_zed2i_depth)
