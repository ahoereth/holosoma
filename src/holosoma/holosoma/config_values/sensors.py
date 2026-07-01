"""Named sensor presets, selectable via the ``sensors:`` subcommand.

Each preset fixes which cameras are created. Camera fields can be overridden at the CLI
by the camera's dict key (e.g. ``sensors:g1-cameras --sensors.cameras.head_cam.width 224``), but
the set of cameras is fixed.

Extensions add presets by inserting into ``DEFAULTS`` on import (e.g. via a
``holosoma.config.sensors`` entry point).
"""

from holosoma.config_types.sensors import CameraSensorConfig, SensorMountConfig, SensorsConfig
from holosoma.config_values.wbt.g1.sensors import (
    head_and_wrist_cameras,
    stereo_head_and_wrist_cameras,
    stereo_head_cameras,
    waist_cameras,
)

# No sensors (default)
none = SensorsConfig()

# Free-floating overview camera: fixed at an elevated corner of each env, looking down at the scene
# center in an angled ISOMETRIC view. The ``world`` mount anchors to the env frame (not any body),
# so it never moves with the robot — useful for logging/overview. Placed at (2.5, -2.5, 2.5) m
# (front-right, up); the orientation is the look-at quaternion aiming the optical axis (-Z) at the
# origin with +Y up, giving a ~35.26deg downward elevation (the true isometric angle). Verified: the
# camera's -Z maps to the normalized eye->origin direction and +Y stays up. Robot/scene-agnostic,
# 640x480.
_ISO_LOOK_AT_ORIGIN_WXYZ = [0.820473, 0.424708, 0.17592, 0.339851]
overview_camera = SensorsConfig(
    cameras={
        "overview_cam": CameraSensorConfig(
            mount=SensorMountConfig(
                target_kind="world", position=[2.5, -2.5, 2.5], orientation=_ISO_LOOK_AT_ORIGIN_WXYZ
            ),
            width=640,
            height=480,
            vertical_fov=60.0,
            data_types=["rgb"],
        )
    }
)

DEFAULTS = {
    "none": none,
    # Egocentric head camera plus one grasp camera per wrist.
    "g1-cameras": head_and_wrist_cameras,
    # Generic G1 stereo head pair (head_cam_left / head_cam_right).
    "g1-stereo": stereo_head_cameras,
    # Stereo head pair plus the two wrist grasp cameras.
    "g1-stereo-wrists": stereo_head_and_wrist_cameras,
    # Waist-height forward/back depth cameras (waist_front_cam / waist_back_cam).
    "g1-waist": waist_cameras,
    # Free-floating angled isometric overview camera looking at the scene center (640x480).
    "overview": overview_camera,
}
