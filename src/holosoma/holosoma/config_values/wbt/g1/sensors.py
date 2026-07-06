"""Camera sensor presets for the G1 robot.

Mounts on ``g1_29dof`` body names and frames (see ``data/robots/g1/g1_29dof.xml``):

- Head: the ``torso_link`` frame origin is at the waist-pitch joint; the head and ``mid360``
  Livox site sit at ``z ~= 0.41`` in that frame.
- Wrists: in the ``*_wrist_yaw_link`` frame the hand points along +X (``*_palm`` site at
  ``[0.08,0,0]``).

The mount orientations below reorient the camera's optical -Z to look along body +X.
"""

from __future__ import annotations

from holosoma.config_types.sensors import CameraSensorConfig, SensorMountConfig, SensorsConfig

head_camera = SensorsConfig(
    cameras={
        "head_cam": CameraSensorConfig(
            mount=SensorMountConfig(
                target_kind="robot_link",
                target="torso_link",
                position=[0.05, 0.0, 0.41],
                # forward = body +X, up = body +Z (level, forward-facing)
                orientation=[0.5, 0.5, -0.5, -0.5],
            ),
            width=128,
            height=128,
            data_types=["rgb"],
            update_decimation="50Hz",
        ),
    }
)

stereo_head_cameras = SensorsConfig(
    cameras={
        f"head_cam_{side}": CameraSensorConfig(
            mount=SensorMountConfig(
                target_kind="robot_link",
                target="torso_link",
                # Default eye separation (meters) along body +Y (human interpupillary distance) is 0.063
                position=[0.05, sign * 0.063 / 2, 0.41],
                # forward = body +X, up = body +Z (level, forward-facing)
                orientation=[0.5, 0.5, -0.5, -0.5],
            ),
            width=128,
            height=128,
            vertical_fov=45,
            data_types=["rgb"],
            update_decimation="50Hz",
        )
        for side, sign in [("left", +1), ("right", -1)]
    }
)

wrist_cameras = SensorsConfig(
    cameras={
        # One grasp camera per wrist: offset up off the back of the wrist (+Z, clear of the hand)
        # and pitched down to watch the palm/grasp region ahead.
        f"{side}_wrist_cam": CameraSensorConfig(
            mount=SensorMountConfig(
                target_kind="robot_link",
                target=f"{side}_wrist_yaw_link",
                position=[0.03, 0.0, 0.06],
                # forward = +X pitched 30deg toward -Z, up near +Z,
                orientation=[0.61237244, 0.35355339, -0.35355339, -0.61237244],
            ),
            width=128,
            height=128,
            near=0.02,
            data_types=["rgb"],
            update_decimation="50Hz",
        )
        for side in ["left", "right"]
    }
)

g1_cameras = SensorsConfig(
    cameras={
        # Egocentric head camera (torso-frame z~=0.41, by the mid360 mount), looking forward and
        # upright.
        "head_cam": CameraSensorConfig(
            mount=SensorMountConfig(
                target_kind="robot_link",
                target="torso_link",
                position=[0.05, 0.0, 0.41],
                # forward = body +X, up = body +Z (level, forward-facing)
                orientation=[0.5, 0.5, -0.5, -0.5],
            ),
            width=128,
            height=128,
            data_types=["rgb"],
            update_decimation="50Hz",
        ),
    },
)

# Waist-height depth cameras, one looking forward and one back, both on the torso_link frame whose
# origin sits at the waist-pitch joint (z~=0 => waist height). Depth-only, for near-body obstacle
# sensing / whole-body awareness fore and aft.
waist_cameras = SensorsConfig(
    cameras={
        # forward = body +X pitched 70deg down toward -Z, up in the +X/+Z plane; just ahead of the
        # torso, looking down at the ground ahead of the feet.
        "waist_front_cam": CameraSensorConfig(
            mount=SensorMountConfig(
                target_kind="robot_link",
                target="torso_link",
                position=[0.1, 0.0, 0.0],
                orientation=[0.69636424, 0.1227878, -0.1227878, -0.69636424],
            ),
            width=128,
            height=128,
            data_types=["depth"],
            update_decimation="50Hz",
        ),
        # forward = body -X, up = body +Z (level), just behind the torso (180deg yaw of the front).
        "waist_back_cam": CameraSensorConfig(
            mount=SensorMountConfig(
                target_kind="robot_link",
                target="torso_link",
                position=[-0.1, 0.0, 0.0],
                orientation=[0.5, 0.5, 0.5, 0.5],
            ),
            width=128,
            height=128,
            data_types=["depth"],
            update_decimation="50Hz",
        ),
    },
)

head_and_wrist_cameras = SensorsConfig(cameras={**head_camera.cameras, **wrist_cameras.cameras})

stereo_head_and_wrist_cameras = SensorsConfig(cameras={**stereo_head_cameras.cameras, **wrist_cameras.cameras})
