"""Sensor-egress configuration types.

Describes how rendered sensor frames leave the simulator. An egress publishes the sim's rendered
camera frames to an external transport (ROS2).

Imported at CLI-build time, so this module must stay ROS-free: avoid top-level transport imports.
Each config maps to its runtime impl via :attr:`EgressInstanceConfig.egress_cls`.

An egress reads the raw rendered buffer from ``BaseSimulator.get_camera_data`` (rgb ``uint8``
R,G,B; depth ``float32`` meters).
"""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, Literal

from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass

from holosoma.config_types.frequency import DecimationLike, validate_decimation_like

if TYPE_CHECKING:
    # Type-only import, keeps this module ROS-free at runtime.
    from holosoma.sensor_egress.base import SensorEgress

# Depth colormap names accepted by the viz egress (mapped to cv2.COLORMAP_* there).
_DEPTH_COLORMAPS = ("inferno", "turbo", "viridis", "magma", "jet", "gray")

# Reject unknown fields on every egress config.
_FORBID_EXTRA = ConfigDict(extra="forbid")

# Modalities an egress route can carry.
EgressModality = Literal["rgb", "depth"]

# Wire encodings a ROS2 image route may request:
#   - "rgb8"  : raw sensor_msgs/Image, R,G,B (no BGR swap; that is a cv2/JPEG artifact only).
#   - "jpeg"  : sensor_msgs/CompressedImage, lossy (teleop/viz, not depth or training datasets).
#   - "png"   : sensor_msgs/CompressedImage, lossless RGB.
#   - "32FC1" : raw sensor_msgs/Image depth, float32 meters (matches get_camera_data).
#   - "16UC1" : raw sensor_msgs/Image depth, uint16 millimeters.
#
# A ``depth`` route MAY pick an rgb format (rgb8/jpeg/png): the depth map is then COLORIZED to RGB
# (same colormap the viz egress uses) before encoding, for a human-viewable stream. A ``depth`` route
# with a depth format (32FC1/16UC1) publishes the raw metric depth. An ``rgb`` route may only use an
# rgb format (there is nothing to colorize).
ROS2ImageFormat = Literal["rgb8", "jpeg", "png", "32FC1", "16UC1"]
_RGB_FORMATS = ("rgb8", "jpeg", "png")
_DEPTH_FORMATS = ("32FC1", "16UC1")


@dataclass(frozen=True, config=_FORBID_EXTRA)
class EgressInstanceConfig:
    """Base class for one egress sink. Subclasses declare transport-specific fields.

    Each concrete config returns its :class:`~holosoma.sensor_egress.base.SensorEgress` subclass
    from :attr:`egress_cls`, imported lazily so the transport dependency loads only when selected.
    """

    enabled: bool = True
    """Whether this egress is constructed and stepped. Disabled instances are skipped entirely."""

    @property
    def egress_cls(self) -> type[SensorEgress]:
        """The runtime impl class for this config.

        Subclasses import the impl inside this property, not at module top level, to keep the
        config module importable without the transport dependency (e.g. ``rclpy``).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override egress_cls to return its SensorEgress impl class."
        )


@dataclass(frozen=True, config=_FORBID_EXTRA)
class ROS2ImageRoute:
    """One camera-stream to ROS2-topic mapping within a :class:`ROS2ImageEgressConfig`."""

    camera: str
    """Camera name; must match a ``CameraSensorConfig.name`` in the active ``SensorsConfig``."""

    topic: str
    """ROS2 topic to publish on, used verbatim (no auto-suffixing). For a CompressedImage
    (``jpeg``/``png``) the ROS convention is a ``/compressed`` suffix, e.g.
    ``/sim_cameras/head/image/compressed``; spell it out here if you want it."""

    modality: EgressModality = "rgb"
    """Which rendered modality to publish; must be in the camera's ``data_types``."""

    format: ROS2ImageFormat = "jpeg"
    """Wire encoding (see :data:`ROS2ImageFormat`). An ``rgb`` route needs an rgb format; a ``depth``
    route may pick a depth format (raw metric) OR an rgb format (colorized to RGB before encoding)."""

    depth_colormap: str = "inferno"
    """Colormap used when a ``depth`` route is colorized to RGB (rgb format): inferno (default),
    turbo, viridis, magma, jet, or gray. Ignored for raw-depth and rgb routes."""

    depth_range: list[float] | None = None
    """Fixed ``[min_m, max_m]`` depth range (meters) for stable colorization of a colorized ``depth``
    route; ``None`` means ``[0.01, 5.0]``. ``+inf`` (no hit) maps to the far end. Ignored otherwise."""

    @model_validator(mode="after")
    def validate_route(self) -> ROS2ImageRoute:
        if not self.camera:
            raise ValueError("ROS2ImageRoute.camera must be a non-empty camera name.")
        if not self.topic:
            raise ValueError(f"ROS2ImageRoute for camera '{self.camera}' needs a non-empty topic.")
        rgb_fmt = self.format in _RGB_FORMATS
        # rgb modality must use an rgb format. depth modality accepts either: a depth format (raw) or
        # an rgb format (colorized to RGB before encoding) — so only rgb+depth-format is rejected.
        if self.modality == "rgb" and not rgb_fmt:
            raise ValueError(
                f"ROS2ImageRoute camera '{self.camera}': modality 'rgb' needs an rgb format "
                f"{_RGB_FORMATS}, got '{self.format}'."
            )
        # depth is colorized only when the format is an rgb one; the colormap/range knobs are used
        # then. Validate the colormap and range regardless so a misconfig fails loud at construction.
        if self.depth_colormap not in _DEPTH_COLORMAPS:
            raise ValueError(
                f"ROS2ImageRoute camera '{self.camera}': depth_colormap '{self.depth_colormap}' "
                f"unknown; allowed: {sorted(_DEPTH_COLORMAPS)}."
            )
        if self.depth_range is not None and (len(self.depth_range) != 2 or self.depth_range[0] >= self.depth_range[1]):
            raise ValueError(
                f"ROS2ImageRoute camera '{self.camera}': depth_range must be [min_m, max_m] with "
                f"min<max, got {self.depth_range}."
            )
        return self


@dataclass(frozen=True, config=_FORBID_EXTRA)
class ROS2ImageEgressConfig(EgressInstanceConfig):
    """One ROS2 image-publishing sink: a single node fanning out to the cameras in ``routes``."""

    node_name: str = "sim_cameras"
    """ROS2 node name created for this sink."""

    qos: str = "best_effort"
    """QoS profile: ``best_effort`` (default, matches ZED/sensor drivers) or ``reliable``."""

    async_publish: bool = True
    """True (default): snapshot on the sim thread, encode and publish on per-route worker threads
    (drop-oldest under backpressure). False: encode and publish inline on the sim thread, lossless
    and every-frame (dataset capture); the sim waits."""

    queue_maxlen: int = 2
    """Per-route bounded queue depth when ``async_publish``. Drop-oldest beyond this (latest wins):
    1 keeps the freshest frame only, 2 gives one frame of jitter tolerance. Ignored when not async."""

    publish_camera_info: bool = True
    """Also publish a latched ``sensor_msgs/CameraInfo`` per camera (static K from intrinsics)."""

    jpeg_quality: int = 50
    """JPEG encode quality 1-100 for ``jpeg`` routes (ignored by other formats). Default 50."""

    env_id: int = 0
    """Which environment's view to publish. Default 0 (the single real-time robot); set higher to
    stream a specific env of a vectorized run. One env per node; all routes share it."""

    routes: dict[str, ROS2ImageRoute] = field(default_factory=dict)
    """Camera-to-topic routes this node publishes, keyed by an arbitrary label. The key is a
    CLI handle only (like a list index was) — it does not affect publishing; the route's
    ``camera``/``topic`` fields do."""

    @property
    def egress_cls(self) -> type[SensorEgress]:
        # Deferred import: keeps rclpy/cv2 out of CLI-build import.
        from holosoma.sensor_egress.ros2.ros2_image_egress import ROS2ImageEgress

        return ROS2ImageEgress

    @model_validator(mode="after")
    def validate_egress(self) -> ROS2ImageEgressConfig:
        if self.qos not in ("best_effort", "reliable"):
            raise ValueError(f"ROS2ImageEgressConfig.qos must be 'best_effort' or 'reliable', got '{self.qos}'.")
        if self.queue_maxlen < 1:
            raise ValueError(f"ROS2ImageEgressConfig.queue_maxlen must be >= 1, got {self.queue_maxlen}.")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError(f"ROS2ImageEgressConfig.jpeg_quality must be in [1, 100], got {self.jpeg_quality}.")
        if self.env_id < 0:
            raise ValueError(f"ROS2ImageEgressConfig.env_id must be >= 0, got {self.env_id}.")
        topics = [r.topic for r in self.routes.values()]
        dupes = {t for t in topics if topics.count(t) > 1}
        if dupes:
            raise ValueError(f"ROS2ImageEgressConfig node '{self.node_name}' has duplicate topics: {sorted(dupes)}.")
        return self


@dataclass(frozen=True, config=_FORBID_EXTRA)
class ROS2OdometryEgressConfig(EgressInstanceConfig):
    """One ROS2 node publishing the robot base pose/velocity as ``nav_msgs/Odometry``.

    Self-sourced: reads base pose/velocity straight off ``simulator.robot_root_states`` each control
    step (no camera frames), the sim analog of the robot's onboard sport/odom estimate. A stopgap
    that rides the sensor-egress rclpy transport; a first-class base-state egress can replace it
    later.
    """

    node_name: str = "sim_odometry"
    """ROS2 node name created for this sink."""

    topic: str = "/odom"
    """Topic to publish the ``nav_msgs/Odometry`` on."""

    frame_id: str = "odom"
    """``header.frame_id``: the fixed frame the pose is expressed in (odometry origin)."""

    child_frame_id: str = "base_link"
    """``child_frame_id``: the moving body frame the twist is expressed in."""

    qos: str = "best_effort"
    """QoS profile: ``best_effort`` (default) or ``reliable``."""

    env_id: int = 0
    """Which environment's base state to publish. Default 0 (the single real-time robot)."""

    @property
    def egress_cls(self) -> type[SensorEgress]:
        # Deferred import: keeps rclpy out of CLI-build import.
        from holosoma.sensor_egress.ros2.ros2_odometry_egress import ROS2OdometryEgress

        return ROS2OdometryEgress

    @model_validator(mode="after")
    def validate_odometry(self) -> ROS2OdometryEgressConfig:
        if self.qos not in ("best_effort", "reliable"):
            raise ValueError(f"ROS2OdometryEgressConfig.qos must be 'best_effort' or 'reliable', got '{self.qos}'.")
        if not self.topic:
            raise ValueError("ROS2OdometryEgressConfig.topic must be a non-empty topic.")
        if self.env_id < 0:
            raise ValueError(f"ROS2OdometryEgressConfig.env_id must be >= 0, got {self.env_id}.")
        return self


@dataclass(frozen=True, config=_FORBID_EXTRA)
class VizEgressConfig(EgressInstanceConfig):
    """Local visualization egress: tile mounted-camera views into a live cv2 window and/or an mp4.

    Tiles all watched (camera, modality, env) panels into one grid per step. Inline only:
    ``publish`` composes the grid and shows or buffers it on the calling thread.
    """

    live_window: bool = False
    """Show a live ``cv2`` window of the camera view(s). Needs a display; ignored headless."""

    record_video: bool = False
    """Buffer frames and encode an mp4 (H.264) at stop."""

    env_ids: list[int] = field(default_factory=lambda: [0])
    """Environments to visualize (default ``[0]``). Multiple env ids tile as a grid: one row per
    env, one column per (camera, modality) panel."""

    cameras: list[str] | None = None
    """Camera names to show; ``None`` (default) means all configured cameras."""

    modalities: list[EgressModality] | None = None
    """Modalities to show; ``None`` (default) means every modality each selected camera produces."""

    depth_range: list[float] | None = None
    """Fixed ``[min_m, max_m]`` depth range (meters) for stable colorization; ``None`` means
    ``[0.01, 5.0]``. ``+inf`` (no hit) maps to the far end."""

    depth_colormap: str = "inferno"
    """OpenCV colormap for depth: inferno (default), turbo, viridis, magma, jet, or gray."""

    update_decimation: DecimationLike = 1
    """Int visualizes every Nth rendered frame of the fastest watched camera; a frequency string
    ("10Hz") is a target against the control rate, converted to a frame-decimation by the recorder."""

    playback_rate: float = 1.0
    """Video playback speed factor; 1.0 plays back at true wall-clock speed."""

    save_dir: str | None = None
    """Output directory for the video; ``None`` derives it from the experiment/video dir."""

    @property
    def egress_cls(self) -> type[SensorEgress]:
        # Deferred import: keeps cv2/video utils out of CLI-build import.
        from holosoma.sensor_egress.viz.viz_egress import VizEgress

        return VizEgress

    @model_validator(mode="after")
    def validate_recorder(self) -> VizEgressConfig:
        validate_decimation_like(self.update_decimation, field="VizEgressConfig.update_decimation")
        if not self.env_ids:
            raise ValueError("VizEgressConfig.env_ids must be non-empty (default [0]).")
        if any(e < 0 for e in self.env_ids):
            raise ValueError(f"VizEgressConfig.env_ids must all be >= 0, got {self.env_ids}.")
        if self.depth_range is not None and (len(self.depth_range) != 2 or self.depth_range[0] >= self.depth_range[1]):
            raise ValueError(
                f"VizEgressConfig.depth_range must be [min_m, max_m] with min<max, got {self.depth_range}."
            )
        if self.depth_colormap not in _DEPTH_COLORMAPS:
            raise ValueError(
                f"VizEgressConfig.depth_colormap '{self.depth_colormap}' unknown; allowed: {sorted(_DEPTH_COLORMAPS)}."
            )
        return self


@dataclass(frozen=True, config=_FORBID_EXTRA)
class SensorEgressConfig:
    """Top-level egress composition: a (possibly heterogeneous) collection of sinks.

    ``instances`` may mix transports (e.g. a ROS2 image sink plus a video recorder). Each value
    keeps its concrete ``EgressInstanceConfig`` subclass through pydantic and tyro, so flat
    per-field CLI overrides work directly on a mixed collection, keyed by the dict label, e.g.
    ``sensor_egress:mixed --sensor_egress.instances.head.node_name cams``.
    """

    instances: dict[str, EgressInstanceConfig] = field(default_factory=dict)
    """Egress sinks to construct, keyed by an arbitrary label. The key is a CLI handle only
    (like a list index was) — it does not affect behavior. Empty (default, the ``none``
    preset) means no egress."""
