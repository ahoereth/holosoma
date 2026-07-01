"""Backend-agnostic, ROS-free substrate for sensor egress.

Defines the :class:`SensorEgress` ABC (the outbound, sensor-side sibling of ``BasicSdk2Bridge``),
the :class:`FramePacket` the driver hands it, the :class:`CameraIntrinsics` it carries, and the
:class:`EgressContext` of sim-level facts every egress is built with. Concrete egress live in their
own modules and import their heavy deps (rclpy, …) at module top — they are imported only when an
:class:`~holosoma.config_types.sensor_egress.EgressInstanceConfig.egress_cls` fires, so this base
and the config layer stay importable without those deps.

Stream identity is a :data:`StreamKey` = ``(camera, modality, env_id)``: an egress advertises the
exact streams it needs via :meth:`SensorEgress.wanted_streams`, and the driver snapshots only the
union of those (one GPU→host copy per (camera, modality), serving all wanted envs in one read) and
hands each egress a per-step batch of just its streams. ROS2 wants env 0 only; the recorder may want
several envs to tile — env selection is thus a per-egress lever, not a global bake-in. Egress are
egress-only and run AFTER ``render_sensors`` — never in the physics-coupled bridge slot.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:
    from holosoma.config_types.sensor_egress import EgressInstanceConfig
    from holosoma.config_types.sensors import SensorsConfig
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator

# (camera_name, modality, env_id). The unit of stream identity across the egress system.
# ``typing.Tuple`` (not the builtin ``tuple[...]``) so this runtime alias is valid on the
# mypy target (python_version = 3.8), where builtin generics aren't subscriptable at runtime.
StreamKey = Tuple[str, str, int]


@dataclass(frozen=True)
class CameraIntrinsics:
    """Static intrinsics of one camera, carried on every :class:`FramePacket`.

    Backend-agnostic core fields (the same set on every backend), sufficient to derive a
    pinhole projection matrix ``K`` for a ``CameraInfo`` message.
    """

    width: int
    height: int
    vertical_fov: float
    """Vertical field of view, degrees."""
    near: float
    far: float


@dataclass
class FramePacket:
    """One rendered frame, already copied to host, ready for an egress to consume.

    Produced by :class:`~holosoma.sensor_egress.driver.SensorEgressDriver` (which owns the single
    device→host copy) and handed to every egress whose ``wanted_streams`` includes this
    ``(camera, modality, env_id)``. Holds no GPU tensor, so it is safe to pass to a worker thread.
    """

    camera: str
    modality: str
    """``"rgb"`` or ``"depth"``."""
    env_id: int
    array: np.ndarray
    """Host-side copy: ``uint8 [H, W, 3]`` R,G,B for rgb; ``float32 [H, W, 1]`` meters for depth."""
    sim_time: float
    """Simulation time of this frame (``simulator.time()``), for the message timestamp."""
    intrinsics: CameraIntrinsics

    @property
    def key(self) -> StreamKey:
        return (self.camera, self.modality, self.env_id)


class SensorEgress(ABC):
    """Abstract base for one outbound sensor sink (the egress-side sibling of ``BasicSdk2Bridge``).

    Lifecycle: the driver constructs every enabled egress with the simulator (mirroring how
    ``BasicSdk2Bridge`` takes the simulator), calls :meth:`start` once at setup, :meth:`publish`
    once per control step with that step's fresh frames, and :meth:`stop` once at teardown.
    Implementations own their transport (node, sockets, worker threads, cv2 window/encoder); the
    driver owns the snapshot and the fan-out. Egress read what they need off the simulator
    (``sensor_config``, ``num_envs``, ``headless``, ``video_config``, ``simulator_config.sim`` …).
    """

    def __init__(self, config: EgressInstanceConfig, simulator: BaseSimulator) -> None:
        self.config = config
        self.simulator = simulator

    @property
    def sensors_config(self) -> SensorsConfig:
        return self.simulator.sensor_config

    @property
    def control_hz(self) -> float:
        """Control-step rate (sim fps / control_decimation), the base for resolving frequency strings."""
        sim = self.simulator.simulator_config.sim
        return sim.fps / sim.control_decimation_steps

    @abstractmethod
    def wanted_streams(self) -> set[StreamKey]:
        """The ``(camera, modality, env_id)`` triples this egress needs.

        The driver unions this across all egress to decide which buffers to snapshot, and hands each
        egress only its own streams. The sole generic hook — no transport vocabulary (topics/formats)
        leaks to the driver. ROS2 typically returns ``{(cam, mod, 0)}``; a tiling recorder returns
        one triple per (camera, modality, env) it watches.
        """

    @abstractmethod
    def start(self) -> None:
        """Open the transport (node/sockets/threads/window) and publish any static/latched data."""

    @abstractmethod
    def publish(self, frames: dict[StreamKey, FramePacket]) -> None:
        """Handle one control step's fresh frames for THIS egress (only its wanted streams present).

        ``frames`` contains an entry per wanted stream that rendered this step (slower cameras are
        absent on steps they did not render). A single-stream sink (ROS2) iterates and sends each; a
        correlated sink (recorder) composes across the batch. Called only when ``frames`` is
        non-empty.
        """

    @abstractmethod
    def stop(self) -> None:
        """Drain/stop workers, finalize output (encode video), and close the transport/window."""
