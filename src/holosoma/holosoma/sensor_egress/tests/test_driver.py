"""Unit tests for the sensor-egress driver and config layer (pure, no simulator, no ROS).

Covers the Phase-1 contract: the driver snapshots only freshly-rendered wanted streams, fans
out to the right egress, isolates a failing egress, validates routes against the SensorsConfig,
and (critically) the config layer imports without pulling in any transport dependency (rclpy).
A ``FakeEgress``/``FakeEgressConfig`` test double stands in for a real transport.
"""

from __future__ import annotations

import sys
from dataclasses import field

import numpy as np
import pytest
import tyro
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from typing_extensions import Annotated

from holosoma.config_types.sensor_egress import EgressInstanceConfig, SensorEgressConfig
from holosoma.config_types.sensors import CameraSensorConfig, SensorMountConfig, SensorsConfig
from holosoma.sensor_egress.base import FramePacket, SensorEgress
from holosoma.sensor_egress.driver import SensorEgressDriver
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim

_MOUNT = SensorMountConfig(target_kind="robot_link", target="pelvis")


# ----- test double: a fully in-memory egress with no transport -----


class FakeEgress(SensorEgress):
    """Records every per-step batch it receives. ``wanted_streams`` comes from its config."""

    def __init__(self, config, simulator):
        super().__init__(config, simulator)
        self.started = False
        self.stopped = False
        self.batches: list[dict] = []  # each control step's frames dict

    @property
    def received(self) -> list[FramePacket]:
        """Flattened packets across all batches, for convenience in assertions."""
        return [pkt for batch in self.batches for pkt in batch.values()]

    def wanted_streams(self):
        return set(self.config.streams)

    def start(self):
        self.started = True

    def publish(self, frames):
        self.batches.append(frames)

    def stop(self):
        self.stopped = True


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class FakeEgressConfig(EgressInstanceConfig):
    # list of (camera, modality, env_id) triples this fake wants
    streams: list[tuple[str, str, int]] = field(default_factory=list)

    @property
    def egress_cls(self):
        return FakeEgress


class FailingEgress(FakeEgress):
    def publish(self, frames):
        raise RuntimeError("boom")


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class FailingEgressConfig(FakeEgressConfig):
    @property
    def egress_cls(self):
        return FailingEgress


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class OtherEgressConfig(EgressInstanceConfig):
    """A second, distinct egress type — used to prove one SensorEgressConfig holds a mix."""

    tag: str = "default"

    @property
    def egress_cls(self):
        return FakeEgress


# Module-level so tyro can resolve the field's string annotation against module globals
# (the test file uses ``from __future__ import annotations``).
_MIXED_DEFAULTS = {
    "none": SensorEgressConfig(instances={}),
    "mixed": SensorEgressConfig(
        instances={"fake": FakeEgressConfig(streams=[("head", "rgb", 0)]), "other": OtherEgressConfig(tag="rec")}
    ),
}


@dataclass(frozen=True)
class _MixedRoot:
    sensor_egress: Annotated[
        SensorEgressConfig,
        tyro.conf.arg(constructor=tyro.extras.subcommand_type_from_defaults(_MIXED_DEFAULTS)),
    ] = _MIXED_DEFAULTS["none"]


# ----- a minimal fake simulator exposing only what the driver touches -----


class _FakeSensorManager:
    def __init__(self):
        self.last_due: set[str] = set()


class _FakeSimEngineCfg:
    fps = 200.0
    control_decimation_steps = 4  # control_hz = 50


class _FakeSimulatorConfig:
    sim = _FakeSimEngineCfg()


class _FakeVideoConfig:
    save_dir = None


class _FakeTrainingConfig:
    def __init__(self, num_envs: int):
        self.num_envs = num_envs


class _FakeSimulator:
    """Minimal stand-in exposing only what SensorEgressDriver / egress touch on the simulator."""

    def __init__(self, sensors_config: SensorsConfig, frames: dict[tuple[str, str], np.ndarray], num_envs: int = 1):
        self.sensor_config = sensors_config
        self.sensor_manager = _FakeSensorManager()
        # num_envs is read off training_config (set at __init__ on real backends), NOT self.num_envs
        # (which IsaacSim only populates later, in create_envs).
        self.training_config = _FakeTrainingConfig(num_envs)
        self.headless = True
        self.simulator_config = _FakeSimulatorConfig()
        self.video_config = _FakeVideoConfig()
        self._frames = frames  # (camera, modality) -> [N, H, W, C]
        self._t = 0.0
        self.reads: list[tuple[str, str]] = []

    def time(self) -> float:
        return self._t

    def sensor_config_by_name(self, name):
        return self.sensor_config.cameras[name]

    def get_camera_data(self, name, data_type="rgb", env_ids=None):
        # The driver now reads the full [N, ...] buffer once and indexes envs itself.
        self.reads.append((name, data_type))
        return torch.from_numpy(self._frames[(name, data_type)])


def _cam(name, data_types=("rgb",)):
    """Return a ``(name, CameraSensorConfig)`` pair for building a cameras dict."""
    return name, CameraSensorConfig(mount=_MOUNT, data_types=list(data_types))


def _sensors(*cams):
    return SensorsConfig(cameras=dict(cams))


def _rgb(h=2, w=2, n=1):
    """An [N, H, W, 3] batch (the shape get_camera_data returns)."""
    return np.zeros((n, h, w, 3), dtype=np.uint8)


# ----- tests -----


def test_no_instances_means_inactive_driver():
    sim = _FakeSimulator(_sensors(_cam("head")), {("head", "rgb"): _rgb()})
    driver = SensorEgressDriver(sim, SensorEgressConfig(instances={}))
    assert not driver.is_active
    driver.start()
    driver.publish_due()  # must be a no-op, no reads
    assert sim.reads == []


def test_publishes_only_fresh_wanted_streams():
    sim = _FakeSimulator(
        _sensors(_cam("head"), _cam("wrist")),
        {("head", "rgb"): _rgb(), ("wrist", "rgb"): _rgb()},
    )
    cfg = SensorEgressConfig(instances={"a": FakeEgressConfig(streams=[("head", "rgb", 0), ("wrist", "rgb", 0)])})
    driver = SensorEgressDriver(sim, cfg)
    driver.start()
    egress = driver.egress[0]
    assert egress.started

    # Only 'head' rendered this step -> only head is published, wrist is not read.
    sim.sensor_manager.last_due = {"head"}
    driver.publish_due()
    assert [(p.camera, p.modality) for p in egress.received] == [("head", "rgb")]
    assert ("wrist", "rgb") not in sim.reads


def test_snapshot_once_fans_out_to_multiple_egress():
    sim = _FakeSimulator(_sensors(_cam("head")), {("head", "rgb"): _rgb()})
    a = FakeEgressConfig(streams=[("head", "rgb", 0)])
    b = FakeEgressConfig(streams=[("head", "rgb", 0)])
    driver = SensorEgressDriver(sim, SensorEgressConfig(instances={"a": a, "b": b}))
    driver.start()
    sim.sensor_manager.last_due = {"head"}
    driver.publish_due()
    # Two egress both received the frame, but the buffer was read from the sim only ONCE.
    assert sim.reads == [("head", "rgb")]
    assert len(driver.egress[0].received) == 1
    assert len(driver.egress[1].received) == 1


def test_one_read_serves_multiple_envs():
    # A single get_camera_data read of the [N,...] buffer serves every wanted env (no per-env read).
    sim = _FakeSimulator(_sensors(_cam("head")), {("head", "rgb"): _rgb(n=3)}, num_envs=3)
    cfg = SensorEgressConfig(instances={"a": FakeEgressConfig(streams=[("head", "rgb", 0), ("head", "rgb", 2)])})
    driver = SensorEgressDriver(sim, cfg)
    driver.start()
    sim.sensor_manager.last_due = {"head"}
    driver.publish_due()
    assert sim.reads == [("head", "rgb")]  # ONE read for both envs
    envs = sorted(p.env_id for p in driver.egress[0].received)
    assert envs == [0, 2]


def test_packet_carries_intrinsics_sim_time_and_env():
    sim = _FakeSimulator(_sensors(_cam("head")), {("head", "rgb"): _rgb(h=4, w=6)})
    sim._t = 1.25
    driver = SensorEgressDriver(
        sim, SensorEgressConfig(instances={"a": FakeEgressConfig(streams=[("head", "rgb", 0)])})
    )
    driver.start()
    sim.sensor_manager.last_due = {"head"}
    driver.publish_due()
    pkt = driver.egress[0].received[0]
    assert pkt.sim_time == 1.25
    assert pkt.env_id == 0
    assert (pkt.intrinsics.width, pkt.intrinsics.height) == (128, 128)  # config defaults
    assert pkt.array.shape == (4, 6, 3) and pkt.array.dtype == np.uint8


def test_failing_egress_is_isolated():
    sim = _FakeSimulator(_sensors(_cam("head")), {("head", "rgb"): _rgb()})
    bad = FailingEgressConfig(streams=[("head", "rgb", 0)])
    good = FakeEgressConfig(streams=[("head", "rgb", 0)])
    driver = SensorEgressDriver(sim, SensorEgressConfig(instances={"bad": bad, "good": good}))
    driver.start()
    sim.sensor_manager.last_due = {"head"}
    driver.publish_due()  # must NOT raise despite the failing egress
    # the good egress (constructed second) still got its frame
    assert len(driver.egress[1].received) == 1


def test_disabled_instance_skipped():
    sim = _FakeSimulator(_sensors(_cam("head")), {("head", "rgb"): _rgb()})
    cfg = SensorEgressConfig(instances={"a": FakeEgressConfig(enabled=False, streams=[("head", "rgb", 0)])})
    driver = SensorEgressDriver(sim, cfg)
    assert driver.egress == []
    assert not driver.is_active


def test_validation_rejects_unknown_camera():
    sim = _FakeSimulator(_sensors(_cam("head")), {("head", "rgb"): _rgb()})
    cfg = SensorEgressConfig(instances={"a": FakeEgressConfig(streams=[("nonexistent", "rgb", 0)])})
    with pytest.raises(ValueError, match="not in the active SensorsConfig"):
        SensorEgressDriver(sim, cfg)


def test_validation_rejects_unrendered_modality():
    # camera renders rgb only; egress asks for depth -> fail loud at driver construction.
    sim = _FakeSimulator(_sensors(_cam("head", data_types=("rgb",))), {("head", "rgb"): _rgb()})
    cfg = SensorEgressConfig(instances={"a": FakeEgressConfig(streams=[("head", "depth", 0)])})
    with pytest.raises(ValueError, match="renders only"):
        SensorEgressDriver(sim, cfg)


def test_validation_rejects_out_of_range_env():
    sim = _FakeSimulator(_sensors(_cam("head")), {("head", "rgb"): _rgb()}, num_envs=1)
    cfg = SensorEgressConfig(instances={"a": FakeEgressConfig(streams=[("head", "rgb", 5)])})
    with pytest.raises(ValueError, match="env 5"):
        SensorEgressDriver(sim, cfg)


def test_stop_tears_down_all():
    sim = _FakeSimulator(_sensors(_cam("head")), {("head", "rgb"): _rgb()})
    cfg = SensorEgressConfig(instances={"a": FakeEgressConfig(streams=[("head", "rgb", 0)])})
    driver = SensorEgressDriver(sim, cfg)
    driver.start()
    driver.stop()
    assert driver.egress[0].stopped


def test_config_layer_imports_without_rclpy():
    # The whole optional-dependency guarantee: importing the config + config_values modules must
    # not import rclpy (the deferred egress_cls import is the only path that would). Guards against
    # a regression where someone top-level-imports a transport dep in the ROS-free config layer.
    import holosoma.config_types.sensor_egress
    import holosoma.config_values.sensor_egress
    import holosoma.sensor_egress  # noqa: F401  (package root + base + driver)

    assert "rclpy" not in sys.modules, "config/driver layer must stay ROS-free; rclpy leaked in."


def test_one_config_type_holds_heterogeneous_instances_through_tyro():
    # The core requirement: a single SensorEgressConfig (base-typed instances) holds a MIX of
    # different egress types in one preset, and selecting that preset via tyro preserves each
    # element's concrete subclass + fields (no slicing) AND allows a flat per-field CLI override.
    # Select the mixed preset: both concrete subclasses survive with their fields.
    cfg = tyro.cli(_MixedRoot, args=["sensor-egress:mixed"])
    insts = cfg.sensor_egress.instances
    assert [type(i).__name__ for i in insts.values()] == ["FakeEgressConfig", "OtherEgressConfig"]
    assert insts["other"].tag == "rec"

    # Flat per-field override of a keyed element on the base-typed dict (no nested subcommand needed).
    cfg2 = tyro.cli(_MixedRoot, args=["sensor-egress:mixed", "--sensor-egress.instances.other.tag", "ds"])
    assert cfg2.sensor_egress.instances["other"].tag == "ds"
    assert type(cfg2.sensor_egress.instances["fake"]).__name__ == "FakeEgressConfig"
