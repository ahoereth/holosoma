"""The sensor-egress driver: snapshot once, fan out per step, isolate failures. ROS-free.

Owned by ``BaseSimulator`` and stepped once per control step AFTER ``render_sensors`` (NOT in the
physics-coupled bridge slot). It constructs each enabled egress directly via the config's
``egress_cls`` (no registry/factory), passing a read-only :class:`EgressContext`. It unions every
egress's ``wanted_streams`` (``(camera, modality, env_id)`` triples), does ONE device→host copy per
freshly-rendered (camera, modality) — serving every wanted env from that single read — and hands
each egress a per-step batch of just the streams it asked for.

This is the sensor-side analog of ``SimulatorBridge``, but plural (a list of egress, not one bridge)
and outbound-only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from holosoma.sensor_egress.base import CameraIntrinsics, FramePacket, SensorEgress, StreamKey

if TYPE_CHECKING:
    from holosoma.config_types.sensor_egress import SensorEgressConfig
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator


class SensorEgressDriver:
    """Constructs, drives, and tears down the configured sensor egress for one simulator."""

    def __init__(self, simulator: BaseSimulator, egress_config: SensorEgressConfig) -> None:
        self.simulator = simulator
        # config → impl is a direct typed reference (no registry, no string, no factory). Each egress
        # gets the simulator (mirroring BasicSdk2Bridge) and reads what it needs off it.
        self.egress = [inst.egress_cls(inst, simulator) for inst in egress_config.instances.values() if inst.enabled]
        # Union of every egress's needs; only these (camera, modality, env) triples are snapshotted.
        self._wanted: set[StreamKey] = set()
        for e in self.egress:
            self._wanted |= e.wanted_streams()
        # Per-egress wanted set, cached so publish_due routes without re-querying each step.
        self._wanted_by_egress: list[set[StreamKey]] = [e.wanted_streams() for e in self.egress]
        self._validate_against_sensors()

    @property
    def is_active(self) -> bool:
        return bool(self.egress)

    def _validate_against_sensors(self) -> None:
        """Fail loud if a stream names a camera/modality/env the active sim does not provide."""
        cams = dict(self.simulator.sensor_config.cameras)
        # training_config.num_envs is set at __init__ on every backend; self.num_envs is not yet
        # populated when IsaacSim builds the egress during scene setup (it lands in create_envs).
        num_envs = self.simulator.training_config.num_envs
        for cam, mod, env in sorted(self._wanted):
            if cam not in cams:
                raise ValueError(
                    f"Sensor egress references camera '{cam}', which is not in the active SensorsConfig "
                    f"(cameras: {sorted(cams)}). Egress streams must name a configured camera."
                )
            if mod not in cams[cam].data_types:
                raise ValueError(
                    f"Sensor egress wants '{mod}' from camera '{cam}', but that camera renders only "
                    f"{list(cams[cam].data_types)}. Add '{mod}' to its data_types or fix the egress."
                )
            if not 0 <= env < num_envs:
                raise ValueError(
                    f"Sensor egress wants env {env} of camera '{cam}', but the sim has {num_envs} env(s) "
                    f"[0, {num_envs})."
                )

    def _intrinsics_of(self, cam: str) -> CameraIntrinsics:
        c = self.simulator.sensor_config_by_name(cam)
        return CameraIntrinsics(width=c.width, height=c.height, vertical_fov=c.vertical_fov, near=c.near, far=c.far)

    def start(self) -> None:
        for e in self.egress:
            e.start()

    def publish_due(self) -> None:
        """Snapshot every freshly-rendered wanted stream once and hand each egress its batch."""
        if not self.egress or self.simulator.sensor_manager is None:
            return
        fresh = self.simulator.sensor_manager.last_due  # set[str] of camera names rendered this step
        sim_time = self.simulator.time()

        # One GPU->host read per (camera, modality) that is both wanted and fresh; index the wanted
        # envs out of that single read. ``packets[key]`` is the FramePacket for each wanted triple.
        packets: dict[StreamKey, FramePacket] = {}
        by_cam_mod: dict[tuple[str, str], list[int]] = {}
        for cam, mod, env in self._wanted:
            if cam in fresh:
                by_cam_mod.setdefault((cam, mod), []).append(env)
        for (cam, mod), envs in by_cam_mod.items():
            buf = self.simulator.get_camera_data(cam, mod)  # [N, H, W, C] on device
            intr = self._intrinsics_of(cam)
            host = buf.detach().cpu().numpy()  # single sync for all wanted envs of this (cam, mod)
            for env in envs:
                packets[(cam, mod, env)] = FramePacket(
                    camera=cam, modality=mod, env_id=env, array=host[env], sim_time=sim_time, intrinsics=intr
                )

        # Hand each egress only the streams it wanted that rendered this step; skip if none.
        for egress, wanted in zip(self.egress, self._wanted_by_egress):
            batch = {k: packets[k] for k in wanted if k in packets}
            if batch:
                _safe_publish(egress, batch)

    def stop(self) -> None:
        for e in self.egress:
            _safe_stop(e)


def _safe_publish(egress: SensorEgress, frames: dict[StreamKey, FramePacket]) -> None:
    """Publish on one egress, swallowing+logging any error so siblings and the sim are unaffected."""
    try:
        egress.publish(frames)
    except Exception as exc:
        logger.error(f"Sensor egress {type(egress).__name__} publish failed: {exc}")


def _safe_stop(egress: SensorEgress) -> None:
    """Stop one egress, swallowing+logging any error so the rest still tear down."""
    try:
        egress.stop()
    except Exception as exc:
        logger.error(f"Sensor egress {type(egress).__name__} stop failed: {exc}")
