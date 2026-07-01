"""Unit tests for the shared ``BaseSimulator.get_camera_data`` accessor (pure, no simulator).

The accessor lives once on the base class and reads buffers filled by a backend's ``render_sensors``.
These pin its two sim-free failure modes and the env-id slicing, on a minimal stub, so the behavior
holds identically regardless of backend.
"""

from __future__ import annotations

import pytest

from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
from holosoma.simulator.shared.camera_sensor import CameraRuntime
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim


class _Manager:
    """Minimal SensorManager-shaped stub: name -> CameraRuntime with prefilled buffers."""

    def __init__(self, runtimes: dict[str, CameraRuntime]):
        self._runtimes = runtimes

    def has_camera(self, name: str) -> bool:
        return name in self._runtimes

    def get(self, name: str) -> CameraRuntime:
        return self._runtimes[name]

    @property
    def names(self) -> list[str]:
        return list(self._runtimes)


def _sim(manager) -> BaseSimulator:
    sim = BaseSimulator.__new__(BaseSimulator)  # skip __init__ (needs a full config)
    sim.sensor_manager = manager
    return sim


def test_no_sensor_manager_raises_not_implemented():
    sim = _sim(None)
    with pytest.raises(NotImplementedError, match="no camera 'head'"):
        sim.get_camera_data("head", "rgb")


def test_missing_camera_raises_not_implemented():
    sim = _sim(_Manager({}))
    with pytest.raises(NotImplementedError, match="no camera 'head'"):
        sim.get_camera_data("head", "rgb")


def test_missing_buffer_raises_runtime_error():
    rt = CameraRuntime(config=None)  # no buffers filled (render_sensors never ran for this type)
    sim = _sim(_Manager({"head": rt}))
    with pytest.raises(RuntimeError, match="no 'rgb' frame"):
        sim.get_camera_data("head", "rgb")


def test_returns_full_buffer_and_env_subset():
    buf = torch.arange(3 * 2 * 2 * 3, dtype=torch.uint8).reshape(3, 2, 2, 3)
    rt = CameraRuntime(config=None, buffers={"rgb": buf})
    sim = _sim(_Manager({"head": rt}))
    assert torch.equal(sim.get_camera_data("head", "rgb"), buf)  # env_ids=None -> all
    subset = sim.get_camera_data("head", "rgb", env_ids=[0, 2])
    assert torch.equal(subset, buf[[0, 2]])
