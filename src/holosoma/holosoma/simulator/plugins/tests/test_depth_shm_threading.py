"""The depth-shm plugin's render thread contract (pure, no simulator, no GL).

Rendering a camera costs far more than a 500 Hz physics step's 2 ms budget, so this plugin renders on
its own thread instead of on ``FRAME_END``. Three properties keep that from silently regressing:
the camera must be taken off the control-loop schedule (or the cost returns to the physics thread),
``publish`` must stay a no-op (same reason), and the thread must be started before the sim loop and
joined before the shared memory it writes into is released.

A fake simulator stands in for the real one: these tests are about scheduling and lifecycle, not
about pixels — the numeric side is covered by ``test_depth_shm_plugin.py``.
"""

from __future__ import annotations

import threading
import time
from multiprocessing import shared_memory
from uuid import uuid4

import numpy as np
import pytest

from holosoma.config_types.plugin import DepthShmPluginConfig
from holosoma.config_types.sensor import CameraSensorConfig, SensorMountConfig
from holosoma.simulator.base_simulator.hooks import HookRegistry, Phase
from holosoma.simulator.plugins.depth_shm_plugin import DepthShmPlugin
from holosoma.simulator.shared.camera_sensor import SensorManager
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim

CAM = "front_depth"
SHM_NAME = "depth_img_shm_test_threading"
RENDER_H, RENDER_W = 60, 106


class _FakeSharedMemory:
    blocks: dict[str, bytearray] = {}

    def __init__(self, *, name: str, create: bool = False, size: int = 0) -> None:
        if create:
            if name in self.blocks:
                raise FileExistsError(name)
            self.blocks[name] = bytearray(size)
        elif name not in self.blocks:
            raise FileNotFoundError(name)
        self.name = name
        self.buf = self.blocks[name]
        self.size = len(self.buf)

    def close(self) -> None:
        pass

    def unlink(self) -> None:
        if self.name not in self.blocks:
            raise FileNotFoundError(self.name)
        del self.blocks[self.name]


@pytest.fixture(autouse=True)
def fake_shared_memory(monkeypatch: pytest.MonkeyPatch):
    _FakeSharedMemory.blocks.clear()
    monkeypatch.setattr(shared_memory, "SharedMemory", _FakeSharedMemory)
    yield
    _FakeSharedMemory.blocks.clear()


def _camera_config() -> CameraSensorConfig:
    return CameraSensorConfig(
        mount=SensorMountConfig(target_kind="robot_link", target="pelvis"),
        width=RENDER_W,
        height=RENDER_H,
        data_types=["depth"],
        update_decimation=1,
    )


class _FakeBackend:
    """Counts render calls and records which thread they came from."""

    def __init__(self) -> None:
        self.calls = 0
        self.threads: set[int] = set()

    def render_cameras(self, cameras) -> None:
        self.calls += 1
        self.threads.add(threading.get_ident())


class _FakeTrainingConfig:
    num_envs = 1


class _FakeSimulator:
    """Minimal surface DepthShmPlugin touches: hooks, sensor_manager, backend, camera reads."""

    def __init__(self) -> None:
        self.hooks = HookRegistry()
        self.sensor_manager = SensorManager("cpu", control_hz=125.0)
        self.sensor_manager.register_camera(CAM, _camera_config())
        self.backend = _FakeBackend()
        self.sensor_config = {CAM: _camera_config()}
        self.training_config = _FakeTrainingConfig()

    def get_camera_data(self, name, data_type="rgb", env_ids=None, device=None):
        # Metric depth, [N, H, W, C], mid-range so it normalizes to something non-degenerate.
        return torch.full((1, RENDER_H, RENDER_W, 1), 1.5, dtype=torch.float32)

    def time(self) -> float:
        return 0.0

    def sensor_config_by_name(self, name):
        return self.sensor_config[name]


def _config(**overrides) -> DepthShmPluginConfig:
    kwargs = {
        "camera": CAM,
        "shm_name": SHM_NAME,
        "render_hz": 200.0,  # fast so tests do not sleep long
        "near_clip": 0.3,
        "far_clip": 3.0,
        "crop_top": 2,
        "crop_left": 4,
        "crop_right": 4,
    }
    kwargs.update(overrides)
    return DepthShmPluginConfig(**kwargs)


@pytest.fixture
def plugin():
    sim = _FakeSimulator()
    p = DepthShmPlugin(_config(), sim)
    yield p, sim
    p.stop()  # idempotent; releases shm even if a test already stopped it


def test_camera_is_removed_from_the_control_loop_schedule(plugin) -> None:
    """The whole point: ``collect_due`` must never return a camera this plugin renders itself."""
    _, sim = plugin
    sim.hooks.emit(Phase.EPISODE_START, 0)

    # Many control steps' worth of scheduling; the claimed camera must never come due.
    assert all(sim.sensor_manager.collect_due() == [] for _ in range(50))


def test_publish_is_a_noop_so_frame_end_costs_nothing(plugin) -> None:
    """``FRAME_END`` must not render or write: that would be back on the physics thread."""
    p, sim = plugin
    sim.hooks.emit(Phase.EPISODE_START, 0)
    p._stop.set()  # freeze the render thread so only the hook can act
    if p._thread is not None:
        p._thread.join(timeout=2.0)

    before = sim.backend.calls
    sim.hooks.emit(Phase.FRAME_END)
    assert sim.backend.calls == before


def test_render_thread_publishes_frames_off_the_main_thread(plugin) -> None:
    """Frames must be produced, and produced by the plugin's thread rather than the caller's."""
    p, sim = plugin
    sim.hooks.emit(Phase.EPISODE_START, 0)

    deadline = time.monotonic() + 5.0
    while sim.backend.calls < 3 and time.monotonic() < deadline:
        time.sleep(0.01)

    assert sim.backend.calls >= 3, "render thread produced no frames"
    assert threading.get_ident() not in sim.backend.threads, "rendering ran on the calling thread"
    # 1.5 m within [0.3, 3.0] -> (1.5-0.3)/2.7 - 0.5
    assert p._array is not None
    np.testing.assert_allclose(p._array, (1.5 - 0.3) / 2.7 - 0.5, atol=1e-6)


def test_start_blocks_until_the_first_frame_is_published(plugin) -> None:
    """Startup must not race: the GL/first-frame cost belongs before the timed sim loop."""
    p, sim = plugin
    assert sim.backend.calls == 0

    sim.hooks.emit(Phase.EPISODE_START, 0)

    # start() returned, so a frame already exists — no polling needed.
    assert sim.backend.calls >= 1
    assert p._first_frame.is_set()


def test_stop_joins_the_thread_before_releasing_shared_memory(plugin) -> None:
    """Dropping the shm mapping under a live render would fault on the buffer it writes into."""
    p, sim = plugin
    sim.hooks.emit(Phase.EPISODE_START, 0)
    thread = p._thread
    assert thread is not None and thread.is_alive()

    sim.hooks.emit(Phase.CLOSE)

    assert not thread.is_alive()
    assert p._thread is None
    assert p._array is None


def test_render_thread_survives_a_render_error(plugin) -> None:
    """A transient backend failure must not kill the thread and silently freeze the depth stream."""
    _, sim = plugin
    sim.hooks.emit(Phase.EPISODE_START, 0)

    calls = {"n": 0}
    original = sim.backend.render_cameras

    def flaky(cameras):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise RuntimeError("transient GL failure")
        original(cameras)

    sim.backend.render_cameras = flaky
    before = sim.backend.calls
    deadline = time.monotonic() + 5.0
    while sim.backend.calls <= before and time.monotonic() < deadline:
        time.sleep(0.01)

    assert calls["n"] > 2, "thread stopped after the error instead of retrying"
    assert sim.backend.calls > before, "thread never resumed publishing"


def test_render_hz_must_be_positive() -> None:
    """A zero/negative rate would divide by zero in the loop period."""
    with pytest.raises(ValueError, match="render_hz must be > 0"):
        _config(render_hz=0.0)


def test_producer_rejects_existing_block_with_different_shape() -> None:
    name = f"depth_img_shm_test_{uuid4().hex}"
    expected_bytes = 58 * 87 * np.dtype(np.float32).itemsize
    stale = shared_memory.SharedMemory(name=name, create=True, size=expected_bytes + 4)
    plugin = DepthShmPlugin(_config(shm_name=name), _FakeSimulator())

    try:
        with pytest.raises(ValueError, match=f"is {expected_bytes + 4} bytes"):
            plugin.start()
    finally:
        plugin.stop()
        stale.close()
        stale.unlink()
