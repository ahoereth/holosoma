"""Publish preprocessed depth frames to shared memory for a policy process.

Sim-side counterpart of ``holosoma_inference.sensors.depth_shm.DepthShmSensor``. The
two agree on a small implicit contract — block name, array shape, dtype, and value
range — so any change here must be mirrored there.

The preprocessing (resize, clip, normalize) lives on this side so the policy reads a
tensor it can hand straight to its depth backbone. The resize deliberately uses torch
bicubic with antialiasing to match ``torchvision.transforms.Resize(..., BICUBIC)`` as
used during training; a cheaper bilinear resize shifts the depth statistics enough to
degrade a stair policy's foothold placement.

Rendering runs on its OWN daemon thread at ``render_hz`` rather than inline on the
``FRAME_END`` hook. A MuJoCo GL depth render costs ~0.25 ms on a GPU context and several
ms on a software one, which does not fit the 2 ms budget of a 500 Hz physics step: inline,
every render lands on one unlucky step, blows its slot, and the rate limiter then sprints
to catch up — the sim reports 400/600 Hz swings and cannot hold its target. Off-thread,
the camera cannot steal from the physics budget at all (measured 845 -> 7781 FPS uncapped
on the sim2sim rig). This mirrors the reference deployment, which runs its depth server on
a separate thread at the camera's real publish rate.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from holosoma.simulator.plugins.camera_consumer import CameraConsumerPlugin, FramePacket, StreamKey

if TYPE_CHECKING:
    from holosoma.config_types.plugin import DepthShmPluginConfig
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator


class DepthShmPlugin(CameraConsumerPlugin):
    """Write normalized depth for one camera into a shared-memory block.

    Renders on a background thread at ``cfg.render_hz`` (see the module docstring); the
    ``FRAME_END`` hook is left unused so the physics loop pays nothing for the camera.
    """

    def __init__(self, config: DepthShmPluginConfig, simulator: BaseSimulator) -> None:
        # Read by wanted_streams(), which the base calls during __init__.
        self._camera = config.camera
        self._env_id = config.env_id
        super().__init__(config, simulator)

        self._shape = (1, 1, config.resized_height, config.resized_width)
        self._shm = None
        self._array: np.ndarray | None = None
        # Ring buffer for modeled latency; sized so index -1-latency is always valid.
        self._history: list[np.ndarray] = []
        self._history_len = config.latency_frames + 1

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._first_frame = threading.Event()

        # The base class starts a consumer lazily on its first FRAME_END, i.e. from inside the
        # physics loop. Constructing a GL context there costs ~150-450 ms and holds the GIL, which
        # shows up as a large stall in the first FPS window. EPISODE_START fires once after setup
        # and before the loop begins, so start there instead and take the cost off the clock.
        from holosoma.simulator.base_simulator.hooks import Phase

        simulator.hooks.add(Phase.EPISODE_START, self._on_episode_start, name=f"{self._label()}.start")

    def _on_episode_start(self, env_id: int = 0) -> None:
        """Start the render thread before the sim loop, so its GL setup is not timed as a stall."""
        if self._started:
            return
        try:
            self.start()
            self._started = True
        except Exception as exc:  # isolation: match the base class's fail-soft behavior
            logger.error(f"[DepthShmPlugin] start failed: {exc}")

    def wanted_streams(self) -> set[StreamKey]:
        return {(self._camera, "depth", self._env_id)}

    def start(self) -> None:
        """Create (or adopt) the shared-memory block, then start the render thread."""
        from multiprocessing import shared_memory

        nbytes = int(np.prod(self._shape)) * np.dtype(np.float32).itemsize
        try:
            self._shm = shared_memory.SharedMemory(create=True, size=nbytes, name=self.cfg.shm_name)
            logger.info(f"[DepthShmPlugin] created '{self.cfg.shm_name}' ({nbytes} bytes) shape={self._shape}")
        except FileExistsError:
            # A stale block from a previous run, or a consumer that started first.
            self._shm = shared_memory.SharedMemory(name=self.cfg.shm_name)
            if self._shm.size != nbytes:
                size = self._shm.size
                self._shm.close()
                self._shm = None
                raise ValueError(
                    f"Existing shared memory '{self.cfg.shm_name}' is {size} bytes but this camera "
                    f"needs {nbytes}. Remove /dev/shm/{self.cfg.shm_name} (a stale block from a run "
                    f"with a different resolution) and retry."
                ) from None
            logger.info(f"[DepthShmPlugin] attached existing '{self.cfg.shm_name}' shape={self._shape}")
        self._array = np.ndarray(self._shape, dtype=np.float32, buffer=self._shm.buf)

        # Take this camera off the control-loop render schedule: we render it ourselves, and letting
        # `sensors.render` also render it would put the GL cost right back on the physics thread
        # (and build a second, main-thread renderer for it).
        manager = self.simulator.sensor_manager
        if manager is not None:
            manager.claim_external_rendering(self._camera)

        self._stop.clear()
        self._thread = threading.Thread(target=self._render_loop, name="depth-shm-render", daemon=True)
        self._thread.start()
        # Block until the thread has built its GL context and published one frame. Constructing a
        # mujoco.Renderer costs ~150-300 ms and holds the GIL, so letting it happen concurrently
        # would stall the physics loop mid-run; ``start()`` is called before that loop begins, so
        # waiting here moves the cost to a point where nothing is being timed yet. It also means the
        # shm block holds a real frame before any policy can attach.
        if not self._first_frame.wait(timeout=30.0):
            logger.warning("[DepthShmPlugin] render thread produced no frame within 30s")
        logger.info(f"[DepthShmPlugin] render thread started at {self.cfg.render_hz} Hz")

    def publish(self, frames: dict[StreamKey, FramePacket]) -> None:
        """No-op: rendering and publishing both happen on this plugin's own thread.

        The base class still registers this on ``FRAME_END``, but doing the work there would put
        the render back on the physics thread — the exact cost this plugin exists to avoid.
        """

    def _render_loop(self) -> None:
        """Render and publish at ``render_hz`` until stopped.

        Reads ``MjData`` while the physics thread mutates it, without a lock — deliberately, and
        matching the reference deployment. ``update_scene`` copies what it needs into its own
        ``MjvScene``, so the worst case is a frame blending poses from two adjacent physics steps,
        which at 500 Hz is a sub-2 ms skew: far smaller than the camera latency the policy was
        trained against, and cheaper than making the physics loop wait on a lock.
        """
        period = 1.0 / self.cfg.render_hz
        # Build this thread's GL context and renderer before entering the cadence. The first render
        # costs ~300 ms (context creation + shader compile); paying it inside the loop would stall
        # the very first frame and, because MuJoCo's GL work is not fully concurrent, visibly hitch
        # the physics thread at startup.
        try:
            self._render_and_publish_once()
        except Exception as exc:
            logger.error(f"[DepthShmPlugin] first render failed: {exc}")
        finally:
            # Release start() either way: a permanently broken camera should not hang startup.
            self._first_frame.set()

        next_at = time.perf_counter()
        while not self._stop.is_set():
            try:
                self._render_and_publish_once()
            except Exception as exc:  # isolation: the render thread must not kill the sim
                logger.error(f"[DepthShmPlugin] render thread error: {exc}")
                self._stop.wait(period)
                continue
            # Absolute schedule so a slow frame does not accumulate drift.
            next_at += period
            sleep_for = next_at - time.perf_counter()
            if sleep_for <= 0:
                next_at = time.perf_counter()  # fell behind; resynchronize rather than sprint
            else:
                self._stop.wait(sleep_for)

    def _render_and_publish_once(self) -> None:
        """Render this plugin's camera on the calling thread and write the result to shm."""
        if self._array is None:
            return
        manager = self.simulator.sensor_manager
        if manager is None:
            return

        # Render only OUR camera, on this thread. The backend builds a per-thread renderer on
        # first use, so this does not touch the renderer the main thread may hold.
        self.simulator.backend.render_cameras([manager.get(self._camera)])
        buf = self.simulator.get_camera_data(self._camera, "depth", device="cpu")

        # [N, H, W, C] host tensor -> this env's [H, W] metric depth.
        depth = np.asarray(buf[self._env_id].detach().numpy(), dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]

        # Non-hits come back as +inf from the raycast; treat them as maximally far so
        # they normalize to +0.5 rather than poisoning the array with NaN.
        depth = np.nan_to_num(depth, nan=self.cfg.far_clip, posinf=self.cfg.far_clip, neginf=self.cfg.near_clip)

        frame = self._resize_clip_normalize(depth)

        self._history.append(frame)
        if len(self._history) > self._history_len:
            self._history.pop(0)
        # Serve the oldest frame still in the window until the buffer fills, so early
        # steps publish real data rather than zeros.
        delayed = self._history[0] if len(self._history) < self._history_len else self._history[-self._history_len]

        np.copyto(self._array, delayed[None, None])

    def _crop(self, depth: np.ndarray) -> np.ndarray:
        """Drop the configured border rows/cols, mirroring training's pre-resize crop.

        Training crops the raw camera frame (``depth[:, 2:, 4:-4]`` for the D435i rig) *before*
        resizing, so the resize sees that field of view. Cropping after the resize — or not at
        all — changes the geometry the backbone sees without any error."""
        cfg = self.cfg
        top, left = cfg.crop_top, cfg.crop_left
        # Translate "drop N from the end" into an index; None keeps the full extent.
        bottom = -cfg.crop_bottom if cfg.crop_bottom else None
        right = -cfg.crop_right if cfg.crop_right else None
        if not (top or left or bottom or right):
            return depth
        cropped = depth[top:bottom, left:right]
        if cropped.size == 0:
            raise ValueError(
                f"DepthShmPlugin crop removed the whole {depth.shape} frame "
                f"(top={cfg.crop_top}, bottom={cfg.crop_bottom}, left={cfg.crop_left}, right={cfg.crop_right})."
            )
        return cropped

    def _resize_clip_normalize(self, depth: np.ndarray) -> np.ndarray:
        """Crop, resize to the backbone's input size, then clip and normalize to [-0.5, 0.5]."""
        depth = self._crop(depth)

        # Clamp to [near_clip, far_clip] BEFORE the resize, matching training, which clamps at
        # capture time and only then crops/resizes. This is not redundant with the post-resize clip:
        # the bicubic kernel spreads out-of-range values into their neighbors, so clamping after the
        # resize leaves a halo the training pipeline never had. MuJoCo also reports the scene-extent
        # plane (tens of meters) where a ray escapes, which would smear badly.
        near, far = self.cfg.near_clip, self.cfg.far_clip
        depth = np.clip(depth, near, far)

        target = (self.cfg.resized_height, self.cfg.resized_width)
        if depth.shape != target:
            import torch

            tensor = torch.from_numpy(np.ascontiguousarray(depth))[None, None]
            tensor = torch.nn.functional.interpolate(
                tensor, size=target, mode="bicubic", align_corners=False, antialias=True
            )
            depth = tensor[0, 0].numpy()

        # Post-resize handling, mirroring training term-for-term:
        #   1. clamp the FAR side only (bicubic overshoots at sharp depth edges),
        #   2. map anything below `empty_threshold` to far — "too close to be real" reads as empty,
        #      which is how an invalid/dropped pixel presents on the physical camera,
        #   3. normalize with NO near-side clamp.
        # Step 3 is deliberate: bicubic ringing can dip a hair under `near`, and training lets that
        # through, so the output range is *approximately* [-0.5, 0.5] rather than strictly bounded.
        # Clamping it here would look tidier but would feed the backbone a distribution it never saw.
        depth = np.minimum(depth, far)
        depth = np.where(depth < self.cfg.empty_threshold, far, depth)
        return ((depth - near) / (far - near) - 0.5).astype(np.float32)

    def stop(self) -> None:
        """Stop the render thread, then release the block (unlinked for a clean next run)."""
        self._stop.set()
        if self._thread is not None:
            # Join before dropping ``_array``: the thread writes into that buffer, so tearing it out
            # from under a live render would fault on the shm mapping.
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logger.warning("[DepthShmPlugin] render thread did not exit within 2s")
            self._thread = None

        self._array = None
        if self._shm is not None:
            self._shm.close()
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass  # a consumer already unlinked it
            self._shm = None
            logger.info(f"[DepthShmPlugin] released '{self.cfg.shm_name}'")
