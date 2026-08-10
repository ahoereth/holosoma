"""Publish preprocessed depth frames to shared memory for a policy process.

Sim-side counterpart of ``holosoma_inference.sensors.depth_shm.DepthShmSensor``. The
two agree on a small implicit contract — block name, array shape, dtype, and value
range — so any change here must be mirrored there.

The preprocessing (resize, clip, normalize) lives on this side so the policy reads a
tensor it can hand straight to its depth backbone. The resize deliberately uses torch
bicubic with antialiasing to match ``torchvision.transforms.Resize(..., BICUBIC)`` as
used during training; a cheaper bilinear resize shifts the depth statistics enough to
degrade a stair policy's foothold placement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from holosoma.simulator.plugins.camera_consumer import CameraConsumerPlugin, FramePacket, StreamKey

if TYPE_CHECKING:
    from holosoma.config_types.plugin import DepthShmPluginConfig
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator


class DepthShmPlugin(CameraConsumerPlugin):
    """Write normalized depth for one camera into a shared-memory block each step."""

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

    def wanted_streams(self) -> set[StreamKey]:
        return {(self._camera, "depth", self._env_id)}

    def start(self) -> None:
        """Create (or adopt) the shared-memory block."""
        from multiprocessing import shared_memory

        nbytes = int(np.prod(self._shape)) * np.dtype(np.float32).itemsize
        try:
            self._shm = shared_memory.SharedMemory(create=True, size=nbytes, name=self.cfg.shm_name)
            logger.info(f"[DepthShmPlugin] created '{self.cfg.shm_name}' ({nbytes} bytes) shape={self._shape}")
        except FileExistsError:
            # A stale block from a previous run, or a consumer that started first.
            self._shm = shared_memory.SharedMemory(name=self.cfg.shm_name)
            if self._shm.size < nbytes:
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

    def publish(self, frames: dict[StreamKey, FramePacket]) -> None:
        packet = frames.get((self._camera, "depth", self._env_id))
        if packet is None or self._array is None:
            return

        # Host copy is float32 [H, W, 1] in meters; drop the channel axis.
        depth = np.asarray(packet.array, dtype=np.float32)
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
        """Release the block. Unlinks so the next run starts from a clean slate."""
        self._array = None
        if self._shm is not None:
            self._shm.close()
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass  # a consumer already unlinked it
            self._shm = None
            logger.info(f"[DepthShmPlugin] released '{self.cfg.shm_name}'")
