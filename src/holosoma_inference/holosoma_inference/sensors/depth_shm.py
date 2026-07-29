"""Shared-memory depth sensor.

Attaches to a POSIX shared-memory block published by an external depth server
(e.g. the MuJoCo image server in the ``holosoma`` sim package, or an on-robot
camera daemon) and hands the latest frame stack to a policy.

The producer owns the block and writes a float32 array of shape
``(num_cameras, 1, height, width)`` containing depth that is already resized,
clipped and normalized — this sensor performs no preprocessing, it only reads.
Reading is a plain memory copy, so it adds no measurable latency to the control
loop.

Implements :class:`~holosoma_inference.sensors.base.Sensor`, so a policy polls
it through the same ``get_latest()`` seam as any other injected sensor.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from holosoma_inference.sensors.base import Sensor


class DepthShmSensor(Sensor):
    """Read-only view of a depth-image shared-memory block.

    Parameters
    ----------
    shape
        Expected array shape ``(num_cameras, channels, height, width)``. Must
        match the producer exactly — a mismatch means the two sides disagree
        about resolution, which would silently corrupt the depth latent.
    name
        Shared-memory block name (the producer's ``shared_memory`` name).
    required
        When ``True``, a missing block raises at ``start()``. When ``False``,
        the sensor logs a warning and serves zero frames, which keeps the
        control loop runnable without a depth producer.
    """

    def __init__(self, shape: tuple[int, ...], name: str = "depth_img_shm", required: bool = True):
        self._shape = tuple(shape)
        self._name = name
        self._required = required
        self._shm = None
        self._array: np.ndarray | None = None
        self._zeros = np.zeros(self._shape, dtype=np.float32)
        self._warned_missing = False

    def start(self) -> None:
        """Attach to the shared-memory block.

        Raises
        ------
        FileNotFoundError
            If the block does not exist and ``required`` is True.
        ValueError
            If the block is too small for ``shape`` — i.e. the producer and the
            policy disagree on the image dimensions.
        """
        from multiprocessing import shared_memory

        try:
            self._shm = shared_memory.SharedMemory(name=self._name)
        except FileNotFoundError:
            if self._required:
                raise FileNotFoundError(
                    f"Depth shared memory '{self._name}' not found. Start the depth image "
                    f"server before the policy, or set --task.depth-shm.no-required to run "
                    f"with zero-filled depth frames."
                ) from None
            logger.warning(f"[DepthShmSensor] '{self._name}' not found — serving zero depth frames")
            return

        expected_bytes = int(np.prod(self._shape)) * np.dtype(np.float32).itemsize
        if self._shm.size < expected_bytes:
            actual = self._shm.size
            self._shm.close()
            self._shm = None
            raise ValueError(
                f"Depth shared memory '{self._name}' is {actual} bytes but shape {self._shape} "
                f"needs {expected_bytes}. The producer's camera resolution does not match this "
                f"policy's camera config."
            )

        self._array = np.ndarray(self._shape, dtype=np.float32, buffer=self._shm.buf)
        logger.info(f"[DepthShmSensor] attached to '{self._name}' shape={self._shape}")

    def get_latest(self) -> np.ndarray:
        """Return a copy of the latest depth stack, shape ``self._shape``.

        Copied because the producer may overwrite the buffer mid-read; the
        policy must operate on a stable frame.
        """
        if self._array is None:
            if not self._warned_missing:
                logger.warning(f"[DepthShmSensor] '{self._name}' unavailable — using zeros")
                self._warned_missing = True
            return self._zeros.copy()
        return self._array.copy()

    def stop(self) -> None:
        """Detach from the block. Does not unlink — the producer owns it."""
        self._array = None
        if self._shm is not None:
            self._shm.close()
            self._shm = None
