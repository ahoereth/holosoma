"""Tracker controller — the holosoma-side control logic (NOT a ROS node).

Owns the two G1 high-level client proxies and turns ``ExoskeletonCmd``
targets into client actions. Knows nothing about rclpy.

The control loop is owned here: ``run()`` ticks at a fixed rate and each tick
polls the newest command from the injected ``source`` (anything with
``get_latest() -> ExoskeletonCmd | None``, e.g. ``TeleopListener``).
Re-publishing every tick gives arm_sdk the steady ~250 Hz it needs to hold/
track smoothly even between (slower) incoming messages.

Arm: re-published every tick. Loco: ``LocoClient.Move(continous_move=True)``
latches, so velocity is only re-issued when it changes.
"""

from __future__ import annotations

import threading
from typing import Protocol

import numpy as np
from loguru import logger

from holosoma_inference.teleop.holosoma_teleop_msgs._ensure_msgs import ExoskeletonCmd
from holosoma_inference.utils.rate import RateLimiter

CONTROL_HZ = 250.0


class CommandSource(Protocol):
    def get_latest(self) -> ExoskeletonCmd | None: ...


class TrackerController:
    def __init__(self, source: CommandSource, arm=None, loco=None, control_hz: float = CONTROL_HZ):
        self._source = source
        self._arm = arm
        self._loco = loco
        self._control_hz = control_hz
        self._rate = RateLimiter(control_hz)

        self._last_vel: tuple[float, float, float] | None = None
        self._stop = threading.Event()

    def tick(self) -> None:
        """Poll the newest command and push it to the clients."""
        target = self._source.get_latest()
        if target is None:
            return

        if self._arm is not None:
            q = np.concatenate([np.array(target.q_left_arm), np.array(target.q_right_arm)])
            self._arm.track_dual_arm(q.tolist())  # clip+publish, child-side

        if self._loco is not None:
            v = target.base_velocity
            vel = (v.linear.x, v.linear.y, v.angular.z)
            self._loco.set_velocity(*vel)
            self._last_vel = vel

    def run(self) -> None:
        """Block in the steady control loop until :meth:`stop`."""
        logger.info(f"[tracker-controller] loop running at {self._control_hz:.0f} Hz")
        while not self._stop.is_set():
            self.tick()
            self._rate.sleep()  # drift-compensated; accounts for tick() work time

    def stop(self) -> None:
        self._stop.set()
        if self._loco is not None:
            self._loco.stop()
