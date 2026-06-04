"""Tracker controller — the holosoma-side control logic (NOT a ROS node).

Owns the two G1 high-level client proxies and turns ``UnitreeTrackerCommand``
targets into client actions. Knows nothing about rclpy.

Control split (Option 3 — callback wakes, loop sustains):

* ``set_target(msg)`` is cheap and non-blocking — called straight from the ROS
  subscription callback (executor thread) so the freshest target is visible
  immediately, without blocking message reception on a proxy round-trip.
* ``tick()`` pushes the *held* target to the clients. Run it from a steady loop
  (``run()``) so arm_sdk keeps getting ~250 Hz commands to hold/track smoothly
  even between (slower) tracking messages — the heartbeat.

Arm: re-published every tick (it needs continuous commands to hold position).
Loco: ``LocoClient.Move(continous_move=True)`` latches, so we only re-issue
velocity when it changes (the callback path makes a change visible at once).
"""

from __future__ import annotations

import threading
import time

import numpy as np
from loguru import logger

from holosoma_inference.teleop.holosoma_teleop_msgs._ensure_msgs import UnitreeTrackerCommand

CONTROL_HZ = 250.0


class TrackerController:
    def __init__(self, arm=None, loco=None, control_hz: float = CONTROL_HZ):
        self._arm = arm
        self._loco = loco
        self._dt = 1.0 / control_hz

        self._lock = threading.Lock()
        self._target: UnitreeTrackerCommand | None = None
        self._last_vel: tuple[float, float, float] | None = None
        self._stop = threading.Event()

    # --- called from the ROS callback (executor thread): fast, non-blocking ---
    def set_target(self, msg: UnitreeTrackerCommand) -> None:
        with self._lock:
            self._target = msg

    # --- called from the control loop: pushes held target to the clients ---
    def tick(self) -> None:
        with self._lock:
            target = self._target
        if target is None:
            return

        if self._arm is not None:
            q = np.concatenate([np.array(target.q_left_arm), np.array(target.q_right_arm)])
            self._arm.track_dual_arm(q.tolist())  # clip+publish, child-side

        if self._loco is not None:
            v = target.base_velocity
            vel = (v.linear.x, v.linear.y, v.angular.z)
            if vel != self._last_vel:  # Move latches; only re-issue on change
                self._loco.set_velocity(*vel)
                self._last_vel = vel

    def run(self) -> None:
        """Block in the steady control loop until :meth:`stop`."""
        logger.info(f"[tracker-controller] loop running at {1.0 / self._dt:.0f} Hz")
        while not self._stop.is_set():
            self.tick()
            time.sleep(self._dt)

    def stop(self) -> None:
        self._stop.set()
        if self._loco is not None:
            self._loco.stop()
