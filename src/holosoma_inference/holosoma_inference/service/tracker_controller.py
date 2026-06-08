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

CONTROL_HZ = 50.0


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
        self._tick_count = 0

    def tick(self) -> None:
        """Poll the newest command and push it to the clients."""
        target = self._source.get_latest()
        if target is None:
            return

        if self._arm is not None:
            q = np.concatenate([np.array(target.q_left_arm), np.array(target.q_right_arm)])
            self._arm.track_dual_arm(q.tolist())  # clip+publish, child-side
        
        if self._loco is not None:
            self._tick_count += 1
            if self._tick_count % 10 == 0:
                v = target.base_velocity
                lx = v.linear.x if abs(v.linear.x) >= 0.05 else 0.0
                ly = v.linear.y if abs(v.linear.y) >= 0.05 else 0.0
                vel = (lx, ly, v.angular.z)

                # --- diagnostics: log every velocity dispatch ---
                changed = vel != self._last_vel
                if changed or (self._tick_count % 250 == 0):
                    logger.info(
                        f"[loco] tick={self._tick_count} vel_cmd={vel} "
                        f"raw=({v.linear.x:.3f},{v.linear.y:.3f},{v.angular.z:.3f}) "
                        f"changed={changed} last={self._last_vel}"
                    )

                code = self._loco.set_velocity(*vel)
                if code is not None and code != 0:
                    logger.warning(f"[loco] set_velocity error code={code}")
                self._last_vel = vel

            # Periodic FSM health check every ~1 second (offset from velocity ticks)
            if self._tick_count % 50 == 25:
                try:
                    healthy, fsm_state = self._loco.check_fsm_healthy()
                    if not healthy:
                        logger.warning(f"[loco] FSM NOT IN WALK MODE: {fsm_state}")
                    else:
                        logger.debug(f"[loco] FSM OK: {fsm_state}")
                except Exception as e:
                    logger.warning(f"[loco] FSM health check failed: {e}")

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
