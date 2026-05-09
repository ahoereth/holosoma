"""DampingPolicy — hold the last observed joint positions.

Used as the safety idle state of the Controller. On entry, captures
the robot's current joint positions; on each tick, publishes a
``send_low_command`` with that pose and the policy's KP/KD gains so
the robot stays energized at the same place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from holosoma_inference.controllers.protocol import Command

if TYPE_CHECKING:
    from holosoma_inference.controllers.controller import Controller
    from holosoma_inference.inputs.api.commands import StateCommand, VelCmd


class DampingPolicy:
    """Hold last-observed joint positions with the policy's KP/KD."""

    name = "damping"

    def __init__(self, kp_scale: float = 1.0, kd_scale: float = 1.0):
        self.kp_scale = kp_scale
        self.kd_scale = kd_scale
        self._q_hold: np.ndarray | None = None

    def on_activate(self, ctx: Controller) -> None:
        # Capture on the first act() call so we use the freshest state.
        self._q_hold = None

    def on_deactivate(self, ctx: Controller) -> None:
        self._q_hold = None

    def apply_velocity(self, vc: VelCmd) -> None:
        return None

    def apply_command(self, cmd: StateCommand) -> bool:
        return False

    def act(self, ctx: Controller, state: np.ndarray) -> Command:
        n = ctx.num_dofs
        if self._q_hold is None:
            self._q_hold = state[7 : 7 + n].copy()
        kp = ctx.motor_kp * self.kp_scale
        kd = ctx.motor_kd * self.kd_scale
        zeros = np.zeros(n)
        return Command(
            q=self._q_hold + ctx.joint_offsets,
            dq=zeros,
            tau=zeros,
            kp_override=kp,
            kd_override=kd,
            dof_pos_latest=state[7 : 7 + n],
        )
