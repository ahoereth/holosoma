"""InitPolicy — interpolate from current pose to a target over N ticks.

Replaces the legacy ``get_ready_state`` / ``init_count`` flag pair on
``BasePolicy``. On activation, captures the current joint positions
and ramps linearly toward ``target_q`` over ``n_steps`` control ticks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from holosoma_inference.controllers.protocol import Command

if TYPE_CHECKING:
    from holosoma_inference.controllers.controller import Controller
    from holosoma_inference.inputs.api.commands import StateCommand, VelCmd


class InitPolicy:
    """Linear interpolation from current dof_pos to ``target_q``."""

    name = "init"

    def __init__(self, target_q: np.ndarray | tuple[float, ...], n_steps: int = 500):
        self.target_q = np.asarray(target_q, dtype=np.float64)
        self.n_steps = int(n_steps)
        self._counter = 0
        self._q0: np.ndarray | None = None

    def on_activate(self, ctx: Controller) -> None:
        self._counter = 0
        state = ctx.interface.get_low_state()
        self._q0 = state[0, 7 : 7 + ctx.num_dofs].copy()

    def on_deactivate(self, ctx: Controller) -> None:
        self._counter = 0
        self._q0 = None

    def is_done(self) -> bool:
        return self._counter >= self.n_steps

    def apply_velocity(self, vc: VelCmd) -> None:
        return None

    def apply_command(self, cmd: StateCommand) -> bool:
        return False

    def act(self, ctx: Controller, state: np.ndarray) -> Command:
        n = ctx.num_dofs
        if self._q0 is None:
            self._q0 = state[7 : 7 + n].copy()
        alpha = min(self._counter / max(self.n_steps, 1), 1.0)
        self._counter += 1
        q = self._q0 + (self.target_q - self._q0) * alpha
        return Command(
            q=q + ctx.joint_offsets,
            dof_pos_latest=state[7 : 7 + n],
        )
