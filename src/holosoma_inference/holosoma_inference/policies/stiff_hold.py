"""StiffHoldPolicy — hold a fixed pose with explicit KP/KD overrides.

Replaces the ``_stiff_hold_active`` flag on the WBT policy. Used as
the WBT startup state: hold the configured stiff-startup pose with
high KP/KD until the operator triggers the motion clip.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from holosoma_inference.controllers.protocol import Command

if TYPE_CHECKING:
    from holosoma_inference.controllers.controller import Controller
    from holosoma_inference.inputs.api.commands import StateCommand, VelCmd


class StiffHoldPolicy:
    """Hold a fixed ``q`` with overridden ``kp`` / ``kd`` gains."""

    name = "stiff_hold"

    def __init__(self, q, kp, kd):
        self.q = np.asarray(q, dtype=np.float64).reshape(-1)
        self.kp = np.asarray(kp, dtype=np.float64).reshape(-1)
        self.kd = np.asarray(kd, dtype=np.float64).reshape(-1)

    def on_activate(self, ctx: Controller) -> None:
        return None

    def on_deactivate(self, ctx: Controller) -> None:
        return None

    def apply_velocity(self, vc: VelCmd) -> None:
        return None

    def apply_command(self, cmd: StateCommand) -> bool:
        return False

    def act(self, ctx: Controller, state: np.ndarray) -> Command:
        n = ctx.num_dofs
        return Command(
            q=self.q + ctx.joint_offsets,
            kp_override=self.kp,
            kd_override=self.kd,
            dof_pos_latest=state[7 : 7 + n],
        )
