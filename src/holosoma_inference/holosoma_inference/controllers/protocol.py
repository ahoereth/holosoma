"""PolicyProtocol — the contract every policy implements.

A policy is anything that maps robot state to a low-level command.
Locomotion ONNX, WBT ONNX, damping (hold last pose), init ramp,
WBT stiff-hold startup — they're all the same kind of object.

The protocol has five members:

  * ``act(ctx, state) -> Command``        — the hot path, called every tick
  * ``on_activate(ctx)``                  — called when this policy becomes active
  * ``on_deactivate(ctx)``                — called when another policy takes over
  * ``apply_velocity(vc)``                — VelCmd side-channel
  * ``apply_command(cmd) -> bool``        — StateCommand side-channel; True if handled

``Command`` is the dataclass returned from ``act`` — same fields as
``BaseInterface.send_low_command`` reified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from holosoma_inference.controllers.controller import Controller
    from holosoma_inference.inputs.api.commands import StateCommand, VelCmd


@dataclass
class Command:
    """Low-level command returned from ``PolicyProtocol.act``.

    ``q`` is required; everything else is optional. The Controller
    fills in zeros for ``dq`` / ``tau`` if not supplied. ``kp_override``
    / ``kd_override`` cause the interface to use those gains instead of
    the configured ``motor_kp`` / ``motor_kd``.
    """

    q: np.ndarray
    dq: np.ndarray | None = None
    tau: np.ndarray | None = None
    kp_override: np.ndarray | None = None
    kd_override: np.ndarray | None = None
    dof_pos_latest: np.ndarray | None = None


@runtime_checkable
class PolicyProtocol(Protocol):
    """Maps robot state to a low-level command."""

    name: str

    def act(self, ctx: Controller, state: np.ndarray) -> Command:
        """One tick of the control loop."""
        ...

    def on_activate(self, ctx: Controller) -> None:
        """Called when this policy becomes the Controller's active one."""
        ...

    def on_deactivate(self, ctx: Controller) -> None:
        """Called when another policy takes over."""
        ...

    def apply_velocity(self, vc: VelCmd) -> None:
        """Process a velocity command (one of two side-channels)."""
        ...

    def apply_command(self, cmd: StateCommand) -> bool:
        """Process a discrete command. Returns True if handled, False to
        fall through to the Controller's builtin dispatch."""
        ...
