"""Deprecated import path — use ``holosoma_inference.controllers``.

This module exists for one release cycle so external callers can update
their imports. New code should ``from holosoma_inference.controllers
import Controller, PolicyProtocol, Command``.
"""

from __future__ import annotations

import warnings

from holosoma_inference.controllers import (
    Command,
    Controller,
    ControllerState,
    PolicyProtocol,
    build_default_hardware,
)

warnings.warn(
    "holosoma_inference.controller is deprecated; import from holosoma_inference.controllers instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "Command",
    "Controller",
    "ControllerState",
    "PolicyProtocol",
    "build_default_hardware",
]
