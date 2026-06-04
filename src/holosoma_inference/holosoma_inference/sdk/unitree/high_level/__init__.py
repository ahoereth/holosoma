"""G1 high-level clients (arm_sdk + loco), each runnable in an isolated process.

Convenience constructors wrap each client in an :class:`MPClientProxy` so the
unitree SDK's CycloneDDS stays out of the parent's (rclpy) address space.
"""

from __future__ import annotations

from typing import Any

from .multiprocess_proxy import MPClientProxy

_ARM = "holosoma_inference.sdk.unitree.high_level.arm_client.arm_client:G1j29ArmController"
_LOCO = "holosoma_inference.sdk.unitree.high_level.loco_client.loco_client:G1LocoClient"


def make_mp_arm_client(**kwargs: Any) -> MPClientProxy:
    """Arm controller in its own subprocess. kwargs -> G1j29ArmController."""
    return MPClientProxy(_ARM, **kwargs)


def make_mp_loco_client(**kwargs: Any) -> MPClientProxy:
    """Loco client in its own subprocess. kwargs -> G1LocoClient."""
    return MPClientProxy(_LOCO, **kwargs)


__all__ = ["MPClientProxy", "make_mp_arm_client", "make_mp_loco_client"]
