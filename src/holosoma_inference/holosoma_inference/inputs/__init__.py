from __future__ import annotations

from typing import TYPE_CHECKING

from holosoma_inference.config.config_types.task import InputSource
from holosoma_inference.inputs.api.base import StateCommandProvider, VelCmdProvider
from holosoma_inference.inputs.impl.interface import InterfaceInput
from holosoma_inference.inputs.impl.keyboard import KEYBOARD_VELOCITY_LOCOMOTION, KeyboardInput
from holosoma_inference.inputs.impl.ros2 import Ros2Input

if TYPE_CHECKING:
    from holosoma_inference.config.config_types.inference import InferenceConfig
    from holosoma_inference.sdk.base.base_interface import BaseInterface


def create_input(
    source: InputSource,
    role: str,
    interface: BaseInterface,
    config: InferenceConfig,
    use_joystick: bool,
) -> VelCmdProvider | StateCommandProvider:
    """Create an input provider for *source* and *role* ("velocity" or "command").

    Hardware ownership lives on the Controller, so the factory takes the
    interface and InferenceConfig directly rather than a Policy
    reference.
    """
    if not use_joystick and source in ("interface", "joystick"):
        source = "keyboard"

    if source in ("interface", "joystick"):
        return InterfaceInput(interface)

    if source == "keyboard":
        vel_keys = KEYBOARD_VELOCITY_LOCOMOTION if role == "velocity" else None
        return KeyboardInput.create(velocity_keys=vel_keys)

    if source == "ros2":
        return Ros2Input(
            config.task.ros_cmd_vel_topic,
            config.task.ros_state_input_topic,
            vel_timeout=config.task.ros_vel_timeout,
        )

    raise ValueError(f"Unknown input source: {source}")
