"""Dual-mode policy with runtime switching between two policy instances.

After Step 5 of the Controller refactor, DualModePolicy is a thin
swap object: it holds two BasePolicy instances that share the
Controller's hardware (interface, inputs) and flips
``controller.policy`` between them on SWITCH_MODE. There is no
separate run loop — Controller drives both.
"""

from __future__ import annotations

from loguru import logger
from termcolor import colored

from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.inputs.api.commands import StateCommand


def _select_policy_class(config: InferenceConfig):
    """Determine policy class based on observation config and robot type."""
    from holosoma_inference.compat import entry_points
    from holosoma_inference.policies.locomotion import LocomotionPolicy
    from holosoma_inference.policies.wbt import WholeBodyTrackingPolicy

    robot_type = config.robot.robot_type
    actor_obs = config.observation.obs_dict.get("actor_obs", [])

    if "motion_command" in actor_obs:
        for ep in entry_points(group="holosoma.policies.wbt"):
            if ep.name == robot_type:
                return ep.load()
        return WholeBodyTrackingPolicy

    for ep in entry_points(group="holosoma.policies.locomotion"):
        if ep.name == robot_type:
            return ep.load()
    return LocomotionPolicy


class DualModePolicy:
    """Holds two policies and swaps which one the Controller drives."""

    def __init__(
        self,
        primary_config: InferenceConfig,
        secondary_config: InferenceConfig,
        interface,
    ):
        primary_cls = _select_policy_class(primary_config)
        secondary_cls = _select_policy_class(secondary_config)

        logger.info(
            colored(
                f"Dual-mode: primary={primary_cls.__name__}, secondary={secondary_cls.__name__}",
                "magenta",
            )
        )

        self.primary = primary_cls(config=primary_config, interface=interface)
        logger.info(colored("Initializing secondary policy (shared hardware)...", "magenta"))
        self.secondary = secondary_cls(config=secondary_config, interface=interface)

        self.controller = None
        self.active_label = "primary"
        self._orig_dispatch: dict = {}

    @property
    def active(self):
        return self.primary if self.active_label == "primary" else self.secondary

    def bind_controller(self, controller) -> None:
        """Wire the Controller; intercept SWITCH_MODE on its command provider."""
        self.controller = controller
        controller.set_policy(self.primary)
        # Inject SWITCH_MODE into the shared command provider (joystick X / keyboard x)
        mapping = getattr(controller.command_provider, "_mapping", None)
        if mapping is not None:
            mapping["X"] = StateCommand.SWITCH_MODE
            mapping["x"] = StateCommand.SWITCH_MODE

        # Each policy keeps its own dispatch table. The Controller calls
        # the active policy's _dispatch_command. We intercept SWITCH_MODE
        # by replacing each policy's dispatch with a wrapper that defers
        # to the original.
        self._orig_dispatch = {
            id(self.primary): self.primary._dispatch_command,
            id(self.secondary): self.secondary._dispatch_command,
        }

        def patched(policy, cmd):
            if cmd == StateCommand.SWITCH_MODE:
                self._handle_mode_switch()
            else:
                self._orig_dispatch[id(policy)](cmd)

        self.primary._dispatch_command = lambda cmd: patched(self.primary, cmd)
        self.secondary._dispatch_command = lambda cmd: patched(self.secondary, cmd)

    def _handle_mode_switch(self):
        """Stop the active policy, swap in the inactive one."""
        active = self.active
        active._handle_stop_policy()

        target_label = "secondary" if self.active_label == "primary" else "primary"
        target = self.secondary if target_label == "secondary" else self.primary

        # Push the target policy's KP/KD onto the shared interface
        target._resolve_control_gains()

        # Carry over joystick key_states so edge detection doesn't see a false
        # rising edge on the X button (which is still physically held down).
        from holosoma_inference.inputs.impl.interface import InterfaceInput

        ctrl = self.controller
        if ctrl is not None and isinstance(ctrl.velocity_input, InterfaceInput):
            # Velocity input is shared between policies; nothing to copy.
            pass

        self.active_label = target_label
        if ctrl is not None:
            ctrl.set_policy(target)

        # Re-initialize phase and activate the new active policy.
        target._init_phase_components()
        target._handle_start_policy()

        logger.info(
            colored(
                f"Switched to {self.active_label} policy ({type(target).__name__})",
                "magenta",
                attrs=["bold"],
            )
        )

    def run(self) -> None:
        """Run the controller. The active policy may change mid-loop."""
        if self.controller is None:
            raise RuntimeError("DualModePolicy.run() called before bind_controller()")
        self.controller.run()
