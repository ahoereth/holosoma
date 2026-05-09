"""Controller — orchestrates a BasePolicy with hardware and inputs.

Step 2 of the Controller refactor (see ``docs/controller-design.md``):
the Controller now owns hardware. It is constructed with an interface,
velocity/command input providers, and a rate limiter. The Policy keeps
``self.interface`` as a back-reference for the inner policy_action loop
but no longer creates or owns the hardware lifecycle.

ControllerState is still a read-only projection of the existing flags;
Step 3 makes it load-bearing.
"""

from __future__ import annotations

import itertools
from enum import Enum
from typing import TYPE_CHECKING

from loguru import logger as _default_logger

if TYPE_CHECKING:
    from holosoma_inference.inputs.api.base import StateCommandProvider, VelCmdProvider
    from holosoma_inference.policies.base import BasePolicy
    from holosoma_inference.sdk.base.base_interface import BaseInterface
    from holosoma_inference.utils.rate import RateLimiter


class ControllerState(Enum):
    """High-level Controller FSM states.

    Today these are read-only labels — the underlying flags
    (``use_policy_action``, ``get_ready_state``, etc.) still drive
    behavior. They become load-bearing in Step 3.
    """

    IDLE = "idle"
    INIT = "init"
    DAMP = "damp"
    STIFF_HOLD = "stiff_hold"
    RUN_POLICY = "run_policy"


class Controller:
    """Drives a BasePolicy through its rl_rate loop.

    Owns hardware (interface, inputs, rate). Holds a reference to the
    policy and exposes itself back to the policy so subclass dispatch
    handlers can reach the velocity input (e.g. ``zero()`` in
    LocomotionPolicy).
    """

    def __init__(
        self,
        policy: BasePolicy,
        interface: BaseInterface,
        velocity_input: VelCmdProvider,
        command_provider: StateCommandProvider,
        rate: RateLimiter,
        logger=None,
        use_joystick: bool = False,
        use_keyboard: bool = False,
    ):
        self.policy = policy
        self.interface = interface
        self.velocity_input = velocity_input
        self.command_provider = command_provider
        self.rate = rate
        self.logger = logger if logger is not None else _default_logger
        self.use_joystick = use_joystick
        self.use_keyboard = use_keyboard

        # DAMP state holds last-observed joint positions with the policy's
        # KP/KD gains. The intent is "robot stays energized at the pose it
        # was at when teleop released the handle" — not low gains, full
        # gains tracking a stationary target.
        self._damp_active = False
        self._damp_q = None
        self.damp_kp_scale = 1.0
        self.damp_kd_scale = 1.0

        # Two-way wiring: policy can call back into the controller for
        # operations like velocity_input.zero() that live on hardware.
        policy.controller = self
        # Policy keeps a reference to interface for its inner per-tick path
        # (read state, send command, KP/KD resolution).
        if not hasattr(policy, "interface") or policy.interface is None:
            policy.interface = interface

    @property
    def state(self) -> ControllerState:
        """Current Controller FSM state.

        Backed by the legacy flags on the policy
        (``use_policy_action`` / ``get_ready_state`` / ``_stiff_hold_active``)
        plus a DAMP override on the controller itself. Reads project the
        flag set onto a single state.
        """
        if self._damp_active:
            return ControllerState.DAMP
        if getattr(self.policy, "get_ready_state", False):
            return ControllerState.INIT
        if getattr(self.policy, "use_policy_action", False):
            return ControllerState.RUN_POLICY
        if getattr(self.policy, "_stiff_hold_active", False):
            return ControllerState.STIFF_HOLD
        return ControllerState.IDLE

    def set_state(self, new_state: ControllerState) -> None:
        """Transition the FSM. Updates the legacy flags atomically.

        ``DAMP`` is held by a controller-side flag; on entry, the current
        joint positions are captured for the hold target. Other states
        clear the damp flag.
        """
        policy = self.policy
        # Clear all flags first so transitions are idempotent.
        policy.use_policy_action = False
        policy.get_ready_state = False
        if hasattr(policy, "_stiff_hold_active"):
            policy._stiff_hold_active = False
        self._damp_active = False
        self._damp_q = None

        if new_state is ControllerState.RUN_POLICY:
            policy.use_policy_action = True
        elif new_state is ControllerState.INIT:
            policy.get_ready_state = True
            policy.init_count = 0
        elif new_state is ControllerState.STIFF_HOLD:
            if hasattr(policy, "_stiff_hold_active"):
                policy._stiff_hold_active = True
        elif new_state is ControllerState.DAMP:
            self._damp_active = True
            try:
                state = self.interface.get_low_state()
                self._damp_q = state[0, 7 : 7 + policy.num_dofs].copy()
            except Exception:
                self.logger.warning("DAMP entry: get_low_state() failed; will capture on first tick")

    def set_policy(self, policy: BasePolicy) -> None:
        """Swap the active policy. Used by dual-mode SWITCH_MODE in Step 5."""
        self.policy = policy
        policy.controller = self
        if not hasattr(policy, "interface") or policy.interface is None:
            policy.interface = self.interface

    def step(self) -> None:
        """Execute one rl_rate tick of the run loop body."""
        policy = self.policy
        policy.latency_tracker.start_cycle()

        vc = self.velocity_input.poll_velocity()
        if vc is not None:
            policy._apply_velocity(vc)

        commands = self.command_provider.poll_commands()
        for cmd in commands:
            policy._dispatch_command(cmd)
        if commands:
            policy._print_control_status()

        if self._damp_active:
            self._publish_damp_command()
        else:
            if policy.use_phase:
                policy.update_phase_time()
            policy.policy_action()

        policy.latency_tracker.end_cycle()

    def _publish_damp_command(self) -> None:
        """Hold last-observed joint positions with low KP/KD."""
        import numpy as np

        policy = self.policy
        state = self.interface.get_low_state()
        if self._damp_q is None:
            self._damp_q = state[0, 7 : 7 + policy.num_dofs].copy()
        kp_full = np.asarray(policy.robot_config.motor_kp, dtype=np.float64) * self.damp_kp_scale
        kd_full = np.asarray(policy.robot_config.motor_kd, dtype=np.float64) * self.damp_kd_scale
        zeros = np.zeros(policy.num_dofs)
        self.interface.send_low_command(
            self._damp_q + policy.joint_offsets,
            zeros,
            zeros,
            state[0, 7 : 7 + policy.num_dofs],
            kp_override=kp_full,
            kd_override=kd_full,
        )

    def run(self) -> None:
        """Run until KeyboardInterrupt."""
        try:
            for it in itertools.count():
                self.step()
                if it % 50 == 0 and self.policy.use_policy_action:
                    lt = self.policy.latency_tracker
                    debug_str = f"RL FPS: {lt.get_fps():.2f} | {lt.get_stats_str()}"
                    self.logger.info(debug_str, flush=True)
                self.rate.sleep()
        except KeyboardInterrupt:
            pass


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------
def build_default_hardware(config) -> tuple:
    """Build a real interface + input providers + rate from *config*.

    Used by ``run_policy.py``. Returns
    ``(interface, velocity_input, command_provider, rate, use_joystick, use_keyboard)``.
    Side-effect: starts each input provider.
    """
    import sys

    from holosoma_inference.inputs import create_input
    from holosoma_inference.sdk import create_interface
    from holosoma_inference.utils.rate import RateLimiter

    sources = {config.task.velocity_input, config.task.state_input}
    need_joystick_hw = bool({"interface", "joystick"} & sources)

    interface = create_interface(
        config.robot,
        config.task.domain_id,
        config.task.interface,
        need_joystick_hw,
    )

    use_joystick = need_joystick_hw and sys.platform != "darwin"
    use_keyboard = "keyboard" in sources
    if use_keyboard:
        from holosoma_inference.inputs.impl.keyboard import get_keyboard_listener

        listener = get_keyboard_listener()
        use_keyboard = listener.start()

    velocity_input = create_input(config.task.velocity_input, "velocity", interface, config, use_joystick)
    if config.task.velocity_input == config.task.state_input:
        command_provider = velocity_input
    else:
        command_provider = create_input(config.task.state_input, "command", interface, config, use_joystick)

    velocity_input.start()
    if command_provider is not velocity_input:
        command_provider.start()

    rate = RateLimiter(config.task.rl_rate)
    return interface, velocity_input, command_provider, rate, use_joystick, use_keyboard
