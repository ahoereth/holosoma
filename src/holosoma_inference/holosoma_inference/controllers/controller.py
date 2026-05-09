"""Controller — orchestrates one or more policies sharing hardware.

After Step 8, the Controller holds a dict of policies (each conforming
to ``PolicyProtocol``) and one is "active" at a time. ``step()`` calls
the active policy's ``act(ctx, state)`` to get a ``Command`` and
publishes it via the SDK interface. ``transition_to(name)`` switches
the active policy.

The 5-state FSM (``IDLE / INIT / DAMP / STIFF_HOLD / RUN_POLICY``) is
no longer modelled as an enum — each former state is a concrete
policy:

  IDLE / DAMP   -> DampingPolicy
  INIT          -> InitPolicy
  STIFF_HOLD    -> StiffHoldPolicy
  RUN_POLICY    -> OnnxLocomotionPolicy / OnnxWBTPolicy / extension policies

``ControllerState`` remains as a legacy alias mapping enum members back
to the canonical policy names; nothing in the new code path uses it.
"""

from __future__ import annotations

import itertools
import sys
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger as _default_logger

from holosoma_inference.controllers.protocol import Command, PolicyProtocol
from holosoma_inference.inputs.api.commands import StateCommand

if TYPE_CHECKING:
    from holosoma_inference.config.config_types.inference import InferenceConfig
    from holosoma_inference.inputs.api.base import StateCommandProvider, VelCmdProvider
    from holosoma_inference.sdk.base.base_interface import BaseInterface
    from holosoma_inference.utils.rate import RateLimiter


# ---------------------------------------------------------------------------
# Legacy state enum — kept for one release cycle for callers that still
# reference ControllerState.RUN_POLICY etc. New code uses policy names.
# ---------------------------------------------------------------------------
class ControllerState(Enum):
    IDLE = "idle"
    INIT = "init"
    DAMP = "damping"
    STIFF_HOLD = "stiff_hold"
    RUN_POLICY = "run_policy"

    @property
    def policy_name(self) -> str:
        # IDLE collapses to damping; the others map 1:1 to policy keys.
        return "damping" if self is ControllerState.IDLE else self.value


class Controller:
    """Drives one of N policies through the rl_rate loop."""

    def __init__(
        self,
        policies: dict[str, PolicyProtocol],
        initial: str,
        *,
        interface: BaseInterface,
        velocity_input: VelCmdProvider,
        command_provider: StateCommandProvider,
        rate: RateLimiter,
        robot_config,
        joint_offsets: np.ndarray,
        latency_tracker=None,
        logger=None,
        use_joystick: bool = False,
        use_keyboard: bool = False,
        default_run_policy: str | None = None,
    ):
        if initial not in policies:
            raise ValueError(f"initial policy {initial!r} not in policies dict {list(policies)}")

        self.policies: dict[str, PolicyProtocol] = dict(policies)
        self._active_name = initial
        self.interface = interface
        self.velocity_input = velocity_input
        self.command_provider = command_provider
        self.rate = rate
        self._robot_config = robot_config
        self._joint_offsets = np.asarray(joint_offsets, dtype=np.float64)
        self.latency_tracker = latency_tracker
        self.logger = logger if logger is not None else _default_logger
        self.use_joystick = use_joystick
        self.use_keyboard = use_keyboard

        # Which policy START maps to. Defaults to the first registered
        # policy that isn't damping/init/stiff_hold.
        self._default_run_policy = default_run_policy or self._infer_default_run_policy()

        # Two-way wiring so policy.controller works for legacy code.
        for p in self.policies.values():
            if hasattr(p, "controller"):
                p.controller = self

        self.active.on_activate(self)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def active(self) -> PolicyProtocol:
        return self.policies[self._active_name]

    @property
    def active_name(self) -> str:
        return self._active_name

    def transition_to(self, name: str) -> None:
        if name == self._active_name:
            return
        if name not in self.policies:
            raise KeyError(f"unknown policy {name!r}; have {list(self.policies)}")
        self.active.on_deactivate(self)
        self._active_name = name
        self.active.on_activate(self)
        self.logger.info("Active policy: {}", name)

    @property
    def num_dofs(self) -> int:
        return int(self._robot_config.num_joints)

    @property
    def motor_kp(self) -> np.ndarray:
        return np.asarray(self._robot_config.motor_kp, dtype=np.float64)

    @property
    def motor_kd(self) -> np.ndarray:
        return np.asarray(self._robot_config.motor_kd, dtype=np.float64)

    @property
    def joint_offsets(self) -> np.ndarray:
        return self._joint_offsets

    # ------------------------------------------------------------------
    # Per-tick driver
    # ------------------------------------------------------------------
    def step(self) -> None:
        if self.latency_tracker is not None:
            self.latency_tracker.start_cycle()

        vc = self.velocity_input.poll_velocity()
        if vc is not None:
            self.active.apply_velocity(vc)

        commands = self.command_provider.poll_commands()
        for cmd in commands:
            self.dispatch(cmd)
        if commands:
            self._print_status()

        state = self.interface.get_low_state()[0]
        command = self.active.act(self, state)
        self._send(command)

        if self.latency_tracker is not None:
            self.latency_tracker.end_cycle()

    def dispatch(self, cmd: StateCommand) -> None:
        """Route a StateCommand to the active policy then to builtin handlers."""
        if self.active.apply_command(cmd):
            return
        self._builtin_dispatch(cmd)

    def run(self) -> None:
        try:
            for it in itertools.count():
                self.step()
                if self.latency_tracker is not None and it % 50 == 0:
                    fps = self.latency_tracker.get_fps()
                    if fps > 0:
                        debug_str = f"RL FPS: {fps:.2f} | {self.latency_tracker.get_stats_str()}"
                        self.logger.info(debug_str)
                self.rate.sleep()
        except KeyboardInterrupt:
            pass

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _builtin_dispatch(self, cmd: StateCommand) -> None:
        if cmd is StateCommand.START:
            self.transition_to(self._default_run_policy)
        elif cmd is StateCommand.STOP:
            if "damping" in self.policies:
                self.transition_to("damping")
            else:
                self.logger.warning("STOP requested but no 'damping' policy registered")
        elif cmd is StateCommand.INIT:
            if "init" in self.policies:
                self.transition_to("init")
            else:
                self.logger.warning("INIT requested but no 'init' policy registered")
        elif cmd is StateCommand.DAMP:
            if "damping" in self.policies:
                self.transition_to("damping")
            else:
                self.logger.warning("DAMP requested but no 'damping' policy registered")
        elif cmd is StateCommand.KILL:
            self.logger.info("Kill command received")
            sys.exit(0)
        elif cmd is StateCommand.SWITCH_MODE:
            self._cycle_run_policies()
        elif cmd is StateCommand.NEXT_POLICY or cmd in _MULTI_MODEL_COMMANDS:
            # Multi-model select on the active policy.
            if hasattr(self.active, "_dispatch_command"):
                self.active._dispatch_command(cmd)
        elif cmd in (
            StateCommand.KP_UP,
            StateCommand.KP_DOWN,
            StateCommand.KP_UP_FINE,
            StateCommand.KP_DOWN_FINE,
            StateCommand.KP_RESET,
        ):
            self._adjust_kp(cmd)

    def _cycle_run_policies(self) -> None:
        """SWITCH_MODE: cycle between the non-utility policies."""
        run_policies = [n for n in self.policies if n not in {"damping", "init", "stiff_hold"}]
        if len(run_policies) < 2:
            return
        if self._active_name in run_policies:
            i = (run_policies.index(self._active_name) + 1) % len(run_policies)
        else:
            i = 0
        self.transition_to(run_policies[i])

    def _adjust_kp(self, cmd: StateCommand) -> None:
        if not hasattr(self.interface, "kp_level"):
            return
        delta = {
            StateCommand.KP_UP: 0.1,
            StateCommand.KP_DOWN: -0.1,
            StateCommand.KP_UP_FINE: 0.01,
            StateCommand.KP_DOWN_FINE: -0.01,
        }
        if cmd is StateCommand.KP_RESET:
            self.interface.kp_level = 1.0
        else:
            self.interface.kp_level += delta[cmd]

    def _print_status(self) -> None:
        if hasattr(self.active, "_print_control_status"):
            self.active._print_control_status()

    def _send(self, c: Command) -> None:
        n = self.num_dofs
        zeros = np.zeros(n)
        self.interface.send_low_command(
            c.q,
            c.dq if c.dq is not None else zeros,
            c.tau if c.tau is not None else zeros,
            c.dof_pos_latest,
            kp_override=c.kp_override,
            kd_override=c.kd_override,
        )

    def _infer_default_run_policy(self) -> str:
        for name in self.policies:
            if name not in {"damping", "init", "stiff_hold"}:
                return name
        return next(iter(self.policies))

    # ------------------------------------------------------------------
    # Legacy single-policy compatibility
    # ------------------------------------------------------------------
    @classmethod
    def from_single_policy(
        cls,
        policy: PolicyProtocol,
        *,
        interface,
        velocity_input,
        command_provider,
        rate,
        latency_tracker=None,
        logger=None,
        use_joystick: bool = False,
        use_keyboard: bool = False,
    ) -> Controller:
        """Build a Controller from a single OnnxBasePolicy plus default
        damping/init policies. Used by run_policy.py and the harness.
        """
        from holosoma_inference.policies.damping import DampingPolicy
        from holosoma_inference.policies.init_ramp import InitPolicy

        policies: dict[str, PolicyProtocol] = {
            policy.name: policy,
            "damping": DampingPolicy(),
            "init": InitPolicy(target_q=policy.default_dof_angles),
        }
        return cls(
            policies=policies,
            initial=policy.name,
            interface=interface,
            velocity_input=velocity_input,
            command_provider=command_provider,
            rate=rate,
            robot_config=policy.robot_config,
            joint_offsets=policy.joint_offsets,
            latency_tracker=latency_tracker if latency_tracker is not None else policy.latency_tracker,
            logger=logger,
            use_joystick=use_joystick,
            use_keyboard=use_keyboard,
            default_run_policy=policy.name,
        )

    @property
    def policy(self) -> PolicyProtocol:
        """Legacy alias for ``active``. New code uses ``controller.active``."""
        return self.active


# Module-level set of multi-model select commands.
_MULTI_MODEL_COMMANDS = frozenset(StateCommand[f"SWITCH_POLICY_{n}"] for n in range(1, 10))


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------
def build_default_hardware(config: InferenceConfig) -> tuple:
    """Build a real interface + input providers + rate from *config*.

    Returns ``(interface, velocity_input, command_provider, rate, use_joystick, use_keyboard)``.
    Side-effect: starts each input provider.
    """
    import sys as _sys

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

    use_joystick = need_joystick_hw and _sys.platform != "darwin"
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
