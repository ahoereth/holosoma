"""Controller — orchestrates a BasePolicy with hardware and inputs.

This is Step 1 of the Controller refactor described in
``docs/controller-design.md``. At this step the Controller owns the run
loop only — hardware (``interface``, ``_velocity_input``,
``_command_provider``, ``rate``, ``latency_tracker``) still lives on the
Policy. Subsequent steps move ownership.

ControllerState is exported here so call sites in tests and (eventually)
``run_policy.py`` can target the new API even before later steps move
the underlying flags.
"""

from __future__ import annotations

import itertools
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from holosoma_inference.policies.base import BasePolicy


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

    Step 1 contract: hardware ownership remains on the policy. The
    Controller is a thin wrapper around the loop body that previously
    lived in ``BasePolicy.run()``. Behavior must match the previous
    implementation exactly.
    """

    def __init__(self, policy: BasePolicy):
        self.policy = policy

    @property
    def state(self) -> ControllerState:
        """Best-effort projection of policy flags onto ControllerState.

        Read-only at Step 1. Reflects the existing flag-based FSM.
        """
        if getattr(self.policy, "get_ready_state", False):
            return ControllerState.INIT
        if getattr(self.policy, "use_policy_action", False):
            return ControllerState.RUN_POLICY
        if getattr(self.policy, "_stiff_hold_active", False):
            return ControllerState.STIFF_HOLD
        return ControllerState.IDLE

    def step(self) -> None:
        """Execute one rl_rate tick of the run loop body.

        Equivalent to the per-iteration body of ``BasePolicy.run()``,
        minus iteration counting and FPS logging.
        """
        policy = self.policy
        policy.latency_tracker.start_cycle()

        vc = policy._velocity_input.poll_velocity()
        if vc is not None:
            policy._apply_velocity(vc)

        commands = policy._command_provider.poll_commands()
        for cmd in commands:
            policy._dispatch_command(cmd)
        if commands:
            policy._print_control_status()

        if policy.use_phase:
            policy.update_phase_time()

        policy.policy_action()

        policy.latency_tracker.end_cycle()

    def run(self) -> None:
        """Run until KeyboardInterrupt."""
        policy = self.policy
        try:
            for it in itertools.count():
                self.step()
                if it % 50 == 0 and policy.use_policy_action:
                    debug_str = (
                        f"RL FPS: {policy.latency_tracker.get_fps():.2f} | {policy.latency_tracker.get_stats_str()}"
                    )
                    policy.logger.info(debug_str, flush=True)
                policy.rate.sleep()
        except KeyboardInterrupt:
            pass
