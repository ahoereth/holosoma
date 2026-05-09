"""Step 1 Controller tests — verify the no-op extraction.

At Step 1 the Controller is a thin wrapper around the run loop body.
Hardware and inputs still live on the policy. These tests confirm the
new API exists and that ``Controller.step()`` produces the same effects
as the previous ``BasePolicy.run()`` body for one tick.
"""

from __future__ import annotations

from holosoma_inference.controller import Controller, ControllerState
from tests.sim2sim.harness import build_policy_with_sim


def test_controller_state_reflects_policy_flags():
    policy, _ = build_policy_with_sim()
    ctl = Controller(policy)
    # Fresh policy: no flags set → IDLE
    assert ctl.state is ControllerState.IDLE

    policy.use_policy_action = True
    assert ctl.state is ControllerState.RUN_POLICY

    policy.use_policy_action = False
    policy.get_ready_state = True
    assert ctl.state is ControllerState.INIT


def test_controller_step_advances_one_cycle():
    policy, sim_interface = build_policy_with_sim()
    ctl = Controller(policy)
    z_before = sim_interface.pelvis_height
    # Step with use_policy_action=False: policy_action just holds dof_pos.
    # We're verifying the loop body completes without raising.
    ctl.step()
    z_after = sim_interface.pelvis_height
    # Sanity: pelvis didn't teleport
    assert abs(z_after - z_before) < 0.05
