"""Step 4 DAMP tests rewritten for Step 8.

DAMP is now a concrete policy (``DampingPolicy``). The Controller's
``transition_to("damping")`` activates it; the policy captures
``q_hold`` lazily on the first ``act()`` call.
"""

from __future__ import annotations

import numpy as np

from holosoma_inference.policies.damping import DampingPolicy
from tests.sim2sim.harness import build_policy_with_sim


def _drain_initial_commands(controller):
    """Pop the harness stub's queued [START, STAND_TOGGLE] so subsequent
    transitions stick. The stub emits these once on the first poll to
    set the locomotion test up; damping tests don't need them."""
    controller.command_provider.poll_commands()


def test_damp_holds_pose():
    """After entering DAMP, pelvis stays above 0.6 m for 2 s of sim time."""
    _, controller, sim_interface = build_policy_with_sim()

    # Step a few ticks so the model settles into walking.
    for _ in range(5):
        controller.step()

    controller.transition_to("damping")
    assert controller.active_name == "damping"
    assert isinstance(controller.active, DampingPolicy)

    # Drive the loop for 2 seconds of sim time at 50 Hz.
    z_min = sim_interface.pelvis_height
    for _ in range(100):
        controller.step()
        z_min = min(z_min, sim_interface.pelvis_height)

    assert z_min > 0.6, f"Pelvis dropped below 0.6 m in DAMP: min={z_min:.3f} m"


def test_damp_to_run_transition_resets_capture():
    """Transitioning out of damping clears the q_hold so re-entry recaptures."""
    _, controller, sim_interface = build_policy_with_sim()
    _drain_initial_commands(controller)

    controller.transition_to("damping")
    damper = controller.active
    assert isinstance(damper, DampingPolicy)

    # Call act directly so we don't have to drive the full step pipeline.
    state = sim_interface.get_low_state()[0]
    damper.act(controller, state)
    assert damper._q_hold is not None

    controller.transition_to("locomotion")
    assert damper._q_hold is None  # on_deactivate clears it


def test_damp_q_captured_from_interface():
    """Damping captures current joint state on the first act(), not default pose."""
    policy, controller, sim_interface = build_policy_with_sim()
    _drain_initial_commands(controller)

    # Push joints to a non-default pose so we can detect the capture.
    sim_interface.data.qpos[7 : 7 + 29] += 0.1
    import mujoco

    mujoco.mj_forward(sim_interface.model, sim_interface.data)

    controller.transition_to("damping")
    damper = controller.active
    assert isinstance(damper, DampingPolicy)
    state = sim_interface.get_low_state()[0]
    damper.act(controller, state)
    assert damper._q_hold is not None
    assert not np.allclose(damper._q_hold, np.array(policy.config.robot.default_dof_angles))
