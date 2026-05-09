"""Step 4 DAMP state tests.

DAMP holds the last observed joint positions with low KP/KD. The robot
must remain energized — pelvis stays up, joints don't go slack — and
must remain roughly stationary (no commanded motion).
"""

from __future__ import annotations

import numpy as np

from holosoma_inference.controller import ControllerState
from holosoma_inference.inputs.api.commands import StateCommand
from tests.sim2sim.harness import build_policy_with_sim


def test_damp_holds_pose():
    """After entering DAMP, pelvis stays above 0.6 m for 2 s of sim time."""
    _, controller, sim_interface = build_policy_with_sim()

    # Step a few ticks in IDLE to settle the model.
    for _ in range(5):
        controller.step()

    # Enter DAMP — capture current joint positions as the hold target.
    controller.policy._dispatch_command(StateCommand.DAMP)
    assert controller.state is ControllerState.DAMP
    assert controller._damp_q is not None

    # Drive the loop for 2 seconds of sim time at 50 Hz.
    z_min = sim_interface.pelvis_height
    for _ in range(100):
        controller.step()
        z_min = min(z_min, sim_interface.pelvis_height)

    assert z_min > 0.6, f"Pelvis dropped below 0.6 m in DAMP: min={z_min:.3f} m"


def test_damp_to_run_transition_clears_damp_flag():
    """Entering DAMP then RUN_POLICY must clear the damp flag and resume policy."""
    _, controller, _ = build_policy_with_sim()

    controller.set_state(ControllerState.DAMP)
    assert controller._damp_active is True

    controller.set_state(ControllerState.RUN_POLICY)
    assert controller._damp_active is False
    assert controller.state is ControllerState.RUN_POLICY


def test_damp_q_captured_from_interface():
    """DAMP entry captures current joint state, not default pose."""
    policy, controller, sim_interface = build_policy_with_sim()

    # Push joints to a non-default pose so we can detect the capture.
    sim_interface.data.qpos[7 : 7 + 29] += 0.1
    import mujoco

    mujoco.mj_forward(sim_interface.model, sim_interface.data)

    controller.set_state(ControllerState.DAMP)
    assert controller._damp_q is not None
    assert not np.allclose(controller._damp_q, np.array(policy.config.robot.default_dof_angles))
