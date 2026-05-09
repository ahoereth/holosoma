"""Step 1 Controller tests — verify the no-op extraction.

Originally written when Controller.state was a flag-projection enum;
after Step 8 the equivalent fact is ``controller.active_name`` /
``controller.active``. These tests still gate the harness's basic
"build a controller, step it once" path.
"""

from __future__ import annotations

from tests.sim2sim.harness import build_policy_with_sim


def test_controller_active_reflects_initial_policy():
    policy, ctl, _ = build_policy_with_sim()
    assert ctl.active_name == "locomotion"
    assert ctl.active is policy


def test_controller_step_advances_one_cycle():
    _, ctl, sim_interface = build_policy_with_sim()
    z_before = sim_interface.pelvis_height
    ctl.step()
    z_after = sim_interface.pelvis_height
    assert abs(z_after - z_before) < 0.05
