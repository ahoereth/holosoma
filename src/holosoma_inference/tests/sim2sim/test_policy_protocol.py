"""Step 8b protocol conformance tests."""

from __future__ import annotations

import numpy as np

from holosoma_inference.controllers.protocol import Command, PolicyProtocol
from tests.sim2sim.harness import build_policy_with_sim


def test_locomotion_policy_conforms_to_protocol():
    policy, _, _ = build_policy_with_sim()
    assert isinstance(policy, PolicyProtocol)
    assert policy.name == "locomotion"


def test_act_returns_command_with_correct_shape():
    policy, _, sim_interface = build_policy_with_sim()
    state = sim_interface.get_low_state()[0]
    cmd = policy.act(None, state)
    assert isinstance(cmd, Command)
    assert cmd.q.shape == (policy.num_dofs,)
    # dof_pos_latest helps the interface clamp velocity tracking
    assert cmd.dof_pos_latest is not None
    assert cmd.dof_pos_latest.shape == (policy.num_dofs,)


def test_apply_velocity_does_not_raise():
    from holosoma_inference.inputs.api.commands import VelCmd

    policy, _, _ = build_policy_with_sim()
    policy.apply_velocity(VelCmd(lin_vel=np.array([0.5, 0.0]), ang_vel=0.0))


def test_on_activate_on_deactivate_no_op_on_base():
    policy, ctl, _ = build_policy_with_sim()
    policy.on_activate(ctl)
    policy.on_deactivate(ctl)
