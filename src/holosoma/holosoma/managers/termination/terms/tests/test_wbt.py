from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from holosoma.config_values.wbt.g1.termination import g1_29dof_wbt_termination
from holosoma.managers.termination.terms.wbt import motion_ends

pytestmark = pytest.mark.no_sim


def test_motion_ends_uses_each_environment_selected_clip() -> None:
    motion_command = SimpleNamespace(
        time_steps=torch.tensor([8, 9, 18, 19]),
        motion_ids=torch.tensor([0, 0, 1, 1]),
        motion=SimpleNamespace(
            motion_end_idx=torch.tensor([10, 20]),
            time_step_total=20,
        ),
    )
    env = SimpleNamespace(
        command_manager=SimpleNamespace(
            get_state=lambda name: motion_command if name == "motion_command" else None,
        )
    )

    assert motion_ends(env).tolist() == [False, True, False, True]


def test_motion_ends_rejects_a_missing_command() -> None:
    env = SimpleNamespace(command_manager=SimpleNamespace(get_state=lambda _name: None))

    with pytest.raises(RuntimeError, match="no stateful term"):
        motion_ends(env)


def test_default_wbt_termination_treats_motion_end_as_timeout() -> None:
    motion_end = g1_29dof_wbt_termination.terms["motion_end"]

    assert motion_end.func == "holosoma.managers.termination.terms.wbt:motion_ends"
    assert motion_end.is_timeout
