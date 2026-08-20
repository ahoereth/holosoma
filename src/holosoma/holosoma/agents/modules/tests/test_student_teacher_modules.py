from __future__ import annotations

import math

import pytest

from holosoma.agents.modules.student_teacher_modules import MIN_ACTION_STD, StudentTeacher
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim


def _module(*, num_teachers: int = 1, init_noise_std: float = 0.1) -> StudentTeacher:
    return StudentTeacher(
        num_actor_obs=2,
        num_teacher_obs=3,
        num_actions=2,
        student_hidden_dims=[4],
        teacher_hidden_dims=[4],
        init_noise_std=init_noise_std,
        num_teachers=num_teachers,
    )


def _load_constant_teachers(module: StudentTeacher) -> None:
    for index, teacher in enumerate(module.teachers):
        state = {key: torch.zeros_like(value) for key, value in teacher.state_dict().items()}
        state[f"{len(teacher) - 1}.bias"].fill_(index + 1)
        module.load_teacher_state_dict(state, teacher_index=index)


def test_teacher_action_requires_loaded_weights() -> None:
    module = _module()

    with pytest.raises(RuntimeError, match="not loaded"):
        module.teacher_act(torch.zeros(2, 3))


def test_partial_non_strict_load_does_not_mark_teacher_ready() -> None:
    module = _module()

    module.load_teacher_state_dict({}, strict=False)

    assert module.loaded_teachers == (False,)
    with pytest.raises(RuntimeError, match="not loaded"):
        module.require_loaded_teachers()


def test_multi_teacher_routing_and_no_teacher_sentinel() -> None:
    module = _module(num_teachers=2)
    _load_constant_teachers(module)

    actions = module.teacher_act(
        torch.zeros(3, 3),
        torch.tensor([0, 1, -1]),
    )

    torch.testing.assert_close(
        actions,
        torch.tensor([[1.0, 1.0], [2.0, 2.0], [0.0, 0.0]]),
    )


def test_multiple_teachers_require_explicit_routing() -> None:
    module = _module(num_teachers=2)
    _load_constant_teachers(module)

    with pytest.raises(ValueError, match="teacher_idx is required"):
        module.teacher_act(torch.zeros(2, 3))


@pytest.mark.parametrize(
    ("teacher_idx", "error", "message"),
    [
        (torch.tensor([[0], [1]]), ValueError, "must have shape"),
        (torch.tensor([True, False]), TypeError, "integer-valued"),
        (torch.tensor([0.0, 0.5]), ValueError, "finite integer-valued"),
        (torch.tensor([0, 2]), IndexError, "invalid values"),
        (torch.tensor([-2, 0]), IndexError, "invalid values"),
    ],
)
def test_invalid_teacher_routing_fails_loudly(
    teacher_idx: torch.Tensor,
    error: type[Exception],
    message: str,
) -> None:
    module = _module(num_teachers=2)
    _load_constant_teachers(module)

    with pytest.raises(error, match=message):
        module.teacher_act(torch.zeros(2, 3), teacher_idx)


def test_teacher_observation_shape_is_exact() -> None:
    module = _module()
    _load_constant_teachers(module)

    with pytest.raises(ValueError, match=r"shape \[N, 3\]"):
        module.teacher_act(torch.zeros(2, 4))


def test_action_std_is_projected_positive() -> None:
    module = _module(init_noise_std=-1.0)

    torch.testing.assert_close(module.std, torch.full((2,), MIN_ACTION_STD))
    module.update_distribution(torch.zeros(3, 2))
    assert torch.all(module.action_std > 0)


def test_non_finite_action_std_is_rejected() -> None:
    with pytest.raises(FloatingPointError, match="non-finite"):
        _module(init_noise_std=math.nan)

    module = _module()
    with torch.no_grad():
        module.std.fill_(math.inf)
    with pytest.raises(FloatingPointError, match="non-finite"):
        module.update_distribution(torch.zeros(1, 2))


def test_legacy_single_teacher_state_dict_is_upgraded() -> None:
    source = _module()
    legacy_state = {
        f"teacher.{key[len('teachers.0.') :]}" if key.startswith("teachers.0.") else key: value
        for key, value in source.state_dict().items()
    }
    target = _module()

    target.load_training_state_dict(legacy_state)

    for key, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[key], value)


def test_mixed_teacher_state_dict_keys_are_rejected() -> None:
    module = _module()
    state = dict(module.state_dict())
    state["teacher.0.weight"] = state["teachers.0.0.weight"]

    with pytest.raises(ValueError, match="mixes legacy"):
        module.load_training_state_dict(state)


def test_teacher_readiness_metadata_is_strict() -> None:
    module = _module(num_teachers=2)

    with pytest.raises(ValueError, match="Expected readiness state"):
        module.set_loaded_teachers([True])
    with pytest.raises(TypeError, match="only booleans"):
        module.set_loaded_teachers([1, 0])
