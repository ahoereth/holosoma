from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from holosoma.agents.distillation.distillation import Distillation
from holosoma.agents.distillation_ppo.distillation_ppo import DistillationPPO
from holosoma.agents.modules.student_teacher_modules import StudentTeacher
from holosoma.train_agent import infer_num_teachers_from_checkpoint
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim


@pytest.mark.parametrize("algo_cls", [Distillation, DistillationPPO])
@pytest.mark.parametrize(("start_iteration", "expected"), [(1, True), (5, False)])
def test_warmup_uses_absolute_iteration(
    algo_cls: type[Distillation | DistillationPPO],
    start_iteration: int,
    expected: bool,
) -> None:
    algo = object.__new__(algo_cls)
    algo.current_learning_iteration = start_iteration
    algo.config = SimpleNamespace(
        num_learning_iterations=1,
        distillation_warmup_steps=3,
        save_interval=100,
    )
    algo._train_mode = lambda: None
    algo.policy = SimpleNamespace(require_loaded_teachers=lambda: None)
    algo.env = SimpleNamespace(reset_all=dict)
    algo.logging_helper = SimpleNamespace(
        record_collection_time=nullcontext,
        record_learn_time=nullcontext,
    )
    observed: list[bool] = []
    algo._rollout_step = lambda obs: observed.append(algo._warmup_active) or obs
    algo._training_step = dict
    algo.is_main_process = False
    algo.log_dir = None
    if algo_cls is DistillationPPO:
        algo.adjust_ppo_dagger_coeff = lambda _iteration: None
        algo._set_std_lr_from_ppo_coef = lambda: None

    algo_cls.learn(algo, num_learning_iterations=1)

    assert observed == [expected]


class _TinyPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.student = torch.nn.Linear(2, 1, bias=False)
        self.critic = torch.nn.Linear(2, 1, bias=False)
        torch.nn.init.zeros_(self.student.weight)
        torch.nn.init.zeros_(self.critic.weight)

    def _student_input(self, actor_obs: torch.Tensor, _depth_obs: torch.Tensor) -> torch.Tensor:
        return actor_obs

    def project_action_std(self) -> None:
        pass


def test_warmup_update_is_dagger_only() -> None:
    algo = object.__new__(DistillationPPO)
    algo.policy = _TinyPolicy()
    algo.config = SimpleNamespace(dagger_loss_coef=2.0, max_grad_norm=10.0)
    algo.distill_loss_fn = torch.nn.functional.mse_loss
    algo.optimizer = torch.optim.SGD(algo.policy.parameters(), lr=0.1)
    algo.is_multi_gpu = False
    algo._warmup_active = True
    critic_before = algo.policy.critic.weight.detach().clone()

    losses = algo._update_step(
        {
            "actor_obs": torch.tensor([[1.0, 1.0]]),
            "depth_obs": torch.zeros(1, 1),
            "critic_obs": torch.tensor([[1.0, 1.0]]),
            "actions": torch.zeros(1, 1),
            "teacher_actions": torch.ones(1, 1),
        }
    )

    assert not torch.equal(algo.policy.student.weight, torch.zeros_like(algo.policy.student.weight))
    torch.testing.assert_close(algo.policy.critic.weight, critic_before)
    assert losses["value_loss"].item() == 0.0
    assert losses["surrogate_loss"].item() == 0.0


def test_global_advantage_normalization_handles_uneven_shards(monkeypatch: pytest.MonkeyPatch) -> None:
    algo = object.__new__(DistillationPPO)
    local = torch.tensor([1.0, 2.0, 3.0])
    remote = torch.tensor([10.0, 20.0], dtype=torch.float64)

    def fake_all_reduce(stats: torch.Tensor, op: object) -> None:
        del op
        stats += torch.stack((remote.sum(), remote.square().sum(), remote.new_tensor(remote.numel())))

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    normalized = algo._normalize_advantages_multi_gpu(local)
    combined = torch.cat((local, remote.to(local.dtype)))
    expected = (local - combined.mean()) / (combined.std() + 1e-8)
    torch.testing.assert_close(normalized, expected)


def test_teacher_count_is_inferred_from_legacy_state_dict() -> None:
    checkpoint = {
        "model_state_dict": {
            "teachers.0.0.weight": torch.zeros(1),
            "teachers.1.0.weight": torch.zeros(1),
        }
    }

    assert infer_num_teachers_from_checkpoint(checkpoint) == 2


def test_teacher_count_is_inferred_from_single_teacher_state_dict() -> None:
    checkpoint = {"model_state_dict": {"teacher.0.weight": torch.zeros(1)}}

    assert infer_num_teachers_from_checkpoint(checkpoint) == 1


def test_mixed_teacher_key_formats_are_rejected() -> None:
    checkpoint = {
        "model_state_dict": {
            "teacher.0.weight": torch.zeros(1),
            "teachers.0.0.weight": torch.zeros(1),
        }
    }

    with pytest.raises(ValueError, match="mixes legacy"):
        infer_num_teachers_from_checkpoint(checkpoint)


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "2"])
def test_invalid_explicit_teacher_count_is_rejected(value: object) -> None:
    with pytest.raises(ValueError, match="Invalid num_teachers"):
        infer_num_teachers_from_checkpoint({"num_teachers": value})


def test_teacher_count_metadata_must_match_state_dict() -> None:
    checkpoint = {
        "num_teachers": 1,
        "model_state_dict": {
            "teachers.0.0.weight": torch.zeros(1),
            "teachers.1.0.weight": torch.zeros(1),
        },
    }

    with pytest.raises(ValueError, match="contains 2 teacher slots"):
        infer_num_teachers_from_checkpoint(checkpoint)


def test_non_contiguous_teacher_slots_are_rejected() -> None:
    checkpoint = {
        "model_state_dict": {
            "teachers.0.0.weight": torch.zeros(1),
            "teachers.2.0.weight": torch.zeros(1),
        }
    }

    with pytest.raises(ValueError, match="non-contiguous"):
        infer_num_teachers_from_checkpoint(checkpoint)


@pytest.mark.parametrize("algo_cls", [Distillation, DistillationPPO])
def test_legacy_checkpoint_does_not_certify_teacher_weights(
    algo_cls: type[Distillation | DistillationPPO],
    tmp_path,
) -> None:
    policy = StudentTeacher(
        num_actor_obs=2,
        num_teacher_obs=3,
        num_actions=1,
        student_hidden_dims=[2],
        teacher_hidden_dims=[2],
    )
    checkpoint = tmp_path / "legacy.pt"
    torch.save({"model_state_dict": policy.state_dict()}, checkpoint)

    algo = object.__new__(algo_cls)
    algo.device = "cpu"
    algo.policy = policy
    algo._restore_env_state = lambda _state: None
    algo_cls.load(algo, checkpoint)

    assert policy.loaded_teachers == (False,)
    with pytest.raises(RuntimeError, match="not loaded"):
        policy.require_loaded_teachers()
