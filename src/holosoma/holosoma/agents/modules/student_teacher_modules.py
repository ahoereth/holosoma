"""Student/teacher modules for distillation algorithms.

Parallels ``ppo_modules.py`` (``PPOActor``/``PPOCritic``). Produced by the
distillation algorithm's ``setup()``. Weights for the teacher MLP are loaded
from a standalone PPO-actor checkpoint and frozen; the student MLP is trained
against teacher actions (DAgger) plus PPO advantages via the critic.

Obs layout (matches the distillation rollout storage):
- ``actor_obs``: proprioception only, consumed by the student.
- ``depth_obs``: raw depth image, consumed by the depth CNN backbone whose
  latent is concatenated onto ``actor_obs`` before the student MLP.
- ``teacher_obs``: privileged obs seen by the frozen teacher (same layout as
  what the teacher's original PPO training saw in its actor group).
- ``critic_obs``: privileged obs seen by the critic.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
from torch import nn
from torch.distributions import Normal


def _build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int, activation: str) -> nn.Sequential:
    """Build a plain MLP: [input_dim] -> hidden_dims... -> output_dim with the given activation between hidden layers."""
    act_cls = getattr(nn, activation)
    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(act_cls())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class StudentTeacher(nn.Module):
    """Proprio-only student + one or more frozen privileged teachers.

    Student consumes ``actor_obs`` (+ optional depth latent from a subclass).
    Each teacher consumes the full ``teacher_obs`` and has its weights loaded
    via :meth:`load_teacher_state_dict`, then frozen (``eval()`` + no grad).

    Routing across teachers is driven by an explicit ``teacher_idx`` argument
    to :meth:`teacher_act` rather than a reserved column of the observation:
    where that index comes from, and what counts as "no teacher", are decisions
    for the application. See :meth:`teacher_act`.
    """

    def __init__(
        self,
        num_actor_obs: int,
        num_teacher_obs: int,
        num_actions: int,
        student_hidden_dims: list[int],
        teacher_hidden_dims: list[int],
        activation: str = "ELU",
        init_noise_std: float = 0.01,
        num_teachers: int = 1,
    ):
        super().__init__()
        if num_teachers < 1:
            raise ValueError(f"num_teachers must be >= 1, got {num_teachers}")
        self.num_actor_obs = num_actor_obs
        self.num_teacher_obs = num_teacher_obs
        self.num_actions = num_actions
        # Each teacher MLP sees the whole teacher_obs. An application that
        # smuggles routing metadata through the observation is responsible for
        # stripping it before calling teacher_act, and for declaring
        # num_teacher_obs as the width its teachers actually expect.
        self._teacher_in_dim = num_teacher_obs

        self.student = _build_mlp(num_actor_obs, student_hidden_dims, num_actions, activation)
        self.teachers = nn.ModuleList(
            [
                _build_mlp(self._teacher_in_dim, teacher_hidden_dims, num_actions, activation)
                for _ in range(num_teachers)
            ]
        )

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

        self._loaded_teacher = [False] * num_teachers
        for t in self.teachers:
            t.eval()
            for p in t.parameters():
                p.requires_grad_(False)

        print(f"Student MLP: {self.student}")
        print(f"Teachers ({num_teachers}x): {self.teachers[0]}")

    # Keep ``policy.teacher`` as a legacy alias so existing call sites like
    # ``algo._train_mode`` / ``algo.load`` (which defensively do
    # ``self.policy.teacher.eval()`` / iterate parameters) keep working. New
    # code should iterate ``self.teachers`` instead.
    @property
    def teacher(self) -> nn.Module:
        return self.teachers

    @property
    def loaded_teacher(self) -> bool:
        return all(self._loaded_teacher)

    @property
    def num_teachers(self) -> int:
        return len(self.teachers)

    def load_teacher_state_dict(self, ckpt: dict, strict: bool = True, teacher_index: int = 0) -> None:
        """Load a teacher-PPO checkpoint into ``self.teachers[teacher_index]``.

        Accepts either a raw state dict or a holosoma algo checkpoint dict.
        Renames ``actor.*`` / ``actor_module.module.*`` keys to ``<bare>`` so
        they match a plain ``nn.Sequential`` layout.
        """
        if teacher_index < 0 or teacher_index >= len(self.teachers):
            raise IndexError(
                f"teacher_index={teacher_index} out of range for {len(self.teachers)} teachers"
            )

        state: dict = ckpt
        # unwrap common holosoma/rsl_rl ckpt wrappers. holosoma PPO saves as
        # {'actor_model_state_dict': ..., 'critic_model_state_dict': ...}; other
        # tooling stores a nested 'model_state_dict' / 'model' / 'state_dict'.
        for key in (
            "actor_model_state_dict",
            "model_state_dict",
            "model",
            "state_dict",
        ):
            if isinstance(state, dict) and key in state and isinstance(state[key], dict):
                state = state[key]
                break

        # collect prefix candidates, longest first
        prefixes = (
            "actor_module.module.",
            "actor_module.",
            "actor.",
        )
        renamed: dict[str, torch.Tensor] = {}
        for k, v in state.items():
            new_k = k
            for p in prefixes:
                if new_k.startswith(p):
                    new_k = new_k[len(p):]
                    break
            renamed[new_k] = v

        target = self.teachers[teacher_index]
        teacher_keys = set(dict(target.named_parameters()).keys()) | set(dict(target.named_buffers()).keys())
        filtered = {k: v for k, v in renamed.items() if k in teacher_keys}
        missing = [k for k in teacher_keys if k not in filtered]
        if missing and strict:
            raise RuntimeError(
                f"load_teacher_state_dict[{teacher_index}]: missing keys after renaming: "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}. "
                f"Available (post-rename): {list(renamed.keys())[:10]}"
            )
        target.load_state_dict(filtered, strict=strict)
        target.eval()
        for p in target.parameters():
            p.requires_grad_(False)
        self._loaded_teacher[teacher_index] = True

    # ---------- action interfaces (parallel PPOActor) ----------

    def _student_input(self, actor_obs: torch.Tensor, depth_obs: torch.Tensor | None = None) -> torch.Tensor:
        return actor_obs

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, actor_obs: torch.Tensor, depth_obs: torch.Tensor | None = None) -> None:
        mean = self.student(self._student_input(actor_obs, depth_obs))
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, policy_state_dict: dict) -> torch.Tensor:
        self.update_distribution(
            policy_state_dict["actor_obs"],
            policy_state_dict.get("depth_obs"),
        )
        return self.distribution.sample()

    def act_inference(self, policy_state_dict: dict) -> torch.Tensor:
        return self.student(
            self._student_input(
                policy_state_dict["actor_obs"], policy_state_dict.get("depth_obs")
            )
        )

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    @torch.no_grad()
    def teacher_act(
        self, teacher_obs: torch.Tensor, teacher_idx: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Deterministic teacher forward. Used to generate DAgger labels.

        ``teacher_obs`` is the privileged observation, passed to the teacher
        MLPs unchanged. ``teacher_idx`` optionally selects a teacher per env:

        - ``None`` (default): every env goes to ``self.teachers[0]``.
        - an integer tensor of shape ``[N]``: env ``n`` is routed to
          ``self.teachers[teacher_idx[n]]``. Negative entries mean "no teacher
          applies" and yield a zero action, which callers may use to mark
          samples to exclude from the behaviour-cloning loss.

        The index is an argument rather than a reserved observation column so
        that this module has no opinion on how an application derives it. See
        the ``DistillationPPO._dagger_loss`` docstring for the corresponding
        loss-side extension point.
        """
        if teacher_obs.shape[-1] != self.num_teacher_obs:
            raise RuntimeError(
                f"teacher_act: expected last dim {self.num_teacher_obs}, "
                f"got {teacher_obs.shape[-1]}"
            )
        out = self.teachers[0](teacher_obs)
        if teacher_idx is None:
            return out
        idx = teacher_idx.long().reshape(-1)
        for i in range(1, len(self.teachers)):
            mask = idx == i
            if mask.any():
                out[mask] = self.teachers[i](teacher_obs[mask])
        out[idx < 0] = 0.0
        return out

    def reset(self, dones=None) -> None:
        pass


class DepthStudentTeacher(StudentTeacher):
    """Student-teacher composite with a CNN depth backbone on the student side.

    The backbone is built externally and passed in so the config layer can swap
    architectures without touching this module.
    """

    def __init__(
        self,
        num_actor_obs: int,
        num_teacher_obs: int,
        num_actions: int,
        depth_backbone: nn.Module,
        depth_output_dim: int,
        student_hidden_dims: list[int],
        teacher_hidden_dims: list[int],
        activation: str = "ELU",
        init_noise_std: float = 0.01,
        num_teachers: int = 1,
    ):
        # Student sees actor_obs (proprio) concatenated with depth latent.
        super().__init__(
            num_actor_obs=num_actor_obs + depth_output_dim,
            num_teacher_obs=num_teacher_obs,
            num_actions=num_actions,
            student_hidden_dims=student_hidden_dims,
            teacher_hidden_dims=teacher_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            num_teachers=num_teachers,
        )
        self.proprio_dim = num_actor_obs
        self.depth_output_dim = depth_output_dim
        self.depth_backbone = depth_backbone

    def _student_input(self, actor_obs: torch.Tensor, depth_obs: torch.Tensor | None = None) -> torch.Tensor:
        if depth_obs is None:
            raise ValueError("DepthStudentTeacher.student requires depth_obs")
        depth_latent = self.depth_backbone(depth_obs)
        return torch.cat([actor_obs, depth_latent], dim=-1)


class StudentTeacherCritic(StudentTeacher):
    """Student-teacher with a privileged critic for PPO updates."""

    def __init__(
        self,
        num_actor_obs: int,
        num_teacher_obs: int,
        num_critic_obs: int,
        num_actions: int,
        student_hidden_dims: list[int],
        teacher_hidden_dims: list[int],
        critic_hidden_dims: list[int],
        activation: str = "ELU",
        init_noise_std: float = 0.01,
        num_teachers: int = 1,
    ):
        super().__init__(
            num_actor_obs=num_actor_obs,
            num_teacher_obs=num_teacher_obs,
            num_actions=num_actions,
            student_hidden_dims=student_hidden_dims,
            teacher_hidden_dims=teacher_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            num_teachers=num_teachers,
        )
        self.num_critic_obs = num_critic_obs
        self.critic = _build_mlp(num_critic_obs, critic_hidden_dims, 1, activation)
        print(f"Critic MLP: {self.critic}")

    def evaluate(self, policy_state_dict: dict) -> torch.Tensor:
        return self.critic(policy_state_dict["critic_obs"])


class DepthStudentTeacherCritic(DepthStudentTeacher):
    """Depth-aware student + frozen teacher + privileged critic. The target class used by ``Warp-Distillation-Finetune``."""

    def __init__(
        self,
        num_actor_obs: int,
        num_teacher_obs: int,
        num_critic_obs: int,
        num_actions: int,
        depth_backbone: nn.Module,
        depth_output_dim: int,
        student_hidden_dims: list[int],
        teacher_hidden_dims: list[int],
        critic_hidden_dims: list[int],
        activation: str = "ELU",
        init_noise_std: float = 0.01,
        num_teachers: int = 1,
    ):
        super().__init__(
            num_actor_obs=num_actor_obs,
            num_teacher_obs=num_teacher_obs,
            num_actions=num_actions,
            depth_backbone=depth_backbone,
            depth_output_dim=depth_output_dim,
            student_hidden_dims=student_hidden_dims,
            teacher_hidden_dims=teacher_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            num_teachers=num_teachers,
        )
        self.num_critic_obs = num_critic_obs
        self.critic = _build_mlp(num_critic_obs, critic_hidden_dims, 1, activation)
        print(f"Critic MLP: {self.critic}")

    def evaluate(self, policy_state_dict: dict) -> torch.Tensor:
        return self.critic(policy_state_dict["critic_obs"])
