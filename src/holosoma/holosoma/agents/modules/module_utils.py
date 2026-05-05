from __future__ import annotations

from holosoma.agents.modules.ppo_modules import PPOActor, PPOActorEncoder, PPOCritic, PPOCriticEncoder
from holosoma.agents.modules.student_teacher_modules import (
    DepthStudentTeacherCritic,
    StudentTeacher,
    StudentTeacherCritic,
)
from holosoma.utils.helpers import get_class


def setup_ppo_actor_module(
    obs_dim_dict,
    module_config,
    num_actions,
    init_noise_std,
    device,
    history_length: dict[str, int],
):
    module_type = module_config.type
    if module_type in ["MLPEncoder", "CNNEncoder"]:
        return PPOActorEncoder(
            obs_dim_dict=obs_dim_dict,
            module_config_dict=module_config,
            num_actions=num_actions,
            init_noise_std=init_noise_std,
        ).to(device)
    if module_type == "MLP":
        return PPOActor(
            obs_dim_dict=obs_dim_dict,
            module_config_dict=module_config,
            num_actions=num_actions,
            init_noise_std=init_noise_std,
            history_length=history_length,
        ).to(device)

    raise ValueError(f"Invalid actor type: {module_type}")


def setup_ppo_critic_module(
    obs_dim_dict,
    module_config,
    device,
    history_length: dict[str, int],
):
    module_type = module_config.type
    if module_type in ["MLPEncoder", "CNNEncoder"]:
        return PPOCriticEncoder(
            obs_dim_dict=obs_dim_dict,
            module_config_dict=module_config,
        ).to(device)
    if module_type == "MLP":
        return PPOCritic(
            obs_dim_dict=obs_dim_dict,
            module_config_dict=module_config,
            history_length=history_length,
        ).to(device)
    raise ValueError(f"Invalid critic type: {module_type}")


def setup_student_teacher_module(
    num_actor_obs: int,
    num_teacher_obs: int,
    num_critic_obs: int,
    num_actions: int,
    module_config,
    device,
):
    """Build a ``DepthStudentTeacherCritic`` (the only variant we ship today).

    ``module_config`` is a :class:`StudentTeacherModuleConfig`. The depth
    backbone is resolved from ``module_config.depth_backbone`` (dotted path)
    and instantiated with ``(depth_output_dim,)``.
    """
    backbone_cls = get_class(module_config.depth_backbone)
    depth_backbone = backbone_cls(module_config.depth_output_dim)

    return DepthStudentTeacherCritic(
        num_actor_obs=num_actor_obs,
        num_teacher_obs=num_teacher_obs,
        num_critic_obs=num_critic_obs,
        num_actions=num_actions,
        depth_backbone=depth_backbone,
        depth_output_dim=module_config.depth_output_dim,
        student_hidden_dims=list(module_config.student_hidden_dims),
        teacher_hidden_dims=list(module_config.teacher_hidden_dims),
        critic_hidden_dims=list(module_config.critic_hidden_dims),
        activation=module_config.activation,
        init_noise_std=module_config.init_noise_std,
    ).to(device)
