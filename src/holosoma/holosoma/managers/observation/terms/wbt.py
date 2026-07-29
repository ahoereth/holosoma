"""Whole body tracking observation terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.utils.rotations import quat_rotate_inverse, quaternion_to_matrix, subtract_frame_transforms
from holosoma.utils.torch_utils import get_axis_params, to_torch

if TYPE_CHECKING:
    from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager


#########################################################################################################
## terms same to managers/observation/terms/locomotion.py
#########################################################################################################
def _base_quat(env: WholeBodyTrackingManager) -> torch.Tensor:
    return env.base_quat


def gravity_vector(env: WholeBodyTrackingManager, up_axis_idx: int = 2) -> torch.Tensor:
    axis = to_torch(get_axis_params(-1.0, up_axis_idx), device=env.device)
    return axis.unsqueeze(0).expand(env.num_envs, -1)


def base_forward_vector(env: WholeBodyTrackingManager) -> torch.Tensor:
    axis = to_torch([1.0, 0.0, 0.0], device=env.device)
    return axis.unsqueeze(0).expand(env.num_envs, -1)


def get_base_lin_vel(env: WholeBodyTrackingManager) -> torch.Tensor:
    root_states = env.simulator.robot_root_states
    lin_vel_world = root_states[:, 7:10]
    return quat_rotate_inverse(_base_quat(env), lin_vel_world, w_last=True)


def get_base_ang_vel(env: WholeBodyTrackingManager) -> torch.Tensor:
    ang_vel_world = env.simulator.robot_root_states[:, 10:13]
    return quat_rotate_inverse(_base_quat(env), ang_vel_world, w_last=True)


def get_projected_gravity(env: WholeBodyTrackingManager) -> torch.Tensor:
    return quat_rotate_inverse(_base_quat(env), gravity_vector(env), w_last=True)


def base_lin_vel(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Base linear velocity in base frame.

    Returns:
        Tensor of shape [num_envs, 3]

    Equivalent to:
        env._get_obs_base_lin_vel()
    """
    return get_base_lin_vel(env)


def base_ang_vel(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Base angular velocity in base frame.

    Returns:
        Tensor of shape [num_envs, 3]

    Equivalent to:
        env._get_obs_base_ang_vel()
    """
    return get_base_ang_vel(env)


def projected_gravity(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Gravity vector projected into base frame.

    Returns:
        Tensor of shape [num_envs, 3]

    Equivalent to:
        env._get_obs_projected_gravity()
    """
    return get_projected_gravity(env)


def dof_pos(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Joint positions relative to default positions.

    Returns:
        Tensor of shape [num_envs, num_dof]

    Equivalent to:
        env._get_obs_dof_pos()
    """
    return env.simulator.dof_pos - env.default_dof_pos


def dof_vel(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Joint velocities.

    Returns:
        Tensor of shape [num_envs, num_dof]

    Equivalent to:
        env._get_obs_dof_vel()
    """
    return env.simulator.dof_vel


def actions(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Last actions taken by the policy.

    Returns:
        Tensor of shape [num_envs, num_actions]

    Equivalent to:
        env._get_obs_actions()
    """
    return env.action_manager.action


#########################################################################################################
## terms specific to Whole Body Tracking
#########################################################################################################


def _get_motion_command_and_assert_type(env: WholeBodyTrackingManager) -> MotionCommand:
    motion_command = env.command_manager.get_state("motion_command")
    assert motion_command is not None, "motion_command not found in command manager"
    assert isinstance(motion_command, MotionCommand), f"Expected MotionCommand, got {type(motion_command)}"
    return motion_command


def motion_command(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    return motion_command.command


def which_motion(
    env: WholeBodyTrackingManager,
    bad_ref_pos_threshold: float = 0.5,
    bad_ref_ori_threshold: float = 0.8,
    bad_motion_body_pos_threshold: float = 0.5,
    bad_motion_body_pos_body_names: tuple[str, ...] = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    ),
) -> torch.Tensor:
    """Per-env current motion index (shape [num_envs, 1]).

    Used as the leading column of ``teacher_obs`` to route samples to the
    matching teacher in ``StudentTeacher.teacher_act``. When the motion file
    lacks a ``motion_idxs`` field, ``MotionLoader`` already zero-fills
    (command/terms/wbt.py:157) so single-motion runs collapse to
    ``teachers[0]`` automatically.

    When the teacher's anchor / orientation / tracked-body errors exceed the
    given thresholds the env is flagged as "expert lost" and the index is
    overwritten with ``-1``; ``StudentTeacher.teacher_act`` zeros that env's
    teacher action, which then drives DistillationPPO's ``expert_terminate``
    mask (``teacher_actions == 0``) so the DAgger loss is suppressed for that
    sample. Mirrors far-tracking ``observations.py:190-197``.
    """
    cmd = _get_motion_command_and_assert_type(env)
    idx = cmd.motion.motion_idxs[cmd.time_steps].to(torch.float32)

    bad_ref_pos = (
        torch.norm(cmd.ref_pos_w - cmd.robot_ref_pos_w, dim=1) > bad_ref_pos_threshold
    )

    g = gravity_vector(env)
    motion_g_b = quat_rotate_inverse(cmd.ref_quat_w, g, w_last=True)
    robot_g_b = quat_rotate_inverse(cmd.robot_ref_quat_w, g, w_last=True)
    bad_ref_ori = torch.abs(motion_g_b[:, 2] - robot_g_b[:, 2]) > bad_ref_ori_threshold

    body_names_to_track = list(cmd.motion_cfg.body_names_to_track)
    body_idx = torch.tensor(
        [body_names_to_track.index(n) for n in bad_motion_body_pos_body_names],
        dtype=torch.long,
        device=env.device,
    )
    body_err = torch.norm(
        cmd.body_pos_relative_w[:, body_idx] - cmd.robot_body_pos_w[:, body_idx], dim=-1
    )
    bad_motion_body_pos = torch.any(body_err > bad_motion_body_pos_threshold, dim=-1)

    expert_terminate = bad_ref_pos | bad_ref_ori | bad_motion_body_pos
    idx = torch.where(expert_terminate, torch.full_like(idx, -1.0), idx)
    return idx.view(env.num_envs, 1)


def robot_anchor_projected_gravity(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Gravity vector projected into the motion-command anchor (ref) frame.

    Mirrors far-tracking's ``robot_anchor_projected_gravity``
    (tracking/mdp/observations.py:82-88). Gives the student a static
    orientation cue relative to the reference pose — complements
    :func:`base_ang_vel` (rates) and :func:`dof_pos` (joint angles) which
    don't encode absolute torso attitude.

    Returns:
        Tensor of shape [num_envs, 3].
    """
    motion_command = _get_motion_command_and_assert_type(env)
    return quat_rotate_inverse(
        motion_command.robot_ref_quat_w, gravity_vector(env), w_last=True
    ).view(env.num_envs, -1)


def motion_ref_pos_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    pos, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        motion_command.ref_pos_w,
        motion_command.ref_quat_w,
    )
    return pos.view(env.num_envs, -1)


def motion_ref_ori_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    _, ori = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        motion_command.ref_pos_w,
        motion_command.ref_quat_w,
    )
    mat = quaternion_to_matrix(ori, w_last=True)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_pos_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)

    num_bodies = len(motion_command.motion_cfg.body_names_to_track)
    pos_b, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w[:, None, :].repeat(1, num_bodies, 1),
        motion_command.robot_ref_quat_w[:, None, :].repeat(1, num_bodies, 1),
        motion_command.robot_body_pos_w,
        motion_command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)

    num_bodies = len(motion_command.motion_cfg.body_names_to_track)
    _, ori_b = subtract_frame_transforms(
        motion_command.robot_ref_pos_w[:, None, :].repeat(1, num_bodies, 1),
        motion_command.robot_ref_quat_w[:, None, :].repeat(1, num_bodies, 1),
        motion_command.robot_body_pos_w,
        motion_command.robot_body_quat_w,
    )
    mat = quaternion_to_matrix(ori_b, w_last=True)
    return mat[..., :2].reshape(mat.shape[0], -1)


def obj_pos_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    pos, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        motion_command.simulator_object_pos_w,
        motion_command.simulator_object_quat_w,
    )
    return pos.view(env.num_envs, -1)


def obj_ori_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    _, ori = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        motion_command.simulator_object_pos_w,
        motion_command.simulator_object_quat_w,
    )
    mat = quaternion_to_matrix(ori, w_last=True)
    return mat[..., :2].reshape(mat.shape[0], -1)


def obj_lin_vel_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    unit_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    vel_b, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w.clone(),
        motion_command.robot_ref_quat_w.clone(),
        motion_command.simulator_object_lin_vel_w,
        unit_quat,
    )
    return vel_b.view(env.num_envs, -1)


def velocity_command(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Deprecated: forwards to the application that owns this observation.

    The one-hot velocity command is not part of holosoma's motion format — it
    was introduced by an application (FAR-pi's ``wbt_training``) along with the
    command term that supplies it, and only that application consumes it. The
    implementation now lives beside its command term.

    This forwarder exists solely because the dotted path is serialized into
    checkpoints trained before the move, so ``resolve_callable`` must keep
    resolving it. It reads ``vel_cmd`` off whatever command term is registered,
    which keeps core free of any import of the application package.
    """
    motion_command = _get_motion_command_and_assert_type(env)
    vel_cmd = getattr(motion_command, "vel_cmd", None)
    if vel_cmd is None:
        raise TypeError(
            f"velocity_command: command term {type(motion_command).__name__} has no "
            f"'vel_cmd'. This observation requires a command term that supplies one "
            f"(e.g. wbt_training.config_values.motion_command:PhpMotionCommand)."
        )
    return vel_cmd.view(env.num_envs, -1)
