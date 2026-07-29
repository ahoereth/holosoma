"""Default observation configurations for holosoma_inference.

This module provides pre-configured observation spaces for different
robot types and tasks, converted from the original YAML configurations.
"""

from __future__ import annotations

from holosoma_inference.config.config_types.observation import ObservationConfig
from holosoma_inference.utils.config_registry import (
    ConfigRegistry,
    deprecated_defaults_alias,
    deprecated_get_defaults,
)

OBSERVATION_REGISTRY = ConfigRegistry(ObservationConfig, group="holosoma.config.observation")

# =============================================================================
# Locomotion Observation Configurations
# =============================================================================

loco_g1_29dof = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "base_ang_vel",
            "projected_gravity",
            "command_lin_vel",
            "command_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
            "sin_phase",
            "cos_phase",
        ]
    },
    obs_dims={
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "command_lin_vel": 2,
        "command_ang_vel": 1,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
        "sin_phase": 2,
        "cos_phase": 2,
    },
    obs_scales={
        "base_lin_vel": 2.0,
        "base_ang_vel": 0.25,
        "projected_gravity": 1.0,
        "command_lin_vel": 1.0,
        "command_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 0.05,
        "actions": 1.0,
        "sin_phase": 1.0,
        "cos_phase": 1.0,
    },
    history_length_dict={
        "actor_obs": 1,
    },
)

loco_t1_29dof = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "base_ang_vel",
            "projected_gravity",
            "command_lin_vel",
            "command_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
            "sin_phase",
            "cos_phase",
        ]
    },
    obs_dims={
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "command_lin_vel": 2,
        "command_ang_vel": 1,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
        "sin_phase": 2,
        "cos_phase": 2,
    },
    obs_scales={
        "base_lin_vel": 1.0,  # T1 uses 1.0 (vs G1's 2.0)
        "base_ang_vel": 1.0,  # T1 uses 1.0 (vs G1's 0.25)
        "projected_gravity": 1.0,
        "command_lin_vel": 1.0,
        "command_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 0.1,  # T1 uses 0.1 (vs G1's 0.05)
        "actions": 1.0,
        "sin_phase": 1.0,
        "cos_phase": 1.0,
    },
    history_length_dict={
        "actor_obs": 1,
    },
)


# =============================================================================
# WBT (Whole Body Tracking) Observation Configurations
# =============================================================================

wbt = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "motion_command",
            "motion_ref_ori_b",
            "base_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
        ]
    },
    obs_dims={
        "motion_command": 58,
        "motion_ref_pos_b": 3,
        "motion_ref_ori_b": 6,
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
    },
    obs_scales={
        "actions": 1.0,
        "motion_command": 1.0,
        "motion_ref_pos_b": 1.0,
        "motion_ref_ori_b": 1.0,
        "base_lin_vel": 1.0,
        "base_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 1.0,
        "robot_body_pos_b": 1.0,
        "robot_body_ori_b": 1.0,
    },
    history_length_dict={
        "actor_obs": 1,
    },
)

# =============================================================================
# Depth Distillation Observation Configurations
# =============================================================================

# Vision-based locomotion (stairs / rough terrain). The student consumes
# ``[actor_obs | velocity_command | depth_latent]``; only ``actor_obs`` is built
# by the generic history machinery, so the term order below IS the wire order
# the checkpoint expects.
wbt_distillation_g1 = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "projected_gravity",
            "base_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
        ],
        "depth_obs": [
            "cam_depth",
        ],
    },
    obs_dims={
        "projected_gravity": 3,
        "base_ang_vel": 3,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
        # resized_height * resized_width = 58 * 87
        "cam_depth": 5046,
        # One-hot direction command. Not part of actor_obs — the policy
        # concatenates it between the proprioceptive terms and the depth latent.
        "velocity_command": 15,
    },
    obs_scales={
        "projected_gravity": 1.0,
        "base_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 1.0,
        "actions": 1.0,
        "cam_depth": 1.0,
    },
    history_length_dict={
        "actor_obs": 1,
        "depth_obs": 3,
    },
    # These checkpoints were exported with terms in declaration order, so the
    # list above is the literal wire layout: [gravity(3) | ang_vel(3) |
    # dof_pos(29) | dof_vel(29) | actions(29)] = 93 dims.
    sort_obs_terms=False,
)


# =============================================================================
# Default Configurations Dictionary
# =============================================================================

# Register core presets. Keys use hyphen-case naming convention for CLI compatibility.
OBSERVATION_REGISTRY.add("loco-g1-29dof", loco_g1_29dof)
OBSERVATION_REGISTRY.add("loco-t1-29dof", loco_t1_29dof)
OBSERVATION_REGISTRY.add("wbt", wbt)
OBSERVATION_REGISTRY.add("wbt-distillation-g1", wbt_distillation_g1)

__getattr__ = deprecated_defaults_alias(__name__, OBSERVATION_REGISTRY)
get_defaults = deprecated_get_defaults(__name__, OBSERVATION_REGISTRY)
