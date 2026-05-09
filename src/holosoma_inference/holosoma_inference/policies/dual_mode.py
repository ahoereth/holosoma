"""Dual-mode helpers — build a Controller with two run-policies.

After Step 8, "dual mode" is just a Controller whose ``policies`` dict
has more than one non-utility policy. ``SWITCH_MODE`` cycles through
them. There is no DualModePolicy class anymore — this module is left
as the home for the policy-class selection helper used by
``run_policy.py``.
"""

from __future__ import annotations

from holosoma_inference.config.config_types.inference import InferenceConfig


def _select_policy_class(config: InferenceConfig):
    """Determine policy class based on observation config and robot type."""
    from holosoma_inference.compat import entry_points
    from holosoma_inference.policies.locomotion import LocomotionPolicy
    from holosoma_inference.policies.wbt import WholeBodyTrackingPolicy

    robot_type = config.robot.robot_type
    actor_obs = config.observation.obs_dict.get("actor_obs", [])

    if "motion_command" in actor_obs:
        for ep in entry_points(group="holosoma.policies.wbt"):
            if ep.name == robot_type:
                return ep.load()
        return WholeBodyTrackingPolicy

    for ep in entry_points(group="holosoma.policies.locomotion"):
        if ep.name == robot_type:
            return ep.load()
    return LocomotionPolicy
