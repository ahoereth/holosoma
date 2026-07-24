"""Synthetic (offline) interface for ``--task.debug.dryer-run``.

Fabricates a plausible standing robot state and never opens a DDS connection,
so the policy loop runs with no sim bridge or hardware driver present at all.
Used to smoke-test config loading, ONNX inference, and the observation pipeline
off-robot. ``send_low_command`` is a no-op (dryer_run implies dry_run), but it's
implemented defensively in case it's ever called directly.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from holosoma_inference.config.config_types import RobotConfig
from holosoma_inference.sdk.base.base_interface import BaseInterface


class SyntheticInterface(BaseInterface):
    """Returns a fixed, upright standing state; sends nothing.

    Layout matches :meth:`BaseInterface.get_low_state`:
    ``[base_pos(3), quat(4), dof_pos(N), lin_vel(3), ang_vel(3), dof_vel(N)]``
    followed by an optional ``projected_gravity(3)`` (we append it, upright).
    """

    def __init__(
        self,
        robot_config: RobotConfig,
        domain_id: int = 0,
        interface_str: str | None = None,
        use_joystick: bool = False,
    ):
        super().__init__(robot_config, domain_id, interface_str, use_joystick)

        default_angles = getattr(robot_config, "default_dof_angles", None)
        if default_angles is not None and len(default_angles) > 0:
            self._dof_pos = np.asarray(default_angles, dtype=float)
        else:
            n = len(getattr(robot_config, "dof_names", []) or [])
            self._dof_pos = np.zeros(n if n > 0 else 1)
        self._num_dof = self._dof_pos.shape[0]

        # Upright: identity quaternion (w,x,y,z), gravity straight down in the
        # base frame, zero linear/angular velocity, zero joint velocity.
        self._quat = np.array([1.0, 0.0, 0.0, 0.0])
        self._projected_gravity = np.array([0.0, 0.0, -1.0])

        self._kp_level = 1.0
        self._kd_level = 1.0

        logger.warning(
            "DRYER RUN: using SyntheticInterface — state is fabricated (upright, "
            f"default pose, zero velocities across {self._num_dof} DOF) and NO "
            "connection to any bridge/driver is opened."
        )

    def get_low_state(self) -> np.ndarray:
        return np.concatenate(
            [
                np.zeros(3),  # base_pos
                self._quat,  # quat (w,x,y,z)
                self._dof_pos,  # dof_pos
                np.zeros(3),  # base lin_vel
                np.zeros(3),  # base ang_vel
                np.zeros(self._num_dof),  # dof_vel
                self._projected_gravity,  # projected_gravity (upright)
            ]
        ).reshape(1, -1)

    def send_low_command(self, *args, **kwargs) -> None:
        return None

    def get_joystick_msg(self):
        return None

    def get_joystick_key(self, wc_msg=None):
        return None

    @property
    def kp_level(self):
        return self._kp_level

    @kp_level.setter
    def kp_level(self, value):
        self._kp_level = float(value)

    @property
    def kd_level(self):
        return self._kd_level

    @kd_level.setter
    def kd_level(self, value):
        self._kd_level = float(value)
