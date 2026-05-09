"""In-process MuJoCo robot interface for headless sim2sim tests.

Implements the same get_low_state / send_low_command contract as
:class:`holosoma_inference.sdk.unitree.unitree_interface.UnitreeInterface`
but reads/writes a local mujoco.MjData instead of the unitree binding.

State layout (matches :class:`BaseInterface.get_low_state`):
    [base_pos(3) | quat_wxyz(4) | joint_pos(N) | base_lin_vel(3)
     | base_ang_vel(3) | joint_vel(N)]
"""

from __future__ import annotations

import mujoco
import numpy as np


class MujocoSimInterface:
    """Drives a MuJoCo G1 model with PD torque control matching real-robot semantics."""

    def __init__(
        self,
        model_path: str,
        kp: np.ndarray,
        kd: np.ndarray,
        torque_limit: np.ndarray,
        num_joints: int = 29,
        steps_per_control: int = 4,
        initial_qpos: np.ndarray | None = None,
        initial_height: float = 0.78,
    ):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self._n = num_joints
        self._steps = steps_per_control

        self._kp_base = np.asarray(kp, dtype=np.float64)
        self._kd_base = np.asarray(kd, dtype=np.float64)
        self._kp = self._kp_base.copy()
        self._kd = self._kd_base.copy()
        self.torque_limit = np.asarray(torque_limit, dtype=np.float64)
        self._kp_level = 1.0
        self._kd_level = 1.0

        # Place robot at standing pose so it doesn't free-fall on tick 0
        if initial_qpos is not None:
            self.data.qpos[7 : 7 + self._n] = initial_qpos
        self.data.qpos[2] = initial_height
        # wxyz identity quaternion
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        mujoco.mj_forward(self.model, self.data)

        # Match BaseInterface contract: real interfaces expose robot_config
        # but the controller uses it for KP/KD only; harness-side stubs suffice.
        self._buf = np.zeros((1, 3 + 4 + num_joints + 3 + 3 + num_joints), dtype=np.float64)

    # ------------------------------------------------------------------
    # BaseInterface contract
    # ------------------------------------------------------------------
    def get_low_state(self) -> np.ndarray:
        n = self._n
        b = self._buf
        b[0, 0:3] = self.data.qpos[0:3]
        b[0, 3:7] = self.data.qpos[3:7]
        b[0, 7 : 7 + n] = self.data.qpos[7 : 7 + n]
        b[0, 7 + n : 7 + n + 3] = self.data.qvel[0:3]
        b[0, 7 + n + 3 : 7 + n + 6] = self.data.qvel[3:6]
        b[0, 7 + n + 6 : 7 + 2 * n + 6] = self.data.qvel[6 : 6 + n]
        return b

    def send_low_command(
        self,
        cmd_q,
        cmd_dq=None,
        cmd_tau=None,
        dof_pos_latest=None,
        kp_override=None,
        kd_override=None,
    ):
        target_q = np.asarray(cmd_q, dtype=np.float64).flatten()
        kp = (
            np.asarray(kp_override, dtype=np.float64) * self._kp_level
            if kp_override is not None
            else self._kp_base * self._kp_level
        )
        kd = (
            np.asarray(kd_override, dtype=np.float64) * self._kd_level
            if kd_override is not None
            else self._kd_base * self._kd_level
        )
        for _ in range(self._steps):
            n = self._n
            tau = (target_q - self.data.qpos[7 : 7 + n]) * kp - self.data.qvel[6 : 6 + n] * kd
            self.data.ctrl[:] = np.clip(tau, -self.torque_limit, self.torque_limit)
            mujoco.mj_step(self.model, self.data)

    def update_config(self, robot_config):
        if getattr(robot_config, "motor_kp", None) is not None:
            self._kp_base = np.asarray(robot_config.motor_kp, dtype=np.float64)
        if getattr(robot_config, "motor_kd", None) is not None:
            self._kd_base = np.asarray(robot_config.motor_kd, dtype=np.float64)

    # ------------------------------------------------------------------
    # Joystick stubs — harness has no joystick
    # ------------------------------------------------------------------
    def get_joystick_msg(self):
        return None

    def get_joystick_key(self, wc_msg=None):
        return None

    @property
    def kp_level(self):
        return self._kp_level

    @kp_level.setter
    def kp_level(self, value):
        self._kp_level = value

    @property
    def kd_level(self):
        return self._kd_level

    @kd_level.setter
    def kd_level(self, value):
        self._kd_level = value

    @property
    def pelvis_height(self) -> float:
        return float(self.data.qpos[2])
