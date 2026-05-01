"""Unitree robot interface using unitree_sdk2py (pure-Python DDS).

Replaces the earlier ``unitree_interface`` wheel path. The wheel's hardcoded
CycloneDDS template uses ``<NetworkInterface name="$iface">`` which, on
gmp hosts where a single NIC carries both the ROS 2 subnet (192.168.50.x)
and the Unitree subnet (192.168.123.x), binds to the wrong primary IP.

unitree_sdk2py accepts a full CycloneDDS XML config (``DEFAULT_CYCLONEDDS_URI``
in rfmpi/ros_workspace/src/gmp_unitree/utils/default_cyclonedds_uri.py),
which uses ``address="192.168.123.100"`` to explicitly pick the robot's
subnet address. This is the same code path the non-holosoma
g1_arm_client.py uses — which is known to drive the robot.

API parity with the previous wheel-backed ``UnitreeInterface`` is preserved:
``get_low_state``, ``send_low_command``, ``get_joystick_msg``,
``get_joystick_key``, ``kp_level`` / ``kd_level`` properties, and a
``self.unitree_interface`` attribute that exposes a ``read_low_state()``
shim for telemetry's extended-state read path.
"""

from __future__ import annotations

import logging
import os
import sys
import time

import numpy as np

from holosoma_inference.config.config_types import RobotConfig
from holosoma_inference.sdk.base.base_interface import BaseInterface

_DEFAULT_CYCLONEDDS_URI = """<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS>
    <Domain Id="any">
        <General>
            <Interfaces>
                <NetworkInterface address="192.168.123.100" priority="default" multicast="default"/>
            </Interfaces>
        </General>
        <Tracing>
            <Verbosity>config</Verbosity>
            <OutputFile>/tmp/cdds.LOG</OutputFile>
        </Tracing>
    </Domain>
</CycloneDDS>"""

# DDS topic names shared by all HG-family robots (G1, H1_2).
_TOPIC_LOW_COMMAND_MOTION = "rt/arm_sdk"
_TOPIC_LOW_COMMAND_DEBUG = "rt/lowcmd"
_TOPIC_LOW_STATE = "rt/lowstate"


class _RawStateShim:
    """Adapts ``unitree_sdk2py``'s LowState into the field shape the telemetry
    layer expects from the old wheel's ``unitree_interface.read_low_state()``.

    The wheel returned an object with ``.motor.{q,dq,tau_est,voltage,temperature}``
    and ``.imu.{quat,omega,accel}``. sdk2py's dataclass has
    ``motor_state[i].{q,dq,tau_est,temperature,vol}`` and
    ``imu_state.{quaternion,gyroscope,accelerometer}``. This shim translates.
    """

    __slots__ = ("motor", "imu")

    def __init__(self, raw, num_motors: int):
        motor_q = [float(raw.motor_state[i].q) for i in range(num_motors)]
        motor_dq = [float(raw.motor_state[i].dq) for i in range(num_motors)]
        motor_tau = [float(raw.motor_state[i].tau_est) for i in range(num_motors)]
        motor_vol = [float(getattr(raw.motor_state[i], "vol", 0.0)) for i in range(num_motors)]
        # temperature is an array [motor_temp, driver_temp]; expose motor side.
        motor_temp = [
            float(raw.motor_state[i].temperature[0]) if len(raw.motor_state[i].temperature) else 0.0
            for i in range(num_motors)
        ]
        self.motor = _RawMotorBlob(motor_q, motor_dq, motor_tau, motor_vol, motor_temp)
        self.imu = _RawImuBlob(
            list(raw.imu_state.quaternion),
            list(raw.imu_state.gyroscope),
            list(raw.imu_state.accelerometer),
        )


class _RawMotorBlob:
    __slots__ = ("q", "dq", "tau_est", "voltage", "temperature")

    def __init__(self, q, dq, tau_est, voltage, temperature):
        self.q = q
        self.dq = dq
        self.tau_est = tau_est
        self.voltage = voltage
        self.temperature = temperature


class _RawImuBlob:
    __slots__ = ("quat", "omega", "accel")

    def __init__(self, quat, omega, accel):
        self.quat = quat
        self.omega = omega
        self.accel = accel


class _Sdk2pyInnerInterface:
    """Object attached as ``self.unitree_interface`` for API parity with the
    wheel. Exposes ``read_low_state()`` (raw shim) and ``read_wireless_controller()``.
    """

    def __init__(self, subscriber, num_motors: int):
        self._sub = subscriber
        self._num_motors = num_motors

    def read_low_state(self):
        raw = self._sub.Read()
        if raw is None:
            # Return a zeroed shim so callers don't crash on early reads
            class _Zero:
                motor_state = [
                    type("M", (), {"q": 0.0, "dq": 0.0, "tau_est": 0.0, "vol": 0.0, "temperature": [0, 0]})()
                    for _ in range(self._num_motors)
                ]
                imu_state = type("I", (), {"quaternion": [1.0, 0.0, 0.0, 0.0], "gyroscope": [0.0] * 3, "accelerometer": [0.0] * 3})()

            raw = _Zero()
        return _RawStateShim(raw, self._num_motors)

    def read_wireless_controller(self):
        # sdk2py exposes the wireless controller on a separate channel; the
        # holosoma driver does not currently consume it. Return None.
        return None


class UnitreeInterface(BaseInterface):
    """Unitree G1/H1_2 DDS interface via unitree_sdk2py (pure-Python).

    The old wheel-backed implementation lived in this same file; this replaces
    it because the wheel's hardcoded CycloneDDS config (``<NetworkInterface
    name="$iface">``) can't pick the robot-DDS IP when a NIC carries multiple
    subnets. sdk2py accepts a full XML config and is the same transport the
    non-holosoma g1_arm_client uses.
    """

    _channel_factory_initialized = False

    def __init__(self, robot_config: RobotConfig, domain_id=0, interface_str=None, use_joystick=True):
        super().__init__(robot_config, domain_id, "eno1", use_joystick)
        self._kp_level = 1.0
        self._kd_level = 1.0
        self._logger = logging.getLogger(__name__)
        # HG path runs in motion mode (rt/arm_sdk) — matches g1_arm_client's
        # default and the only path that reliably drives the G1 on current
        # firmware. Set HOLOSOMA_UNITREE_MOTION_MODE=0 to fall back to
        # rt/lowcmd for debug use on a released robot.
        self._motion_mode = os.environ.get("HOLOSOMA_UNITREE_MOTION_MODE", "1") != "0"
        self._init_binding()

    # ── one-shot DDS bootstrap ─────────────────────────────────────────────

    def _init_binding(self):
        try:
            from unitree_sdk2py.core.channel import (
                ChannelFactoryInitialize,
                ChannelPublisher,
                ChannelSubscriber,
            )
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as hg_LowCmd
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as hg_LowState
            from unitree_sdk2py.utils.crc import CRC
        except ImportError as e:
            raise ImportError(
                "unitree_sdk2py not available. Add @unitree_sdk2py//:unitree_sdk2py to the "
                "driver's BUILD deps."
            ) from e

        print(
            f"[unitree_interface] pid={os.getpid()} iface={self.interface_str!r} "
            f"domain_id={self.domain_id} motion_mode={self._motion_mode} "
            f"using unitree_sdk2py + DEFAULT_CYCLONEDDS_URI(192.168.123.100)",
            file=sys.stderr,
            flush=True,
        )

        # ChannelFactory is a process-wide singleton — only initialize once.
        if not UnitreeInterface._channel_factory_initialized:
            ChannelFactoryInitialize(self.domain_id, config=_DEFAULT_CYCLONEDDS_URI)
            UnitreeInterface._channel_factory_initialized = True

        cmd_topic = _TOPIC_LOW_COMMAND_MOTION if self._motion_mode else _TOPIC_LOW_COMMAND_DEBUG
        self._lowcmd_publisher = ChannelPublisher(cmd_topic, hg_LowCmd)
        self._lowcmd_publisher.Init()
        self._lowstate_subscriber = ChannelSubscriber(_TOPIC_LOW_STATE, hg_LowState)
        self._lowstate_subscriber.Init()

        # Wait for first lowstate message so mode_machine is populated before
        # we start publishing commands. Matches g1_arm_client's bootstrap.
        first_state = None
        for attempt in range(200):  # up to 20s at 0.1s intervals
            first_state = self._lowstate_subscriber.Read(timeout=0.05)
            if first_state is not None:
                break
            if attempt % 20 == 0:
                print(
                    f"[unitree_interface] Waiting for first /{_TOPIC_LOW_STATE} ({attempt * 0.1:.1f}s elapsed)...",
                    file=sys.stderr,
                    flush=True,
                )
            time.sleep(0.1)
        if first_state is None:
            print(
                "[unitree_interface] WARNING: no lowstate received after 20s — "
                "robot may be unreachable. Continuing with mode_machine=0.",
                file=sys.stderr,
                flush=True,
            )
            self._mode_machine = 0
        else:
            self._mode_machine = int(getattr(first_state, "mode_machine", 0))
            print(
                f"[unitree_interface] First lowstate received, mode_machine={self._mode_machine}",
                file=sys.stderr,
                flush=True,
            )

        # Pre-allocated reusable command message + CRC util.
        self._crc = CRC()
        self._msg_ctor = unitree_hg_msg_dds__LowCmd_

        # Telemetry parity: expose a self.unitree_interface attribute that
        # behaves like the wheel's (.motor, .imu on read_low_state()).
        self.unitree_interface = _Sdk2pyInnerInterface(self._lowstate_subscriber, self.robot_config.num_motors)

        while True:
            state = self.get_low_state()
            with open("/tmp/blah.txt", "a") as _f:
                _f.write(
                    f"[{time.time():.3f}] pid={os.getpid()} "
                    f"{state}"
                )
            time.sleep(2)

    # ── BaseInterface impl ─────────────────────────────────────────────────

    def get_low_state(self) -> np.ndarray:
        """Return robot state as (1, 3+4+N+3+3+N) row:
        [base_pos(3), quat(4), joint_pos(N), lin_vel(3), ang_vel(3), joint_vel(N)].
        """
        num_joints = self.robot_config.num_joints
        num_motors = self.robot_config.num_motors
        raw = self._lowstate_subscriber.Read()

        if raw is None:
            # No state yet — return zeros with a valid identity quat.
            base_pos = np.zeros(3)
            quat = np.array([1.0, 0.0, 0.0, 0.0])
            joint_pos = np.zeros(num_joints)
            base_lin_vel = np.zeros(3)
            base_ang_vel = np.zeros(3)
            joint_vel = np.zeros(num_joints)
        else:
            base_pos = np.zeros(3)
            quat = np.array(list(raw.imu_state.quaternion))
            base_lin_vel = np.zeros(3)
            base_ang_vel = np.array(list(raw.imu_state.gyroscope))
            motor_pos = np.array([raw.motor_state[i].q for i in range(num_motors)], dtype=np.float64)
            motor_vel = np.array([raw.motor_state[i].dq for i in range(num_motors)], dtype=np.float64)
            joint_pos = np.zeros(num_joints)
            joint_vel = np.zeros(num_joints)
            motor_order = self.robot_config.joint2motor
            for j_id in range(num_joints):
                m_id = motor_order[j_id]
                joint_pos[j_id] = float(motor_pos[m_id])
                joint_vel[j_id] = float(motor_vel[m_id])

        return np.concatenate([base_pos, quat, joint_pos, base_lin_vel, base_ang_vel, joint_vel]).reshape(1, -1)

    def send_low_command(
        self,
        cmd_q: np.ndarray,
        cmd_dq: np.ndarray,
        cmd_tau: np.ndarray,
        dof_pos_latest: np.ndarray = None,
        kp_override: np.ndarray = None,
        kd_override: np.ndarray = None,
    ):
        """Publish one lowcmd to rt/arm_sdk (or rt/lowcmd in debug)."""
        num_motors = self.robot_config.num_motors
        num_joints = self.robot_config.num_joints

        # Scatter the policy's per-joint targets into motor-indexed arrays.
        cmd_q_target = np.zeros(num_motors)
        cmd_dq_target = np.zeros(num_motors)
        cmd_tau_target = np.zeros(num_motors)
        use_kp_override = kp_override is not None
        use_kd_override = kd_override is not None
        cmd_kp = np.zeros(num_motors) if use_kp_override else None
        cmd_kd = np.zeros(num_motors) if use_kd_override else None

        motor_order = self.robot_config.joint2motor
        for j_id in range(num_joints):
            m_id = motor_order[j_id]
            cmd_q_target[m_id] = float(cmd_q[j_id])
            cmd_dq_target[m_id] = float(cmd_dq[j_id])
            cmd_tau_target[m_id] = float(cmd_tau[j_id])
            if use_kp_override:
                cmd_kp[m_id] = float(kp_override[j_id])
            if use_kd_override:
                cmd_kd[m_id] = float(kd_override[j_id])

        motor_kp = np.array(cmd_kp if use_kp_override else self.robot_config.motor_kp)
        motor_kd = np.array(cmd_kd if use_kd_override else self.robot_config.motor_kd)
        motor_kp = motor_kp * self._kp_level
        motor_kd = motor_kd * self._kd_level

        msg = self._msg_ctor()
        msg.mode_pr = 0
        msg.mode_machine = self._mode_machine

        for m_id in range(num_motors):
            mc = msg.motor_cmd[m_id]
            mc.mode = 1  # position control
            mc.q = float(cmd_q_target[m_id])
            mc.dq = float(cmd_dq_target[m_id])
            mc.tau = float(cmd_tau_target[m_id])
            mc.kp = float(motor_kp[m_id])
            mc.kd = float(motor_kd[m_id])

        # In motion mode, motor_cmd[29] ("k_not_used_joint_0") carries the
        # "release" token — q=1.0 hands control over to the rt/arm_sdk path.
        # Matches ``g1_arm_client.publish_command`` (line 140).
        # Set AFTER the main loop so we don't overwrite the release slot.
        if self._motion_mode:
            _release_idx = 29
            if _release_idx < len(msg.motor_cmd):
                msg.motor_cmd[_release_idx].q = 1.0

        msg.crc = self._crc.Crc(msg)

        # Heartbeat — append to /tmp/blah.txt every 100 frames so we can
        # confirm this path is hot.
        if not hasattr(self, "_wlc_counter"):
            self._wlc_counter = 0
        self._wlc_counter += 1
        if self._wlc_counter % 100 == 0:
            try:
                state = self.get_low_state()
                with open("/tmp/blah.txt", "a") as _f:
                    _f.write(
                        f"[{time.time():.3f}] pid={os.getpid()} wlc_count={self._wlc_counter} "
                        f"{state}"
                    )
            except Exception:  # noqa: BLE001
                pass

        self._lowcmd_publisher.Write(msg)

    def get_joystick_msg(self):
        # Wireless controller is on a separate sdk2py channel. Holosoma doesn't
        # use it in the WBT path; keep API but return None.
        return None

    def get_joystick_key(self, wc_msg=None):
        if wc_msg is None:
            wc_msg = self.get_joystick_msg()
        if wc_msg is None:
            return None
        return self._wc_key_map.get(getattr(wc_msg, "keys", 0), None)

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
