"""Headless sim2sim harness for locomotion policy.

Composes a real :class:`LocomotionPolicy`, replaces the SDK interface and
input providers with in-process stubs, and drives the per-cycle step for
a fixed sim-time horizon. Reports whether the robot fell.

This is intentionally a thin coverage gate, not a behavioural replica of
the live joystick/keyboard flow. Pass = "policy can produce stable
torques on plausible obs and the robot remains upright in walking mode
when commanded forward at 0.5 m/s for ~10 s of sim time".
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

import numpy as np

from holosoma_inference.config.config_values.inference import g1_29dof_loco
from holosoma_inference.inputs.api.commands import StateCommand, VelCmd
from tests.sim2sim.mujoco_interface import MujocoSimInterface

G1_MJCF = os.path.expanduser(
    "~/projects/holosoma/src/holosoma/holosoma/data/robots/g1/scenes/scene_g1_29dof_wbt_plane.xml"
)
G1_LOCO_ONNX = os.path.expanduser(
    "~/projects/holosoma/src/holosoma_inference/holosoma_inference/models/loco/g1_29dof/fastsac_g1_29dof.onnx"
)


class _StubVelInput:
    """Replaces VelCmdProvider — feeds a fixed forward velocity each cycle."""

    def __init__(self, lin_vel=(0.5, 0.0), ang_vel=0.0):
        self._vc = VelCmd(lin_vel=np.array(lin_vel, dtype=np.float32), ang_vel=float(ang_vel))
        self._first = True

    def start(self):
        pass

    def poll_velocity(self):
        return self._vc

    def zero(self):
        self._vc = VelCmd(lin_vel=np.zeros(2, dtype=np.float32), ang_vel=0.0)


class _StubCmdProvider:
    """Replaces StateCommandProvider — emits a scripted command sequence."""

    def __init__(self):
        # Emit START (activate policy) then STAND_TOGGLE (enter walking mode).
        # After that, no further commands.
        self._queue: list[list[StateCommand]] = [
            [StateCommand.START, StateCommand.STAND_TOGGLE],
        ]
        self._mapping: dict[str, StateCommand] = {}

    def start(self):
        pass

    def poll_commands(self):
        if self._queue:
            return self._queue.pop(0)
        return []


@dataclass
class HarnessResult:
    final_pelvis_z: float
    min_pelvis_z: float
    steps: int
    fell: bool

    def summary(self) -> str:
        verdict = "FELL" if self.fell else "OK"
        return (
            f"[{verdict}] pelvis final={self.final_pelvis_z:.3f} m, min={self.min_pelvis_z:.3f} m, steps={self.steps}"
        )


def build_policy_with_sim(
    config=None,
    sim_steps_per_control: int = 4,
    initial_pelvis_height: float = 0.78,
):
    """Construct a real LocomotionPolicy + Controller wired to a MuJoCo
    in-process interface and stub input providers.

    Returns ``(policy, controller, sim_interface)``.
    """
    from holosoma_inference.controllers import Controller
    from holosoma_inference.policies.locomotion import LocomotionPolicy

    config = config if config is not None else replace(g1_29dof_loco, secondary=None)
    config = replace(
        config,
        task=replace(
            config.task,
            model_path=G1_LOCO_ONNX,
            velocity_input="keyboard",
            state_input="keyboard",
            interface="lo",
        ),
    )

    sim_interface = MujocoSimInterface(
        model_path=G1_MJCF,
        kp=np.array(config.robot.motor_kp) if config.robot.motor_kp is not None else np.ones(29) * 80.0,
        kd=np.array(config.robot.motor_kd) if config.robot.motor_kd is not None else np.ones(29) * 2.0,
        torque_limit=_g1_torque_limits(),
        num_joints=29,
        steps_per_control=sim_steps_per_control,
        initial_qpos=np.array(config.robot.default_dof_angles),
        initial_height=initial_pelvis_height,
    )

    policy = LocomotionPolicy(config=config, interface=sim_interface)

    controller = Controller.from_single_policy(
        policy,
        interface=sim_interface,
        velocity_input=_StubVelInput(),
        command_provider=_StubCmdProvider(),
        rate=_NullRate(),
        logger=_NullLogger(),
    )

    return policy, controller, sim_interface


def run_harness(
    duration_s: float = 10.0,
    initial_pelvis_height: float = 0.78,
    fall_threshold: float = 0.3,
    render: bool = False,
) -> HarnessResult:
    """Drive policy.policy_action() for *duration_s* seconds of sim time.

    When *render* is True, opens a passive MuJoCo viewer and paces the loop
    at real-time (sleeps between control ticks). The viewer requires a
    DISPLAY; on a headless host this raises at viewer construction.
    """
    import time

    policy, controller, sim_interface = build_policy_with_sim(initial_pelvis_height=initial_pelvis_height)

    n_steps = int(duration_s * policy.config.task.rl_rate)
    min_z = sim_interface.pelvis_height
    control_dt = 1.0 / policy.config.task.rl_rate

    viewer = None
    if render:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(sim_interface.model, sim_interface.data)

    try:
        for _ in range(n_steps):
            tick_start = time.perf_counter()
            controller.step()
            z = sim_interface.pelvis_height
            min_z = min(min_z, z)
            if min_z < fall_threshold:
                break
            if viewer is not None:
                viewer.sync()
                elapsed = time.perf_counter() - tick_start
                remaining = control_dt - elapsed
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        # Intentionally do not call viewer.close(). The passive viewer
        # races with the GL context teardown at interpreter shutdown and
        # prints GLXBadWindow to stderr from libX11. Letting Python's
        # normal cleanup handle it produces a quiet exit.
        pass

    final_z = sim_interface.pelvis_height
    return HarnessResult(
        final_pelvis_z=final_z,
        min_pelvis_z=min_z,
        steps=n_steps,
        fell=min_z < fall_threshold,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
# From g1_29dof.xml ctrlrange (matched to dof_names order).
# Layout: 6 left-leg | 6 right-leg | 3 waist | 7 left-arm | 7 right-arm.
_G1_TORQUE_LIMITS = (
    *(88, 88, 88, 139, 50, 50),
    *(88, 88, 88, 139, 50, 50),
    *(88, 50, 50),
    *(25, 25, 25, 25, 25, 5, 5),
    *(25, 25, 25, 25, 25, 5, 5),
)


def _g1_torque_limits() -> np.ndarray:
    return np.array(_G1_TORQUE_LIMITS, dtype=np.float64)


class _NullLogger:
    def info(self, *a, **kw):
        pass

    def warning(self, *a, **kw):
        pass

    def error(self, *a, **kw):
        pass


class _NullRate:
    def sleep(self):
        pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true", help="Open a passive MuJoCo viewer (requires DISPLAY).")
    parser.add_argument("--duration", type=float, default=10.0, help="Sim time in seconds.")
    args = parser.parse_args()

    result = run_harness(duration_s=args.duration, render=args.render)
    print(result.summary())
    raise SystemExit(0 if not result.fell else 1)
