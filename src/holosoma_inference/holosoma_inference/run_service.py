#!/usr/bin/env python3
"""Holosoma teleop service. The API is the teleop ROS messages (SmplhCmd, etc.);
this entrypoint routes them to one of two backends:

    controller : in-process split-body Unitree control (arm_sdk + loco)
    policy     : a WBT policy tracking a retargeted SmplhCmd stream

    python run_service.py controller --no-loco
    python run_service.py policy --preset g1-29dof-holosoma-wbt --model-path m.onnx --urdf-path g1.urdf
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass

import tyro
from loguru import logger

DENSE_TOPIC = "/holosoma/dense_tracking_command"


@dataclass
class ControllerConfig:
    """In-process split-body controller (arm_sdk + loco)."""

    iface: str = "eth0"
    """network interface the G1 MCU is on."""
    no_arms: bool = False
    no_loco: bool = False


@dataclass
class PolicyConfig:
    """WBT policy driven by a retargeted SmplhCmd stream (two subprocesses)."""

    preset: str
    model_path: str
    urdf_path: str
    rl_rate_hz: float = 50.0
    smplh_topic: str = "/holosoma/smplh_command"


def _run_controller(cfg: ControllerConfig) -> None:
    from holosoma_inference.sdk.unitree_high_level import make_mp_arm_client, make_mp_loco_client
    from holosoma_inference.service.tracker_controller import TrackerController
    from holosoma_inference.teleop.holosoma_teleop_listener_node import TeleopListener

    arm = loco = None
    if not cfg.no_loco:  # loco first: enter FSM-501 before arm_sdk
        logger.info("starting loco client subprocess …")
        loco = make_mp_loco_client(iface=cfg.iface)
        loco.start()
        loco.set_walk_mode()
    if not cfg.no_arms:
        logger.info("starting arm client subprocess …")
        arm = make_mp_arm_client(iface=cfg.iface, motion_mode=True)
        arm.ctrl_dual_arm_initialization_pose()
        arm.speed_gradual_max()

    listener = TeleopListener()
    listener.start()
    controller = TrackerController(source=listener, arm=arm, loco=loco)
    try:
        controller.run()
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("shutting down …")
        controller.stop()
        listener.stop()
        if loco is not None:
            loco.close()  # type: ignore[attr-defined]
        if arm is not None:
            arm.close()  # type: ignore[attr-defined]


def _run_policy(cfg: PolicyConfig) -> None:
    # The WBT policy lives in FAR-pi (wbt_wrappers), so we can't import it from
    # holosoma core — orchestrate the two pieces as subprocesses instead.
    cmds = [
        [
            sys.executable,
            "-m",
            "holosoma_inference.teleop.retargeting.retargeter_node",
            "--urdf-path",
            cfg.urdf_path,
            "--rl-rate-hz",
            str(cfg.rl_rate_hz),
        ],
        [
            sys.executable,
            "-m",
            "wbt_wrappers_inference.run_policy",
            cfg.preset,
            "--task.model-path",
            cfg.model_path,
            "--task.teleop-topic",
            DENSE_TOPIC,
        ],
    ]
    procs = [subprocess.Popen(c) for c in cmds]
    try:
        while all(p.poll() is None for p in procs):  # run until one exits or Ctrl-C
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("shutting down …")
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()


def main() -> None:
    cfg = tyro.cli(ControllerConfig | PolicyConfig)
    if isinstance(cfg, ControllerConfig):
        _run_controller(cfg)
    else:
        _run_policy(cfg)


if __name__ == "__main__":
    main()
