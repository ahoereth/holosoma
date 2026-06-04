#!/usr/bin/env python3
"""Unitree-as-tracker service entrypoint (experimental glue).

Composition root. Wires three separable pieces:

    TeleopListener               (ROS transport — owns rclpy in a bg thread)
            │  get_latest()                        ← controller polls newest msg
            ▼
    TrackerController            (holosoma control logic — owns the clients)
            │  tick() @ 250 Hz                      ← loop we own: arm heartbeat
            ├─ q_left_arm + q_right_arm ─▶ arm proxy  ─▶ G1j29ArmController (rt/arm_sdk)
            └─ base_velocity (Twist)    ─▶ loco proxy ─▶ G1LocoClient       (LocoClient.Move)

The spin thread just caches the freshest message; the controller's loop owns
the ~250 Hz rate and polls it via get_latest() each tick.
Each SDK client runs in its own spawned child process (separate DDS).

Runs on the Jetson. No retargeting — arms are expected pre-solved. Loco enters
FSM-501 (arms-decoupled walk) BEFORE arm init, else arm_sdk is ignored.

    python run_service.py
    python run_service.py --no-loco          # arms only
"""

from __future__ import annotations

from dataclasses import dataclass

import tyro
from loguru import logger

from holosoma_inference.sdk.unitree_high_level import make_mp_arm_client, make_mp_loco_client
from holosoma_inference.service.tracker_controller import TrackerController
from holosoma_inference.teleop.holosoma_teleop_listener_node import TeleopListener


@dataclass
class ServiceConfig:
    """Unitree-as-tracker service."""

    dds_uri: str | None = None
    """CYCLONEDDS_URI config for the arm client."""
    no_arms: bool = False
    """skip the arm client."""
    no_loco: bool = False
    """skip the loco client."""


def main(cfg: ServiceConfig | None = None) -> None:
    if cfg is None:
        cfg = tyro.cli(ServiceConfig)

    # --- SDK clients, each isolated in its own child process ---
    arm = None
    loco = None
    # Loco first: enter FSM-501 so arm_sdk isn't owned by the loco controller.
    if not cfg.no_loco:
        logger.info("starting loco client subprocess …")
        loco = make_mp_loco_client()
        loco.start()
        loco.set_walk_mode()
    # Then bring arms to the init pose and ramp velocity.
    if not cfg.no_arms:
        logger.info("starting arm client subprocess …")
        arm = make_mp_arm_client(dds_uri_config=cfg.dds_uri, motion_mode=True)
        arm.ctrl_dual_arm_initialization_pose()
        arm.speed_gradual_max()

    # TeleopListener owns the rclpy lifecycle in a bg thread and caches the
    # newest command; the controller's loop polls it via get_latest().
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
            loco.close()  # type: ignore[attr-defined]  # close() lives on the proxy, not G1LocoClient
        if arm is not None:
            arm.close()  # type: ignore[attr-defined]  # close() lives on the proxy, not G1j29ArmController


if __name__ == "__main__":
    main()
