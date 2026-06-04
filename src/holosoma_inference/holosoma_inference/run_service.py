#!/usr/bin/env python3
"""Unitree-as-tracker service entrypoint (experimental glue).

Composition root. Wires three separable pieces:

    TeleopListener               (ROS transport — owns rclpy in a bg thread)
            │  on_command=ctrl.set_target          ← IoC: rclpy calls us
            ▼
    TrackerController            (holosoma control logic — owns the clients)
            │  tick() @ 250 Hz                      ← loop we own: arm heartbeat
            ├─ q_left_arm + q_right_arm ─▶ arm proxy  ─▶ G1j29ArmController (rt/arm_sdk)
            └─ base_velocity (Twist)    ─▶ loco proxy ─▶ G1LocoClient       (LocoClient.Move)

The node's callback only stores the freshest target (non-blocking, runs in the
executor thread); the controller's own loop sustains the ~250 Hz arm commands.
Each SDK client runs in its own spawned child process (separate DDS).

Runs on the Jetson. No retargeting — arms are expected pre-solved. Loco enters
FSM-501 (arms-decoupled walk) BEFORE arm init, else arm_sdk is ignored.

    python run_service.py --bridge-host 192.168.123.164
    python run_service.py --no-loco          # arms only
    python run_service.py --dry-run          # listener + controller, no SDK clients
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

    bridge_host: str | None = None
    """arm DDS bridge host[:port]; omit for direct DDS."""
    dds_uri: str | None = None
    """CYCLONEDDS_URI config for the arm client (direct-DDS mode)."""
    no_arms: bool = False
    """skip the arm client."""
    no_loco: bool = False
    """skip the loco client."""
    dry_run: bool = False
    """listener + controller, no SDK clients."""
    fsm_id: int = 501
    """loco FSM id (501 = arms-decoupled walk)."""


def main(cfg: ServiceConfig | None = None) -> None:
    if cfg is None:
        cfg = tyro.cli(ServiceConfig)

    # --- SDK clients, each isolated in its own child process ---
    arm = None
    loco = None
    if not cfg.dry_run:
        # Loco first: enter FSM-501 so arm_sdk isn't owned by the loco controller.
        if not cfg.no_loco:
            logger.info("starting loco client subprocess …")
            loco = make_mp_loco_client()
            loco.start()
            loco.set_walk_mode(cfg.fsm_id)
        # Then bring arms to the init pose and ramp velocity.
        if not cfg.no_arms:
            logger.info("starting arm client subprocess …")
            arm = make_mp_arm_client(dds_uri_config=cfg.dds_uri, motion_mode=True, bridge_host=cfg.bridge_host)
            arm.ctrl_dual_arm_initialization_pose()
            arm.speed_gradual_max()

    # --- Controller (holosoma logic) injected into the listener (ROS transport) ---
    controller = TrackerController(arm=arm, loco=loco)

    # TeleopListener owns the rclpy lifecycle in a bg thread (callback freshness);
    # the controller's steady loop owns this thread (arm heartbeat).
    try:
        with TeleopListener(on_command=controller.set_target):
            controller.run()
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("shutting down …")
        controller.stop()
        if loco is not None:
            loco.close()  # type: ignore[attr-defined]  # close() lives on the proxy, not G1LocoClient
        if arm is not None:
            arm.close()  # type: ignore[attr-defined]  # close() lives on the proxy, not G1j29ArmController


if __name__ == "__main__":
    main()
