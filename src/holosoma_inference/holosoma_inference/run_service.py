#!/usr/bin/env python3
"""Unitree-as-tracker service entrypoint (experimental glue).

    UnitreeTrackerCommand  ──▶  HolosomaTeleopListenerNode   (parent: rclpy)
                                       │  get_latest()
                                       ▼
                          ┌─ q_left_arm + q_right_arm ─▶ arm proxy  ─▶ G1j29ArmController (rt/arm_sdk)
                          └─ base_velocity (Twist)    ─▶ loco proxy ─▶ G1LocoClient       (LocoClient.Move)

Each SDK client runs in its own spawned child process (see clients_mp) so the
arm SDK's CycloneDDS, the loco SDK's CycloneDDS, and the parent's rclpy all
live in separate address spaces. Runs on the Jetson. No retargeting here —
arms are expected pre-solved.

Startup order matters: loco goes to FSM-501 (arms-decoupled walk) BEFORE the
arm init trajectory, else the loco controller owns the arm joints and arm_sdk
is silently ignored.

    python run_service.py --bridge-host 192.168.123.164
    python run_service.py --no-loco          # arms only
    python run_service.py --dry-run          # listener only, no SDK clients
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import rclpy
from loguru import logger

from holosoma_inference.teleop.holosoma_teleop_listener_node import HolosomaTeleopListenerNode

CONTROL_HZ = 250.0  # arm_sdk control rate


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unitree-as-tracker service")
    p.add_argument("--bridge-host", default=None, help="arm DDS bridge host[:port]; omit for direct DDS")
    p.add_argument("--dds-uri", default=None, help="CYCLONEDDS_URI config for the arm client (direct-DDS mode)")
    p.add_argument("--no-arms", action="store_true", help="skip the arm client")
    p.add_argument("--no-loco", action="store_true", help="skip the loco client")
    p.add_argument("--dry-run", action="store_true", help="listener only, no SDK clients")
    p.add_argument("--fsm-id", type=int, default=501, help="loco FSM id (501 = arms-decoupled walk)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # --- SDK clients, each isolated in its own child process ---
    arm = None
    loco = None
    if not args.dry_run:
        from holosoma_inference.sdk.unitree.high_level import make_mp_arm_client, make_mp_loco_client

        # Loco first: enter FSM-501 so arm_sdk isn't owned by the loco controller.
        if not args.no_loco:
            logger.info("starting loco client subprocess …")
            loco = make_mp_loco_client()
            loco.start()
            loco.set_walk_mode(args.fsm_id)
        # Then bring arms to the init pose and ramp velocity.
        if not args.no_arms:
            logger.info("starting arm client subprocess …")
            arm = make_mp_arm_client(dds_uri_config=args.dds_uri, motion_mode=True, bridge_host=args.bridge_host)
            arm.ctrl_dual_arm_initialization_pose()
            arm.speed_gradual_max()

    # --- Listener (parent process owns rclpy) ---
    rclpy.init()
    node = HolosomaTeleopListenerNode()

    dt = 1.0 / CONTROL_HZ
    logger.info(f"service loop running at {CONTROL_HZ:.0f} Hz (dry_run={args.dry_run})")
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)  # non-blocking: drain callbacks
            cmd = node.get_latest()
            if cmd is not None:
                if arm is not None:
                    q = np.concatenate([np.array(cmd.q_left_arm), np.array(cmd.q_right_arm)])
                    arm.track_dual_arm(q.tolist())  # clip+publish, child-side
                if loco is not None:
                    v = cmd.base_velocity
                    loco.set_velocity(v.linear.x, v.linear.y, v.angular.z)
            time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("shutting down …")
        if loco is not None:
            loco.stop()
            loco.close()  # type: ignore[attr-defined]  # close() lives on the proxy, not G1LocoClient
        if arm is not None:
            arm.close()  # type: ignore[attr-defined]  # close() lives on the proxy, not G1j29ArmController
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
