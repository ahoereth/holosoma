"""Console entry: split-body Unitree controller (arm_sdk + loco).

Subscribes ExoskeletonCmd (via HolosomaNode), drives the G1 arms over arm_sdk
and the base over LocoClient. Each SDK client runs in its own spawned process
(separate DDS). Loco enters FSM-501 BEFORE arm init, else arm_sdk is ignored.
"""

from __future__ import annotations

import rclpy
import tyro
from loguru import logger

from holosoma_inference.sdk.unitree_high_level import make_mp_arm_client, make_mp_loco_client
from holosoma_inference.service.tracker_controller import TrackerController
from holosoma_service.holosoma_node import HolosomaNode


def main(iface: str = "eth0", no_arms: bool = False, no_loco: bool = False) -> None:
    arm = loco = None
    if not no_loco:
        logger.info("starting loco client …")
        loco = make_mp_loco_client(iface=iface)
        loco.start()
        loco.set_walk_mode()
    if not no_arms:
        logger.info("starting arm client …")
        arm = make_mp_arm_client(iface=iface, motion_mode=True)
        arm.ctrl_dual_arm_initialization_pose()
        arm.speed_gradual_max()

    rclpy.init()
    node = HolosomaNode()
    node.start()
    controller = TrackerController(source=node, arm=arm, loco=loco)
    try:
        controller.run()
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("shutting down …")
        controller.stop()
        node.stop()
        rclpy.shutdown()
        if loco is not None:
            loco.close()  # type: ignore[attr-defined]
        if arm is not None:
            arm.close()  # type: ignore[attr-defined]


def _cli() -> None:
    tyro.cli(main)


if __name__ == "__main__":
    _cli()
