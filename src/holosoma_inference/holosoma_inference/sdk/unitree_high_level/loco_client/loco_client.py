"""Minimal G1 locomotion client — direct ``LocoClient`` wrapper.

This is the SDK-side subset of FAR-pi's ``g1_loco_bridge.py`` with all
TCP-bridge plumbing removed. It assumes it runs ON THE JETSON, where the
unitree SDK's CycloneDDS version works (the bridge only existed to escape
the laptop/Docker DDS version mismatch — not needed here).

Intended use: a service layer receives ROS2 commands and calls these
methods directly. The class owns DDS init + the ``LocoClient`` session;
callers just push velocity / FSM transitions.

Standalone (smoke test on Jetson):

    PYTHONPATH=~/unitree_sdk2 python3 loco_client.py
"""

from __future__ import annotations

import logging
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as hg_LowState

# Conventional-walk FSM. 501 decouples arms from legs so rt/arm_sdk works
# alongside locomotion; 200/801 (AMP gait) couples them and blocks arm_sdk.
FSM_ID_WALK = 501

JETSON_DDS_CONFIG = """<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS>
    <Domain Id="any">
        <General>
            <Interfaces>
                <NetworkInterface name="eno1" priority="default" multicast="default"/>
            </Interfaces>
        </General>
        <Tracing>
            <Verbosity>warning</Verbosity>
            <OutputFile>/tmp/g1_loco_client_dds.log</OutputFile>
        </Tracing>
    </Domain>
</CycloneDDS>"""


class G1LocoClient:
    """Thin wrapper around Unitree's ``LocoClient`` for base velocity control."""

    def __init__(self, dds_config: str = JETSON_DDS_CONFIG, timeout: float = 5.0, logger=None):
        self._logger = logger or logging.getLogger(__name__)

        self._logger.info("Initializing DDS...")
        ChannelFactoryInitialize(0, config=dds_config)

        # Wait for the MCU to come up on rt/lowstate before opening the
        # LocoClient RPC session (mirrors the bridge's startup gate).
        self._lowstate_sub = ChannelSubscriber("rt/lowstate", hg_LowState)
        self._lowstate_sub.Init()
        self._logger.info("Waiting for MCU DDS connection...")
        while True:
            msg = self._lowstate_sub.Read(timeout=0.1)
            if msg is not None:
                self._logger.info(f"MCU connected! mode_machine={msg.mode_machine}")
                break

        self._loco = LocoClient()
        self._loco.SetTimeout(timeout)
        self._loco.Init()
        self._logger.info("LocoClient ready")

    def start(self) -> None:
        """Bring the robot to balance-stand and start the controller."""
        code = self._loco.BalanceStand(1)
        self._logger.info(f"BalanceStand(1) -> code={code}")
        code = self._loco.Start()
        self._logger.info(f"Start() -> code={code}")

    def set_walk_mode(self, fsm_id: int = FSM_ID_WALK) -> None:
        """Switch to conventional-walk FSM so arm_sdk works alongside loco."""
        fsm_before, fsm_data_before = self._loco._Call(7001, "{}")
        self._logger.info(f"FSM before SetFsmId({fsm_id}): code={fsm_before} data={fsm_data_before}")
        code = self._loco.SetFsmId(fsm_id)
        self._logger.info(f"SetFsmId({fsm_id}) -> code={code}")

    def set_velocity(self, vx: float, vy: float, vyaw: float) -> int:
        """Command base velocity (body frame). Continuous: holds until next call.

        Returns the RPC status code from SetVelocity (0 = success).
        """
        t0 = time.perf_counter()
        code = self._loco.SetVelocity(vx, vy, vyaw, duration=864000.0)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if code != 0:
            self._logger.warning(
                f"SetVelocity FAILED: ({vx:.3f},{vy:.3f},{vyaw:.3f}) -> code={code} ({elapsed_ms:.1f}ms)"
            )
        elif elapsed_ms > 50.0:
            self._logger.warning(
                f"SetVelocity SLOW: ({vx:.3f},{vy:.3f},{vyaw:.3f}) -> code={code} ({elapsed_ms:.1f}ms)"
            )
        elif abs(vx) > 0.01 or abs(vy) > 0.01 or abs(vyaw) > 0.01:
            self._logger.debug(f"SetVelocity({vx:.3f},{vy:.3f},{vyaw:.3f}) -> code={code} ({elapsed_ms:.1f}ms)")
        return code

    def stop(self) -> None:
        """Stop base motion (zero velocity)."""
        self._loco.StopMove()

    def get_fsm_state(self):
        """Return raw (code, data) for the current FSM state (api 7001)."""
        return self._loco._Call(7001, "{}")

    def check_fsm_healthy(self) -> tuple[bool, int | str | None]:
        """Check if FSM is still in walking mode (501).

        Returns (is_healthy, fsm_id_or_error). Caller should log the result.
        """
        try:
            code, data = self._loco._Call(7001, "{}")
        except Exception as exc:
            return False, f"exception: {exc}"
        if code != 0:
            return False, f"rpc_error={code}"
        import json
        try:
            fsm_id = json.loads(data).get("data")
        except Exception:
            fsm_id = data
        if fsm_id != FSM_ID_WALK:
            return False, fsm_id
        return True, fsm_id


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    client = G1LocoClient()
    client.start()
    client.set_walk_mode()
    print("LocoClient up — FSM state:", client.get_fsm_state())
    client.stop()
