"""Per-session rosbag recorder for run_policy.

Spawns ``ros2 bag record --all`` in its own process group for the lifetime of a
policy run, then SIGINTs it for a clean MCAP close on exit. Each run_policy
invocation gets its own short, self-contained bag, so debugging a single run no
longer means hunting for a timestamp inside a long fleet-telemetry recording.

Usage::

    with SessionRecorder.from_debug_config(cfg.task.debug):
        policy.run()
    # bag path is logged on exit (and available as the recorder's .bag_path)
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from typing_extensions import Self  # typing.Self is 3.11+; robot runs 3.10

if TYPE_CHECKING:
    from holosoma_inference.config.config_types.task import DebugConfig


class SessionRecorder:
    """Context manager that records a rosbag for the duration of the block.

    No-op when ``enabled`` is False, so callers can wrap unconditionally. Failure
    to start the recorder is logged and swallowed — a missing ``ros2`` or a bag
    that won't start must never take down the policy run.
    """

    def __init__(self, enabled: bool, out_dir: str = "~/run_policy_sessions", storage: str = "mcap"):
        self.enabled = enabled
        self.out_dir = Path(out_dir).expanduser()
        self.storage = storage
        self.bag_path: Path | None = None
        self._proc: subprocess.Popen | None = None

    @classmethod
    def from_debug_config(cls, debug: DebugConfig) -> SessionRecorder:
        """Build from a task ``DebugConfig`` (reads ``record_rosbag`` / ``rosbag_dir``)."""
        return cls(enabled=debug.record_rosbag, out_dir=debug.rosbag_dir)

    def __enter__(self) -> Self:
        if not self.enabled:
            return self
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self.bag_path = self.out_dir / time.strftime("run_policy_session_%Y%m%d_%H%M%S")
            # fmt: off
            cmd = [
                "ros2", "bag", "record", "--all",
                "--storage", self.storage,
                "--disable-keyboard-controls",
                "-o", str(self.bag_path),
            ]
            # fmt: on
            # setsid: own process group so we can SIGINT only the recorder on exit.
            self._proc = subprocess.Popen(cmd, preexec_fn=os.setsid)  # noqa: PLW1509
            logger.info(f"💾 Recording session rosbag → {self.bag_path} (ros2 bag record --all)")
        except Exception as exc:  # never let recording break the run
            logger.warning(f"Could not start session rosbag recorder: {exc}")
            self._proc = None
            self.bag_path = None
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._proc is None:
            return
        # SIGINT for a clean MCAP close; SIGKILL if it won't exit.
        try:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGINT)
            self._proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
        except OSError:
            pass  # recorder already exited
        if self.bag_path and self.bag_path.exists():
            logger.info(f"💾 Session rosbag saved: {self.bag_path.resolve()}")
