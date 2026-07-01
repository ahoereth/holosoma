"""Local-visualization sensor egress: tile mounted-camera views into a live cv2 window and/or mp4.

The egress-family successor to ``CameraSensorRecorder``. It is a normal :class:`SensorEgress`: the
driver snapshots its wanted ``(camera, modality, env)`` streams once per step (the single GPU→host
copy) and hands them to :meth:`publish` as a batch; this class colorizes depth, tiles the panels
into one grid (rows = envs, cols = (camera, modality)), and shows it live and/or buffers it for an
H.264 file at :meth:`stop`. cv2 + video utils are imported at module top — this module is reached
only via ``VizEgressConfig.egress_cls``, so importing the egress package stays cv2-free.

Inherently inline (composes + shows/buffers on the calling thread) — no async worker path: a live
cv2 window must be driven where the loop runs, and the per-step grid is cheap.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from loguru import logger

from holosoma.config_types.frequency import is_frequency_string, resolve_decimation
from holosoma.sensor_egress.base import SensorEgress
from holosoma.sensor_egress.viz.image_grid import colorize_depth, tile_images
from holosoma.utils.video_utils import create_video

if TYPE_CHECKING:
    from holosoma.config_types.sensor_egress import VizEgressConfig
    from holosoma.sensor_egress.base import FramePacket, StreamKey
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator

_WINDOW = "holosoma camera sensors"


class VizEgress(SensorEgress):
    """Tiling cv2-window / mp4 sink over mounted-camera frames."""

    config: VizEgressConfig

    def __init__(self, config: VizEgressConfig, simulator: BaseSimulator) -> None:
        super().__init__(config, simulator)
        self._frames_video: list[np.ndarray] = []  # buffered grids for the mp4
        self._step = -1  # batches seen (proxy for the fastest watched camera's render count)
        self._last_captured = -1

        # Cameras to watch: configured selection or all cameras in the active SensorsConfig.
        all_cams = list(self.sensors_config.cameras)
        self._cam_names = config.cameras if config.cameras is not None else all_cams
        cams_by_name = dict(self.sensors_config.cameras)

        # A PANEL is one (camera, modality): its own grid column. For each watched camera take the
        # modalities it actually produces, intersected with the config's modality selection.
        self._panels: list[tuple[str, str]] = []
        for name in self._cam_names:
            cam_mods = list(cams_by_name[name].data_types)
            mods = cam_mods if config.modalities is None else [m for m in config.modalities if m in cam_mods]
            self._panels.extend((name, m) for m in mods)
        if not self._panels:
            logger.warning(f"VizEgress: selected modalities {config.modalities} match no camera output.")

        # Panel label disambiguation: append ":modality" only for cameras shown with >1 modality.
        per_cam = Counter(name for name, _ in self._panels)
        self._multi_modality = {name for name, n in per_cam.items() if n > 1}

        self._env_ids = list(config.env_ids)
        self._depth_range = (config.depth_range[0], config.depth_range[1]) if config.depth_range else (0.01, 5.0)
        self._frame_decimation = self._resolve_frame_decimation()

        # Live window viable only with a non-headless sim AND a usable display.
        self._show_live = config.live_window and not simulator.headless and bool(os.environ.get("DISPLAY"))
        if config.live_window and not self._show_live:
            logger.warning("VizEgress: live_window requested but no display; window disabled.")
        self._window_open = False
        self._cell_wh = {name: (c.width, c.height) for name, c in self.sensors_config.cameras.items()}

    def _resolve_frame_decimation(self) -> int:
        # update_decimation is in units of the fastest watched camera's RENDERED frames. An int is
        # already that; a frequency string is a target against the control rate, converted to frames
        # via the fastest watched camera's render decimation (d_min control steps between its frames).
        cfg = self.config
        if not is_frequency_string(cfg.update_decimation):
            return int(cfg.update_decimation)
        # We do not see SensorManager here; approximate d_min as 1 (publish() already only fires on
        # steps a panel rendered, so the batch cadence is the fastest camera's frame cadence).
        n_viz = resolve_decimation(cfg.update_decimation, self.control_hz, field="recorder update_decimation")
        return max(1, n_viz)

    def wanted_streams(self) -> set[StreamKey]:
        return {(cam, mod, env) for (cam, mod) in self._panels for env in self._env_ids}

    def start(self) -> None:
        logger.info(
            f"VizEgress active: panels={[f'{n}:{m}' for n, m in self._panels]} "
            f"envs={self._env_ids} live_window={self._show_live} record_video={self.config.record_video}"
        )

    def publish(self, frames: dict[StreamKey, FramePacket]) -> None:
        if not self._panels or not (self._show_live or self.config.record_video):
            return
        # Batch arrival == this step's fastest watched camera rendered. Gate by frame-decimation.
        self._step += 1
        if self._step - self._last_captured < self._frame_decimation:
            return
        self._last_captured = self._step

        views: list[np.ndarray] = []
        labels: list[str] = []
        for env in self._env_ids:
            for name, modality in self._panels:
                packet = frames.get((name, modality, env))
                if packet is None:
                    w, h = self._cell_wh.get(name, (128, 128))
                    views.append(self._missing_tile(w, h))
                else:
                    img = packet.array
                    if modality == "depth":
                        img = colorize_depth(img, self._depth_range, self.config.depth_colormap)
                    views.append(img)
                label = f"env{env}/{name}"
                labels.append(f"{label}:{modality}" if name in self._multi_modality else label)
        grid = tile_images(views, layout=(len(self._env_ids), len(self._panels)), labels=labels)  # RGB

        if self._show_live:
            self._show(grid)
        if self.config.record_video:
            self._frames_video.append(grid)

    def _show(self, grid: np.ndarray) -> None:
        cv2.imshow(_WINDOW, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        cv2.pollKey()  # non-blocking HighGUI pump (unlike waitKey(1))
        if self._window_open and cv2.getWindowProperty(_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            self._show_live = False  # user closed it -> stop drawing live for the rest of the run
            self._window_open = False
        else:
            self._window_open = True

    @staticmethod
    def _missing_tile(width: int, height: int, cell: int = 16) -> np.ndarray:
        """Source-style magenta/black 'missing texture' for a panel with no frame yet (RGB)."""
        ys = (np.arange(height) // cell)[:, None]
        xs = (np.arange(width) // cell)[None, :]
        tile = np.zeros((height, width, 3), dtype=np.uint8)
        tile[(ys + xs) % 2 == 1] = (255, 0, 255)
        return tile

    def stop(self) -> None:
        if self.config.record_video and self._frames_video:
            # Capture cadence = the recorder's frame-decimation (batches already track the fastest
            # camera's render rate). Encode fps so the video plays at true wall-clock speed.
            fps = self.control_hz / self._frame_decimation * self.config.playback_rate
            create_video(
                np.array(self._frames_video, dtype=np.uint8),
                fps=fps,
                save_dir=str(self._save_dir()),
                output_format="h264",
                wandb_logging=False,
            )
            self._frames_video = []
        if self._window_open:
            cv2.destroyWindow(_WINDOW)
            self._window_open = False

    def _save_dir(self) -> Path:
        if self.config.save_dir is not None:
            return Path(self.config.save_dir)
        spectator_dir = self.simulator.video_config.save_dir
        return Path(spectator_dir) if spectator_dir else Path("logs/camera_sensors")
