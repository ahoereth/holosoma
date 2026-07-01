from __future__ import annotations

from dataclasses import field

from pydantic.dataclasses import dataclass

from holosoma.config_types.experiment import TrainingConfig
from holosoma.config_types.logger import LoggerConfig
from holosoma.config_types.robot import RobotConfig
from holosoma.config_types.scene import SceneConfig
from holosoma.config_types.sensor_egress import SensorEgressConfig
from holosoma.config_types.sensors import SensorsConfig
from holosoma.config_types.simulator import SimulatorInitConfig


@dataclass(frozen=True)
class FullSimConfig:
    """Collection of configs needed for constructing simulator classes."""

    simulator: SimulatorInitConfig
    robot: RobotConfig
    training: TrainingConfig
    logger: LoggerConfig
    """Logger configuration for video recording and output directories."""

    scene: SceneConfig = field(default_factory=SceneConfig)
    """Scene composition (rigid objects, scene files)."""

    sensors: SensorsConfig = field(default_factory=SensorsConfig)
    """Sensor composition (mounted cameras; depth/seg + non-camera sensors reserved)."""

    sensor_egress: SensorEgressConfig = field(default_factory=SensorEgressConfig)
    """Outbound sensor-frame publishing (ROS2 camera egress). Empty (default) = no egress."""

    experiment_dir: str | None = None
    """Experiment directory path (computed from logger config in base_task)."""
