"""Console entry: WBT policy tracking a live DenseTrackingCmd stream.

Builds holosoma's own ``WholeBodyTrackingPolicy``, injects a ``DenseTargetSource``
(subscribes DenseTrackingCmd) as its target source, and runs the policy's own
``run()`` loop. Holosoma-native — no FAR-pi import. rclpy lives only in the
injected source; the policy core stays agnostic.
"""

from __future__ import annotations

import tyro

from holosoma_inference.config.config_values.inference import get_annotated_inference_config
from holosoma_inference.config.utils import TYRO_CONFIG
from holosoma_inference.policies.wbt import WholeBodyTrackingPolicy
from holosoma_service.dense_target_source import DenseTargetSource


def main() -> None:
    config = tyro.cli(get_annotated_inference_config(), config=TYRO_CONFIG)
    policy = WholeBodyTrackingPolicy(config=config)
    policy._target_source = DenseTargetSource(config.robot.num_joints)
    policy.run()  # owns its RateLimiter + SDK I/O; reads get_target() each step


if __name__ == "__main__":
    main()
