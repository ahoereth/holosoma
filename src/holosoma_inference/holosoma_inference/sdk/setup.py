import platform

from setuptools import setup

UNITREE_VERSION = "0.1.1"
UNITREE_REPO = "https://github.com/amazon-far/unitree_sdk2"
BOOSTER_VERSION = "0.1.0"
BOOSTER_REPO = "https://github.com/amazon-far/booster_robotics_sdk"

PLATFORM_MAP = {
    "x86_64": "linux_x86_64",
    "aarch64": "linux_aarch64",
}

platform_tag = PLATFORM_MAP.get(platform.machine(), "linux_x86_64")

unitree_url = (
    f"{UNITREE_REPO}/releases/download/{UNITREE_VERSION}/unitree_sdk2-{UNITREE_VERSION}-cp310-cp310-{platform_tag}.whl"
)
booster_url = f"{BOOSTER_REPO}/releases/download/{BOOSTER_VERSION}/booster_robotics_sdk-{BOOSTER_VERSION}-cp310-cp310-{platform_tag}.whl"  # noqa: E501

setup(
    extras_require={
        "unitree": [f"unitree_sdk2 @ {unitree_url}"],
        "booster": [f"booster_robotics_sdk @ {booster_url}"],
    },
)
