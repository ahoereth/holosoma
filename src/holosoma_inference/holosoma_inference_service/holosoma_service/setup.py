from glob import glob

from setuptools import find_packages, setup

package_name = "holosoma_service"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),  # noqa: PTH207
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Tomasz Lewicki",
    maintainer_email="jtomasle@amazon.com",
    description="ROS2 service nodes + launch for holosoma teleop.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "retargeter_node = holosoma_service.retargeter_node:main",
            "policy_node = holosoma_service.policy_node:main",
            "teleop_listener_node = holosoma_service.holosoma_teleop_listener_node:main",
            "wasd_controller_node = holosoma_service.wasd_controller_node:main",
        ],
    },
)
