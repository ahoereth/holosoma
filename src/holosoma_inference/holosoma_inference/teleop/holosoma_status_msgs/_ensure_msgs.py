"""Auto-build the ``holosoma_status_msgs`` ROS2 package on first import.

Experimental. Modeled on bxi_sim_bridge's _ensure_communication. Builds the
.msg files in this package (via colcon, into /tmp/holosoma_status_ws) if the
types aren't already importable, then patches sys.path + pre-loads the native
typesupport .so's so rclpy can use them without sourcing a setup.bash.

Requires colcon + ROS2 on the importing machine (e.g. the Jetson).

    from holosoma_inference.teleop.holosoma_status_msgs._ensure_msgs import Heartbeat
"""

from __future__ import annotations

import ctypes
import importlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

_PKG = "holosoma_status_msgs"
_WS = Path("/tmp/holosoma_status_ws")
_PKG_DIR = Path(__file__).resolve().parent  # this dir IS the ament package
_REQUIRED = ("Heartbeat",)

# Native libs, dependency order (generator_c first, generator_py last).
_SO_ORDER = [
    f"lib{_PKG}__rosidl_generator_c.so",
    f"lib{_PKG}__rosidl_typesupport_introspection_c.so",
    f"lib{_PKG}__rosidl_typesupport_fastrtps_c.so",
    f"lib{_PKG}__rosidl_typesupport_c.so",
    f"lib{_PKG}__rosidl_typesupport_introspection_cpp.so",
    f"lib{_PKG}__rosidl_typesupport_fastrtps_cpp.so",
    f"lib{_PKG}__rosidl_typesupport_cpp.so",
    f"lib{_PKG}__rosidl_generator_py.so",
]


def _available() -> bool:
    try:
        mod = importlib.import_module(f"{_PKG}.msg")
    except (ImportError, ModuleNotFoundError):
        return False
    return all(hasattr(mod, n) for n in _REQUIRED)


def _site_packages_dirs(install_dir: Path):
    """All generated python package dirs under an install prefix.

    ament/colcon on Ubuntu uses ``local/lib/pythonX/dist-packages`` (Debian),
    while plain installs use ``lib/pythonX/site-packages`` — match both, with
    or without the ``local/`` prefix.
    """
    return [p for p in install_dir.glob("**/python*/*-packages") if p.is_dir()]


def _build_is_fresh() -> bool:
    """All required bindings present on disk? Check before any dlopen()."""
    snake = lambda n: re.sub(r"(?<!^)(?=[A-Z])", "_", n).lower()  # noqa: E731
    for sp in _site_packages_dirs(_WS / "install" / _PKG):
        msg_dir = sp / _PKG / "msg"
        if all((msg_dir / f"_{snake(n)}.py").exists() for n in _REQUIRED):
            return True
    return False


def _source_workspace() -> None:
    install_dir = _WS / "install" / _PKG
    if not install_dir.exists():
        return
    for sp in _site_packages_dirs(install_dir):
        if str(sp) not in sys.path:
            sys.path.insert(0, str(sp))
    os.environ["AMENT_PREFIX_PATH"] = f"{install_dir}:{os.environ.get('AMENT_PREFIX_PATH', '')}".rstrip(":")
    lib_dir = install_dir / "lib"
    os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}".rstrip(":")
    for so in _SO_ORDER:
        p = lib_dir / so
        if p.exists():
            try:
                ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
            except OSError as exc:
                print(f"[status_msgs] warn: pre-load {so} failed: {exc}")


def _build() -> None:
    # Build against the SAME distro the runtime has sourced. Using
    # sorted(/opt/ros/*) picks alphabetically (e.g. foxy over humble) and can
    # build .so's the runtime can't load — so prefer $ROS_DISTRO when set.
    distro = os.environ.get("ROS_DISTRO")
    ros_setup = Path(f"/opt/ros/{distro}/setup.bash") if distro else None
    if ros_setup is None or not ros_setup.exists():
        ros_setup = next(iter(sorted(Path("/opt/ros").glob("*/setup.bash"))), None)
    if ros_setup is None:
        raise RuntimeError("ROS2 not found (no /opt/ros/*/setup.bash).")
    if not shutil.which("colcon"):
        raise RuntimeError("colcon not found: sudo apt-get install python3-colcon-common-extensions")

    if _WS.exists():
        shutil.rmtree(_WS)
    for m in [m for m in sys.modules if m == _PKG or m.startswith(f"{_PKG}.")]:
        del sys.modules[m]

    (_WS / "src").mkdir(parents=True, exist_ok=True)
    link = _WS / "src" / _PKG
    link.symlink_to(_PKG_DIR)

    print(f"[status_msgs] building {_PKG} in {_WS} …")
    cmd = f"source {ros_setup} && cd {_WS} && colcon build --packages-select {_PKG}"
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"colcon build failed:\n{r.stdout}\n{r.stderr}")
    print("[status_msgs] built.")


def ensure_msgs() -> None:
    if _available():
        return
    if _build_is_fresh():
        _source_workspace()
        if _available():
            return
    _build()
    _source_workspace()
    if not _available():
        # Surface the real import error instead of a generic message.
        importlib.import_module(f"{_PKG}.msg")
        raise RuntimeError(f"built {_PKG} but import succeeded yet a required type is missing: {_REQUIRED}")


ensure_msgs()

from holosoma_status_msgs.msg import Heartbeat  # noqa: E402

__all__ = ["Heartbeat"]
