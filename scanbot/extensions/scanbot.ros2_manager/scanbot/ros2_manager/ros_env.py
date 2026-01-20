"""ROS 2 bootstrap and import helpers for scanbot.ros2_manager."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os
import subprocess
import sys

import carb


@dataclass
class Ros2Imports:
    rclpy: Any
    ActionServer: Any
    ActionClient: Any
    CancelResponse: Any
    GoalResponse: Any
    ReentrantCallbackGroup: Any
    MultiThreadedExecutor: Any
    TargetTcp: Any
    TeleportJointsAction: Any
    TeleportTcpAction: Any
    TeleportJoints: Any
    TeleportTcp: Any
    GetJointLimits: Any
    MarkerPoseArray: Any
    PoseStamped: Any
    Image: Any
    CameraInfo: Any
    PointCloud2: Any
    PointField: Any
    JointState: Any
    Empty: Any
    Trigger: Any
    CAMERA_QOS: Any


_ROS_AVAILABLE = False
_ROS_IMPORT_ERROR = ""
_ROS_ENV_BOOTSTRAPPED = False
_ROS_IMPORTS: Ros2Imports | None = None


def ensure_ros2_available() -> tuple[bool, str, Ros2Imports | None]:
    """Ensure ROS 2 is importable and type support is available."""
    global _ROS_AVAILABLE
    typesupport_ok = _ros2_typesupport_ok() if _ROS_AVAILABLE else False
    if not _ROS_AVAILABLE or not typesupport_ok or not _ros_imports_ready():
        _bootstrap_ros2_env()
        _try_import_ros()
        typesupport_ok = _ros2_typesupport_ok() if _ROS_AVAILABLE else False
    return _ROS_AVAILABLE and typesupport_ok, _ROS_IMPORT_ERROR, _ROS_IMPORTS


def _try_import_ros() -> bool:
    global _ROS_AVAILABLE, _ROS_IMPORT_ERROR, _ROS_IMPORTS
    try:
        import rclpy as _rclpy  # type: ignore
        from rclpy.action import (
            ActionClient as _ActionClient,
            ActionServer as _ActionServer,
            CancelResponse as _CancelResponse,
            GoalResponse as _GoalResponse,
        )
        from rclpy.callback_groups import ReentrantCallbackGroup as _ReentrantCallbackGroup
        from rclpy.executors import MultiThreadedExecutor as _MultiThreadedExecutor
        from rclpy.qos import QoSProfile as _QoSProfile, ReliabilityPolicy as _ReliabilityPolicy, HistoryPolicy as _HistoryPolicy
        # Reload scanbot_msgs.srv to pick up newly generated service classes.
        try:
            import importlib
            import scanbot_msgs.srv as _scanbot_msgs_srv

            importlib.invalidate_caches()
            importlib.reload(_scanbot_msgs_srv)
        except Exception:
            pass
        # Reload scanbot_msgs.action to pick up newly generated action classes.
        try:
            import importlib
            import scanbot_msgs.action as _scanbot_msgs_action

            importlib.invalidate_caches()
            importlib.reload(_scanbot_msgs_action)
        except Exception:
            pass
        # Reload scanbot_msgs.msg to pick up newly generated message classes.
        try:
            import importlib
            import scanbot_msgs.msg as _scanbot_msgs_msg

            importlib.invalidate_caches()
            importlib.reload(_scanbot_msgs_msg)
        except Exception:
            pass
        from scanbot_msgs.action import TargetTcp as _TargetTcp
        from scanbot_msgs.action import TeleportJoints as _TeleportJointsAction
        from scanbot_msgs.action import TeleportTcp as _TeleportTcpAction
        from scanbot_msgs.srv import TeleportJoints as _TeleportJoints
        from scanbot_msgs.srv import TeleportTcp as _TeleportTcp
        try:
            from scanbot_msgs.srv import GetJointLimits as _GetJointLimits
        except Exception:
            _GetJointLimits = None
        from scanbot_msgs.msg import MarkerPoseArray as _MarkerPoseArray
        from geometry_msgs.msg import PoseStamped as _PoseStamped
        from sensor_msgs.msg import Image as _Image
        from sensor_msgs.msg import CameraInfo as _CameraInfo
        from sensor_msgs.msg import PointCloud2 as _PointCloud2
        from sensor_msgs.msg import PointField as _PointField
        from sensor_msgs.msg import JointState as _JointState
        from std_msgs.msg import Empty as _Empty
        from std_srvs.srv import Trigger as _Trigger

        _ROS_IMPORTS = Ros2Imports(
            rclpy=_rclpy,
            ActionServer=_ActionServer,
            ActionClient=_ActionClient,
            CancelResponse=_CancelResponse,
            GoalResponse=_GoalResponse,
            ReentrantCallbackGroup=_ReentrantCallbackGroup,
            MultiThreadedExecutor=_MultiThreadedExecutor,
            TargetTcp=_TargetTcp,
            TeleportJointsAction=_TeleportJointsAction,
            TeleportTcpAction=_TeleportTcpAction,
            TeleportJoints=_TeleportJoints,
            TeleportTcp=_TeleportTcp,
            GetJointLimits=_GetJointLimits,
            MarkerPoseArray=_MarkerPoseArray,
            PoseStamped=_PoseStamped,
            Image=_Image,
            CameraInfo=_CameraInfo,
            PointCloud2=_PointCloud2,
            PointField=_PointField,
            JointState=_JointState,
            Empty=_Empty,
            Trigger=_Trigger,
            CAMERA_QOS=_QoSProfile(
                depth=1,
                reliability=_ReliabilityPolicy.RELIABLE,
                history=_HistoryPolicy.KEEP_LAST,
            ),
        )
        _ROS_AVAILABLE = True
        _ROS_IMPORT_ERROR = ""
        return True
    except Exception as exc:  # pragma: no cover
        _ROS_AVAILABLE = False
        _ROS_IMPORT_ERROR = str(exc)
        _ROS_IMPORTS = None
        return False


def _bootstrap_ros2_env() -> None:
    global _ROS_ENV_BOOTSTRAPPED
    if _ROS_ENV_BOOTSTRAPPED:
        return

    setup_scripts: list[str] = []
    ros_setup = "/opt/ros/humble/setup.bash"
    if os.path.isfile(ros_setup):
        setup_scripts.append(ros_setup)

    # Try workspace location relative to this extension path.
    try:
        ext_path = Path(__file__).resolve()
        for parent in ext_path.parents:
            ws_setup = parent / "scanbot" / "ros2" / "install" / "setup.bash"
            if ws_setup.is_file():
                setup_scripts.append(str(ws_setup))
                break
    except Exception:
        pass

    # Try common workspace locations for scanbot_msgs install.
    candidate_roots = ["/workspace/isaaclab", "/workspace"]
    for root in candidate_roots:
        ws_setup = Path(root) / "scanbot" / "ros2" / "install" / "setup.bash"
        if ws_setup.is_file():
            setup_scripts.append(str(ws_setup))
            break

    if not setup_scripts:
        _ROS_ENV_BOOTSTRAPPED = True
        return

    def _prepend_env_path(var_name: str, value: str) -> None:
        if not value:
            return
        cur = os.environ.get(var_name, "")
        parts = [p for p in cur.split(":") if p]
        if value in parts:
            return
        parts.insert(0, value)
        os.environ[var_name] = ":".join(parts)

    cmd = " && ".join([f"source {script}" for script in setup_scripts]) + " && env -0"
    try:
        output = subprocess.check_output(["bash", "-lc", cmd])
    except Exception:
        _ROS_ENV_BOOTSTRAPPED = True
        return

    for entry in output.split(b"\0"):
        if not entry:
            continue
        key, _, value = entry.partition(b"=")
        if not key:
            continue
        os.environ[key.decode()] = value.decode()

    # Prepend PYTHONPATH entries to sys.path so ROS2 packages take precedence over any
    # bundled / preinstalled modules that may exist in the Kit Python environment.
    py_path = os.environ.get("PYTHONPATH", "")
    py_paths = [p for p in py_path.split(":") if p]
    for path in reversed(py_paths):
        try:
            while path in sys.path:
                sys.path.remove(path)
        except Exception:
            pass
        sys.path.insert(0, path)

    # If any ROS2-related modules were imported before bootstrapping (e.g. from Kit's
    # Python environment), purge them so we can re-import the ROS Humble versions.
    for prefix in ("rclpy", "rosidl_generator_py", "rosidl_parser", "scanbot_msgs"):
        for name in list(sys.modules.keys()):
            if name == prefix or name.startswith(prefix + "."):
                try:
                    del sys.modules[name]
                except Exception:
                    pass

    # Ensure scanbot_msgs libs and python packages are reachable even if hooks are incomplete.
    for root in candidate_roots:
        install_root = Path(root) / "scanbot" / "ros2" / "install"
        pkg_root = install_root / "scanbot_msgs"
        if pkg_root.is_dir():
            lib_dir = pkg_root / "lib"
            if lib_dir.is_dir():
                _prepend_env_path("LD_LIBRARY_PATH", str(lib_dir))
            local_lib = pkg_root / "local" / "lib"
            if local_lib.is_dir():
                _prepend_env_path("LD_LIBRARY_PATH", str(local_lib))
            for py_dir in pkg_root.glob("local/lib/python*/dist-packages"):
                _prepend_env_path("PYTHONPATH", str(py_dir))
                py_dir_str = str(py_dir)
                try:
                    while py_dir_str in sys.path:
                        sys.path.remove(py_dir_str)
                except Exception:
                    pass
                sys.path.insert(0, py_dir_str)
            break

    msg = f"[scanbot.ros2_manager] Bootstrapped ROS2 env with: {setup_scripts}"
    try:
        carb.log_info(msg)
    except Exception:
        pass
    _ROS_ENV_BOOTSTRAPPED = True


def _ros2_typesupport_ok() -> bool:
    try:
        from rosidl_generator_py import import_type_support as _import_type_support

        _import_type_support("scanbot_msgs")
        return True
    except Exception as exc:  # pragma: no cover - runtime only
        global _ROS_IMPORT_ERROR
        _ROS_IMPORT_ERROR = str(exc)
        return False


def _ros_imports_ready() -> bool:
    if _ROS_IMPORTS is None:
        return False
    return all(
        hasattr(_ROS_IMPORTS, attr)
        for attr in (
            "TargetTcp",
            "TeleportJointsAction",
            "TeleportTcpAction",
            "TeleportTcp",
            "TeleportJoints",
            "MarkerPoseArray",
            "PoseStamped",
            "Image",
            "CameraInfo",
            "JointState",
            "Empty",
            "Trigger",
        )
    )


_try_import_ros()
