"""Omniverse extension entrypoint for scanbot.ros2_manager."""

from __future__ import annotations

import threading
import os

import carb
import omni.ext
import omni.kit.app

from scanbot.scripts import scanbot_context
from scanbot.scripts.utilities import pos_util

from .camera_bridge import CameraBridge
from .marker_bridge import MarkerBridge
from .publishers import JointStatePublisher, ScanpointPosePublisher, TcpPosePublisher
from .ros_env import Ros2Imports, ensure_ros2_available
from .services import JointLimitsService, ResetEnvService
from .target_tcp_action import TargetTcpAction
from .teleport_actions import TeleportActions

import isaaclab.utils.math as math_utils


class Extension(omni.ext.IExt):
    def __init__(self) -> None:
        super().__init__()
        self._ext_id = ""
        self._sub = None
        self._timer = None
        self._hook_update_enabled = False
        self._debug = False
        self._update_counter = 0
        self._started = False

        self._ros_initialized = False
        self._ros: Ros2Imports | None = None
        self._node = None
        self._executor = None
        self._executor_thread: threading.Thread | None = None

        self._target_tcp: TargetTcpAction | None = None
        self._teleport: TeleportActions | None = None
        self._tcp_pub: TcpPosePublisher | None = None
        self._sp_pub: ScanpointPosePublisher | None = None
        self._joint_pub: JointStatePublisher | None = None
        self._reset_srv: ResetEnvService | None = None
        self._limits_srv: JointLimitsService | None = None

        self._camera_bridge: CameraBridge | None = None
        self._marker_bridge: MarkerBridge | None = None

    def on_startup(self, ext_id: str) -> None:
        self._ext_id = ext_id
        self._debug = os.getenv("SCANBOT_ROS2_DEBUG", "0") == "1"
        if self._debug:
            carb.log_info(
                "[scanbot.ros2_manager] scanbot_context id=%s file=%s"
                % (id(scanbot_context), scanbot_context.__file__)
            )

        ros_ok, ros_err, ros = ensure_ros2_available()
        if not ros_ok or ros is None:
            msg = f"[scanbot.ros2_manager] ROS2 unavailable: {ros_err}. Extension will stay idle."
            carb.log_error(msg)
            return
        self._ros = ros

        if not ros.rclpy.ok():
            ros.rclpy.init()
            self._ros_initialized = True

        self._node = ros.rclpy.create_node("scanbot_ros2_manager")

        self._executor = ros.MultiThreadedExecutor()
        self._executor.add_node(self._node)
        self._executor_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._executor_thread.start()

        self._target_tcp = TargetTcpAction(self._node, ros, math_utils)
        self._teleport = TeleportActions(self._node, ros, goal_active_fn=self._target_tcp.has_active_goal)
        self._tcp_pub = TcpPosePublisher(self._node, ros.PoseStamped)
        self._sp_pub = ScanpointPosePublisher(self._node, ros.PoseStamped)
        self._joint_pub = JointStatePublisher(self._node, ros.JointState)
        self._reset_srv = ResetEnvService(self._node, ros, goal_active_fn=self._target_tcp.has_active_goal)
        self._limits_srv = JointLimitsService(self._node, ros)

        self._camera_bridge = CameraBridge(
            self._node,
            ros.Image,
            ros.CameraInfo,
            ros.PoseStamped,
            ros.PointCloud2,
            ros.PointField,
            ros.CAMERA_QOS,
        )
        self._marker_bridge = MarkerBridge(self._node, ros.MarkerPoseArray, ros.Empty)

        def _env_is_one(name: str) -> bool:
            return os.getenv(name, "") == "1"

        use_hook = _env_is_one("SCANBOT_ROS2_HOOK_UPDATE")
        use_timer = _env_is_one("SCANBOT_ROS2_TIMER_UPDATE")

        headless = False
        app_launcher = scanbot_context.get_app_launcher()
        if app_launcher is not None:
            headless = bool(getattr(app_launcher, "_headless", False))
        if not headless:
            headless = _env_is_one("HEADLESS") or _env_is_one("SCANBOT_HEADLESS")
        if not headless:
            headless = not os.getenv("DISPLAY")
        if headless and not use_hook and not use_timer:
            use_hook = True
        if use_hook:
            self._hook_update_enabled = True

            def _hook() -> None:
                if not self._hook_update_enabled:
                    return
                self._on_update(None)
                scanbot_context.enqueue_hook(_hook)

            scanbot_context.enqueue_hook(_hook)
        elif use_timer:
            self._timer = self._node.create_timer(0.02, self._on_update)
        else:
            app = omni.kit.app.get_app()
            stream = app.get_update_event_stream()
            self._sub = stream.create_subscription_to_pop(self._on_update, name="scanbot.ros2_manager.update")

        carb.log_info(f"[scanbot.ros2_manager] update mode: hook={use_hook} timer={use_timer}")
        self._node.get_logger().info("Scanbot ROS2 Manager started")
        carb.log_info("[scanbot.ros2_manager] Started")
        self._started = True

    def on_shutdown(self) -> None:
        if not self._started:
            return
        if self._sub is not None:
            self._sub.unsubscribe()
            self._sub = None
        self._hook_update_enabled = False
        if self._timer is not None:
            self._node.destroy_timer(self._timer)
            self._timer = None

        # Stop ROS callbacks early so we don't leak action servers across hot-reloads.
        self._executor.remove_node(self._node)

        self._target_tcp.shutdown()
        self._target_tcp = None

        self._teleport.shutdown()
        self._teleport = None

        self._reset_srv.shutdown()
        self._reset_srv = None

        self._limits_srv.shutdown()
        self._limits_srv = None

        self._tcp_pub.shutdown()
        self._tcp_pub = None

        self._sp_pub.shutdown()
        self._sp_pub = None

        self._joint_pub.shutdown()
        self._joint_pub = None

        self._camera_bridge.shutdown()
        self._camera_bridge = None

        self._marker_bridge.shutdown()
        self._marker_bridge = None

        # Always tear down the global rclpy context on unload. This makes executor
        # threads exit promptly (spin() checks rclpy.ok()) and prevents stale DDS
        # entities (e.g. action servers) from accumulating across hot-reloads.
        if self._ros.rclpy.ok():
            self._ros.rclpy.shutdown()

        self._executor.shutdown()
        self._executor = None

        self._executor_thread.join(timeout=10.0)
        self._executor_thread = None

        self._node.destroy_node()
        self._node = None
        self._ros_initialized = False
        self._started = False

        carb.log_info(f"[scanbot.ros2_manager] Stopped: {self._ext_id}")

    def _on_update(self, _event=None) -> None:
        env = scanbot_context.get_env()
        if self._update_counter == 0:
            carb.log_info(
                f"[scanbot.ros2_manager] _on_update first call (env {'set' if env is not None else 'missing'})"
            )
        self._update_counter += 1
        if env is None:
            self._target_tcp.maybe_timeout_when_env_missing()
            self._teleport.maybe_timeout_when_env_missing()
            return
        pos_util.configure_from_env(env)

        action_term = env.action_manager.get_term("arm_action")
        curr_pos, curr_quat = action_term._compute_frame_pose()

        # Apply teleports before publishing state/sensors to avoid a 1-tick lag where
        # images/joints/poses reflect the pre-teleport state.
        self._teleport.maybe_update(env, action_term, curr_pos, curr_quat)
        curr_pos, curr_quat = action_term._compute_frame_pose()
        self._teleport.maybe_update(env, action_term, curr_pos, curr_quat)

        if curr_pos is not None and curr_quat is not None:
            self._tcp_pub.maybe_publish(curr_pos, curr_quat, frame_id="base")
            self._sp_pub.maybe_publish(curr_pos, curr_quat)

        robot = env.scene["robot"]
        self._joint_pub.maybe_publish(robot)

        self._camera_bridge.maybe_publish(env)
        self._marker_bridge.maybe_render(env)

        if action_term is None or curr_pos is None or curr_quat is None:
            return

        had_goal = self._target_tcp.has_active_goal()
        action = self._target_tcp.maybe_compute_action(env, curr_pos, curr_quat)
        if action is None:
            # If the goal completed/aborted, ensure any previously queued DIK action
            # is not applied again by the launcher loop.
            if had_goal and not self._target_tcp.has_active_goal():
                scanbot_context.clear_actions()
            return

        scanbot_context.clear_actions()
        scanbot_context.enqueue_action(action)
