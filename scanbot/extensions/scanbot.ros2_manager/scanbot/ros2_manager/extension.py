"""Omniverse extension entrypoint for scanbot.ros2_manager."""

from __future__ import annotations

import threading
import os

import carb
import omni.ext
import omni.kit.app

from scanbot.scripts import scanbot_context

from .camera_bridge import CameraBridge
from .marker_bridge import MarkerBridge
from .publishers import JointStatePublisher, ScanpointPosePublisher, TcpPosePublisher
from .ros_env import Ros2Imports, ensure_ros2_available
from .services import JointLimitsService, ResetEnvService
from .target_tcp_action import TargetTcpAction
from .teleport_actions import TeleportActions


try:
    import isaaclab.utils.math as math_utils
except Exception:  # pragma: no cover - Isaac Lab runtime only
    math_utils = None


class Extension(omni.ext.IExt):
    def __init__(self) -> None:
        super().__init__()
        self._ext_id = ""
        self._sub = None
        self._timer = None
        self._hook_update_enabled = False
        self._debug = False
        self._update_counter = 0

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
        self._debug = str(os.getenv("SCANBOT_ROS2_DEBUG", "0")).lower() in {"1", "true", "yes"}
        if self._debug:
            try:
                carb.log_info(
                    "[scanbot.ros2_manager] scanbot_context id=%s file=%s"
                    % (id(scanbot_context), getattr(scanbot_context, "__file__", "n/a"))
                )
            except Exception:
                pass

        ros_ok, ros_err, ros = ensure_ros2_available()
        if not ros_ok or ros is None:
            msg = f"[scanbot.ros2_manager] ROS2 unavailable: {ros_err}. Extension will stay idle."
            try:
                carb.log_error(msg)
            except Exception:
                pass
            return
        self._ros = ros

        if math_utils is None:
            msg = "[scanbot.ros2_manager] Isaac Lab math utils missing; extension idle."
            try:
                carb.log_error(msg)
            except Exception:
                pass
            return

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

        use_hook = str(os.getenv("SCANBOT_ROS2_HOOK_UPDATE", "0")).lower() in {"1", "true", "yes"}
        use_timer = str(os.getenv("SCANBOT_ROS2_TIMER_UPDATE", "0")).lower() in {"1", "true", "yes"}
        headless = str(os.getenv("HEADLESS", "0")).lower() in {"1", "true", "yes"}
        if headless and not use_hook:
            use_timer = True
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

        try:
            carb.log_info(
                f"[scanbot.ros2_manager] update mode: hook={use_hook} timer={use_timer}"
            )
        except Exception:
            pass

        try:
            self._node.get_logger().info("Scanbot ROS2 Manager started")
        except Exception:
            pass
        try:
            carb.log_info("[scanbot.ros2_manager] Started")
        except Exception:
            pass

    def on_shutdown(self) -> None:
        if self._sub is not None:
            self._sub.unsubscribe()
            self._sub = None
        self._hook_update_enabled = False
        if self._timer is not None and self._node is not None:
            try:
                self._node.destroy_timer(self._timer)
            except Exception:
                pass
            self._timer = None

        # Stop ROS callbacks early so we don't leak action servers across hot-reloads.
        if self._executor is not None and self._node is not None:
            try:
                self._executor.remove_node(self._node)
            except Exception:
                pass

        if self._target_tcp is not None:
            self._target_tcp.shutdown()
            self._target_tcp = None

        if self._teleport is not None:
            self._teleport.shutdown()
            self._teleport = None

        if self._reset_srv is not None:
            self._reset_srv.shutdown()
            self._reset_srv = None

        if self._limits_srv is not None:
            self._limits_srv.shutdown()
            self._limits_srv = None

        if self._tcp_pub is not None:
            self._tcp_pub.shutdown()
            self._tcp_pub = None

        if self._sp_pub is not None:
            self._sp_pub.shutdown()
            self._sp_pub = None

        if self._joint_pub is not None:
            self._joint_pub.shutdown()
            self._joint_pub = None

        if self._camera_bridge is not None:
            self._camera_bridge.shutdown()
            self._camera_bridge = None

        if self._marker_bridge is not None:
            self._marker_bridge.shutdown()
            self._marker_bridge = None

        # Always tear down the global rclpy context on unload. This makes executor
        # threads exit promptly (spin() checks rclpy.ok()) and prevents stale DDS
        # entities (e.g. action servers) from accumulating across hot-reloads.
        if self._ros is not None and self._ros.rclpy.ok():
            try:
                self._ros.rclpy.shutdown()
            except Exception:
                pass

        if self._executor is not None:
            try:
                self._executor.shutdown()
            except Exception:
                pass
            self._executor = None

        if self._executor_thread is not None:
            self._executor_thread.join(timeout=10.0)
            self._executor_thread = None

        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
            self._node = None
        self._ros_initialized = False

        try:
            carb.log_info(f"[scanbot.ros2_manager] Stopped: {self._ext_id}")
        except Exception:
            pass

    def _on_update(self, _event=None) -> None:
        env = scanbot_context.get_env()
        if self._update_counter == 0:
            try:
                carb.log_info(
                    f"[scanbot.ros2_manager] _on_update first call (env {'set' if env is not None else 'missing'})"
                )
            except Exception:
                pass
        self._update_counter += 1
        if env is None:
            if self._target_tcp is not None:
                self._target_tcp.maybe_timeout_when_env_missing()
            if self._teleport is not None:
                self._teleport.maybe_timeout_when_env_missing()
            return

        action_term = None
        curr_pos = None
        curr_quat = None

        try:
            action_term = env.action_manager.get_term("arm_action")
        except Exception as exc:
            action_term = None
            if self._target_tcp is not None and self._target_tcp.has_active_goal():
                self._target_tcp.abort_active_goal(f"Missing arm_action term: {exc}")

        if action_term is not None and not hasattr(action_term, "_compute_frame_pose"):
            if self._target_tcp is not None and self._target_tcp.has_active_goal():
                self._target_tcp.abort_active_goal("arm_action does not expose _compute_frame_pose()")

        if action_term is not None and hasattr(action_term, "_compute_frame_pose"):
            try:
                curr_pos, curr_quat = action_term._compute_frame_pose()
            except Exception as exc:
                curr_pos = None
                curr_quat = None
                if self._target_tcp is not None and self._target_tcp.has_active_goal():
                    self._target_tcp.abort_active_goal(f"Failed to read current TCP pose: {exc}")

        if curr_pos is None or curr_quat is None:
            # Fall back to the end-effector frame transformer when action terms don't expose pose.
            try:
                ee_frame = env.scene["ee_frame"]
            except Exception:
                ee_frame = None
            if ee_frame is not None:
                data = getattr(ee_frame, "data", None)
                pos_src = getattr(data, "target_pos_source", None) if data is not None else None
                quat_src = getattr(data, "target_quat_source", None) if data is not None else None
                if pos_src is not None and quat_src is not None:
                    idx = 0
                    names = getattr(data, "target_frame_names", None)
                    if names and "end_effector" in names:
                        idx = names.index("end_effector")
                    try:
                        curr_pos = pos_src[:, idx, :]
                        curr_quat = quat_src[:, idx, :]
                    except Exception:
                        curr_pos = None
                        curr_quat = None

        if self._tcp_pub is not None and curr_pos is not None and curr_quat is not None:
            self._tcp_pub.maybe_publish(curr_pos, curr_quat, frame_id="base")

        if self._sp_pub is not None and curr_pos is not None and curr_quat is not None:
            self._sp_pub.maybe_publish(curr_pos, curr_quat)

        if self._joint_pub is not None:
            try:
                robot = env.scene["robot"]
            except Exception:
                robot = None
            if robot is not None:
                self._joint_pub.maybe_publish(robot)

        if self._camera_bridge is not None:
            self._camera_bridge.maybe_publish(env)
        if self._marker_bridge is not None:
            self._marker_bridge.maybe_render(env)

        if self._teleport is not None:
            self._teleport.maybe_update(env, action_term, curr_pos, curr_quat)

        if self._target_tcp is None:
            return

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
