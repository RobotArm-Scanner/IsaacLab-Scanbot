"""Misc ROS2 services for scanbot.ros2_manager."""

from __future__ import annotations

from scanbot.scripts import scanbot_context

from . import config as _cfg

RESET_ENV_SERVICE = _cfg.RESET_ENV_SERVICE
GET_JOINT_LIMITS_SERVICE = _cfg.GET_JOINT_LIMITS_SERVICE


class ResetEnvService:
    def __init__(self, node, ros, goal_active_fn) -> None:
        self._node = node
        self._ros = ros
        self._goal_active_fn = goal_active_fn
        self._srv = None

        if self._node is None or self._ros is None or self._ros.Trigger is None:
            return

        cb_group = self._ros.ReentrantCallbackGroup()
        try:
            self._srv = self._node.create_service(
                self._ros.Trigger,
                RESET_ENV_SERVICE,
                self._on_reset,
                callback_group=cb_group,
            )
        except Exception as exc:
            self._srv = None
            try:
                self._node.get_logger().warn(f"Failed to create reset_env service: {exc}")
            except Exception:
                pass

    def shutdown(self) -> None:
        if self._node is not None and self._srv is not None:
            try:
                self._node.destroy_service(self._srv)
            except Exception:
                pass
        self._srv = None

    def _on_reset(self, _request, response):
        if self._goal_active_fn():
            response.success = False
            response.message = "target_tcp goal active; reset rejected"
            return response

        env = scanbot_context.get_env()
        if env is None:
            response.success = False
            response.message = "env not ready"
            return response

        def _reset_hook() -> None:
            env2 = scanbot_context.get_env()
            if env2 is None:
                return
            scanbot_context.clear_actions()
            env2.sim.reset()
            env2.reset()

        scanbot_context.enqueue_hook(_reset_hook)
        response.success = True
        response.message = "reset scheduled"
        return response


class JointLimitsService:
    def __init__(self, node, ros) -> None:
        self._node = node
        self._ros = ros
        self._srv = None

        if self._node is None or self._ros is None or self._ros.GetJointLimits is None:
            return

        cb_group = self._ros.ReentrantCallbackGroup()
        try:
            self._srv = self._node.create_service(
                self._ros.GetJointLimits,
                GET_JOINT_LIMITS_SERVICE,
                self._on_get_joint_limits,
                callback_group=cb_group,
            )
        except Exception as exc:
            self._srv = None
            try:
                self._node.get_logger().warn(f"Failed to create get_joint_limits service: {exc}")
            except Exception:
                pass

    def shutdown(self) -> None:
        if self._node is not None and self._srv is not None:
            try:
                self._node.destroy_service(self._srv)
            except Exception:
                pass
        self._srv = None

    def _on_get_joint_limits(self, _request, response):
        env = scanbot_context.get_env()
        if env is None:
            response.success = False
            response.message = "env not ready"
            response.name = []
            response.lower = []
            response.upper = []
            return response

        try:
            robot = env.scene["robot"]
            joint_names = list(getattr(robot, "joint_names", []))
            limits = robot.data.joint_pos_limits[0].detach().cpu().numpy()
        except Exception as exc:
            response.success = False
            response.message = f"failed to read joint limits: {exc}"
            response.name = []
            response.lower = []
            response.upper = []
            return response

        if limits.ndim != 2 or limits.shape[1] < 2:
            response.success = False
            response.message = f"unexpected joint_pos_limits shape: {tuple(limits.shape)}"
            response.name = []
            response.lower = []
            response.upper = []
            return response

        lower = limits[:, 0].astype(float)
        upper = limits[:, 1].astype(float)
        # Ensure python-native types for rosidl.
        response.success = True
        response.message = "ok"
        response.name = [str(n) for n in joint_names]
        response.lower = [float(x) for x in lower.tolist()]
        response.upper = [float(x) for x in upper.tolist()]
        return response
