"""Target TCP action server and DIK command generator."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Optional

import torch

from . import config as _cfg

ACTION_NAME = _cfg.ACTION_NAME
DEFAULT_POS_TOL = _cfg.DEFAULT_POS_TOL
DEFAULT_ROT_TOL = _cfg.DEFAULT_ROT_TOL
DEFAULT_TIMEOUT_SEC = _cfg.DEFAULT_TIMEOUT_SEC
GAIN_POS = _cfg.GAIN_POS
GAIN_ROT = _cfg.GAIN_ROT
MAX_POS_STEP = _cfg.MAX_POS_STEP
MAX_ROT_STEP = _cfg.MAX_ROT_STEP
POS_ONLY_GATE = _cfg.POS_ONLY_GATE
STABLE_STEPS = int(getattr(_cfg, "TARGET_TCP_STABLE_STEPS", 1))


@dataclass
class GoalState:
    goal_handle: any
    target_pos_cpu: torch.Tensor
    target_quat_cpu: torch.Tensor
    pos_tol: float
    rot_tol: float
    timeout_sec: float
    start_time: float
    done_event: threading.Event
    status: str = "pending"  # pending|succeeded|canceled|aborted
    message: str = ""
    pos_error: float = math.inf
    rot_error: float = math.inf
    stable_steps: int = 0


class TargetTcpAction:
    """ROS2 action server for moving the robot TCP via DIK actions."""

    def __init__(self, node, ros, math_utils) -> None:
        self._node = node
        self._ros = ros
        self._math_utils = math_utils

        self._action_server = None
        self._goal_lock = threading.Lock()
        self._active_goal: Optional[GoalState] = None

        if self._node is None or self._ros is None:
            return
        if self._math_utils is None:
            return

        cb_group = self._ros.ReentrantCallbackGroup()
        self._action_server = self._ros.ActionServer(
            self._node,
            self._ros.TargetTcp,
            ACTION_NAME,
            execute_callback=self._execute_callback,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=cb_group,
        )

    def shutdown(self) -> None:
        self.abort_active_goal("TargetTcpAction shutdown")
        if self._action_server is not None:
            try:
                self._action_server.destroy()
            except Exception:
                pass
        self._action_server = None

    def has_active_goal(self) -> bool:
        with self._goal_lock:
            return self._active_goal is not None

    def abort_active_goal(self, message: str) -> None:
        with self._goal_lock:
            goal_state = self._active_goal
            self._active_goal = None
        if goal_state is None:
            return
        goal_state.status = "aborted"
        goal_state.message = message
        goal_state.done_event.set()

    def maybe_compute_action(self, env, curr_pos: torch.Tensor, curr_quat: torch.Tensor) -> torch.Tensor | None:
        """Return a DIK action tensor when a goal is active, else None."""
        goal_state = None
        with self._goal_lock:
            goal_state = self._active_goal

        if goal_state is None:
            return None

        if self._check_timeout(env_ready=True):
            return None

        try:
            target_pos = goal_state.target_pos_cpu.to(env.device).unsqueeze(0)
            target_quat = goal_state.target_quat_cpu.to(env.device).unsqueeze(0)
        except Exception as exc:
            self._fail_goal(f"Failed to move target to device: {exc}")
            return None

        pos_err, rot_err = self._math_utils.compute_pose_error(
            curr_pos[0:1],
            curr_quat[0:1],
            target_pos,
            target_quat,
            rot_error_type="axis_angle",
        )

        pos_err_vec = pos_err[0]
        rot_err_vec = rot_err[0]
        pos_err_norm = float(torch.linalg.norm(pos_err_vec).item())
        rot_err_norm = float(torch.linalg.norm(rot_err_vec).item())

        goal_state.pos_error = pos_err_norm
        goal_state.rot_error = rot_err_norm
        self._publish_feedback(goal_state)

        if pos_err_norm <= goal_state.pos_tol and rot_err_norm <= goal_state.rot_tol:
            goal_state.stable_steps += 1
            if goal_state.stable_steps >= max(1, STABLE_STEPS):
                self._complete_goal(goal_state, "succeeded", "Reached target")
                return None
        else:
            goal_state.stable_steps = 0

        delta_pos = pos_err_vec * GAIN_POS
        if pos_err_norm > 1e-9:
            # Clamp the *step* size, not the raw error magnitude.
            delta_pos_norm = pos_err_norm * GAIN_POS
            if delta_pos_norm > MAX_POS_STEP:
                delta_pos = pos_err_vec / pos_err_norm * MAX_POS_STEP

        if pos_err_norm > POS_ONLY_GATE:
            delta_rot = torch.zeros_like(rot_err_vec)
        else:
            delta_rot = rot_err_vec * GAIN_ROT
            if rot_err_norm > 1e-9:
                delta_rot_norm = rot_err_norm * GAIN_ROT
                if delta_rot_norm > MAX_ROT_STEP:
                    delta_rot = rot_err_vec / rot_err_norm * MAX_ROT_STEP

        try:
            action_dim = env.action_manager.total_action_dim
            if action_dim < 6:
                raise RuntimeError(f"Expected action_dim >= 6, got {action_dim}")
            action = torch.zeros((env.num_envs, action_dim), device=env.device)
            action[:, 0:3] = delta_pos
            action[:, 3:6] = delta_rot
            return action
        except Exception as exc:
            self._fail_goal(f"Failed to build action: {exc}")
            return None

    def _check_timeout(self, env_ready: bool) -> bool:
        with self._goal_lock:
            goal_state = self._active_goal
        if goal_state is None:
            return True

        if goal_state.timeout_sec <= 0:
            return False

        elapsed = time.monotonic() - goal_state.start_time
        if elapsed >= goal_state.timeout_sec:
            msg = "Timed out" if env_ready else "Timed out waiting for env"
            self._complete_goal(goal_state, "aborted", msg)
            return True
        return False

    def maybe_timeout_when_env_missing(self) -> None:
        """Call on update when env isn't ready to allow action goals to timeout."""
        self._check_timeout(env_ready=False)

    def _goal_callback(self, goal_request) -> any:
        if not self._validate_goal(goal_request):
            return self._ros.GoalResponse.REJECT
        return self._ros.GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle) -> any:
        with self._goal_lock:
            if self._active_goal is not None and self._active_goal.goal_handle == goal_handle:
                self._active_goal.status = "canceled"
                self._active_goal.message = "Canceled by client"
                self._active_goal.done_event.set()
                self._active_goal = None
        return self._ros.CancelResponse.ACCEPT

    def _execute_callback(self, goal_handle) -> any:
        goal_request = goal_handle.request
        if not self._validate_goal(goal_request):
            result = self._ros.TargetTcp.Result()
            result.success = False
            result.message = "Invalid goal"
            goal_handle.abort()
            return result

        target_pos, target_quat = self._goal_to_target(goal_request)
        pos_tol = float(goal_request.pos_tolerance) if goal_request.pos_tolerance > 0.0 else DEFAULT_POS_TOL
        rot_tol = float(goal_request.rot_tolerance) if goal_request.rot_tolerance > 0.0 else DEFAULT_ROT_TOL
        timeout = float(goal_request.timeout_sec) if goal_request.timeout_sec > 0.0 else DEFAULT_TIMEOUT_SEC

        goal_state = GoalState(
            goal_handle=goal_handle,
            target_pos_cpu=target_pos,
            target_quat_cpu=target_quat,
            pos_tol=pos_tol,
            rot_tol=rot_tol,
            timeout_sec=timeout,
            start_time=time.monotonic(),
            done_event=threading.Event(),
        )

        with self._goal_lock:
            if self._active_goal is not None:
                self._active_goal.status = "aborted"
                self._active_goal.message = "Preempted by new goal"
                self._active_goal.done_event.set()
            self._active_goal = goal_state

        while not goal_state.done_event.is_set():
            if goal_handle.is_cancel_requested:
                goal_state.status = "canceled"
                goal_state.message = "Canceled by client"
                goal_state.done_event.set()
                break
            time.sleep(0.05)

        result = self._ros.TargetTcp.Result()
        result.success = goal_state.status == "succeeded"
        result.message = goal_state.message

        if goal_state.status == "succeeded":
            goal_handle.succeed()
        elif goal_state.status == "canceled":
            goal_handle.canceled()
        else:
            goal_handle.abort()

        return result

    def _publish_feedback(self, goal_state: GoalState) -> None:
        if self._node is None:
            return
        try:
            feedback = self._ros.TargetTcp.Feedback()
            feedback.pos_error = float(goal_state.pos_error)
            feedback.rot_error = float(goal_state.rot_error)
            goal_state.goal_handle.publish_feedback(feedback)
        except Exception:
            pass

    def _complete_goal(self, goal_state: GoalState, status: str, message: str) -> None:
        with self._goal_lock:
            if self._active_goal == goal_state:
                self._active_goal = None
        goal_state.status = status
        goal_state.message = message
        goal_state.done_event.set()

    def _fail_goal(self, message: str) -> None:
        with self._goal_lock:
            goal_state = self._active_goal
        if goal_state is None:
            return
        self._complete_goal(goal_state, "aborted", message)

    def _validate_goal(self, goal_request) -> bool:
        pose = goal_request.target.pose
        frame_id = getattr(goal_request.target.header, "frame_id", "")
        if frame_id and frame_id not in {"base", "base_link", "robot_base"}:
            if self._node is not None:
                self._node.get_logger().warn(
                    f"Target frame_id '{frame_id}' is not base frame; interpreting as base frame anyway."
                )
        vals = [
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ]
        for val in vals:
            if not math.isfinite(val):
                if self._node is not None:
                    self._node.get_logger().warn("Rejecting goal: non-finite pose value")
                return False
        return True

    @staticmethod
    def _goal_to_target(goal_request) -> tuple[torch.Tensor, torch.Tensor]:
        pose = goal_request.target.pose
        pos = torch.tensor([pose.position.x, pose.position.y, pose.position.z], dtype=torch.float32)
        quat = torch.tensor(
            [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z],
            dtype=torch.float32,
        )
        quat_norm = torch.linalg.norm(quat)
        if quat_norm > 0:
            quat = quat / quat_norm
        else:
            quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        return pos, quat
