"""Teleport actions for quickly moving the robot state."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

import carb

from scanbot.scripts import scanbot_context
from scanbot.common import pos_util

from . import config as _cfg

TELEPORT_JOINT_ACTION = _cfg.TELEPORT_JOINT_ACTION
TELEPORT_TCP_ACTION = _cfg.TELEPORT_TCP_ACTION
PIPER_URDF_PATH = _cfg.PIPER_URDF_PATH


_BASE_FRAMES = {"", "base", "base_link", "robot_base"}
_WORLD_FRAMES = {"world"}


@dataclass
class _GoalState:
    goal_handle: Any
    kind: str  # "joint" | "tcp"
    start_time: float
    timeout_sec: float
    done_event: threading.Event

    status: str = "pending"  # pending|succeeded|canceled|aborted
    message: str = ""

    applied: bool = False

    # Joint goal
    joint_names: list[str] | None = None
    joint_positions: list[float] | None = None
    joint_tol: float = 1e-3
    joint_max_abs_error: float = math.inf

    # TCP goal (base frame)
    target_pos_base: np.ndarray | None = None
    target_quat_wxyz_base: np.ndarray | None = None
    pos_tol: float = 1e-3
    rot_tol: float = 1e-3
    pos_error: float = math.inf
    rot_error: float = math.inf
    ik_ok: bool = True


class TeleportActions:
    """ROS2 action servers for teleporting joints or TCP (IK) and waiting until reached."""

    def __init__(self, node, ros, goal_active_fn) -> None:
        self._node = node
        self._ros = ros
        self._goal_active_fn = goal_active_fn

        self._teleport_tcp_action = None
        self._teleport_joint_action = None

        self._goal_lock = threading.Lock()
        self._active_goal: Optional[_GoalState] = None

        self._pin_model = None
        self._pin_data = None
        self._pin_eef_frame_id = None
        self._pinocchio = None

        if self._node is None or self._ros is None:
            return

        cb_group = self._ros.ReentrantCallbackGroup()

        if self._ros.TeleportTcpAction is not None:
            try:
                self._teleport_tcp_action = self._ros.ActionServer(
                    self._node,
                    self._ros.TeleportTcpAction,
                    TELEPORT_TCP_ACTION,
                    execute_callback=self._execute_tcp,
                    goal_callback=self._goal_tcp,
                    cancel_callback=self._cancel,
                    callback_group=cb_group,
                )
            except Exception as exc:
                self._teleport_tcp_action = None
                try:
                    self._node.get_logger().warn(f"Failed to create teleport_tcp action: {exc}")
                except Exception:
                    pass

        if self._ros.TeleportJointsAction is not None:
            try:
                self._teleport_joint_action = self._ros.ActionServer(
                    self._node,
                    self._ros.TeleportJointsAction,
                    TELEPORT_JOINT_ACTION,
                    execute_callback=self._execute_joint,
                    goal_callback=self._goal_joint,
                    cancel_callback=self._cancel,
                    callback_group=cb_group,
                )
            except Exception as exc:
                self._teleport_joint_action = None
                try:
                    self._node.get_logger().warn(f"Failed to create teleport_joint action: {exc}")
                except Exception:
                    pass

    def shutdown(self) -> None:
        with self._goal_lock:
            goal_state = self._active_goal
            self._active_goal = None
        if goal_state is not None:
            goal_state.status = "aborted"
            goal_state.message = "TeleportActions shutdown"
            goal_state.done_event.set()
        if self._teleport_tcp_action is not None:
            try:
                self._teleport_tcp_action.destroy()
            except Exception:
                pass
        if self._teleport_joint_action is not None:
            try:
                self._teleport_joint_action.destroy()
            except Exception:
                pass
        self._teleport_tcp_action = None
        self._teleport_joint_action = None

    def has_active_goal(self) -> bool:
        with self._goal_lock:
            return self._active_goal is not None

    def maybe_timeout_when_env_missing(self) -> None:
        self._check_timeout(env_ready=False)

    def maybe_update(self, env, action_term, curr_tcp_pos: torch.Tensor | None, curr_tcp_quat: torch.Tensor | None) -> None:
        """Call on each sim update to apply teleports and complete active goals."""
        if env is None:
            return

        with self._goal_lock:
            goal_state = self._active_goal
        if goal_state is None:
            return

        if self._check_timeout(env_ready=True):
            return

        if goal_state.kind == "joint":
            self._step_joint_goal(env, goal_state)
            return

        if goal_state.kind == "tcp":
            self._step_tcp_goal(env, action_term, curr_tcp_pos, curr_tcp_quat, goal_state)
            return

    def _goal_joint(self, goal_request) -> Any:
        if self._goal_active_fn():
            return self._ros.GoalResponse.REJECT
        if not self._validate_joint_goal(goal_request):
            return self._ros.GoalResponse.REJECT
        return self._ros.GoalResponse.ACCEPT

    def _goal_tcp(self, goal_request) -> Any:
        if self._goal_active_fn():
            return self._ros.GoalResponse.REJECT
        if not self._validate_tcp_goal(goal_request):
            return self._ros.GoalResponse.REJECT
        return self._ros.GoalResponse.ACCEPT

    def _cancel(self, goal_handle) -> Any:
        with self._goal_lock:
            if self._active_goal is not None and self._active_goal.goal_handle == goal_handle:
                self._active_goal.status = "canceled"
                self._active_goal.message = "Canceled by client"
                self._active_goal.done_event.set()
                self._active_goal = None
        return self._ros.CancelResponse.ACCEPT

    def _execute_joint(self, goal_handle) -> Any:
        goal_request = goal_handle.request
        if self._goal_active_fn() or not self._validate_joint_goal(goal_request):
            result = self._ros.TeleportJointsAction.Result()
            result.success = False
            result.message = "Invalid goal"
            result.max_abs_error = float("inf")
            goal_handle.abort()
            return result

        names = list(getattr(goal_request, "name", []))
        positions = [float(x) for x in list(getattr(goal_request, "position", []))]
        tol = float(getattr(goal_request, "tolerance", 0.0) or 0.0)
        timeout = float(getattr(goal_request, "timeout_sec", 0.0) or 0.0)

        goal_state = _GoalState(
            goal_handle=goal_handle,
            kind="joint",
            start_time=time.monotonic(),
            timeout_sec=timeout if timeout > 0.0 else float(_cfg.DEFAULT_TIMEOUT_SEC),
            done_event=threading.Event(),
            joint_names=names,
            joint_positions=positions,
            joint_tol=tol if tol > 0.0 else 1e-3,
        )

        with self._goal_lock:
            if self._active_goal is not None:
                self._active_goal.status = "aborted"
                self._active_goal.message = "Preempted by new teleport goal"
                self._active_goal.done_event.set()
            self._active_goal = goal_state

        while not goal_state.done_event.is_set():
            if goal_handle.is_cancel_requested:
                goal_state.status = "canceled"
                goal_state.message = "Canceled by client"
                goal_state.done_event.set()
                break
            time.sleep(0.05)

        result = self._ros.TeleportJointsAction.Result()
        result.success = goal_state.status == "succeeded"
        result.message = goal_state.message
        result.max_abs_error = float(goal_state.joint_max_abs_error)

        if goal_state.status == "succeeded":
            goal_handle.succeed()
        elif goal_state.status == "canceled":
            goal_handle.canceled()
        else:
            goal_handle.abort()
        return result

    def _execute_tcp(self, goal_handle) -> Any:
        goal_request = goal_handle.request
        if self._goal_active_fn() or not self._validate_tcp_goal(goal_request):
            result = self._ros.TeleportTcpAction.Result()
            result.success = False
            result.message = "Invalid goal"
            result.pos_error = float("inf")
            result.rot_error = float("inf")
            goal_handle.abort()
            return result

        target = goal_request.target
        frame_id = getattr(getattr(target, "header", None), "frame_id", "") or ""
        frame_id = frame_id.strip() or "base"
        pos = np.array([target.pose.position.x, target.pose.position.y, target.pose.position.z], dtype=float)
        quat_wxyz = np.array(
            [
                target.pose.orientation.w,
                target.pose.orientation.x,
                target.pose.orientation.y,
                target.pose.orientation.z,
            ],
            dtype=float,
        )
        quat_wxyz = self._normalize_quat_wxyz(quat_wxyz)

        pos_tol = float(getattr(goal_request, "pos_tolerance", 0.0) or 0.0)
        rot_tol = float(getattr(goal_request, "rot_tolerance", 0.0) or 0.0)
        timeout = float(getattr(goal_request, "timeout_sec", 0.0) or 0.0)

        goal_state = _GoalState(
            goal_handle=goal_handle,
            kind="tcp",
            start_time=time.monotonic(),
            timeout_sec=timeout if timeout > 0.0 else float(_cfg.DEFAULT_TIMEOUT_SEC),
            done_event=threading.Event(),
            target_pos_base=pos,
            target_quat_wxyz_base=quat_wxyz,
            pos_tol=pos_tol if pos_tol > 0.0 else 1e-3,
            rot_tol=rot_tol if rot_tol > 0.0 else 1e-3,
        )
        # Stash frame_id in message (we convert later in sim thread).
        goal_state.message = frame_id

        with self._goal_lock:
            if self._active_goal is not None:
                self._active_goal.status = "aborted"
                self._active_goal.message = "Preempted by new teleport goal"
                self._active_goal.done_event.set()
            self._active_goal = goal_state

        while not goal_state.done_event.is_set():
            if goal_handle.is_cancel_requested:
                goal_state.status = "canceled"
                goal_state.message = "Canceled by client"
                goal_state.done_event.set()
                break
            time.sleep(0.05)

        result = self._ros.TeleportTcpAction.Result()
        result.success = goal_state.status == "succeeded"
        result.message = goal_state.message
        result.pos_error = float(goal_state.pos_error)
        result.rot_error = float(goal_state.rot_error)

        if goal_state.status == "succeeded":
            goal_handle.succeed()
        elif goal_state.status == "canceled":
            goal_handle.canceled()
        else:
            goal_handle.abort()
        return result

    def _check_timeout(self, env_ready: bool) -> bool:
        with self._goal_lock:
            goal_state = self._active_goal
        if goal_state is None:
            return True

        if goal_state.timeout_sec <= 0.0:
            return False

        elapsed = time.monotonic() - goal_state.start_time
        if elapsed >= goal_state.timeout_sec:
            msg = "Timed out" if env_ready else "Timed out waiting for env"
            self._complete_goal(goal_state, "aborted", msg)
            return True
        return False

    def _complete_goal(self, goal_state: _GoalState, status: str, message: str) -> None:
        with self._goal_lock:
            if self._active_goal == goal_state:
                self._active_goal = None
        goal_state.status = status
        goal_state.message = message
        goal_state.done_event.set()

    def _publish_joint_feedback(self, goal_state: _GoalState) -> None:
        try:
            fb = self._ros.TeleportJointsAction.Feedback()
            fb.max_abs_error = float(goal_state.joint_max_abs_error)
            fb.elapsed_sec = float(time.monotonic() - goal_state.start_time)
            goal_state.goal_handle.publish_feedback(fb)
        except Exception:
            pass

    def _publish_tcp_feedback(self, goal_state: _GoalState) -> None:
        try:
            fb = self._ros.TeleportTcpAction.Feedback()
            fb.pos_error = float(goal_state.pos_error)
            fb.rot_error = float(goal_state.rot_error)
            fb.elapsed_sec = float(time.monotonic() - goal_state.start_time)
            goal_state.goal_handle.publish_feedback(fb)
        except Exception:
            pass

    @staticmethod
    def _validate_joint_goal(goal_request) -> bool:
        names = list(getattr(goal_request, "name", []))
        positions = list(getattr(goal_request, "position", []))
        if not positions:
            return False
        if names and len(names) != len(positions):
            return False
        for v in positions:
            if not math.isfinite(float(v)):
                return False
        return True

    @staticmethod
    def _validate_tcp_goal(goal_request) -> bool:
        target = getattr(goal_request, "target", None)
        if target is None:
            return False
        frame_id = getattr(getattr(target, "header", None), "frame_id", "") or ""
        frame_id = frame_id.strip()
        if frame_id and frame_id not in _BASE_FRAMES and frame_id not in _WORLD_FRAMES:
            # Unknown frame; still accept but treat as base in sim thread.
            pass
        pose = target.pose
        vals = [
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ]
        return all(math.isfinite(float(v)) for v in vals)

    def _step_joint_goal(self, env, goal_state: _GoalState) -> None:
        if not goal_state.applied:
            if not self._apply_joint_teleport(goal_state.joint_names or [], goal_state.joint_positions or []):
                # Env may be mid-reset; retry until timeout.
                return
            goal_state.applied = True

        try:
            robot = env.scene["robot"]
            joint_names = list(robot.joint_names)
            q = robot.data.joint_pos[0].detach().cpu().numpy()
        except Exception as exc:
            # Env may be mid-reset; retry until timeout.
            return

        names = goal_state.joint_names or []
        targets = goal_state.joint_positions or []
        tol = float(goal_state.joint_tol)
        max_err = math.inf
        if names:
            name_to_idx = {n: i for i, n in enumerate(joint_names)}
            errs = []
            for name, target in zip(names, targets):
                idx = name_to_idx.get(str(name))
                if idx is None or idx >= len(q):
                    continue
                errs.append(abs(float(q[idx]) - float(target)))
            if errs:
                max_err = float(max(errs))
        else:
            if len(targets) == len(q):
                errs = [abs(float(a) - float(b)) for a, b in zip(q.tolist(), targets)]
                if errs:
                    max_err = float(max(errs))
        goal_state.joint_max_abs_error = float(max_err)
        self._publish_joint_feedback(goal_state)

        if math.isfinite(max_err) and max_err <= tol:
            self._complete_goal(goal_state, "succeeded", "Reached joint target")

    def _step_tcp_goal(
        self,
        env,
        action_term,
        curr_tcp_pos: torch.Tensor | None,
        curr_tcp_quat: torch.Tensor | None,
        goal_state: _GoalState,
    ) -> None:
        if action_term is None or curr_tcp_pos is None or curr_tcp_quat is None:
            # Env may be mid-reset; retry until timeout.
            return

        if not goal_state.applied:
            if not self._ensure_pinocchio_setup():
                self._complete_goal(goal_state, "aborted", "pinocchio unavailable or URDF load failed")
                return

            try:
                robot = env.scene["robot"]
            except Exception as exc:
                self._complete_goal(goal_state, "aborted", f"robot unavailable: {exc}")
                return

            # goal_state.message temporarily stores the incoming frame_id.
            frame_id = (goal_state.message or "").strip()
            if frame_id not in _BASE_FRAMES and frame_id not in _WORLD_FRAMES:
                frame_id = "base"

            pos = np.asarray(goal_state.target_pos_base, dtype=float).reshape(3)
            quat_wxyz = self._normalize_quat_wxyz(goal_state.target_quat_wxyz_base)

            try:
                if frame_id in _WORLD_FRAMES:
                    root_pos_w = robot.data.root_pos_w[0].detach().cpu().numpy()
                    root_quat_w = robot.data.root_quat_w[0].detach().cpu().numpy()
                    pos, quat_wxyz = pos_util.world_to_base_pose(pos, quat_wxyz, root_pos_w, root_quat_w)
            except Exception as exc:
                self._complete_goal(goal_state, "aborted", f"frame conversion failed: {exc}")
                return

            try:
                q_init = robot.data.joint_pos[0].detach().cpu().numpy()
            except Exception as exc:
                self._complete_goal(goal_state, "aborted", f"failed to read joints: {exc}")
                return

            try:
                target_link6 = self._tcp_base_to_link6(pos, quat_wxyz)
                ok, q_sol = self._solve_ik_to_link6(q_init, target_link6)
            except Exception as exc:
                self._complete_goal(goal_state, "aborted", f"IK failed: {exc}")
                return

            goal_state.ik_ok = bool(ok)
            goal_state.target_pos_base = np.asarray(pos, dtype=float).reshape(3)
            goal_state.target_quat_wxyz_base = np.asarray(quat_wxyz, dtype=float).reshape(4)

            if not self._apply_joint_teleport([], q_sol.tolist()):
                # Env may be mid-reset; retry until timeout.
                return
            goal_state.applied = True

        # Check current TCP error (base frame).
        target_pos = np.asarray(goal_state.target_pos_base, dtype=float).reshape(3)
        target_quat = np.asarray(goal_state.target_quat_wxyz_base, dtype=float).reshape(4)
        curr_pos = curr_tcp_pos[0].detach().cpu().numpy().reshape(3)
        curr_quat = curr_tcp_quat[0].detach().cpu().numpy().reshape(4)
        curr_quat = self._normalize_quat_wxyz(curr_quat)

        goal_state.pos_error = float(np.linalg.norm(curr_pos - target_pos))
        goal_state.rot_error = float(self._quat_angle_wxyz(curr_quat, target_quat))
        self._publish_tcp_feedback(goal_state)

        if goal_state.pos_error <= float(goal_state.pos_tol) and goal_state.rot_error <= float(goal_state.rot_tol):
            msg = "Reached tcp target"
            if not goal_state.ik_ok:
                msg = "Reached tcp target (IK not fully converged)"
            self._complete_goal(goal_state, "succeeded", msg)

    def _ensure_pinocchio_setup(self) -> bool:
        if self._pin_model is not None and self._pin_data is not None and self._pin_eef_frame_id is not None:
            return True

        if self._pinocchio is None:
            try:
                import pinocchio as _pinocchio  # type: ignore

                self._pinocchio = _pinocchio
            except Exception as exc:
                if self._node is not None:
                    try:
                        self._node.get_logger().warn(f"pinocchio import failed: {exc}")
                    except Exception:
                        pass
                return False

        try:
            model = self._pinocchio.buildModelFromUrdf(PIPER_URDF_PATH)
            data = model.createData()
            eef_frame_id = model.getFrameId("link6")
        except Exception as exc:
            if self._node is not None:
                try:
                    self._node.get_logger().warn(f"pinocchio URDF load failed: {exc}")
                except Exception:
                    pass
            return False

        self._pin_model = model
        self._pin_data = data
        self._pin_eef_frame_id = int(eef_frame_id)
        return True

    def _tcp_base_to_link6(self, tcp_pos_base, tcp_quat_wxyz):
        pos_b = np.asarray(tcp_pos_base, dtype=float).reshape(3)
        quat_wxyz = self._normalize_quat_wxyz(tcp_quat_wxyz)
        quat_xyzw = np.roll(quat_wxyz, -1)
        base_to_tcp = self._pinocchio.SE3(self._pinocchio.Quaternion(quat_xyzw), pos_b)

        try:
            body_offset_pos, body_offset_quat = pos_util.get_body_offset()
        except Exception as exc:
            if self._node is not None:
                try:
                    self._node.get_logger().warn(f"TCP body offset unavailable: {exc}")
                except Exception:
                    pass
            raise
        body_offset_quat_xyzw = np.roll(np.array(body_offset_quat, dtype=float), -1)
        body_offset_se3 = self._pinocchio.SE3(
            self._pinocchio.Quaternion(body_offset_quat_xyzw),
            np.array(body_offset_pos, dtype=float),
        )
        return base_to_tcp * body_offset_se3.inverse()

    @staticmethod
    def _normalize_quat_wxyz(q) -> np.ndarray:
        quat = np.asarray(q, dtype=float).reshape(4)
        norm = np.linalg.norm(quat)
        if norm <= 0.0:
            raise ValueError("Quaternion must have non-zero norm.")
        return quat / norm

    @staticmethod
    def _quat_angle_wxyz(q0_wxyz: np.ndarray, q1_wxyz: np.ndarray) -> float:
        """Return the geodesic angle between two quaternions (wxyz), accounting for q and -q equivalence."""
        q0 = np.asarray(q0_wxyz, dtype=float).reshape(4)
        q1 = np.asarray(q1_wxyz, dtype=float).reshape(4)
        dot = abs(float(np.dot(q0, q1)))
        dot = max(0.0, min(1.0, dot))
        return 2.0 * float(np.arccos(dot))

    @staticmethod
    def _log3_from_R(R: np.ndarray) -> np.ndarray:
        tr = np.trace(R)
        c = (tr - 1.0) * 0.5
        c = np.clip(c, -1.0, 1.0)
        theta = float(np.arccos(c))
        if theta < 1e-9:
            return np.zeros(3, dtype=float)
        s = float(np.sin(theta))
        if abs(s) < 1e-9:
            axis = np.sqrt(np.maximum((np.diag(R) + 1.0) * 0.5, 0.0))
            if axis.sum() < 1e-9:
                axis = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            axis = np.array(
                [
                    (R[2, 1] - R[1, 2]) / (2.0 * s),
                    (R[0, 2] - R[2, 0]) / (2.0 * s),
                    (R[1, 0] - R[0, 1]) / (2.0 * s),
                ],
                dtype=float,
            )
        return axis * theta

    def _solve_ik_to_link6(
        self,
        q_init: np.ndarray,
        target_link6,
        active_dofs: int = 6,
        max_iters: int = 300,
        pos_tol: float = 1e-4,
        rot_tol: float = 1e-3,
        damping: float = 1e-3,
        step_gain: float = 0.6,
    ) -> tuple[bool, np.ndarray]:
        model = self._pin_model
        data = self._pin_data
        eef_frame_id = self._pin_eef_frame_id

        q = np.asarray(q_init, dtype=float).copy()
        if q.shape[0] < model.nq:
            q = np.pad(q, (0, model.nq - q.shape[0]), mode="constant")
        if q.shape[0] > model.nq:
            q = q[: model.nq]

        prev_err = 1e9
        for _ in range(max_iters):
            self._pinocchio.forwardKinematics(model, data, q)
            self._pinocchio.updateFramePlacements(model, data)
            curr = data.oMf[eef_frame_id]
            try:
                ref_frame = self._pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED
            except Exception:
                ref_frame = self._pinocchio.ReferenceFrame.LOCAL
            t_err_w = target_link6.translation - curr.translation
            R_c = curr.rotation
            R_t = target_link6.rotation
            R_err = R_c.T @ R_t
            rot_vec_local = self._log3_from_R(R_err)
            if ref_frame == getattr(self._pinocchio.ReferenceFrame, "LOCAL_WORLD_ALIGNED", ref_frame):
                pos_err = t_err_w
                rot_vec = R_c @ rot_vec_local
            else:
                pos_err = R_c.T @ t_err_w
                rot_vec = rot_vec_local
            err6 = np.hstack([pos_err, rot_vec])
            if np.linalg.norm(pos_err) < pos_tol and np.linalg.norm(rot_vec) < rot_tol:
                return True, q
            J6 = self._pinocchio.computeFrameJacobian(model, data, q, eef_frame_id, ref_frame)
            J = np.asarray(J6)[:, :active_dofs]
            H = J.T @ J + (damping**2) * np.eye(J.shape[1])
            g = J.T @ err6
            try:
                dq = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                dq = np.linalg.lstsq(H, g, rcond=None)[0]
            dq = np.clip(dq, -0.2, 0.2)
            q[:active_dofs] += step_gain * dq
            err_norm = float(np.linalg.norm(err6))
            if err_norm > prev_err * 1.05:
                step_gain = max(0.1, step_gain * 0.5)
                damping = min(1e-1, damping * 2.0)
            prev_err = err_norm
        return False, q

    def _apply_joint_teleport(self, names: list[str], positions: list[float]) -> bool:
        env = scanbot_context.get_env()
        if env is None:
            return False
        if not positions:
            return False

        try:
            robot = env.scene["robot"]
            joint_names = list(robot.joint_names)
            name_to_index = {name: idx for idx, name in enumerate(joint_names)}
        except Exception as exc:
            try:
                carb.log_warn(f"[scanbot.ros2_manager] teleport_joint: robot unavailable: {exc}")
            except Exception:
                pass
            return False

        if names:
            indices: list[int] = []
            pos_vals: list[float] = []
            missing: list[str] = []
            for name, pos in zip(names, positions):
                idx = name_to_index.get(str(name))
                if idx is None:
                    missing.append(str(name))
                    continue
                indices.append(idx)
                pos_vals.append(float(pos))
            if not indices:
                try:
                    carb.log_warn(
                        f"[scanbot.ros2_manager] teleport_joint: no matching joints (missing: {', '.join(missing)})"
                    )
                except Exception:
                    pass
                return False
            idx_tensor = torch.tensor(indices, dtype=torch.long, device=env.device)
            pos_tensor = torch.tensor(pos_vals, dtype=torch.float32, device=env.device).unsqueeze(0)
            vel_tensor = torch.zeros_like(pos_tensor)
            try:
                robot.write_joint_position_to_sim(pos_tensor, joint_ids=idx_tensor, env_ids=None)
                robot.write_joint_velocity_to_sim(vel_tensor, joint_ids=idx_tensor, env_ids=None)
                robot.set_joint_position_target(pos_tensor, joint_ids=idx_tensor, env_ids=None)
                robot.set_joint_velocity_target(vel_tensor, joint_ids=idx_tensor, env_ids=None)
                robot.write_data_to_sim()
            except Exception as exc:
                try:
                    carb.log_warn(f"[scanbot.ros2_manager] teleport_joint failed: {exc}")
                except Exception:
                    pass
                return False
            if missing:
                try:
                    carb.log_warn(f"[scanbot.ros2_manager] teleport_joint: missing joints: {', '.join(missing)}")
                except Exception:
                    pass
            return True

        pos_tensor = torch.tensor(positions, dtype=torch.float32, device=env.device)
        pos_tensor = pos_tensor.unsqueeze(0).repeat(env.num_envs, 1)
        vel_tensor = torch.zeros_like(pos_tensor)
        if len(positions) != len(joint_names):
            try:
                carb.log_warn(
                    f"[scanbot.ros2_manager] teleport_joint: position length {len(positions)} "
                    f"does not match joint count {len(joint_names)}"
                )
            except Exception:
                pass
            return False

        try:
            robot.write_joint_state_to_sim(pos_tensor, vel_tensor, env_ids=None)
            robot.set_joint_position_target(pos_tensor, env_ids=None)
            robot.set_joint_velocity_target(vel_tensor, env_ids=None)
            robot.write_data_to_sim()
        except Exception as exc:
            try:
                carb.log_warn(f"[scanbot.ros2_manager] teleport_joint failed: {exc}")
            except Exception:
                pass
            return False
        return True
