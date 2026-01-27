"""Pose/position utilities shared across Scanbot extensions."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import torch
from pxr import Gf
import isaaclab.sim as sim_utils


def _to_numpy(x, *, shape: Tuple[int, ...] | None = None) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x, dtype=float)
    if shape is not None:
        arr = arr.reshape(shape)
    return arr


def _normalize_quat_wxyz(q) -> np.ndarray:
    q = _to_numpy(q, shape=(4,))
    norm = np.linalg.norm(q)
    if norm <= 0.0:
        raise ValueError("Quaternion must have non-zero norm.")
    return q / norm


def quat_wxyz_to_rotmat(q) -> np.ndarray:
    w, x, y, z = _normalize_quat_wxyz(q)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def quat_mul_wxyz(q1, q2) -> np.ndarray:
    w1, x1, y1, z1 = _normalize_quat_wxyz(q1)
    w2, x2, y2, z2 = _normalize_quat_wxyz(q2)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def quat_inv_wxyz(q) -> np.ndarray:
    qn = _normalize_quat_wxyz(q)
    return np.array([qn[0], -qn[1], -qn[2], -qn[3]], dtype=float)


def _apply_offset(pos, quat_wxyz, offset_pos, offset_quat_wxyz) -> tuple[np.ndarray, np.ndarray]:
    pos = _to_numpy(pos, shape=(3,))
    quat = _normalize_quat_wxyz(quat_wxyz)
    offset_pos = _to_numpy(offset_pos, shape=(3,))
    offset_quat = _normalize_quat_wxyz(offset_quat_wxyz)
    R = quat_wxyz_to_rotmat(quat)
    out_pos = pos + (R @ offset_pos)
    out_quat = quat_mul_wxyz(quat, offset_quat)
    return out_pos, out_quat


def _invert_offset(offset_pos, offset_quat_wxyz) -> tuple[np.ndarray, np.ndarray]:
    offset_pos = _to_numpy(offset_pos, shape=(3,))
    offset_quat = _normalize_quat_wxyz(offset_quat_wxyz)
    R = quat_wxyz_to_rotmat(offset_quat)
    inv_pos = -(R.T @ offset_pos)
    inv_quat = quat_inv_wxyz(offset_quat)
    return inv_pos, inv_quat


def _compose_offsets(
    pos_a,
    quat_a_wxyz,
    pos_b,
    quat_b_wxyz,
) -> tuple[np.ndarray, np.ndarray]:
    pos_a = _to_numpy(pos_a, shape=(3,))
    quat_a = _normalize_quat_wxyz(quat_a_wxyz)
    pos_b = _to_numpy(pos_b, shape=(3,))
    quat_b = _normalize_quat_wxyz(quat_b_wxyz)
    R_a = quat_wxyz_to_rotmat(quat_a)
    pos = pos_a + (R_a @ pos_b)
    quat = quat_mul_wxyz(quat_a, quat_b)
    return pos, quat


def quat_wxyz_from_deg_xyz(deg_xyz) -> tuple[float, float, float, float]:
    """Euler deg -> quat (WXYZ), applied as Z * Y * X to match existing convention."""
    r = (
        Gf.Rotation(Gf.Vec3d(0, 0, 1), float(deg_xyz[2]))
        * Gf.Rotation(Gf.Vec3d(0, 1, 0), float(deg_xyz[1]))
        * Gf.Rotation(Gf.Vec3d(1, 0, 0), float(deg_xyz[0]))
    )
    q = r.GetQuat()
    return (float(q.GetReal()), *map(float, q.GetImaginary()))


# Offsets are configured at runtime from the active environment.
_BODY_OFFSET_POS: np.ndarray | None = None
_BODY_OFFSET_QUAT_WXYZ: np.ndarray | None = None
_MOUNT_OFFSET_POS: np.ndarray | None = None
_MOUNT_OFFSET_QUAT_WXYZ: np.ndarray | None = None
_CAMERA_OFFSET_POS: np.ndarray | None = None
_CAMERA_OFFSET_QUAT_WXYZ: np.ndarray | None = None

_TCP_TO_SCANPOINT_POS: np.ndarray | None = None
_TCP_TO_SCANPOINT_QUAT_WXYZ: np.ndarray | None = None
_SCANPOINT_TO_TCP_POS: np.ndarray | None = None
_SCANPOINT_TO_TCP_QUAT_WXYZ: np.ndarray | None = None

_CONFIGURED = False
_CONFIGURED_ENV_ID: int | None = None
# OpenGL -> ROS camera convention: 180 deg around X axis.
_CAMERA_ROS_CORRECTION_WXYZ = np.array((0.0, 1.0, 0.0, 0.0), dtype=float)


def _require_configured() -> None:
    if not _CONFIGURED:
        raise RuntimeError("Scanbot offsets not configured. Call configure_from_env(env) first.")


def _set_offsets(
    body_pos,
    body_quat_wxyz,
    mount_pos,
    mount_quat_wxyz,
    camera_pos,
    camera_quat_wxyz,
) -> None:
    global _BODY_OFFSET_POS
    global _BODY_OFFSET_QUAT_WXYZ
    global _MOUNT_OFFSET_POS
    global _MOUNT_OFFSET_QUAT_WXYZ
    global _CAMERA_OFFSET_POS
    global _CAMERA_OFFSET_QUAT_WXYZ
    global _TCP_TO_SCANPOINT_POS
    global _TCP_TO_SCANPOINT_QUAT_WXYZ
    global _SCANPOINT_TO_TCP_POS
    global _SCANPOINT_TO_TCP_QUAT_WXYZ

    _BODY_OFFSET_POS = _to_numpy(body_pos, shape=(3,))
    _BODY_OFFSET_QUAT_WXYZ = _normalize_quat_wxyz(body_quat_wxyz)
    _MOUNT_OFFSET_POS = _to_numpy(mount_pos, shape=(3,))
    _MOUNT_OFFSET_QUAT_WXYZ = _normalize_quat_wxyz(mount_quat_wxyz)
    _CAMERA_OFFSET_POS = _to_numpy(camera_pos, shape=(3,))
    _CAMERA_OFFSET_QUAT_WXYZ = _normalize_quat_wxyz(camera_quat_wxyz)

    body_inv_pos, body_inv_quat = _invert_offset(_BODY_OFFSET_POS, _BODY_OFFSET_QUAT_WXYZ)
    tcp_to_scan_pos, tcp_to_scan_quat = _compose_offsets(
        body_inv_pos,
        body_inv_quat,
        _MOUNT_OFFSET_POS,
        _MOUNT_OFFSET_QUAT_WXYZ,
    )
    tcp_to_scan_pos, tcp_to_scan_quat = _compose_offsets(
        tcp_to_scan_pos,
        tcp_to_scan_quat,
        _CAMERA_OFFSET_POS,
        _CAMERA_OFFSET_QUAT_WXYZ,
    )
    _TCP_TO_SCANPOINT_POS = tcp_to_scan_pos
    _TCP_TO_SCANPOINT_QUAT_WXYZ = tcp_to_scan_quat
    _SCANPOINT_TO_TCP_POS, _SCANPOINT_TO_TCP_QUAT_WXYZ = _invert_offset(
        _TCP_TO_SCANPOINT_POS,
        _TCP_TO_SCANPOINT_QUAT_WXYZ,
    )


def configure_from_env(env, *, force: bool = False) -> bool:
    """Configure Scanbot offsets from the live Isaac Lab environment."""
    global _CONFIGURED
    global _CONFIGURED_ENV_ID

    if env is None:
        return False

    env_id = id(env)
    if _CONFIGURED and _CONFIGURED_ENV_ID == env_id and not force:
        return True

    cfg = env.cfg
    action_term = env.action_manager.get_term("arm_action")

    body_name = action_term.cfg.body_name
    body_offset = action_term.cfg.body_offset
    body_pos = body_offset.pos
    body_quat = body_offset.rot

    scene_cfg = cfg.scene
    robot_path = scene_cfg.robot.prim_path
    link6_path = f"{robot_path}/{body_name}"

    tool_path = scene_cfg.tool.prim_path
    camera_path = scene_cfg.wrist_camera.prim_path

    def _first_prim(path: str):
        prims = sim_utils.find_matching_prims(path)
        if not prims:
            raise RuntimeError(f"Prim not found for path: {path}")
        return prims[0]

    link6_prim = _first_prim(link6_path)
    tool_prim = _first_prim(tool_path)
    camera_prim = _first_prim(camera_path)
    mount_pos, mount_quat = sim_utils.resolve_prim_pose(tool_prim, ref_prim=link6_prim)
    camera_pos, camera_quat = sim_utils.resolve_prim_pose(camera_prim, ref_prim=tool_prim)

    _set_offsets(body_pos, body_quat, mount_pos, mount_quat, camera_pos, camera_quat)
    _CONFIGURED = True
    _CONFIGURED_ENV_ID = env_id
    return True


def get_body_offset() -> tuple[np.ndarray, np.ndarray]:
    _require_configured()
    return _BODY_OFFSET_POS.copy(), _BODY_OFFSET_QUAT_WXYZ.copy()  # type: ignore[union-attr]


def get_mount_offset() -> tuple[np.ndarray, np.ndarray]:
    _require_configured()
    return _MOUNT_OFFSET_POS.copy(), _MOUNT_OFFSET_QUAT_WXYZ.copy()  # type: ignore[union-attr]


def get_camera_offset() -> tuple[np.ndarray, np.ndarray]:
    _require_configured()
    return _CAMERA_OFFSET_POS.copy(), _CAMERA_OFFSET_QUAT_WXYZ.copy()  # type: ignore[union-attr]


def get_tcp_to_scanpoint_offset() -> tuple[np.ndarray, np.ndarray]:
    _require_configured()
    return _TCP_TO_SCANPOINT_POS.copy(), _TCP_TO_SCANPOINT_QUAT_WXYZ.copy()  # type: ignore[union-attr]


def base_to_world_point(pos_base, root_pos_w, root_rot_mat) -> np.ndarray:
    pos_b = _to_numpy(pos_base, shape=(3,))
    root_pos = _to_numpy(root_pos_w, shape=(3,))
    R = _to_numpy(root_rot_mat, shape=(3, 3))
    return (R @ pos_b) + root_pos


def world_to_base_point(pos_world, root_pos_w, root_rot_mat) -> np.ndarray:
    pos_w = _to_numpy(pos_world, shape=(3,))
    root_pos = _to_numpy(root_pos_w, shape=(3,))
    R = _to_numpy(root_rot_mat, shape=(3, 3))
    return R.T @ (pos_w - root_pos)


def base_to_world_pose(
    pos_base,
    quat_base_wxyz,
    root_pos_w,
    root_quat_wxyz,
) -> tuple[np.ndarray, np.ndarray]:
    pos_b = _to_numpy(pos_base, shape=(3,))
    root_pos = _to_numpy(root_pos_w, shape=(3,))
    root_q = _normalize_quat_wxyz(root_quat_wxyz)
    R = quat_wxyz_to_rotmat(root_q)
    pos_w = (R @ pos_b) + root_pos
    quat_w = quat_mul_wxyz(root_q, quat_base_wxyz)
    return pos_w, quat_w


def world_to_base_pose(
    pos_world,
    quat_world_wxyz,
    root_pos_w,
    root_quat_wxyz,
) -> tuple[np.ndarray, np.ndarray]:
    pos_w = _to_numpy(pos_world, shape=(3,))
    root_pos = _to_numpy(root_pos_w, shape=(3,))
    root_q = _normalize_quat_wxyz(root_quat_wxyz)
    R = quat_wxyz_to_rotmat(root_q)
    pos_b = R.T @ (pos_w - root_pos)
    quat_b = quat_mul_wxyz(quat_inv_wxyz(root_q), quat_world_wxyz)
    return pos_b, quat_b


def tcp_to_scanpoint(
    tcp_pos_base,
    tcp_quat_wxyz,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert TCP pose to scanpoint pose in robot base frame.

    This uses the fixed Scanbot offsets from the legacy teleoperation pipeline:
    - TCP includes the DIK body offset (link6 -> TCP).
    - Scanpoint is the wrist camera pose under link6 (mount + camera offsets).
    """
    _require_configured()
    tcp_pos = _to_numpy(tcp_pos_base, shape=(3,))
    tcp_q = _normalize_quat_wxyz(tcp_quat_wxyz)
    scan_pos, scan_q = _apply_offset(
        tcp_pos,
        tcp_q,
        _TCP_TO_SCANPOINT_POS,
        _TCP_TO_SCANPOINT_QUAT_WXYZ,
    )
    scan_q = quat_mul_wxyz(_CAMERA_ROS_CORRECTION_WXYZ, scan_q)
    return scan_pos, scan_q


def scanpoint_to_tcp(
    scan_pos_base,
    scan_quat_wxyz,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert scanpoint pose to TCP pose in robot base frame.

    Inverts the fixed Scanbot scanpoint transform (wrist camera -> TCP).
    """
    _require_configured()
    scan_pos = _to_numpy(scan_pos_base, shape=(3,))
    scan_q = _normalize_quat_wxyz(scan_quat_wxyz)
    scan_q = quat_mul_wxyz(_CAMERA_ROS_CORRECTION_WXYZ, scan_q)
    tcp_pos, tcp_q = _apply_offset(scan_pos, scan_q, _SCANPOINT_TO_TCP_POS, _SCANPOINT_TO_TCP_QUAT_WXYZ)
    return tcp_pos, tcp_q
