"""Pose/position utilities shared across Scanbot extensions."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def _to_numpy(x, *, shape: Tuple[int, ...] | None = None) -> np.ndarray:
    if hasattr(x, "detach"):
        # torch.Tensor or similar
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
    tool_offset_pos=(0.0, 0.0, 0.0),
    tool_offset_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Convert TCP (link6) pose to scanpoint pose in robot base frame.

    tool_offset_* describes the rigid transform from TCP to scanpoint.
    """
    tcp_pos = _to_numpy(tcp_pos_base, shape=(3,))
    tcp_q = _normalize_quat_wxyz(tcp_quat_wxyz)
    offset_pos = _to_numpy(tool_offset_pos, shape=(3,))
    offset_q = _normalize_quat_wxyz(tool_offset_quat_wxyz)
    R_tcp = quat_wxyz_to_rotmat(tcp_q)
    scan_pos = tcp_pos + (R_tcp @ offset_pos)
    scan_q = quat_mul_wxyz(tcp_q, offset_q)
    return scan_pos, scan_q


def scanpoint_to_tcp(
    scan_pos_base,
    scan_quat_wxyz,
    tool_offset_pos=(0.0, 0.0, 0.0),
    tool_offset_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Convert scanpoint pose to TCP (link6) pose in robot base frame.

    tool_offset_* describes the rigid transform from TCP to scanpoint.
    """
    scan_pos = _to_numpy(scan_pos_base, shape=(3,))
    scan_q = _normalize_quat_wxyz(scan_quat_wxyz)
    offset_pos = _to_numpy(tool_offset_pos, shape=(3,))
    offset_q = _normalize_quat_wxyz(tool_offset_quat_wxyz)
    tcp_q = quat_mul_wxyz(scan_q, quat_inv_wxyz(offset_q))
    R_tcp = quat_wxyz_to_rotmat(tcp_q)
    tcp_pos = scan_pos - (R_tcp @ offset_pos)
    return tcp_pos, tcp_q
