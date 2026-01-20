"""Pose/position utilities shared across Scanbot extensions."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import math

try:  # Isaac Sim runtime
    from pxr import Gf
except Exception:  # pragma: no cover
    Gf = None


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


def _quat_wxyz_from_deg_z_y_x(deg_xyz) -> np.ndarray:
    """Match collect3d's Gf.Rotation(Z)*Gf.Rotation(Y)*Gf.Rotation(X) convention."""
    if Gf is not None:
        r = (
            Gf.Rotation(Gf.Vec3d(0, 0, 1), float(deg_xyz[2]))
            * Gf.Rotation(Gf.Vec3d(0, 1, 0), float(deg_xyz[1]))
            * Gf.Rotation(Gf.Vec3d(1, 0, 0), float(deg_xyz[0]))
        )
        q = r.GetQuat()
        return np.array([float(q.GetReal()), *map(float, q.GetImaginary())], dtype=float)

    rx, ry, rz = [math.radians(float(d)) for d in deg_xyz]
    cx, sx = math.cos(rx / 2.0), math.sin(rx / 2.0)
    cy, sy = math.cos(ry / 2.0), math.sin(ry / 2.0)
    cz, sz = math.cos(rz / 2.0), math.sin(rz / 2.0)
    # z * y * x order
    qz = np.array([cz, 0.0, 0.0, sz], dtype=float)
    qy = np.array([cy, 0.0, sy, 0.0], dtype=float)
    qx = np.array([cx, sx, 0.0, 0.0], dtype=float)
    return quat_mul_wxyz(quat_mul_wxyz(qz, qy), qx)


# Fixed Scanbot offsets from legacy collect3d (compute_camera_pose_from_tcp_base).
_BODY_OFFSET_POS = np.array((0.0, 0.0, 0.107), dtype=float)
_BODY_OFFSET_QUAT_WXYZ = np.array((1.0, 0.0, 0.0, 0.0), dtype=float)
_MOUNT_OFFSET_POS = np.array((0.0, 0.0, 0.003), dtype=float)
_MOUNT_OFFSET_QUAT_WXYZ = _quat_wxyz_from_deg_z_y_x((180.0, 0.0, 90.0))
_CAMERA_OFFSET_POS = np.array((-0.00171854, -0.00751282, -0.26088225), dtype=float)
_CAMERA_OFFSET_QUAT_WXYZ = _quat_wxyz_from_deg_z_y_x((90.0, 0.0, 0.0))

_BODY_INV_POS, _BODY_INV_QUAT_WXYZ = _invert_offset(_BODY_OFFSET_POS, _BODY_OFFSET_QUAT_WXYZ)
_TCP_TO_SCANPOINT_POS, _TCP_TO_SCANPOINT_QUAT_WXYZ = _compose_offsets(
    _BODY_INV_POS,
    _BODY_INV_QUAT_WXYZ,
    _MOUNT_OFFSET_POS,
    _MOUNT_OFFSET_QUAT_WXYZ,
)
_TCP_TO_SCANPOINT_POS, _TCP_TO_SCANPOINT_QUAT_WXYZ = _compose_offsets(
    _TCP_TO_SCANPOINT_POS,
    _TCP_TO_SCANPOINT_QUAT_WXYZ,
    _CAMERA_OFFSET_POS,
    _CAMERA_OFFSET_QUAT_WXYZ,
)
# OpenGL -> ROS camera convention: 180 deg around X axis.
_CAMERA_ROS_CORRECTION_WXYZ = np.array((0.0, 1.0, 0.0, 0.0), dtype=float)
_SCANPOINT_TO_TCP_POS, _SCANPOINT_TO_TCP_QUAT_WXYZ = _invert_offset(
    _TCP_TO_SCANPOINT_POS,
    _TCP_TO_SCANPOINT_QUAT_WXYZ,
)


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
    tcp_pos = _to_numpy(tcp_pos_base, shape=(3,))
    tcp_q = _normalize_quat_wxyz(tcp_quat_wxyz)
    scan_pos, scan_q = _apply_offset(tcp_pos, tcp_q, _TCP_TO_SCANPOINT_POS, _TCP_TO_SCANPOINT_QUAT_WXYZ)
    scan_q = quat_mul_wxyz(_CAMERA_ROS_CORRECTION_WXYZ, scan_q)
    return scan_pos, scan_q


def scanpoint_to_tcp(
    scan_pos_base,
    scan_quat_wxyz,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert scanpoint pose to TCP pose in robot base frame.

    Inverts the fixed Scanbot scanpoint transform (wrist camera -> TCP).
    """
    scan_pos = _to_numpy(scan_pos_base, shape=(3,))
    scan_q = _normalize_quat_wxyz(scan_quat_wxyz)
    scan_q = quat_mul_wxyz(_CAMERA_ROS_CORRECTION_WXYZ, scan_q)
    tcp_pos, tcp_q = _apply_offset(scan_pos, scan_q, _SCANPOINT_TO_TCP_POS, _SCANPOINT_TO_TCP_QUAT_WXYZ)
    return tcp_pos, tcp_q
