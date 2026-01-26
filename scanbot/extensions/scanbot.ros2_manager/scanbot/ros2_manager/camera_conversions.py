"""Camera tensor -> ROS message conversions for scanbot.ros2_manager."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def maybe_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.is_cuda:
        return tensor.cpu()
    return tensor


def to_image_msg(
    image_type,
    stamp_msg,
    frame_id: str,
    rgb_tensor: torch.Tensor,
    stride: int = 1,
    rgb_cpu: torch.Tensor | None = None,
) -> Any:
    """Convert a torch RGB(A) tensor to sensor_msgs/Image."""
    msg = image_type()
    msg.header.stamp = stamp_msg
    msg.header.frame_id = frame_id

    arr = rgb_cpu if rgb_cpu is not None else maybe_to_cpu(rgb_tensor)
    if arr.dim() == 4:
        arr = arr[0]
    if stride > 1:
        arr = arr[::stride, ::stride, ...]
    if arr.dtype != torch.uint8:
        arr = (arr.clamp(0, 1) * 255).to(dtype=torch.uint8)
    h, w, c = arr.shape
    if c == 4:
        arr = arr[..., :3]
        c = 3
    msg.height = int(h)
    msg.width = int(w)
    msg.encoding = "rgb8"
    msg.step = int(w * c)
    msg.data = arr.contiguous().numpy().tobytes()
    return msg


def to_depth_msg(
    image_type,
    stamp_msg,
    frame_id: str,
    depth_tensor: torch.Tensor,
    stride: int = 1,
    depth_cpu: torch.Tensor | None = None,
) -> Any:
    """Convert a torch depth tensor to sensor_msgs/Image (32FC1)."""
    msg = image_type()
    msg.header.stamp = stamp_msg
    msg.header.frame_id = frame_id

    arr = depth_cpu if depth_cpu is not None else maybe_to_cpu(depth_tensor)
    if arr.dim() == 4:
        arr = arr[0]
    # Handle common shapes: (H,W,1) or (H,W)
    if arr.dim() == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if stride > 1:
        arr = arr[::stride, ::stride]
    if arr.dtype != torch.float32:
        arr = arr.to(dtype=torch.float32)
    h, w = arr.shape
    msg.height = int(h)
    msg.width = int(w)
    msg.encoding = "32FC1"
    msg.step = int(w * 4)
    msg.data = arr.contiguous().numpy().tobytes()
    return msg


def to_camera_info_msg(
    camera_info_type,
    stamp_msg,
    frame_id: str,
    width: int,
    height: int,
    intrinsic_matrix: torch.Tensor,
) -> Any:
    msg = camera_info_type()
    msg.header.stamp = stamp_msg
    msg.header.frame_id = frame_id
    msg.width = int(width)
    msg.height = int(height)

    k = maybe_to_cpu(intrinsic_matrix)
    if k.dim() == 3:
        k = k[0]
    if k.shape != (3, 3):
        return msg
    k_np = k.to(dtype=torch.float64).contiguous().numpy()
    fx = float(k_np[0, 0])
    fy = float(k_np[1, 1])
    cx = float(k_np[0, 2])
    cy = float(k_np[1, 2])

    msg.k = [float(x) for x in k_np.reshape(-1).tolist()]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    msg.d = []
    msg.distortion_model = ""
    return msg


def to_pose_msg(pose_type, stamp_msg, frame_id: str, pos: torch.Tensor, quat_wxyz: torch.Tensor) -> Any:
    msg = pose_type()
    msg.header.stamp = stamp_msg
    msg.header.frame_id = frame_id

    p = maybe_to_cpu(pos)
    q = maybe_to_cpu(quat_wxyz)
    if p.dim() > 1:
        p = p[0]
    if q.dim() > 1:
        q = q[0]

    msg.pose.position.x = float(p[0])
    msg.pose.position.y = float(p[1])
    msg.pose.position.z = float(p[2])
    msg.pose.orientation.w = float(q[0])
    msg.pose.orientation.x = float(q[1])
    msg.pose.orientation.y = float(q[2])
    msg.pose.orientation.z = float(q[3])
    return msg


def _quat_to_matrix_wxyz(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (w, x, y, z) to a 3x3 rotation matrix."""
    q = quat_wxyz.to(dtype=torch.float32)
    if q.dim() > 1:
        q = q[0]
    if q.numel() != 4:
        raise ValueError("Expected quat_wxyz with 4 elements")
    w, x, y, z = q
    n = torch.sqrt(w * w + x * x + y * y + z * z)
    if float(n) <= 0.0:
        return torch.eye(3, dtype=torch.float32)
    w, x, y, z = w / n, x / n, y / n, z / n

    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    return torch.tensor(
        [
            [ww + xx - yy - zz, 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), ww - xx + yy - zz, 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), ww - xx - yy + zz],
        ],
        dtype=torch.float32,
    )


def to_pointcloud2_msg(
    pointcloud2_type,
    pointfield_type,
    stamp_msg,
    frame_id: str,
    points_xyz: np.ndarray,
    colors_rgb: np.ndarray,
) -> Any:
    """Create a PointCloud2 message with x/y/z (float32) and rgb (uint32)."""
    msg = pointcloud2_type()
    msg.header.stamp = stamp_msg
    msg.header.frame_id = frame_id

    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("points_xyz must be (N,3)")
    if colors_rgb.ndim != 2 or colors_rgb.shape[1] != 3:
        raise ValueError("colors_rgb must be (N,3)")
    if points_xyz.shape[0] != colors_rgb.shape[0]:
        raise ValueError("points_xyz and colors_rgb must have the same length")

    n = int(points_xyz.shape[0])
    msg.height = 1
    msg.width = n
    msg.is_bigendian = False
    msg.is_dense = False

    # Standard 16-byte point: x, y, z, rgb
    msg.point_step = 16
    msg.row_step = msg.point_step * n

    msg.fields = [
        pointfield_type(name="x", offset=0, datatype=pointfield_type.FLOAT32, count=1),
        pointfield_type(name="y", offset=4, datatype=pointfield_type.FLOAT32, count=1),
        pointfield_type(name="z", offset=8, datatype=pointfield_type.FLOAT32, count=1),
        pointfield_type(name="rgb", offset=12, datatype=pointfield_type.UINT32, count=1),
    ]

    if n == 0:
        msg.data = b""
        return msg

    pts = np.asarray(points_xyz, dtype=np.float32)
    cols = np.asarray(colors_rgb, dtype=np.uint8)
    rgb_u32 = (cols[:, 0].astype(np.uint32) << 16) | (cols[:, 1].astype(np.uint32) << 8) | cols[:, 2].astype(
        np.uint32
    )

    cloud = np.empty(n, dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4")]))
    cloud["x"] = pts[:, 0]
    cloud["y"] = pts[:, 1]
    cloud["z"] = pts[:, 2]
    cloud["rgb"] = rgb_u32
    msg.data = cloud.tobytes()
    return msg


def compute_pointcloud_world(
    depth: torch.Tensor,
    rgb: torch.Tensor,
    intrinsic: torch.Tensor,
    pos_world: torch.Tensor,
    quat_world_wxyz: torch.Tensor,
    stride: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Compute a downsampled (world) point cloud from depth + rgb + camera pose."""
    depth = maybe_to_cpu(depth)
    rgb = maybe_to_cpu(rgb)
    intrinsic = maybe_to_cpu(intrinsic)
    pos_world = maybe_to_cpu(pos_world)
    quat_world_wxyz = maybe_to_cpu(quat_world_wxyz)

    if depth.dim() == 4:
        depth = depth[0]
    if depth.dim() == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.dim() != 2:
        return None

    if rgb.dim() == 4:
        rgb = rgb[0]
    if rgb.dim() != 3:
        return None
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]

    if rgb.dtype != torch.uint8:
        rgb = (rgb.clamp(0, 1) * 255).to(dtype=torch.uint8)

    depth = depth.to(dtype=torch.float32)

    if intrinsic.dim() == 3:
        intrinsic = intrinsic[0]
    if intrinsic.shape != (3, 3):
        return None
    intrinsic = intrinsic.to(dtype=torch.float32)
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    if fx == 0.0 or fy == 0.0:
        return None

    h0, w0 = int(depth.shape[0]), int(depth.shape[1])
    stride = max(1, int(stride))
    u = torch.arange(0, w0, step=stride, dtype=torch.float32)
    v = torch.arange(0, h0, step=stride, dtype=torch.float32)
    depth_ds = depth[::stride, ::stride]
    rgb_ds = rgb[::stride, ::stride, :]
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    if depth_ds.shape != vv.shape:
        return None

    z = depth_ds
    x = (uu - cx) / fx * z
    y = (vv - cy) / fy * z

    pts_cam = torch.stack((x, y, z), dim=-1).reshape(-1, 3)
    cols = rgb_ds.reshape(-1, 3)

    valid = torch.isfinite(pts_cam).all(dim=1) & torch.isfinite(pts_cam[:, 2]) & (pts_cam[:, 2] > 0.0)
    if not bool(valid.any()):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    pts_cam = pts_cam[valid]
    cols = cols[valid]

    R = _quat_to_matrix_wxyz(quat_world_wxyz)
    pos = pos_world.to(dtype=torch.float32)
    if pos.dim() > 1:
        pos = pos[0]
    if pos.numel() != 3:
        return None
    pts_world = pts_cam @ R.T + pos[None, :]

    pts_np = pts_world.contiguous().cpu().numpy().astype(np.float32, copy=False)
    cols_np = cols.contiguous().cpu().numpy().astype(np.uint8, copy=False)
    return pts_np, cols_np
