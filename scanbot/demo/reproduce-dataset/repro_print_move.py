#!/usr/bin/python3.10
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "datasets" / "episode_e2.t1_f1__siwon1"
REPRO_DIR = DATASET_DIR.parent / f"reproduced.{DATASET_DIR.name}"
COMPARE_MP4 = ROOT_DIR / "compare.mp4"
MOVE_MODE = os.environ.get("REPRO_MOVE_MODE", "target_tcp")  # target_tcp | teleport_tcp | teleport_joint


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _wxyz_to_xyzw(q_wxyz: list[float]) -> list[float]:
    w, x, y, z = q_wxyz
    return [x, y, z, w]


def _xyzw_to_wxyz(q_xyzw: list[float]) -> list[float]:
    x, y, z, w = q_xyzw
    return [w, x, y, z]


def _fmt(xs) -> str:
    return "[" + ", ".join(f"{float(x): .6f}" for x in xs) + "]"


def _pos_l2_m(p1_xyz: list[float], p2_xyz: list[float]) -> float:
    import math

    dx = float(p1_xyz[0]) - float(p2_xyz[0])
    dy = float(p1_xyz[1]) - float(p2_xyz[1])
    dz = float(p1_xyz[2]) - float(p2_xyz[2])
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _quat_angle_rad_xyzw(q1_xyzw: list[float], q2_xyzw: list[float]) -> float:
    import math

    x1, y1, z1, w1 = (float(v) for v in q1_xyzw)
    x2, y2, z2, w2 = (float(v) for v in q2_xyzw)
    n1 = math.sqrt(x1 * x1 + y1 * y1 + z1 * z1 + w1 * w1)
    n2 = math.sqrt(x2 * x2 + y2 * y2 + z2 * z2 + w2 * w2)
    if n1 == 0.0 or n2 == 0.0:
        return float("nan")
    x1, y1, z1, w1 = x1 / n1, y1 / n1, z1 / n1, w1 / n1
    x2, y2, z2, w2 = x2 / n2, y2 / n2, z2 / n2, w2 / n2
    dot = abs(x1 * x2 + y1 * y2 + z1 * z2 + w1 * w2)
    dot = max(-1.0, min(1.0, dot))
    return 2.0 * math.acos(dot)


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    import struct

    length = struct.pack(">I", len(data))
    crc = zlib.crc32(chunk_type)
    crc = zlib.crc32(data, crc) & 0xFFFFFFFF
    return length + chunk_type + data + struct.pack(">I", crc)


def _write_png_rgb8(path: Path, rgb8):
    import struct

    if rgb8.ndim != 3 or rgb8.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 rgb8, got {rgb8.shape}")
    if str(rgb8.dtype) != "uint8":
        rgb8 = rgb8.astype("uint8", copy=False)
    h, w, _c = rgb8.shape

    raw = b"".join(b"\x00" + rgb8[y].tobytes() for y in range(h))
    comp = zlib.compress(raw, level=6)
    ihdr = struct.pack(">IIBBBBB", int(w), int(h), 8, 2, 0, 0, 0)

    png = b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", comp) + _png_chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def _depth_visual_rgb_like_dataset(depth_m):
    import numpy as np

    d = np.asarray(depth_m, dtype=np.float32)
    finite = d[np.isfinite(d)]
    if finite.size == 0:
        gray = np.zeros(d.shape, dtype=np.uint8)
    else:
        mn = float(finite.min())
        mx = float(finite.max())
        if mx <= mn:
            gray = np.zeros(d.shape, dtype=np.uint8)
        else:
            t = (d - mn) / (mx - mn)
            t = np.clip(t, 0.0, 1.0)
            gray = (t * 255.0).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def _voxel_downsample_xyzrgb(xyz, rgb_u32, *, voxel_size_m: float):
    """Simple voxel downsample (centroid + mean color) without Open3D dependency."""
    import numpy as np

    xyz = np.asarray(xyz, dtype=np.float64)
    rgb_u32 = np.asarray(rgb_u32, dtype=np.uint32)
    if xyz.size == 0:
        return xyz.reshape((0, 3)), rgb_u32.reshape((0,))
    if voxel_size_m <= 0.0:
        return xyz, rgb_u32
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Expected Nx3 xyz, got {xyz.shape}")
    if rgb_u32.shape[0] != xyz.shape[0]:
        raise ValueError(f"Expected rgb_u32 len {xyz.shape[0]}, got {rgb_u32.shape}")

    ijk = np.floor(xyz / float(voxel_size_m)).astype(np.int64, copy=False)
    # Structured view for fast unique grouping.
    ijk_struct = np.ascontiguousarray(ijk).view(np.dtype([("i", "<i8"), ("j", "<i8"), ("k", "<i8")])).reshape(-1)
    _uniq, inv = np.unique(ijk_struct, return_inverse=True)
    counts = np.bincount(inv)

    sx = np.bincount(inv, weights=xyz[:, 0])
    sy = np.bincount(inv, weights=xyz[:, 1])
    sz = np.bincount(inv, weights=xyz[:, 2])
    xyz_ds = np.stack([sx, sy, sz], axis=1) / counts[:, None]

    r = ((rgb_u32 >> 16) & 0xFF).astype(np.float64, copy=False)
    g = ((rgb_u32 >> 8) & 0xFF).astype(np.float64, copy=False)
    b = (rgb_u32 & 0xFF).astype(np.float64, copy=False)
    sr = np.bincount(inv, weights=r) / counts
    sg = np.bincount(inv, weights=g) / counts
    sb = np.bincount(inv, weights=b) / counts

    rr = np.clip(np.rint(sr), 0, 255).astype(np.uint32)
    gg = np.clip(np.rint(sg), 0, 255).astype(np.uint32)
    bb = np.clip(np.rint(sb), 0, 255).astype(np.uint32)
    rgb_ds = (rr << 16) | (gg << 8) | bb
    return xyz_ds.astype(np.float64, copy=False), rgb_ds.astype(np.uint32, copy=False)


def _write_ply_xyzrgb_binary(path: Path, xyz, rgb_u32):
    import numpy as np

    xyz = np.asarray(xyz, dtype=np.float64)
    rgb_u32 = np.asarray(rgb_u32, dtype=np.uint32)
    n = int(xyz.shape[0])
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Expected Nx3 xyz, got {xyz.shape}")
    if rgb_u32.shape[0] != n:
        raise ValueError(f"Expected rgb_u32 len {n}, got {rgb_u32.shape}")

    r = ((rgb_u32 >> 16) & 0xFF).astype(np.uint8)
    g = ((rgb_u32 >> 8) & 0xFF).astype(np.uint8)
    b = (rgb_u32 & 0xFF).astype(np.uint8)

    verts = np.empty(
        n,
        dtype=[
            ("x", "<f8"),
            ("y", "<f8"),
            ("z", "<f8"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    verts["x"] = xyz[:, 0]
    verts["y"] = xyz[:, 1]
    verts["z"] = xyz[:, 2]
    verts["red"] = r
    verts["green"] = g
    verts["blue"] = b

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment Created by repro_print_move.py\n"
        f"element vertex {n}\n"
        "property double x\n"
        "property double y\n"
        "property double z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(verts.tobytes())


def _ply_dtype_from_property_type(t: str):
    import numpy as np

    t = t.strip().lower()
    return {
        "char": np.int8,
        "uchar": np.uint8,
        "short": np.int16,
        "ushort": np.uint16,
        "int": np.int32,
        "uint": np.uint32,
        "float": np.float32,
        "double": np.float64,
    }[t]


def _load_ply_xyz_rgb(path: Path):
    import numpy as np

    with path.open("rb") as f:
        header_lines: list[str] = []
        while True:
            line_b = f.readline()
            if not line_b:
                raise RuntimeError(f"Unexpected EOF while reading PLY header: {path}")
            line = line_b.decode("ascii", errors="replace").rstrip("\n")
            header_lines.append(line)
            if line.strip() == "end_header":
                break

        if not header_lines or not header_lines[0].strip().startswith("ply"):
            raise RuntimeError(f"Not a PLY file: {path}")

        fmt = None
        vertex_n = None
        in_vertex = False
        props: list[tuple[str, object]] = []
        for line in header_lines:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
                continue
            if parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_n = int(parts[2])
                continue
            if parts[0] == "property" and in_vertex:
                if len(parts) >= 3 and parts[1] != "list":
                    props.append((parts[2], _ply_dtype_from_property_type(parts[1])))

        if fmt != "binary_little_endian":
            raise RuntimeError(f"Only binary_little_endian PLY supported (got {fmt!r}): {path}")
        if vertex_n is None:
            raise RuntimeError(f"PLY missing 'element vertex N': {path}")
        if not props:
            raise RuntimeError(f"PLY missing vertex properties: {path}")

        dtype = np.dtype(props, align=False)
        data = np.fromfile(f, dtype=dtype, count=int(vertex_n))

    names = set(data.dtype.names or [])
    if not {"x", "y", "z"}.issubset(names):
        raise RuntimeError(f"PLY missing x/y/z fields: {path}")
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32, copy=False)

    if {"red", "green", "blue"}.issubset(names):
        rgb = np.stack([data["red"], data["green"], data["blue"]], axis=1).astype(np.uint8, copy=False)
    elif "rgb" in names:
        rgb_u32 = data["rgb"].astype(np.uint32, copy=False)
        r = ((rgb_u32 >> 16) & 0xFF).astype(np.uint8)
        g = ((rgb_u32 >> 8) & 0xFF).astype(np.uint8)
        b = (rgb_u32 & 0xFF).astype(np.uint8)
        rgb = np.stack([r, g, b], axis=1)
    else:
        rgb = None

    return xyz, rgb


def _compute_xy_bounds(paths: list[Path]) -> tuple[float, float, float, float]:
    import numpy as np

    xmin = float("inf")
    xmax = float("-inf")
    ymin = float("inf")
    ymax = float("-inf")
    for p in paths:
        if not p.exists():
            continue
        xyz, _rgb = _load_ply_xyz_rgb(p)
        if xyz.size == 0:
            continue
        xmin = min(xmin, float(np.min(xyz[:, 0])))
        xmax = max(xmax, float(np.max(xyz[:, 0])))
        ymin = min(ymin, float(np.min(xyz[:, 1])))
        ymax = max(ymax, float(np.max(xyz[:, 1])))
    if not all(map(lambda v: v == v and abs(v) != float("inf"), [xmin, xmax, ymin, ymax])):
        return (-1.0, 1.0, -1.0, 1.0)

    dx = max(1e-6, xmax - xmin)
    dy = max(1e-6, ymax - ymin)
    pad_x = dx * 0.05
    pad_y = dy * 0.05
    return (xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y)


def _render_pcd_xy(xyz, rgb, *, bounds_xy: tuple[float, float, float, float], size_hw: tuple[int, int]):
    import numpy as np

    h, w = (int(size_hw[0]), int(size_hw[1]))
    img = np.zeros((h, w, 3), dtype=np.uint8)
    if xyz is None or xyz.size == 0:
        return img

    xmin, xmax, ymin, ymax = bounds_xy
    x = xyz[:, 0]
    y = xyz[:, 1]
    u = (x - xmin) / max(1e-6, (xmax - xmin)) * (w - 1)
    v = (1.0 - (y - ymin) / max(1e-6, (ymax - ymin))) * (h - 1)
    u = np.clip(u.astype(np.int32), 0, w - 1)
    v = np.clip(v.astype(np.int32), 0, h - 1)

    if rgb is None:
        col = np.full((xyz.shape[0], 3), 200, dtype=np.uint8)
    else:
        col = rgb.astype(np.uint8, copy=False)

    # Convert RGB->BGR for OpenCV.
    bgr = col[:, ::-1]
    img[v, u] = bgr
    return img


def _list_step_timestamps(dataset_dir: Path) -> list[str]:
    steps = []
    for p in sorted(dataset_dir.glob("data_*.json")):
        data = json.loads(p.read_text())
        ts = data.get("timestamp") or p.stem.removeprefix("data_")
        steps.append(str(ts))
    if not steps:
        raise FileNotFoundError(f"No data_*.json found in {dataset_dir}")
    return steps


def _img_mae_gray(path_a: Path, path_b: Path, *, size_hw: tuple[int, int] = (120, 160)) -> float:
    import cv2
    import numpy as np

    a = cv2.imread(str(path_a), cv2.IMREAD_COLOR)
    b = cv2.imread(str(path_b), cv2.IMREAD_COLOR)
    if a is None:
        raise FileNotFoundError(f"Missing image: {path_a}")
    if b is None:
        raise FileNotFoundError(f"Missing image: {path_b}")

    h, w = (int(size_hw[0]), int(size_hw[1]))
    if a.shape[:2] != (h, w):
        a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
    if b.shape[:2] != (h, w):
        b = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
    ag = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    bg = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    return float(np.mean(np.abs(ag.astype(np.int16) - bg.astype(np.int16))))


def _pcd_centroid_stats(path: Path) -> tuple[int, tuple[float, float, float]]:
    import numpy as np

    xyz, _rgb = _load_ply_xyz_rgb(path)
    n = int(xyz.shape[0])
    if n == 0:
        return 0, (float("nan"), float("nan"), float("nan"))
    c = np.mean(xyz, axis=0)
    return n, (float(c[0]), float(c[1]), float(c[2]))


def check_alignment(*, dataset_dir: Path, repro_dir: Path, window: int = 1) -> None:
    """Print best-match dataset index offset for each reproduced step (helps spot off-by-one)."""
    import math

    steps = _list_step_timestamps(dataset_dir)
    offsets = [o for o in range(-int(window), int(window) + 1) if o != 0]
    offsets = [-int(window), 0, int(window)] if int(window) == 1 else list(range(-int(window), int(window) + 1))

    counts: dict[int, int] = {o: 0 for o in offsets}
    total = 0

    for i, rp_ts in enumerate(steps):
        candidates: list[tuple[int, str, float, float, float, int, int]] = []
        for off in offsets:
            j = i + int(off)
            if j < 0 or j >= len(steps):
                continue
            ds_ts = steps[j]

            rgb_mae = _img_mae_gray(dataset_dir / f"rgb_{ds_ts}.png", repro_dir / f"rgb_{rp_ts}.png")
            depth_mae = _img_mae_gray(
                dataset_dir / f"depth_visual_{ds_ts}.png",
                repro_dir / f"depth_visual_{rp_ts}.png",
            )

            ds_n, ds_c = _pcd_centroid_stats(dataset_dir / "pcd" / f"data_{ds_ts}_pcd.ply")
            rp_n, rp_c = _pcd_centroid_stats(repro_dir / "pcd" / f"data_{rp_ts}_pcd.ply")
            if math.isfinite(ds_c[0]) and math.isfinite(rp_c[0]):
                dx = ds_c[0] - rp_c[0]
                dy = ds_c[1] - rp_c[1]
                dz = ds_c[2] - rp_c[2]
                pcd_centroid_l2 = math.sqrt(dx * dx + dy * dy + dz * dz)
            else:
                pcd_centroid_l2 = float("nan")

            candidates.append((int(off), ds_ts, rgb_mae, depth_mae, pcd_centroid_l2, ds_n, rp_n))

        if not candidates:
            continue

        best = min(candidates, key=lambda x: x[2] + x[3])
        best_off, best_ds_ts, best_rgb, best_depth, best_pcd_l2, best_ds_n, best_rp_n = best
        counts[best_off] = counts.get(best_off, 0) + 1
        total += 1

        cand_str = " | ".join(
            f"off={off:+d} rgb={rgb:.1f} depth={dep:.1f} pcd_c_l2={pcd:.4f} n={dsn}/{rpn} ts={ts}"
            for off, ts, rgb, dep, pcd, dsn, rpn in sorted(candidates, key=lambda x: x[0])
        )
        print(
            f"[{i:03d}] rp_ts={rp_ts} best_off={best_off:+d} best_ds_ts={best_ds_ts} "
            f"(rgb={best_rgb:.1f}, depth={best_depth:.1f}, pcd_c_l2={best_pcd_l2:.4f}, n={best_ds_n}/{best_rp_n})"
        )
        print(f"      {cand_str}")

    print("")
    print("alignment_summary:")
    for off in sorted(counts.keys()):
        print(f"  offset {off:+d}: {counts[off]}/{total}")


def make_compare_video(*, dataset_dir: Path, repro_dir: Path, out_path: Path) -> None:
    import cv2

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Missing dataset: {dataset_dir}")
    if not repro_dir.exists():
        raise FileNotFoundError(f"Missing reproduced dataset: {repro_dir}")

    steps = _list_step_timestamps(dataset_dir)

    # Per-step pointcloud bounds (shared across the whole video for stability)
    pcd_step_paths = []
    for ts in steps:
        pcd_step_paths.append(dataset_dir / "pcd" / f"data_{ts}_pcd.ply")
        pcd_step_paths.append(repro_dir / "pcd" / f"data_{ts}_pcd.ply")
    bounds_step_xy = _compute_xy_bounds(pcd_step_paths)

    # Full pointcloud bounds (shared across the whole video for stability)
    pcd_full_paths = []
    for ts in steps:
        pcd_full_paths.append(dataset_dir / "pcd" / f"full_pcd_data_{ts}.ply")
        pcd_full_paths.append(repro_dir / "pcd" / f"full_pcd_data_{ts}.ply")
    bounds_full_xy = _compute_xy_bounds(pcd_full_paths)

    fps = 5.0
    writer = None
    try:
        for i, ts in enumerate(steps):
            ds_rgb = cv2.imread(str(dataset_dir / f"rgb_{ts}.png"), cv2.IMREAD_COLOR)
            ds_depth = cv2.imread(str(dataset_dir / f"depth_visual_{ts}.png"), cv2.IMREAD_COLOR)
            rp_rgb = cv2.imread(str(repro_dir / f"rgb_{ts}.png"), cv2.IMREAD_COLOR)
            rp_depth = cv2.imread(str(repro_dir / f"depth_visual_{ts}.png"), cv2.IMREAD_COLOR)

            if ds_rgb is None or ds_depth is None or rp_rgb is None or rp_depth is None:
                raise RuntimeError(f"Missing images for ts={ts}")

            h, w = ds_rgb.shape[:2]
            rp_rgb = cv2.resize(rp_rgb, (w, h), interpolation=cv2.INTER_AREA) if rp_rgb.shape[:2] != (h, w) else rp_rgb
            ds_depth = cv2.resize(ds_depth, (w, h), interpolation=cv2.INTER_AREA) if ds_depth.shape[:2] != (h, w) else ds_depth
            rp_depth = cv2.resize(rp_depth, (w, h), interpolation=cv2.INTER_AREA) if rp_depth.shape[:2] != (h, w) else rp_depth

            ds_xyz_step, ds_rgb_step = _load_ply_xyz_rgb(dataset_dir / "pcd" / f"data_{ts}_pcd.ply")
            rp_xyz_step, rp_rgb_step = _load_ply_xyz_rgb(repro_dir / "pcd" / f"data_{ts}_pcd.ply")
            ds_pcd_step = _render_pcd_xy(ds_xyz_step, ds_rgb_step, bounds_xy=bounds_step_xy, size_hw=(h, w))
            rp_pcd_step = _render_pcd_xy(rp_xyz_step, rp_rgb_step, bounds_xy=bounds_step_xy, size_hw=(h, w))

            ds_xyz_full, ds_rgb_full = _load_ply_xyz_rgb(dataset_dir / "pcd" / f"full_pcd_data_{ts}.ply")
            rp_xyz_full, rp_rgb_full = _load_ply_xyz_rgb(repro_dir / "pcd" / f"full_pcd_data_{ts}.ply")
            ds_pcd_full = _render_pcd_xy(ds_xyz_full, ds_rgb_full, bounds_xy=bounds_full_xy, size_hw=(h, w))
            rp_pcd_full = _render_pcd_xy(rp_xyz_full, rp_rgb_full, bounds_xy=bounds_full_xy, size_hw=(h, w))

            top = cv2.hconcat([ds_rgb, rp_rgb])
            mid = cv2.hconcat([ds_depth, rp_depth])
            low = cv2.hconcat([ds_pcd_step, rp_pcd_step])
            bot = cv2.hconcat([ds_pcd_full, rp_pcd_full])
            frame = cv2.vconcat([top, mid, low, bot])

            # Separators
            cv2.line(frame, (w, 0), (w, frame.shape[0]), (255, 255, 255), 2)
            cv2.line(frame, (0, h), (frame.shape[1], h), (255, 255, 255), 2)
            cv2.line(frame, (0, 2 * h), (frame.shape[1], 2 * h), (255, 255, 255), 2)
            cv2.line(frame, (0, 3 * h), (frame.shape[1], 3 * h), (255, 255, 255), 2)

            # Labels
            cv2.putText(frame, f"[{i:03d}/{len(steps)-1:03d}] {ts}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(frame, "DATASET", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(frame, "REPRODUCED", (w + 10, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, "RGB", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "DEPTH", (10, h + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "PCD_STEP(xy)", (10, 2 * h + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "FULL_PCD(xy)", (10, 3 * h + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_path), fourcc, fps, (frame.shape[1], frame.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open VideoWriter: {out_path}")
            writer.write(frame)
    finally:
        if writer is not None:
            writer.release()

    print(f"Saved: {out_path}")


def _pointcloud2_xyzrgb_u32(pcd_msg):
    """Fast path for scanbot.ros2_manager PointCloud2 layout (x,y,z float32 + rgb uint32)."""
    import numpy as np

    data = getattr(pcd_msg, "data", b"") or b""
    if not data:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.uint32)

    point_step = int(getattr(pcd_msg, "point_step", 0) or 0)
    if point_step != 16:
        # Fallback to generic parser (slower but robust).
        from sensor_msgs_py import point_cloud2

        pts = list(point_cloud2.read_points(pcd_msg, field_names=("x", "y", "z", "rgb"), skip_nans=True))
        if not pts:
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.uint32)
        arr = np.asarray(pts)
        if getattr(arr.dtype, "fields", None):
            xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64, copy=False)
            rgb_u32 = arr["rgb"].astype(np.uint32, copy=False)
        else:
            xyz = arr[:, :3].astype(np.float64, copy=False)
            rgb_u32 = arr[:, 3].astype(np.uint32, copy=False)
        return xyz, rgb_u32

    n = len(data) // 16
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4")])
    cloud = np.frombuffer(data, dtype=dtype, count=n)
    xyz = np.stack([cloud["x"], cloud["y"], cloud["z"]], axis=1).astype(np.float64, copy=False)
    rgb_u32 = cloud["rgb"].astype(np.uint32, copy=False)

    mask = np.isfinite(xyz).all(axis=1) & np.isfinite(xyz[:, 2]) & (xyz[:, 2] > 0.0)
    if not bool(mask.any()):
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.uint32)
    return xyz[mask], rgb_u32[mask]


def _load_steps(dataset_dir: Path) -> list[dict]:
    json_paths = sorted(dataset_dir.glob("data_*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No data_*.json in {dataset_dir}")
    steps = []
    for p in json_paths:
        steps.append(json.loads(p.read_text()))
    return steps


def main() -> int:
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError(
            f"Use Python 3.10 for ROS2 Humble (rclpy). Current: {sys.version.split()[0]}\n"
            "Hint: run with /usr/bin/python3.10 (or execute this script directly)."
        )

    dataset_dir: Path = DATASET_DIR.resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_dir}\n"
            "Run ./download.sh to fetch it."
        )
    steps = _load_steps(dataset_dir)
    print(f"dataset_dir={dataset_dir}")
    print(f"num_steps={len(steps)}")
    print(f"move_mode={MOVE_MODE}")
    print(f"repro_dir={REPRO_DIR.resolve()}")
    voxel_size_m = max(0.0, _env_float("REPRO_PCD_VOXEL_SIZE_M", 0.0005))
    print(f"repro_pcd_voxel_size_m={voxel_size_m}")
    action_timeout_s = max(1.0, _env_float("REPRO_ACTION_TIMEOUT_SEC", 60.0))
    print(f"repro_action_timeout_s={action_timeout_s}")
    wait_frames = max(1, int(_env_float("REPRO_WAIT_FRAMES", 2.0)))
    sensor_timeout_s = max(1.0, _env_float("REPRO_SENSOR_TIMEOUT_SEC", 12.0))
    max_steps = max(0, int(_env_float("REPRO_MAX_STEPS", 0.0)))
    print(f"repro_wait_frames={wait_frames}")
    print(f"repro_sensor_timeout_s={sensor_timeout_s}")
    if max_steps > 0:
        print(f"repro_max_steps={max_steps}")
    if os.environ.get("ROS_DOMAIN_ID") or os.environ.get("ROS2_DOMAIN"):
        print(f"ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID','')}, ROS2_DOMAIN={os.environ.get('ROS2_DOMAIN','')}")

    try:
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from rclpy.action import ActionClient
        from rclpy.node import Node
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Image, JointState, PointCloud2
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "ROS2 env not sourced. Run:\n"
            "  source /opt/ros/humble/setup.bash"
        ) from e

    try:
        from scanbot_msgs.action import TargetTcp, TeleportJoints, TeleportTcp
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "scanbot_msgs not found. Run:\n"
            "  source /opt/scanbot_ros2/setup.bash"
        ) from e

    class Client(Node):
        def __init__(self) -> None:
            super().__init__("repro_print_move")
            self._action_timeout_s = action_timeout_s
            qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)

            self.joint_state: JointState | None = None
            self.tcp_pose: PoseStamped | None = None
            self.sp_pose: PoseStamped | None = None

            # Scanbot IsaacLab publishes joints under /scanbot/joint_states (not /joint_states).
            self.create_subscription(JointState, "/scanbot/joint_states", self._on_joint, qos)
            self.create_subscription(JointState, "/joint_states", self._on_joint, qos)
            self.create_subscription(PoseStamped, "/scanbot/tcp_pose", self._on_tcp, qos)
            self.create_subscription(PoseStamped, "/scanbot/sp_pose", self._on_sp, qos)

            self.act_target_tcp = ActionClient(self, TargetTcp, "/scanbot/target_tcp")
            self.act_teleport_tcp = ActionClient(self, TeleportTcp, "/scanbot/teleport_tcp")
            self.act_teleport_joint = ActionClient(self, TeleportJoints, "/scanbot/teleport_joint")

        def _on_joint(self, msg: JointState) -> None:
            self.joint_state = msg

        def _on_tcp(self, msg: PoseStamped) -> None:
            self.tcp_pose = msg

        def _on_sp(self, msg: PoseStamped) -> None:
            self.sp_pose = msg

        def wait_initial(self, timeout_s: float) -> None:
            start = time.time()
            while time.time() - start < timeout_s:
                rclpy.spin_once(self, timeout_sec=0.1)
                if self.joint_state and self.tcp_pose and self.sp_pose:
                    return
            missing = []
            if not self.joint_state:
                missing.append("/scanbot/joint_states (or /joint_states)")
            if not self.tcp_pose:
                missing.append("/scanbot/tcp_pose")
            if not self.sp_pose:
                missing.append("/scanbot/sp_pose")
            raise RuntimeError(f"Timed out waiting for: {', '.join(missing)}")

        def _send_tcp_action(self, client: ActionClient, goal, *, action_name: str, pos_xyz: list[float], quat_wxyz: list[float]) -> None:
            if not client.wait_for_server(timeout_sec=5.0):
                raise RuntimeError(f"Action server not available: {action_name}")

            pose = PoseStamped()
            pose.header.frame_id = "base"
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = map(float, pos_xyz)
            q_xyzw = _wxyz_to_xyzw([float(x) for x in quat_wxyz])
            pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = map(float, q_xyzw)

            goal.target = pose
            goal.pos_tolerance = 0.005
            goal.rot_tolerance = 0.02
            goal.timeout_sec = float(self._action_timeout_s)

            fut = client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, fut, timeout_sec=max(10.0, float(self._action_timeout_s)))
            gh = fut.result()
            if gh is None or not gh.accepted:
                raise RuntimeError(f"{action_name} goal rejected")
            res_fut = gh.get_result_async()
            rclpy.spin_until_future_complete(self, res_fut, timeout_sec=float(self._action_timeout_s) + 10.0)
            res = res_fut.result()
            if res is None or not bool(getattr(res.result, "success", False)):
                raise RuntimeError(f"{action_name} failed: {getattr(res.result, 'message', '')}")

        def _send_joint_action(self, *, joint_names: list[str], joint_pos: list[float]) -> None:
            if not self.act_teleport_joint.wait_for_server(timeout_sec=5.0):
                raise RuntimeError("Action server not available: /scanbot/teleport_joint")
            goal = TeleportJoints.Goal()
            goal.name = list(joint_names)
            goal.position = [float(x) for x in joint_pos]
            goal.tolerance = 0.001
            goal.timeout_sec = float(self._action_timeout_s)

            fut = self.act_teleport_joint.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, fut, timeout_sec=max(10.0, float(self._action_timeout_s)))
            gh = fut.result()
            if gh is None or not gh.accepted:
                raise RuntimeError("TeleportJoints goal rejected")
            res_fut = gh.get_result_async()
            rclpy.spin_until_future_complete(self, res_fut, timeout_sec=float(self._action_timeout_s) + 10.0)
            res = res_fut.result()
            if res is None or not bool(getattr(res.result, "success", False)):
                raise RuntimeError(f"TeleportJoints failed: {getattr(res.result, 'message', '')}")

        def move(self, *, mode: str, pos_xyz: list[float], quat_wxyz: list[float], joint_names: list[str], joint_pos: list[float]) -> None:
            if mode == "target_tcp":
                self._send_tcp_action(
                    self.act_target_tcp,
                    TargetTcp.Goal(),
                    action_name="/scanbot/target_tcp",
                    pos_xyz=pos_xyz,
                    quat_wxyz=quat_wxyz,
                )
                return
            if mode == "teleport_tcp":
                self._send_tcp_action(
                    self.act_teleport_tcp,
                    TeleportTcp.Goal(),
                    action_name="/scanbot/teleport_tcp",
                    pos_xyz=pos_xyz,
                    quat_wxyz=quat_wxyz,
                )
                return
            if mode == "teleport_joint":
                self._send_joint_action(joint_names=joint_names, joint_pos=joint_pos)
                return
            raise RuntimeError(f"Unknown REPRO_MOVE_MODE={mode!r} (use target_tcp|teleport_tcp|teleport_joint)")

        def wait_new_all(self, prev_joint, prev_tcp, prev_sp, timeout_s: float) -> None:
            start = time.time()
            while time.time() - start < timeout_s:
                rclpy.spin_once(self, timeout_sec=0.1)
                if self.joint_state and self.tcp_pose and self.sp_pose:
                    if self.joint_state is not prev_joint and self.tcp_pose is not prev_tcp and self.sp_pose is not prev_sp:
                        return
            raise RuntimeError("Timed out waiting for new joint/tcp/sp messages after move")

        def capture_sensors_once(self, *, timeout_s: float, wait_frames: int = 1) -> tuple[Image, Image, PointCloud2]:
            # NOTE: Keeping image/pcd subscriptions open can tank Isaac Sim FPS.
            # So we subscribe only for the short time we need to grab 1 frame.
            sensor_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )
            target = max(1, int(wait_frames))

            rgb_msg: Image | None = None
            depth_msg: Image | None = None
            pcd_msg: PointCloud2 | None = None

            rgb_count = 0
            depth_count = 0
            pcd_count = 0

            def _on_rgb(msg: Image) -> None:
                nonlocal rgb_msg, rgb_count
                rgb_msg = msg
                rgb_count += 1

            def _on_depth(msg: Image) -> None:
                nonlocal depth_msg, depth_count
                depth_msg = msg
                depth_count += 1

            def _on_pcd(msg: PointCloud2) -> None:
                nonlocal pcd_msg, pcd_count
                pcd_msg = msg
                pcd_count += 1

            rgb_sub = self.create_subscription(Image, "/scanbot/cameras/wrist_camera/image_raw", _on_rgb, sensor_qos)
            depth_sub = self.create_subscription(Image, "/scanbot/cameras/wrist_camera/depth_raw", _on_depth, sensor_qos)
            pcd_sub = self.create_subscription(PointCloud2, "/scanbot/cameras/wrist_camera/points", _on_pcd, sensor_qos)
            try:
                start = time.time()
                while time.time() - start < timeout_s:
                    rclpy.spin_once(self, timeout_sec=0.1)
                    if rgb_count >= target and depth_count >= target and pcd_count >= target:
                        break
            finally:
                self.destroy_subscription(rgb_sub)
                self.destroy_subscription(depth_sub)
                self.destroy_subscription(pcd_sub)

            if rgb_msg is None or depth_msg is None or pcd_msg is None:
                raise RuntimeError("Timed out waiting for sensor messages (rgb/depth/pcd)")
            return rgb_msg, depth_msg, pcd_msg

    rclpy.init(args=None)
    node = Client()
    try:
        node.wait_initial(timeout_s=10.0)
        (REPRO_DIR / "pcd").mkdir(parents=True, exist_ok=True)
        full_xyz = None
        full_rgb_u32 = None
        last_ts = None

        for idx, data in enumerate(steps):
            if max_steps > 0 and idx >= max_steps:
                break
            ts = data.get("timestamp", f"{idx:03d}")
            last_ts = str(ts)

            joint_names = data["joint_names"]
            joint_pos = data["joint_positions"]
            tcp_pos = data["tcp_position_base"]
            tcp_q_wxyz = data["tcp_orientation_base_wxyz"]

            print("")
            print(f"[{idx:03d}] {ts}")
            print("  dataset joints:", ", ".join(f"{n}={p: .6f}" for n, p in zip(joint_names, joint_pos)))
            print("  dataset tcp_7d(base):   tcp_position_base + tcp_orientation_base_wxyz =", _fmt(list(tcp_pos) + list(tcp_q_wxyz)))

            prev_joint, prev_tcp, prev_sp = node.joint_state, node.tcp_pose, node.sp_pose
            node.move(
                mode=MOVE_MODE,
                pos_xyz=list(map(float, tcp_pos)),
                quat_wxyz=list(map(float, tcp_q_wxyz)),
                joint_names=list(joint_names),
                joint_pos=list(joint_pos),
            )
            # Best-effort: don't block on convergence (teleport modes can stop slightly off).
            try:
                node.wait_new_all(prev_joint, prev_tcp, prev_sp, timeout_s=1.0)
            except RuntimeError:
                pass
            rgb_msg, depth_msg, pcd_msg = node.capture_sensors_once(timeout_s=sensor_timeout_s, wait_frames=wait_frames)

            js = node.joint_state
            assert js is not None
            name_to_pos = {n: p for n, p in zip(js.name, js.position)}
            ros_joint_str = ", ".join(f"{n}={name_to_pos.get(n, float('nan')): .6f}" for n in joint_names)

            tcp = node.tcp_pose
            sp = node.sp_pose
            assert tcp is not None and sp is not None

            tcp_pos_ros = [tcp.pose.position.x, tcp.pose.position.y, tcp.pose.position.z]
            tcp_q_xyzw_ros = [tcp.pose.orientation.x, tcp.pose.orientation.y, tcp.pose.orientation.z, tcp.pose.orientation.w]
            tcp_q_wxyz_ros = _xyzw_to_wxyz(tcp_q_xyzw_ros)

            sp_pos_ros = [sp.pose.position.x, sp.pose.position.y, sp.pose.position.z]
            sp_q_xyzw_ros = [sp.pose.orientation.x, sp.pose.orientation.y, sp.pose.orientation.z, sp.pose.orientation.w]

            tcp_pos_err_m = _pos_l2_m(list(map(float, tcp_pos)), tcp_pos_ros)
            tcp_rot_err_deg = _quat_angle_rad_xyzw(
                _wxyz_to_xyzw(list(map(float, tcp_q_wxyz))),
                tcp_q_xyzw_ros,
            ) * (180.0 / 3.141592653589793)

            print("  ros joints:", ros_joint_str)
            print(
                f"  ros tcp_pose({tcp.header.frame_id}):  pos+quat_wxyz =",
                _fmt(list(tcp_pos_ros) + list(tcp_q_wxyz_ros)),
            )
            print(f"  tcp_error: pos_l2_m={tcp_pos_err_m:.6f}, rot_deg={tcp_rot_err_deg:.3f}")
            print(
                f"  ros sp_pose({sp.header.frame_id}):   pos+quat_xyzw =",
                _fmt(list(sp_pos_ros) + list(sp_q_xyzw_ros)),
            )

            # Save reproduced assets
            import numpy as np

            if rgb_msg is not None:
                rgb = np.frombuffer(rgb_msg.data, dtype=np.uint8).reshape(int(rgb_msg.height), int(rgb_msg.width), 3)
                _write_png_rgb8(REPRO_DIR / f"rgb_{ts}.png", rgb)

            if depth_msg is not None:
                depth = np.frombuffer(depth_msg.data, dtype=np.float32).reshape(int(depth_msg.height), int(depth_msg.width))
                depth_vis = _depth_visual_rgb_like_dataset(depth)
                _write_png_rgb8(REPRO_DIR / f"depth_visual_{ts}.png", depth_vis)

            if pcd_msg is not None:
                xyz, rgb_u32 = _pointcloud2_xyzrgb_u32(pcd_msg)
                xyz, rgb_u32 = _voxel_downsample_xyzrgb(xyz, rgb_u32, voxel_size_m=voxel_size_m)
                _write_ply_xyzrgb_binary(REPRO_DIR / "pcd" / f"data_{ts}_pcd.ply", xyz, rgb_u32)
                if full_xyz is None:
                    full_xyz = xyz
                    full_rgb_u32 = rgb_u32
                else:
                    full_xyz = np.concatenate([full_xyz, xyz], axis=0)
                    full_rgb_u32 = np.concatenate([full_rgb_u32, rgb_u32], axis=0)
                full_xyz, full_rgb_u32 = _voxel_downsample_xyzrgb(full_xyz, full_rgb_u32, voxel_size_m=voxel_size_m)
                _write_ply_xyzrgb_binary(REPRO_DIR / "pcd" / f"full_pcd_data_{ts}.ply", full_xyz, full_rgb_u32)

        if full_xyz is not None and full_rgb_u32 is not None and last_ts is not None:
            print(
                f"Saved last full PCD: {REPRO_DIR / 'pcd' / f'full_pcd_data_{last_ts}.ply'} (points={int(full_xyz.shape[0])})"
            )

        return 0
    finally:
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--make-compare-video",
        action="store_true",
        help="Generate compare.mp4 from existing assets (no ROS2 needed).",
    )
    parser.add_argument(
        "--check-alignment",
        action="store_true",
        help="Print per-step best-match offset vs dataset (no ROS2 needed).",
    )
    args, _unknown = parser.parse_known_args()
    if args.make_compare_video:
        make_compare_video(dataset_dir=DATASET_DIR, repro_dir=REPRO_DIR, out_path=COMPARE_MP4)
        raise SystemExit(0)
    if args.check_alignment:
        check_alignment(dataset_dir=DATASET_DIR, repro_dir=REPRO_DIR, window=1)
        raise SystemExit(0)
    raise SystemExit(main())
