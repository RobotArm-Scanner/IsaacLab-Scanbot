#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class FrameMetrics:
    timestamp: str
    joint_max_abs: float | None = None
    tcp_pos_err: float | None = None
    tcp_rot_err_rad: float | None = None
    wrist_rgb_mad: float | None = None
    wrist_depth_mae: float | None = None
    global_rgb_mad: float | None = None
    global_depth_mae: float | None = None
    wrist_pcd_points_legacy: int | None = None
    wrist_pcd_points_new: int | None = None
    wrist_pcd_nn_mean_legacy_to_new: float | None = None
    wrist_pcd_nn_mean_new_to_legacy: float | None = None
    wrist_pcd_chamfer_mean: float | None = None
    wrist_pcd_voxel_points_legacy: int | None = None
    wrist_pcd_voxel_points_new: int | None = None
    wrist_pcd_spacing_mean_legacy: float | None = None
    wrist_pcd_spacing_mean_new: float | None = None
    wrist_pcd_color_mad_legacy_to_new_255: float | None = None
    wrist_pcd_color_mad_new_to_legacy_255: float | None = None
    wrist_pcd_color_mad_mean_255: float | None = None


def _quat_angle_rad(q0_wxyz: list[float], q1_wxyz: list[float]) -> float:
    # Use |dot| to account for q and -q equivalence.
    dot = abs(
        float(q0_wxyz[0]) * float(q1_wxyz[0])
        + float(q0_wxyz[1]) * float(q1_wxyz[1])
        + float(q0_wxyz[2]) * float(q1_wxyz[2])
        + float(q0_wxyz[3]) * float(q1_wxyz[3])
    )
    dot = max(0.0, min(1.0, dot))
    return 2.0 * math.acos(dot)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare a new dataset against a legacy dataset.")
    parser.add_argument("--legacy_dir", type=str, required=True)
    parser.add_argument("--new_dir", type=str, required=True)
    parser.add_argument("--max_frames", type=int, default=50)
    parser.add_argument(
        "--pcd_voxel_m",
        type=float,
        default=0.001,
        help="Voxel size (meters) for density comparison (point counts after voxel downsample).",
    )
    parser.add_argument(
        "--pcd_sample_max",
        type=int,
        default=2000,
        help="Max points per cloud to sample for spacing/color checks (0 = use all).",
    )
    args = parser.parse_args()

    legacy_dir = Path(args.legacy_dir).expanduser().resolve()
    new_dir = Path(args.new_dir).expanduser().resolve()

    legacy_jsons = {p.name: p for p in legacy_dir.glob("data_*.json")}
    new_jsons = {p.name: p for p in new_dir.glob("data_*.json")}
    common_names = sorted(set(legacy_jsons.keys()) & set(new_jsons.keys()))[: args.max_frames]
    if not common_names:
        raise SystemExit("No overlapping data_*.json between legacy and new directories.")

    import numpy as np
    from PIL import Image

    metrics: list[FrameMetrics] = []

    for name in common_names:
        legacy_meta: dict[str, Any] = json.loads(legacy_jsons[name].read_text())
        new_meta: dict[str, Any] = json.loads(new_jsons[name].read_text())
        timestamp = legacy_meta.get("timestamp") or new_meta.get("timestamp") or name.replace("data_", "").replace(".json", "")

        m = FrameMetrics(timestamp=str(timestamp))

        # Joint comparison (by name).
        legacy_j = dict(zip(legacy_meta.get("joint_names", []), legacy_meta.get("joint_positions", [])))
        new_j = dict(zip(new_meta.get("joint_names", []), new_meta.get("joint_positions", [])))
        common_joints = sorted(set(legacy_j.keys()) & set(new_j.keys()))
        if common_joints:
            diffs = [abs(float(legacy_j[j]) - float(new_j[j])) for j in common_joints]
            m.joint_max_abs = float(max(diffs)) if diffs else None

        # TCP comparison (base frame).
        if "tcp_position_base" in legacy_meta and "tcp_position_base" in new_meta:
            p0 = np.array(legacy_meta["tcp_position_base"], dtype=float)
            p1 = np.array(new_meta["tcp_position_base"], dtype=float)
            m.tcp_pos_err = float(np.linalg.norm(p0 - p1))
        if "tcp_orientation_base_wxyz" in legacy_meta and "tcp_orientation_base_wxyz" in new_meta:
            m.tcp_rot_err_rad = _quat_angle_rad(legacy_meta["tcp_orientation_base_wxyz"], new_meta["tcp_orientation_base_wxyz"])

        # Wrist RGB / depth comparison (file-based, legacy naming).
        try:
            legacy_rgb = legacy_dir / legacy_meta["rgb_image_path"]
            new_rgb = new_dir / new_meta["rgb_image_path"]
            img0 = Image.open(legacy_rgb).convert("RGB")
            img1 = Image.open(new_rgb).convert("RGB")
            if img0.size == img1.size:
                a0 = np.asarray(img0, dtype=np.int16)
                a1 = np.asarray(img1, dtype=np.int16)
                m.wrist_rgb_mad = float(np.mean(np.abs(a0 - a1)))
        except Exception:
            pass

        try:
            legacy_depth = legacy_dir / legacy_meta["depth_raw_path"]
            new_depth = new_dir / new_meta["depth_raw_path"]
            d0 = np.load(legacy_depth)
            d1 = np.load(new_depth)
            if d0.shape == d1.shape:
                finite = np.isfinite(d0) & np.isfinite(d1)
                if finite.any():
                    m.wrist_depth_mae = float(np.mean(np.abs(d0[finite] - d1[finite])))
        except Exception:
            pass

        # Global camera RGB / depth (if present).
        try:
            legacy_cam = (legacy_meta.get("cameras") or {}).get("global_camera") or {}
            new_cam = (new_meta.get("cameras") or {}).get("global_camera") or {}
            legacy_rgb = legacy_dir / legacy_cam["rgb_image_path"]
            new_rgb = new_dir / new_cam["rgb_image_path"]
            img0 = Image.open(legacy_rgb).convert("RGB")
            img1 = Image.open(new_rgb).convert("RGB")
            if img0.size == img1.size:
                a0 = np.asarray(img0, dtype=np.int16)
                a1 = np.asarray(img1, dtype=np.int16)
                m.global_rgb_mad = float(np.mean(np.abs(a0 - a1)))
        except Exception:
            pass

        try:
            legacy_cam = (legacy_meta.get("cameras") or {}).get("global_camera") or {}
            new_cam = (new_meta.get("cameras") or {}).get("global_camera") or {}
            legacy_depth = legacy_dir / legacy_cam["depth_raw_path"]
            new_depth = new_dir / new_cam["depth_raw_path"]
            d0 = np.load(legacy_depth)
            d1 = np.load(new_depth)
            if d0.shape == d1.shape:
                finite = np.isfinite(d0) & np.isfinite(d1)
                if finite.any():
                    m.global_depth_mae = float(np.mean(np.abs(d0[finite] - d1[finite])))
        except Exception:
            pass

        # Wrist PCD comparison
        try:
            import open3d as o3d

            legacy_ply = legacy_dir / "pcd" / f"data_{m.timestamp}_pcd.ply"
            new_ply = new_dir / "pcd" / f"data_{m.timestamp}_pcd.ply"
            p0 = o3d.io.read_point_cloud(str(legacy_ply)) if legacy_ply.is_file() else None
            p1 = o3d.io.read_point_cloud(str(new_ply)) if new_ply.is_file() else None
            if p0 is not None:
                m.wrist_pcd_points_legacy = int(len(p0.points))
            if p1 is not None:
                m.wrist_pcd_points_new = int(len(p1.points))

            if p0 is not None and p1 is not None and p0.has_points() and p1.has_points():
                d01 = np.asarray(p0.compute_point_cloud_distance(p1), dtype=float)
                d10 = np.asarray(p1.compute_point_cloud_distance(p0), dtype=float)
                if d01.size:
                    m.wrist_pcd_nn_mean_legacy_to_new = float(d01.mean())
                if d10.size:
                    m.wrist_pcd_nn_mean_new_to_legacy = float(d10.mean())
                if m.wrist_pcd_nn_mean_legacy_to_new is not None and m.wrist_pcd_nn_mean_new_to_legacy is not None:
                    m.wrist_pcd_chamfer_mean = float(
                        0.5 * (m.wrist_pcd_nn_mean_legacy_to_new + m.wrist_pcd_nn_mean_new_to_legacy)
                    )

                voxel = float(getattr(args, "pcd_voxel_m", 0.0) or 0.0)
                if voxel > 0.0:
                    p0_vox = p0.voxel_down_sample(voxel)
                    p1_vox = p1.voxel_down_sample(voxel)
                    m.wrist_pcd_voxel_points_legacy = int(len(p0_vox.points))
                    m.wrist_pcd_voxel_points_new = int(len(p1_vox.points))

                # Spacing: mean NN distance within each cloud (sampled).
                def _sample_indices(n: int, sample_max: int) -> np.ndarray:
                    if n <= 0:
                        return np.zeros((0,), dtype=np.int64)
                    if sample_max <= 0 or n <= sample_max:
                        return np.arange(n, dtype=np.int64)
                    rng = np.random.default_rng(0)
                    return rng.choice(n, size=sample_max, replace=False).astype(np.int64)

                sample_max = int(getattr(args, "pcd_sample_max", 0) or 0)

                pts0 = np.asarray(p0.points, dtype=np.float64)
                pts1 = np.asarray(p1.points, dtype=np.float64)
                idx0 = _sample_indices(pts0.shape[0], sample_max)
                idx1 = _sample_indices(pts1.shape[0], sample_max)

                if idx0.size and pts0.shape[0] >= 2:
                    tree0 = o3d.geometry.KDTreeFlann(p0)
                    dists = []
                    for i in idx0.tolist():
                        _, _, d2 = tree0.search_knn_vector_3d(pts0[i], 2)
                        if len(d2) >= 2:
                            dists.append(float(math.sqrt(float(d2[1]))))
                    if dists:
                        m.wrist_pcd_spacing_mean_legacy = float(np.mean(np.asarray(dists, dtype=float)))

                if idx1.size and pts1.shape[0] >= 2:
                    tree1 = o3d.geometry.KDTreeFlann(p1)
                    dists = []
                    for i in idx1.tolist():
                        _, _, d2 = tree1.search_knn_vector_3d(pts1[i], 2)
                        if len(d2) >= 2:
                            dists.append(float(math.sqrt(float(d2[1]))))
                    if dists:
                        m.wrist_pcd_spacing_mean_new = float(np.mean(np.asarray(dists, dtype=float)))

                # Color: mean absolute diff (0..255) via nearest-neighbor correspondences.
                if p0.has_colors() and p1.has_colors():
                    cols0 = np.asarray(p0.colors, dtype=np.float64)
                    cols1 = np.asarray(p1.colors, dtype=np.float64)

                    tree1 = o3d.geometry.KDTreeFlann(p1)
                    diffs = []
                    for i in idx0.tolist():
                        _, idx, _ = tree1.search_knn_vector_3d(pts0[i], 1)
                        if idx:
                            j = int(idx[0])
                            diffs.append(np.mean(np.abs(cols0[i] - cols1[j])) * 255.0)
                    if diffs:
                        m.wrist_pcd_color_mad_legacy_to_new_255 = float(np.mean(np.asarray(diffs, dtype=float)))

                    tree0 = o3d.geometry.KDTreeFlann(p0)
                    diffs = []
                    for i in idx1.tolist():
                        _, idx, _ = tree0.search_knn_vector_3d(pts1[i], 1)
                        if idx:
                            j = int(idx[0])
                            diffs.append(np.mean(np.abs(cols1[i] - cols0[j])) * 255.0)
                    if diffs:
                        m.wrist_pcd_color_mad_new_to_legacy_255 = float(np.mean(np.asarray(diffs, dtype=float)))

                    if (
                        m.wrist_pcd_color_mad_legacy_to_new_255 is not None
                        and m.wrist_pcd_color_mad_new_to_legacy_255 is not None
                    ):
                        m.wrist_pcd_color_mad_mean_255 = float(
                            0.5 * (m.wrist_pcd_color_mad_legacy_to_new_255 + m.wrist_pcd_color_mad_new_to_legacy_255)
                        )
        except Exception:
            pass

        metrics.append(m)

    # Print per-frame summary.
    print(f"Compared {len(metrics)} frames")
    for m in metrics:
        print(
            f"- {m.timestamp}: joint_max={m.joint_max_abs} rad, tcp_pos={m.tcp_pos_err} m, tcp_rot={m.tcp_rot_err_rad} rad, "
            f"wrist_rgb_mad={m.wrist_rgb_mad}, wrist_depth_mae={m.wrist_depth_mae}, "
            f"global_rgb_mad={m.global_rgb_mad}, global_depth_mae={m.global_depth_mae}, "
            f"pcd_pts={m.wrist_pcd_points_legacy}->{m.wrist_pcd_points_new}, "
            f"pcd_chamfer_mean={m.wrist_pcd_chamfer_mean}"
        )

    # Aggregate.
    joint_max = [x.joint_max_abs for x in metrics if x.joint_max_abs is not None]
    tcp_pos = [x.tcp_pos_err for x in metrics if x.tcp_pos_err is not None]
    tcp_rot = [x.tcp_rot_err_rad for x in metrics if x.tcp_rot_err_rad is not None]
    rgb = [x.wrist_rgb_mad for x in metrics if x.wrist_rgb_mad is not None]
    depth = [x.wrist_depth_mae for x in metrics if x.wrist_depth_mae is not None]
    global_rgb = [x.global_rgb_mad for x in metrics if x.global_rgb_mad is not None]
    global_depth = [x.global_depth_mae for x in metrics if x.global_depth_mae is not None]
    pcd_chamfer = [x.wrist_pcd_chamfer_mean for x in metrics if x.wrist_pcd_chamfer_mean is not None]
    pcd_vox_legacy = [x.wrist_pcd_voxel_points_legacy for x in metrics if x.wrist_pcd_voxel_points_legacy is not None]
    pcd_vox_new = [x.wrist_pcd_voxel_points_new for x in metrics if x.wrist_pcd_voxel_points_new is not None]
    pcd_spacing_legacy = [x.wrist_pcd_spacing_mean_legacy for x in metrics if x.wrist_pcd_spacing_mean_legacy is not None]
    pcd_spacing_new = [x.wrist_pcd_spacing_mean_new for x in metrics if x.wrist_pcd_spacing_mean_new is not None]
    pcd_color_mad = [x.wrist_pcd_color_mad_mean_255 for x in metrics if x.wrist_pcd_color_mad_mean_255 is not None]

    print("Averages (over available frames)")
    print(f"- joint_max_abs: {_mean([float(x) for x in joint_max])}")
    print(f"- tcp_pos_err:   {_mean([float(x) for x in tcp_pos])}")
    print(f"- tcp_rot_err:   {_mean([float(x) for x in tcp_rot])}")
    print(f"- wrist_rgb_mad: {_mean([float(x) for x in rgb])}")
    print(f"- wrist_depth_mae: {_mean([float(x) for x in depth])}")
    print(f"- global_rgb_mad: {_mean([float(x) for x in global_rgb])}")
    print(f"- global_depth_mae: {_mean([float(x) for x in global_depth])}")
    print(f"- wrist_pcd_chamfer_mean: {_mean([float(x) for x in pcd_chamfer])}")
    print(f"- wrist_pcd_voxel_points_legacy: {_mean([float(x) for x in pcd_vox_legacy])}")
    print(f"- wrist_pcd_voxel_points_new: {_mean([float(x) for x in pcd_vox_new])}")
    print(f"- wrist_pcd_spacing_mean_legacy: {_mean([float(x) for x in pcd_spacing_legacy])}")
    print(f"- wrist_pcd_spacing_mean_new: {_mean([float(x) for x in pcd_spacing_new])}")
    print(f"- wrist_pcd_color_mad_mean_255: {_mean([float(x) for x in pcd_color_mad])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
