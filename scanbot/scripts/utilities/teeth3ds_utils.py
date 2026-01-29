"""Utilities for teeth3ds coverage preprocessing and caching."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - optional dependency
    cKDTree = None


def _parse_obj_vertices_faces(path: str) -> Tuple[np.ndarray, List[List[int]]]:
    vertices: List[List[float]] = []
    faces: List[List[int]] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            if raw.startswith("v "):
                _, x, y, z, *_ = raw.split()
                vertices.append([float(x), float(y), float(z)])
            elif raw.startswith("f "):
                parts = raw.split()[1:]
                face: List[int] = []
                for part in parts:
                    idx = part.split("/", 1)[0]
                    face.append(int(idx) - 1)
                if len(face) >= 3:
                    faces.append(face)
    if not vertices:
        raise ValueError(f"No vertices found in OBJ: {path}")
    if not faces:
        raise ValueError(f"No faces found in OBJ: {path}")
    return np.asarray(vertices, dtype=np.float32), faces


def _read_labels(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    labels = np.asarray(payload["labels"], dtype=np.int32)
    return labels


def _triangulate_face(face: List[int]) -> Iterable[Tuple[int, int, int]]:
    if len(face) == 3:
        yield (face[0], face[1], face[2])
        return
    anchor = face[0]
    for i in range(1, len(face) - 1):
        yield (anchor, face[i], face[i + 1])


def _majority_label(face: List[int], labels: np.ndarray) -> int:
    face_labels = labels[np.asarray(face, dtype=np.int64)]
    unique, counts = np.unique(face_labels, return_counts=True)
    return int(unique[np.argmax(counts)])


def _build_triangles(
    vertices: np.ndarray, faces: List[List[int]], labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    triangles: List[List[int]] = []
    tri_labels: List[int] = []
    for face in faces:
        label = _majority_label(face, labels)
        for tri in _triangulate_face(face):
            triangles.append(list(tri))
            tri_labels.append(label)
    if not triangles:
        raise ValueError("No triangles built from faces.")
    return np.asarray(triangles, dtype=np.int64), np.asarray(tri_labels, dtype=np.int32)


def _triangle_areas(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    return areas


def _sample_points_on_triangles(
    vertices: np.ndarray,
    triangles: np.ndarray,
    tri_labels: np.ndarray,
    num_samples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    areas = _triangle_areas(vertices, triangles)
    total_area = float(np.sum(areas))
    if total_area <= 0.0:
        raise ValueError("Triangle areas sum to zero.")
    probs = areas / total_area
    rng = np.random.default_rng(seed)
    tri_indices = rng.choice(len(triangles), size=num_samples, p=probs)
    u = rng.random(num_samples)
    v = rng.random(num_samples)
    sqrt_u = np.sqrt(u)
    w0 = 1.0 - sqrt_u
    w1 = sqrt_u * (1.0 - v)
    w2 = sqrt_u * v
    v0 = vertices[triangles[tri_indices, 0]]
    v1 = vertices[triangles[tri_indices, 1]]
    v2 = vertices[triangles[tri_indices, 2]]
    points = (w0[:, None] * v0) + (w1[:, None] * v1) + (w2[:, None] * v2)
    labels = tri_labels[tri_indices]
    return points.astype(np.float32), labels.astype(np.int32)


def _hash_params(params: Dict[str, object]) -> str:
    payload = json.dumps(params, sort_keys=True).encode("utf-8")
    return hashlib.md5(payload).hexdigest()  # nosec - used only for cache key


@dataclass
class TeethSurfaceCache:
    dataset_id: str
    points: np.ndarray
    labels: np.ndarray
    tooth_ids: np.ndarray
    gum_tooth_ids: np.ndarray
    params: Dict[str, object]
    _kdtree: object | None = None

    @property
    def kdtree(self):
        if self._kdtree is None:
            if cKDTree is None:
                raise RuntimeError("scipy is required for KDTree coverage updates.")
            self._kdtree = cKDTree(self.points)
        return self._kdtree

    @property
    def tooth_mask(self) -> np.ndarray:
        return self.labels > 0

    @property
    def gum_mask(self) -> np.ndarray:
        return self.labels == 0


def _resolve_dataset_paths(resources_root: str, dataset_id: str) -> Tuple[str, str, str]:
    dataset = dataset_id
    if dataset.endswith("_lower"):
        dataset = dataset[:-6]
    base = os.path.join(resources_root, "teeth3ds", dataset)
    obj_path = os.path.join(base, f"{dataset}_lower.obj")
    json_path = os.path.join(base, f"{dataset}_lower.json")
    return dataset, obj_path, json_path


def _compute_gum_tooth_ids(
    points: np.ndarray,
    labels: np.ndarray,
    tooth_ids: np.ndarray,
    gum_assign_radius: float,
) -> np.ndarray:
    gum_mask = labels == 0
    gum_points = points[gum_mask]
    if gum_points.size == 0:
        return np.full((0,), -1, dtype=np.int32)
    tooth_mask = labels > 0
    tooth_points = points[tooth_mask]
    if tooth_points.size == 0:
        return np.full((gum_points.shape[0],), -1, dtype=np.int32)
    if cKDTree is None:
        # Fallback: brute force nearest tooth point for small inputs.
        diff = gum_points[:, None, :] - tooth_points[None, :, :]
        dists = np.linalg.norm(diff, axis=2)
        nearest = np.argmin(dists, axis=1)
        min_dist = dists[np.arange(dists.shape[0]), nearest]
    else:
        tree = cKDTree(tooth_points)
        min_dist, nearest = tree.query(gum_points, k=1)
    tooth_labels = labels[tooth_mask]
    assigned = np.where(min_dist <= gum_assign_radius, tooth_labels[nearest], -1)
    return assigned.astype(np.int32)


def load_teeth_surface_cache(
    resources_root: str,
    dataset_id: str,
    num_samples: int = 20000,
    seed: int = 0,
    gum_assign_radius: float = 0.002,
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    cache_dir: str | None = None,
) -> TeethSurfaceCache:
    dataset, obj_path, json_path = _resolve_dataset_paths(resources_root, dataset_id)
    if cache_dir is None:
        cache_dir = os.path.join(resources_root, "teeth", "t3ds", "cache")
    os.makedirs(cache_dir, exist_ok=True)

    scale_val = float(scale[0])
    params = {
        "dataset_id": dataset,
        "num_samples": num_samples,
        "seed": seed,
        "gum_assign_radius": gum_assign_radius,
        "scale": scale_val,
    }
    cache_key = _hash_params(params)
    cache_path = os.path.join(cache_dir, f"{dataset}_samples_{cache_key}.npz")

    if os.path.isfile(cache_path):
        npz = np.load(cache_path, allow_pickle=True)
        points = npz["points"]
        labels = npz["labels"]
        tooth_ids = npz["tooth_ids"]
        gum_tooth_ids = npz["gum_tooth_ids"]
        return TeethSurfaceCache(dataset, points, labels, tooth_ids, gum_tooth_ids, params)

    vertices, faces = _parse_obj_vertices_faces(obj_path)
    labels = _read_labels(json_path)
    if vertices.shape[0] != labels.shape[0]:
        raise ValueError("vertex/label mismatch in teeth3ds dataset")

    triangles, tri_labels = _build_triangles(vertices, faces, labels)
    points, point_labels = _sample_points_on_triangles(vertices, triangles, tri_labels, num_samples, seed)
    points *= scale_val

    tooth_ids = np.unique(point_labels[point_labels > 0])
    gum_tooth_ids = _compute_gum_tooth_ids(points, point_labels, tooth_ids, gum_assign_radius)

    np.savez_compressed(
        cache_path,
        points=points,
        labels=point_labels,
        tooth_ids=tooth_ids,
        gum_tooth_ids=gum_tooth_ids,
    )
    return TeethSurfaceCache(dataset, points, point_labels, tooth_ids, gum_tooth_ids, params)


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if points.size == 0 or voxel_size <= 0.0:
        return points
    coords = np.floor(points / voxel_size).astype(np.int32)
    _, idx = np.unique(coords, axis=0, return_index=True)
    return points[idx]


def compute_tooth_centers(surface: TeethSurfaceCache) -> Dict[str, np.ndarray]:
    """Compute per-tooth centers in the surface local frame."""
    centers: Dict[str, np.ndarray] = {}
    for tooth_id in surface.tooth_ids:
        mask = surface.labels == tooth_id
        if not np.any(mask):
            continue
        center = surface.points[mask].mean(axis=0)
        centers[str(int(tooth_id))] = center.astype(np.float32)
    return centers


def compute_teeth_center(surface: TeethSurfaceCache) -> np.ndarray:
    """Compute the center of tooth centers in the surface local frame."""
    centers = compute_tooth_centers(surface)
    if not centers:
        return np.zeros(3, dtype=np.float32)
    stacked = np.stack(list(centers.values()), axis=0)
    return stacked.mean(axis=0).astype(np.float32)


class CoverageTracker:
    """Incremental coverage tracker with cached surface samples."""

    def __init__(
        self,
        surface: TeethSurfaceCache,
        coverage_radius: float,
        num_envs: int,
    ) -> None:
        self.surface = surface
        self.coverage_radius = float(coverage_radius)
        self.hit_masks = [np.zeros(surface.points.shape[0], dtype=bool) for _ in range(num_envs)]

    def reset(self, env_ids: Iterable[int]) -> None:
        for env_id in env_ids:
            self.hit_masks[env_id].fill(False)

    def update(self, env_id: int, points_local: np.ndarray) -> None:
        if points_local.size == 0:
            return
        if cKDTree is None:
            diff = points_local[:, None, :] - self.surface.points[None, :, :]
            dists = np.linalg.norm(diff, axis=2)
            hit = np.any(dists <= self.coverage_radius, axis=0)
            self.hit_masks[env_id] |= hit
            return
        idxs = self.surface.kdtree.query_ball_point(points_local, self.coverage_radius)
        flat = [idx for sub in idxs if len(sub) for idx in sub]
        if not flat:
            return
        self.hit_masks[env_id][flat] = True

    def compute_metrics(self, env_id: int) -> Dict[str, Dict[str, float]]:
        hit = self.hit_masks[env_id]
        labels = self.surface.labels
        tooth_ids = self.surface.tooth_ids
        gum_tooth_ids = self.surface.gum_tooth_ids

        teeth_mask = labels > 0
        gum_mask = labels == 0

        teeth_total = int(np.sum(teeth_mask))
        teeth_hit = int(np.sum(hit[teeth_mask]))
        gum_total = int(np.sum(gum_mask))
        gum_hit = int(np.sum(hit[gum_mask]))

        gum_points_count = int(np.sum(gum_tooth_ids >= 0))
        gum_points_hit = int(np.sum(hit[gum_mask][gum_tooth_ids >= 0]))

        teeth = {}
        teeth_gum = {}
        for tooth_id in tooth_ids:
            t_mask = labels == tooth_id
            t_total = int(np.sum(t_mask))
            t_hit = int(np.sum(hit[t_mask]))
            teeth[str(int(tooth_id))] = {
                "coverage": float(t_hit / t_total) if t_total else 0.0,
                "total": t_total,
                "hit": t_hit,
            }

            g_mask = gum_tooth_ids == tooth_id
            g_total = int(np.sum(g_mask))
            g_hit = int(np.sum(hit[gum_mask][g_mask]))
            teeth_gum[str(int(tooth_id))] = {
                "coverage": float(g_hit / g_total) if g_total else 0.0,
                "total": g_total,
                "hit": g_hit,
            }

        teeth["all"] = {
            "coverage": float(teeth_hit / teeth_total) if teeth_total else 0.0,
            "total": teeth_total,
            "hit": teeth_hit,
        }
        teeth_gum["all"] = {
            "coverage": float(gum_points_hit / gum_points_count) if gum_points_count else 0.0,
            "total": gum_points_count,
            "hit": gum_points_hit,
        }

        gum = {
            "coverage": float(gum_hit / gum_total) if gum_total else 0.0,
            "total": gum_total,
            "hit": gum_hit,
        }
        total = {
            "coverage": float((teeth_hit + gum_hit) / (teeth_total + gum_total))
            if (teeth_total + gum_total)
            else 0.0,
            "total": teeth_total + gum_total,
            "hit": teeth_hit + gum_hit,
        }
        return {"teeth": teeth, "teeth_gum": teeth_gum, "gum": gum, "total": total}
