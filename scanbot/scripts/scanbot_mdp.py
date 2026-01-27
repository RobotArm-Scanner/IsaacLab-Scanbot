"""Scanbot custom MDP utilities (coverage + rewards)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.camera import utils as camera_utils
from isaaclab.utils import math as math_utils

from scanbot.scripts.utilities.teeth3ds_utils import CoverageTracker
from scanbot.scripts.utilities.teeth3ds_utils import load_teeth_surface_cache
from scanbot.scripts.utilities.teeth3ds_utils import voxel_downsample


@dataclass
class _CoverageState:
    tracker: CoverageTracker
    metrics: List[Dict[str, Dict[str, float]]]
    last_update_step: np.ndarray
    last_coverage_sum: torch.Tensor
    rewarded_teeth: List[set]
    rewarded_total: np.ndarray
    pcd_voxel_size: float
    pcd_max_points: int
    update_every: int
    camera_name: str
    data_type: str
    teeth_name: str

    def reset_envs(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        ids = env_ids.detach().cpu().numpy().tolist()
        self.tracker.reset(ids)
        for env_id in ids:
            self.metrics[env_id] = {}
            self.last_update_step[env_id] = -1
            self.last_coverage_sum[env_id] = 0.0
            self.rewarded_teeth[env_id].clear()
            self.rewarded_total[env_id] = False

    def maybe_update(self, env) -> None:
        step_buf = env.episode_length_buf
        reset_ids = (step_buf == 0).nonzero(as_tuple=False).flatten()
        if reset_ids.numel() > 0:
            self.reset_envs(reset_ids)

        camera = env.scene[self.camera_name]
        if self.data_type not in camera.data.output:
            raise KeyError(f"Camera output '{self.data_type}' not available for {self.camera_name}")
        depth = camera.data.output[self.data_type]
        intrinsics = camera.data.intrinsic_matrices
        cam_pos = camera.data.pos_w
        cam_quat = camera.data.quat_w_world

        teeth = env.scene[self.teeth_name]
        teeth_pos = teeth.data.root_pos_w
        teeth_quat = teeth.data.root_quat_w

        num_envs = env.num_envs
        for env_id in range(num_envs):
            step = int(step_buf[env_id].item())
            if self.update_every > 1 and step % self.update_every != 0:
                continue
            if self.last_update_step[env_id] == step:
                continue

            depth_img = depth[env_id]
            if depth_img.numel() == 0:
                continue
            if depth_img.ndim == 3 and depth_img.shape[-1] == 1:
                depth_img = depth_img[..., 0]

            points_w = camera_utils.create_pointcloud_from_depth(
                intrinsic_matrix=intrinsics[env_id],
                depth=depth_img,
                keep_invalid=False,
                position=cam_pos[env_id],
                orientation=cam_quat[env_id],
                device=depth_img.device,
            )
            if points_w.numel() == 0:
                continue
            points_w = points_w[torch.isfinite(points_w).all(dim=1)]
            if points_w.numel() == 0:
                continue

            points_local = math_utils.quat_apply_inverse(
                teeth_quat[env_id],
                points_w - teeth_pos[env_id],
            )
            points_np = points_local.detach().cpu().numpy()
            points_np = voxel_downsample(points_np, self.pcd_voxel_size)
            if self.pcd_max_points > 0 and points_np.shape[0] > self.pcd_max_points:
                idx = np.random.choice(points_np.shape[0], size=self.pcd_max_points, replace=False)
                points_np = points_np[idx]

            self.tracker.update(env_id, points_np)
            self.metrics[env_id] = self.tracker.compute_metrics(env_id)
            self.last_update_step[env_id] = step


def _get_state(env, params: Dict[str, object]) -> _CoverageState:
    num_envs = env.num_envs
    cache_key = (
        params["dataset_id"],
        params["num_samples"],
        params["seed"],
        params["gum_assign_radius"],
        params["coverage_radius"],
        params["scale"],
    )
    state = getattr(env, "_scanbot_coverage_state", None)
    if state is None or getattr(state, "_cache_key", None) != cache_key or len(state.metrics) != num_envs:
        surface = load_teeth_surface_cache(
            resources_root=params["resources_root"],
            dataset_id=params["dataset_id"],
            num_samples=params["num_samples"],
            seed=params["seed"],
            gum_assign_radius=params["gum_assign_radius"],
            scale=params["scale"],
        )
        tracker = CoverageTracker(surface, coverage_radius=params["coverage_radius"], num_envs=num_envs)
        state = _CoverageState(
            tracker=tracker,
            metrics=[{} for _ in range(num_envs)],
            last_update_step=np.full((num_envs,), -1, dtype=np.int64),
            last_coverage_sum=torch.zeros(num_envs, device=env.device),
            rewarded_teeth=[set() for _ in range(num_envs)],
            rewarded_total=np.zeros((num_envs,), dtype=bool),
            pcd_voxel_size=float(params["pcd_voxel_size"]),
            pcd_max_points=int(params["pcd_max_points"]),
            update_every=int(params["coverage_update_every"]),
            camera_name=str(params["camera_name"]),
            data_type=str(params["data_type"]),
            teeth_name=str(params["teeth_name"]),
        )
        state._cache_key = cache_key  # type: ignore[attr-defined]
        env._scanbot_coverage_state = state
    return state


def ee_delta_l2(env, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_pos = env.scene[ee_frame_cfg.name].data.target_pos_w[:, 0, :] - env.scene.env_origins[:, 0:3]
    last_pos = getattr(env, "_scanbot_last_ee_pos", None)
    if last_pos is None or last_pos.shape != ee_pos.shape:
        env._scanbot_last_ee_pos = ee_pos.clone()
        return torch.zeros(env.num_envs, device=env.device)

    reset_ids = (env.episode_length_buf == 0).nonzero(as_tuple=False).flatten()
    if reset_ids.numel() > 0:
        last_pos[reset_ids] = ee_pos[reset_ids]

    delta = torch.norm(ee_pos - last_pos, dim=1)
    env._scanbot_last_ee_pos = ee_pos
    return delta


def _build_params(
    resources_root: str,
    dataset_id: str,
    num_samples: int,
    seed: int,
    gum_assign_radius: float,
    coverage_radius: float,
    scale: tuple,
    pcd_voxel_size: float,
    pcd_max_points: int,
    coverage_update_every: int,
    camera_name: str,
    data_type: str,
    teeth_name: str,
) -> Dict[str, object]:
    return {
        "resources_root": resources_root,
        "dataset_id": dataset_id,
        "num_samples": num_samples,
        "seed": seed,
        "gum_assign_radius": gum_assign_radius,
        "coverage_radius": coverage_radius,
        "scale": scale,
        "pcd_voxel_size": pcd_voxel_size,
        "pcd_max_points": pcd_max_points,
        "coverage_update_every": coverage_update_every,
        "camera_name": camera_name,
        "data_type": data_type,
        "teeth_name": teeth_name,
    }


def coverage_delta_reward(
    env,
    resources_root: str,
    dataset_id: str,
    num_samples: int,
    seed: int,
    gum_assign_radius: float,
    coverage_radius: float,
    scale: tuple,
    pcd_voxel_size: float,
    pcd_max_points: int,
    coverage_update_every: int,
    camera_name: str,
    data_type: str,
    teeth_name: str,
) -> torch.Tensor:
    params = _build_params(
        resources_root,
        dataset_id,
        num_samples,
        seed,
        gum_assign_radius,
        coverage_radius,
        scale,
        pcd_voxel_size,
        pcd_max_points,
        coverage_update_every,
        camera_name,
        data_type,
        teeth_name,
    )
    state = _get_state(env, params)
    state.maybe_update(env)

    coverage_sum = torch.zeros(env.num_envs, device=env.device)
    for env_id in range(env.num_envs):
        metrics = state.metrics[env_id]
        if not metrics:
            continue
        teeth_all = metrics["teeth"]["all"]["coverage"]
        teeth_gum_all = metrics["teeth_gum"]["all"]["coverage"]
        coverage_sum[env_id] = float(teeth_all + teeth_gum_all)

    delta = coverage_sum - state.last_coverage_sum
    state.last_coverage_sum = coverage_sum
    return delta


def per_tooth_coverage_bonus(
    env,
    threshold: float,
    resources_root: str,
    dataset_id: str,
    num_samples: int,
    seed: int,
    gum_assign_radius: float,
    coverage_radius: float,
    scale: tuple,
    pcd_voxel_size: float,
    pcd_max_points: int,
    coverage_update_every: int,
    camera_name: str,
    data_type: str,
    teeth_name: str,
) -> torch.Tensor:
    params = _build_params(
        resources_root,
        dataset_id,
        num_samples,
        seed,
        gum_assign_radius,
        coverage_radius,
        scale,
        pcd_voxel_size,
        pcd_max_points,
        coverage_update_every,
        camera_name,
        data_type,
        teeth_name,
    )
    state = _get_state(env, params)
    state.maybe_update(env)

    reward = torch.zeros(env.num_envs, device=env.device)
    for env_id in range(env.num_envs):
        metrics = state.metrics[env_id]
        if not metrics:
            continue
        teeth = metrics["teeth"]
        for tooth_id, data in teeth.items():
            if tooth_id == "all":
                continue
            if data["coverage"] >= threshold and tooth_id not in state.rewarded_teeth[env_id]:
                state.rewarded_teeth[env_id].add(tooth_id)
                reward[env_id] += 1.0
    return reward


def total_coverage_bonus(
    env,
    threshold: float,
    resources_root: str,
    dataset_id: str,
    num_samples: int,
    seed: int,
    gum_assign_radius: float,
    coverage_radius: float,
    scale: tuple,
    pcd_voxel_size: float,
    pcd_max_points: int,
    coverage_update_every: int,
    camera_name: str,
    data_type: str,
    teeth_name: str,
) -> torch.Tensor:
    params = _build_params(
        resources_root,
        dataset_id,
        num_samples,
        seed,
        gum_assign_radius,
        coverage_radius,
        scale,
        pcd_voxel_size,
        pcd_max_points,
        coverage_update_every,
        camera_name,
        data_type,
        teeth_name,
    )
    state = _get_state(env, params)
    state.maybe_update(env)

    reward = torch.zeros(env.num_envs, device=env.device)
    for env_id in range(env.num_envs):
        if state.rewarded_total[env_id]:
            continue
        metrics = state.metrics[env_id]
        if not metrics:
            continue
        teeth_all = metrics["teeth"]["all"]["coverage"]
        teeth_gum_all = metrics["teeth_gum"]["all"]["coverage"]
        if (teeth_all + teeth_gum_all) >= threshold:
            state.rewarded_total[env_id] = True
            reward[env_id] = 1.0
    return reward
