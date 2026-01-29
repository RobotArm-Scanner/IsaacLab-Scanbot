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
from scanbot.scripts.utilities.teeth3ds_utils import compute_teeth_center
from scanbot.scripts.utilities.teeth3ds_utils import load_teeth_surface_cache
from scanbot.scripts.utilities.teeth3ds_utils import voxel_downsample

import isaacsim.util.debug_draw._debug_draw as omni_debug_draw


def _get_scanpoint_debug_state(env, max_distance: float):
    state = getattr(env, "_scanbot_scanpoint_debug_state", None)
    radius = float(max_distance)
    if state is None or state["radius"] != radius or state["num_envs"] != env.num_envs:
        state = {
            "radius": radius,
            "num_envs": env.num_envs,
            "draw": None,
        }
        env._scanbot_scanpoint_debug_state = state
    return state


def _maybe_draw_scanpoint_debug(
    env,
    max_distance: float,
    cam_pos: torch.Tensor,
    support_pos: torch.Tensor,
    dist: torch.Tensor,
    debug_draw_interval: int,
) -> None:
    step = getattr(env, "common_step_counter", 0)
    interval = int(debug_draw_interval) if debug_draw_interval else 1
    if interval > 1 and step % interval != 0:
        return
    state = _get_scanpoint_debug_state(env, max_distance)
    draw = state["draw"]
    if draw is None:
        state["draw"] = omni_debug_draw.acquire_debug_draw_interface()
        draw = state["draw"]
    if draw is None:
        return

    support_np = support_pos.detach().cpu().numpy()
    if support_np.ndim == 1:
        support_np = support_np.reshape(1, 3)
    cam_np = cam_pos.detach().cpu().numpy()
    if cam_np.ndim == 1:
        cam_np = cam_np.reshape(1, 3)
    dist_np = dist.detach().cpu().numpy().reshape(-1)

    # Rebuild the wireframe each update to avoid stale debug lines.
    draw.clear_lines()

    starts: list[list[float]] = []
    ends: list[list[float]] = []
    colors: list[list[float]] = []
    thickness: list[float] = []

    theta = np.linspace(0.0, 2.0 * np.pi, 49)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    sphere_color = [0.2, 0.7, 1.0, 0.9]

    for i in range(support_np.shape[0]):
        center = support_np[i]
        r = float(max_distance)
        # XY circle
        for j in range(len(theta) - 1):
            p0 = center + np.array([cos_t[j], sin_t[j], 0.0]) * r
            p1 = center + np.array([cos_t[j + 1], sin_t[j + 1], 0.0]) * r
            starts.append(p0.tolist())
            ends.append(p1.tolist())
            colors.append(sphere_color)
            thickness.append(1.5)
        # XZ circle
        for j in range(len(theta) - 1):
            p0 = center + np.array([cos_t[j], 0.0, sin_t[j]]) * r
            p1 = center + np.array([cos_t[j + 1], 0.0, sin_t[j + 1]]) * r
            starts.append(p0.tolist())
            ends.append(p1.tolist())
            colors.append(sphere_color)
            thickness.append(1.5)
        # YZ circle
        for j in range(len(theta) - 1):
            p0 = center + np.array([0.0, cos_t[j], sin_t[j]]) * r
            p1 = center + np.array([0.0, cos_t[j + 1], sin_t[j + 1]]) * r
            starts.append(p0.tolist())
            ends.append(p1.tolist())
            colors.append(sphere_color)
            thickness.append(1.5)

        # Support -> camera line (green inside, red outside).
        starts.append(center.tolist())
        ends.append(cam_np[i].tolist())
        inside = dist_np[i] <= max_distance
        colors.append([0.1, 0.9, 0.2, 0.9] if inside else [1.0, 0.2, 0.2, 0.9])
        thickness.append(2.0)

    draw.draw_lines(starts, ends, colors, thickness)


def _get_reward_plot_state(env, key: tuple) -> dict:
    state = getattr(env, "_scanbot_reward_plot_state", None)
    if state is None or state.get("key") != key:
        state = {
            "key": key,
            "fig": None,
            "ax": None,
            "line": None,
            "history": None,
        }
        env._scanbot_reward_plot_state = state
    return state


def _update_reward_plot(
    env,
    update_interval: int = 1,
    max_points: int = 200,
    pause: float = 0.001,
    env_ids: list[int] | None = None,
) -> None:
    step = getattr(env, "common_step_counter", 0)
    interval = int(update_interval) if update_interval else 1
    if interval > 1 and step % interval != 0:
        return

    if env_ids is None:
        env_ids = list(range(env.num_envs))

    if not env_ids:
        return

    import matplotlib
    import matplotlib.pyplot as plt
    from collections import deque

    key = (tuple(env_ids), int(max_points))
    state = _get_reward_plot_state(env, key)
    fig = state.get("fig")
    ax = state.get("ax")
    if fig is None or not plt.fignum_exists(fig.number):
        plt.ion()
        plt.rcParams["figure.raise_window"] = False
        fig, ax = plt.subplots()
        fig.canvas.manager.set_window_title("Scanbot Mean Reward")
        ax.set_xlabel("step")
        ax.set_ylabel("mean reward")
        ax.grid(True, alpha=0.3)
        state["fig"] = fig
        state["ax"] = ax
        state["line"], = ax.plot([], [], label="mean_reward")
        ax.legend(loc="upper right", fontsize="small")
        state["history"] = {
            "t": deque(maxlen=max_points),
            "r": deque(maxlen=max_points),
        }
        # Prevent the plot window from stealing focus in Qt.
        if "Qt" in str(matplotlib.get_backend()):
            from matplotlib.backends.qt_compat import QtCore

            window = fig.canvas.manager.window
            window.setAttribute(QtCore.Qt.WA_ShowWithoutActivating, True)
            window.setWindowFlag(QtCore.Qt.Tool, True)
            window.setFocusPolicy(QtCore.Qt.NoFocus)
            window.show()

    rewards = env.reward_manager._reward_buf[env_ids].detach().cpu().numpy()
    mean_reward = float(rewards.mean())
    history = state["history"]
    history["t"].append(step)
    history["r"].append(mean_reward)
    state["line"].set_data(history["t"], history["r"])

    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(pause)


def _get_coverage_plot_state(env, key: tuple) -> dict:
    state = getattr(env, "_scanbot_coverage_plot_state", None)
    if state is None or state.get("key") != key:
        state = {
            "key": key,
            "fig": None,
            "axes_left": None,
            "axes_right": None,
            "lines": None,
            "summary_lines": None,
            "history": None,
        }
        env._scanbot_coverage_plot_state = state
    return state


def _get_tcp_traj_plot_state(env, key: tuple) -> dict:
    state = getattr(env, "_scanbot_tcp_traj_plot_state", None)
    if state is None or state.get("key") != key:
        state = {
            "key": key,
            "fig": None,
            "axes": None,
            "lines": None,
            "history": None,
            "last_episode_step": None,
        }
        env._scanbot_tcp_traj_plot_state = state
    return state


def _update_tcp_traj_plot(
    env,
    frame_name: str = "ee_frame",
    update_interval: int = 1,
    max_points: int = 500,
    pause: float = 0.001,
    env_ids: list[int] | None = None,
) -> None:
    step = getattr(env, "common_step_counter", 0)
    interval = int(update_interval) if update_interval else 1
    if interval > 1 and step % interval != 0:
        return

    if env_ids is None:
        env_ids = list(range(env.num_envs))

    if not env_ids:
        return

    import matplotlib
    import matplotlib.pyplot as plt
    from collections import deque

    key = (tuple(env_ids), frame_name, int(max_points))
    state = _get_tcp_traj_plot_state(env, key)
    fig = state.get("fig")
    axes = state.get("axes")
    if fig is None or not plt.fignum_exists(fig.number):
        plt.ion()
        plt.rcParams["figure.raise_window"] = False
        fig, axes = plt.subplots(
            len(env_ids),
            1,
            sharex=False,
            subplot_kw={"projection": "3d"},
            figsize=(8, 3 * len(env_ids)),
        )
        if len(env_ids) == 1:
            axes = [axes]
        fig.canvas.manager.set_window_title("Scanbot TCP Trajectory")

        lines: dict[int, object] = {}
        history: dict[int, dict[str, object]] = {}
        last_episode_step: dict[int, int] = {}
        for ax, env_id in zip(axes, env_ids):
            ax.set_title(f"env {env_id}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.grid(True, alpha=0.3)
            line, = ax.plot([], [], [], label="tcp")
            lines[env_id] = line
            history[env_id] = {
                "x": deque(maxlen=max_points),
                "y": deque(maxlen=max_points),
                "z": deque(maxlen=max_points),
            }
            last_episode_step[env_id] = -1
        state["fig"] = fig
        state["axes"] = axes
        state["lines"] = lines
        state["history"] = history
        state["last_episode_step"] = last_episode_step

        # Prevent the plot window from stealing focus in Qt.
        if "Qt" in str(matplotlib.get_backend()):
            from matplotlib.backends.qt_compat import QtCore

            window = fig.canvas.manager.window
            window.setAttribute(QtCore.Qt.WA_ShowWithoutActivating, True)
            window.setWindowFlag(QtCore.Qt.Tool, True)
            window.setFocusPolicy(QtCore.Qt.NoFocus)
            window.show()

    frame = env.scene[frame_name]
    tcp_pos = frame.data.target_pos_w[:, 0, :] - env.scene.env_origins[:, 0:3]
    tcp_np = tcp_pos.detach().cpu().numpy()

    lines = state["lines"]
    history = state["history"]
    axes = state["axes"]
    last_episode_step = state.get("last_episode_step", {})

    for ax, env_id in zip(axes, env_ids):
        episode_step = int(env.episode_length_buf[env_id].item())
        prev_step = int(last_episode_step.get(env_id, -1))
        if episode_step <= 1 or episode_step < prev_step:
            env_hist = history[env_id]
            env_hist["x"].clear()
            env_hist["y"].clear()
            env_hist["z"].clear()
            line = lines[env_id]
            line.set_data([], [])
            line.set_3d_properties([])
        last_episode_step[env_id] = episode_step

        pos = tcp_np[env_id]
        env_hist = history[env_id]
        env_hist["x"].append(float(pos[0]))
        env_hist["y"].append(float(pos[1]))
        env_hist["z"].append(float(pos[2]))
        line = lines[env_id]
        line.set_data(env_hist["x"], env_hist["y"])
        line.set_3d_properties(env_hist["z"])

        xs = np.asarray(env_hist["x"], dtype=float)
        ys = np.asarray(env_hist["y"], dtype=float)
        zs = np.asarray(env_hist["z"], dtype=float)
        if xs.size:
            pad = 0.01
            ax.set_xlim(xs.min() - pad, xs.max() + pad)
            ax.set_ylim(ys.min() - pad, ys.max() + pad)
            ax.set_zlim(zs.min() - pad, zs.max() + pad)

    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(pause)

def _update_coverage_plot(
    env,
    state: "_CoverageState",
    update_interval: int = 1,
    max_points: int = 200,
    pause: float = 0.001,
    env_ids: list[int] | None = None,
    show_legend: bool = False,
    show_summary: bool = True,
) -> None:
    step = getattr(env, "common_step_counter", 0)
    interval = int(update_interval) if update_interval else 1
    if interval > 1 and step % interval != 0:
        return

    if env_ids is None:
        env_ids = list(range(env.num_envs))

    if not env_ids:
        return

    tooth_ids = [str(int(tid)) for tid in state.tracker.surface.tooth_ids]
    key = (tuple(env_ids), tuple(tooth_ids), int(max_points), bool(show_legend), bool(show_summary))

    import matplotlib
    import matplotlib.pyplot as plt
    from collections import deque

    plot_state = _get_coverage_plot_state(env, key)
    fig = plot_state.get("fig")
    axes_left = plot_state.get("axes_left")
    axes_right = plot_state.get("axes_right")
    if fig is None or not plt.fignum_exists(fig.number):
        plt.ion()
        plt.rcParams["figure.raise_window"] = False
        if show_summary:
            fig, axes = plt.subplots(
                len(env_ids),
                2,
                sharex=True,
                figsize=(12, 3 * len(env_ids)),
            )
            axes = np.asarray(axes)
            if axes.ndim == 1:
                axes = axes.reshape(1, 2)
            axes_left = [axes[i][0] for i in range(len(env_ids))]
            axes_right = [axes[i][1] for i in range(len(env_ids))]
        else:
            fig, axes = plt.subplots(len(env_ids), 1, sharex=True, figsize=(10, 3 * len(env_ids)))
            if len(env_ids) == 1:
                axes = [axes]
            axes_left = list(axes)
            axes_right = None
        fig.canvas.manager.set_window_title("Scanbot Coverage per Tooth")

        lines: dict[int, dict[str, object]] = {}
        summary_lines: dict[int, object] | None = {} if show_summary else None
        history: dict[int, dict[str, object]] = {}
        for row, env_id in enumerate(env_ids):
            left_ax = axes_left[row]
            left_ax.set_ylabel(f"env {env_id}")
            left_ax.set_ylim(0.0, 1.0)
            left_ax.grid(True, alpha=0.3)
            env_lines: dict[str, object] = {}
            env_hist: dict[str, object] = {"t": deque(maxlen=max_points), "y": {}, "teeth_all": deque(maxlen=max_points)}
            for tooth_id in tooth_ids:
                line, = left_ax.plot([], [], label=tooth_id)
                env_lines[tooth_id] = line
                env_hist["y"][tooth_id] = deque(maxlen=max_points)
            if show_legend:
                left_ax.legend(loc="upper right", ncol=4, fontsize="x-small")
            lines[env_id] = env_lines
            history[env_id] = env_hist
            if show_summary and axes_right is not None:
                right_ax = axes_right[row]
                right_ax.set_ylabel("teeth/all")
                right_ax.set_ylim(0.0, 1.0)
                right_ax.grid(True, alpha=0.3)
                summary_line, = right_ax.plot([], [], label="teeth/all")
                summary_lines[env_id] = summary_line
        axes_left[-1].set_xlabel("step")
        if show_summary and axes_right is not None:
            axes_right[-1].set_xlabel("step")

        plot_state["fig"] = fig
        plot_state["axes_left"] = axes_left
        plot_state["axes_right"] = axes_right
        plot_state["lines"] = lines
        plot_state["summary_lines"] = summary_lines
        plot_state["history"] = history

        # Prevent the plot window from stealing focus in Qt.
        if "Qt" in str(matplotlib.get_backend()):
            from matplotlib.backends.qt_compat import QtCore

            window = fig.canvas.manager.window
            window.setAttribute(QtCore.Qt.WA_ShowWithoutActivating, True)
            window.setWindowFlag(QtCore.Qt.Tool, True)
            window.setFocusPolicy(QtCore.Qt.NoFocus)
            window.show()

    lines = plot_state["lines"]
    summary_lines = plot_state.get("summary_lines")
    history = plot_state["history"]
    axes_left = plot_state["axes_left"]
    axes_right = plot_state.get("axes_right")

    for env_id in env_ids:
        metrics = state.metrics[env_id]
        if not metrics:
            continue
        teeth = metrics.get("teeth", {})
        env_hist = history[env_id]
        env_hist["t"].append(step)
        for tooth_id in tooth_ids:
            value = float(teeth.get(tooth_id, {}).get("coverage", 0.0))
            env_hist["y"][tooth_id].append(value)
            lines[env_id][tooth_id].set_data(env_hist["t"], env_hist["y"][tooth_id])
        if summary_lines is not None:
            teeth_all = float(teeth.get("all", {}).get("coverage", 0.0))
            env_hist["teeth_all"].append(teeth_all)
            summary_lines[env_id].set_data(env_hist["t"], env_hist["teeth_all"])

    for ax in axes_left:
        ax.relim()
        ax.autoscale_view(scalex=True, scaley=False)
    if axes_right is not None:
        for ax in axes_right:
            ax.relim()
            ax.autoscale_view(scalex=True, scaley=False)

    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(pause)


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
        step_np = step_buf.detach().cpu().numpy().astype(np.int64)
        reset_mask = (step_np <= 1) | (step_np < self.last_update_step)
        if np.any(reset_mask):
            reset_ids = torch.as_tensor(np.nonzero(reset_mask)[0], device=step_buf.device)
            self.reset_envs(reset_ids)

        camera = env.scene[self.camera_name]
        if self.data_type not in camera.data.output:
            raise KeyError(f"Camera output '{self.data_type}' not available for {self.camera_name}")
        depth = camera.data.output[self.data_type]
        intrinsics = camera.data.intrinsic_matrices
        num_envs = env.num_envs
        cam_pos, cam_quat_opengl = camera._view.get_world_poses()
        cam_pos = cam_pos.to(depth.device)
        cam_quat = math_utils.convert_camera_frame_orientation_convention(
            cam_quat_opengl.to(depth.device), origin="opengl", target="ros"
        )

        teeth = env.scene[self.teeth_name]
        teeth_pos = teeth.data.root_pos_w
        teeth_quat = teeth.data.root_quat_w

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


def _get_teeth_center_state(env, params: Dict[str, object]) -> dict:
    cache_key = (
        params["dataset_id"],
        params["num_samples"],
        params["seed"],
        params["gum_assign_radius"],
        params["scale"],
    )
    state = getattr(env, "_scanbot_teeth_center_state", None)
    if state is None or state.get("key") != cache_key:
        surface = load_teeth_surface_cache(
            resources_root=params["resources_root"],
            dataset_id=params["dataset_id"],
            num_samples=params["num_samples"],
            seed=params["seed"],
            gum_assign_radius=params["gum_assign_radius"],
            scale=params["scale"],
        )
        state = {
            "key": cache_key,
            "center_local": compute_teeth_center(surface),
        }
        env._scanbot_teeth_center_state = state
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


def step_progress_penalty(env, power: float = 1.0) -> torch.Tensor:
    progress = env.episode_length_buf.float() / max(1, env.max_episode_length)
    if power != 1.0:
        progress = torch.pow(progress, float(power))
    return progress


def ee_far_from_teeth(
    env,
    max_distance: float,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    teeth_name: str = "teeth",
) -> torch.Tensor:
    ee_pos = env.scene[ee_frame_cfg.name].data.target_pos_w[:, 0, :] - env.scene.env_origins[:, 0:3]
    teeth_pos = env.scene[teeth_name].data.root_pos_w - env.scene.env_origins[:, 0:3]
    dist = torch.norm(ee_pos - teeth_pos, dim=1)
    return dist > max_distance


def scanpoint_far_from_support(
    env,
    max_distance: float,
    camera_name: str = "wrist_camera",
    support_name: str = "teeth_support",
    debug_draw: bool = False,
    debug_draw_interval: int = 1,
    reward_plot: bool = False,
    reward_plot_interval: int = 1,
    reward_plot_max_points: int = 200,
    reward_plot_pause: float = 0.001,
    reward_plot_env_ids: list[int] | None = None,
) -> torch.Tensor:
    camera = env.scene[camera_name]
    support = env.scene[support_name]
    cam_pos, _ = camera._view.get_world_poses()
    cam_pos = cam_pos.to(support.data.root_pos_w.device)
    support_pos = support.data.root_pos_w
    dist = torch.norm(cam_pos - support_pos, dim=1)
    if debug_draw:
        _maybe_draw_scanpoint_debug(
            env,
            max_distance=max_distance,
            cam_pos=cam_pos,
            support_pos=support_pos,
            dist=dist,
            debug_draw_interval=debug_draw_interval,
        )
    if reward_plot:
        _update_reward_plot(
            env,
            update_interval=reward_plot_interval,
            max_points=reward_plot_max_points,
            pause=float(reward_plot_pause),
            env_ids=reward_plot_env_ids,
        )
    return dist > max_distance


def scanpoint_far_from_teeth_center(
    env,
    max_distance: float,
    resources_root: str,
    dataset_id: str,
    num_samples: int,
    seed: int,
    gum_assign_radius: float,
    scale: tuple,
    camera_name: str = "wrist_camera",
    teeth_name: str = "teeth",
    debug_draw: bool = False,
    debug_draw_interval: int = 1,
    reward_plot: bool = False,
    reward_plot_interval: int = 1,
    reward_plot_max_points: int = 200,
    reward_plot_pause: float = 0.001,
    reward_plot_env_ids: list[int] | None = None,
    tcp_traj_plot: bool = False,
    tcp_traj_plot_frame: str = "ee_frame",
    tcp_traj_plot_interval: int = 1,
    tcp_traj_plot_max_points: int = 500,
    tcp_traj_plot_pause: float = 0.001,
    tcp_traj_plot_env_ids: list[int] | None = None,
) -> torch.Tensor:
    params = {
        "resources_root": resources_root,
        "dataset_id": dataset_id,
        "num_samples": num_samples,
        "seed": seed,
        "gum_assign_radius": gum_assign_radius,
        "scale": scale,
    }
    state = _get_teeth_center_state(env, params)
    camera = env.scene[camera_name]
    cam_pos, _ = camera._view.get_world_poses()

    teeth = env.scene[teeth_name]
    teeth_pos = teeth.data.root_pos_w
    teeth_quat = teeth.data.root_quat_w
    cam_pos = cam_pos.to(teeth_pos.device)

    center_local = torch.as_tensor(state["center_local"], device=teeth_pos.device)
    center_local = center_local.unsqueeze(0).expand(teeth_pos.shape[0], -1)
    center_world = math_utils.quat_apply(teeth_quat, center_local) + teeth_pos

    dist = torch.norm(cam_pos - center_world, dim=1)
    if debug_draw:
        _maybe_draw_scanpoint_debug(
            env,
            max_distance=max_distance,
            cam_pos=cam_pos,
            support_pos=center_world,
            dist=dist,
            debug_draw_interval=debug_draw_interval,
        )
    if reward_plot:
        _update_reward_plot(
            env,
            update_interval=reward_plot_interval,
            max_points=reward_plot_max_points,
            pause=float(reward_plot_pause),
            env_ids=reward_plot_env_ids,
        )
    if tcp_traj_plot:
        _update_tcp_traj_plot(
            env,
            frame_name=tcp_traj_plot_frame,
            update_interval=tcp_traj_plot_interval,
            max_points=tcp_traj_plot_max_points,
            pause=float(tcp_traj_plot_pause),
            env_ids=tcp_traj_plot_env_ids,
        )
    return dist > max_distance


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


def teeth_coverage_reached(
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

    coverage = torch.zeros(env.num_envs, device=env.device)
    for env_id in range(env.num_envs):
        metrics = state.metrics[env_id]
        if not metrics:
            continue
        coverage[env_id] = float(metrics["teeth"]["all"]["coverage"])

    return coverage >= float(threshold)


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
    coverage_plot: bool = False,
    coverage_plot_interval: int = 1,
    coverage_plot_max_points: int = 200,
    coverage_plot_pause: float = 0.001,
    coverage_plot_env_ids: list[int] | None = None,
    coverage_plot_show_legend: bool = False,
    coverage_plot_show_summary: bool = True,
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
    if coverage_plot:
        _update_coverage_plot(
            env,
            state,
            update_interval=coverage_plot_interval,
            max_points=coverage_plot_max_points,
            pause=float(coverage_plot_pause),
            env_ids=coverage_plot_env_ids,
            show_legend=coverage_plot_show_legend,
            show_summary=coverage_plot_show_summary,
        )
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
