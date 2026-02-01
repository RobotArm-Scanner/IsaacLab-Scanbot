"""TCP 3D trajectory plotting."""

from __future__ import annotations

from collections import deque

import numpy as np

from . import manager
from . import plot_common


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


def update_tcp_traj_plot(
    env,
    frame_name: str = "ee_frame",
    update_interval: int = 1,
    max_points: int = 500,
    pause: float = 0.001,
    env_ids: list[int] | None = None,
    enabled: bool = True,
) -> None:
    if not enabled:
        manager.close_plot(env, "tcp_traj_plot")
        return
    manager.ensure_control_window(env)
    if not manager.is_enabled("tcp_traj_plot"):
        manager.close_plot(env, "tcp_traj_plot")
        return

    step = getattr(env, "common_step_counter", 0)
    if not plot_common.should_update(step, update_interval):
        return

    env_ids = plot_common.resolve_env_ids(env, env_ids)
    if not env_ids:
        return

    import matplotlib.pyplot as plt

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

        plot_common.apply_qt_focus_flags(fig)

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
