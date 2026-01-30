"""Mean reward plotting."""

from __future__ import annotations

from collections import deque

import numpy as np

from . import manager
from . import plot_common


def _get_reward_plot_state(env, key: tuple) -> dict:
    state = getattr(env, "_scanbot_reward_plot_state", None)
    if state is None or state.get("key") != key:
        state = {
            "key": key,
            "fig": None,
            "ax": None,
            "lines": None,
            "history": None,
            "cum_rewards": None,
            "last_episode_step": None,
        }
        env._scanbot_reward_plot_state = state
    return state


def update_reward_plot(
    env,
    update_interval: int = 1,
    max_points: int = 200,
    pause: float = 0.001,
    env_ids: list[int] | None = None,
    enabled: bool = True,
) -> None:
    if not enabled:
        manager.close_plot(env, "reward_plot")
        return
    manager.ensure_control_window(env)
    if not manager.is_enabled("reward_plot"):
        manager.close_plot(env, "reward_plot")
        return

    step = getattr(env, "common_step_counter", 0)
    if not plot_common.should_update(step, update_interval):
        return

    env_ids = plot_common.resolve_env_ids(env, env_ids)
    if not env_ids:
        return

    import matplotlib.pyplot as plt

    key = (tuple(env_ids), int(max_points))
    state = _get_reward_plot_state(env, key)
    fig = state.get("fig")
    ax = state.get("ax")
    if fig is None or not plt.fignum_exists(fig.number):
        plt.ion()
        plt.rcParams["figure.raise_window"] = False
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
        if not isinstance(axes, (list, tuple, np.ndarray)):
            axes = [axes]
        ax_mean = axes[0]
        ax_env = axes[1]
        fig.canvas.manager.set_window_title("Scanbot Reward")
        ax_mean.set_ylabel("mean reward")
        ax_env.set_xlabel("step")
        ax_env.set_ylabel("cumulative reward")
        ax_mean.grid(True, alpha=0.3)
        ax_env.grid(True, alpha=0.3)
        state["fig"] = fig
        state["ax"] = ax_mean
        state["ax_env"] = ax_env
        line_mean, = ax_mean.plot([], [], label="mean_reward")
        lines: dict[int, object] = {}
        history: dict[int, dict[str, object]] = {}
        cum_rewards: dict[int, float] = {}
        last_episode_step: dict[int, int] = {}
        for env_id in env_ids:
            line, = ax_env.plot([], [], label=f"env {env_id}")
            lines[env_id] = line
            history[env_id] = {
                "t": deque(maxlen=max_points),
                "r": deque(maxlen=max_points),
            }
            cum_rewards[env_id] = 0.0
            last_episode_step[env_id] = -1
        ax_mean.legend(loc="upper right", fontsize="small")
        ax_env.legend(loc="upper right", fontsize="x-small", ncol=4)
        state["lines"] = lines
        state["line_mean"] = line_mean
        state["history"] = history
        state["mean_history"] = {
            "t": deque(maxlen=max_points),
            "r": deque(maxlen=max_points),
        }
        state["cum_rewards"] = cum_rewards
        state["last_episode_step"] = last_episode_step
        plot_common.apply_qt_focus_flags(fig)

    rewards = env.reward_manager._reward_buf[env_ids].detach().cpu().numpy()
    lines = state["lines"]
    history = state["history"]
    cum_rewards = state["cum_rewards"]
    last_episode_step = state["last_episode_step"]
    line_mean = state["line_mean"]
    mean_history = state["mean_history"]

    for idx, env_id in enumerate(env_ids):
        episode_step = int(env.episode_length_buf[env_id].item())
        prev_step = int(last_episode_step.get(env_id, -1))
        if episode_step <= 1 or episode_step < prev_step:
            cum_rewards[env_id] = 0.0
        last_episode_step[env_id] = episode_step

        cum_rewards[env_id] += float(rewards[idx])
        env_hist = history[env_id]
        env_hist["t"].append(step)
        env_hist["r"].append(cum_rewards[env_id])
        lines[env_id].set_data(env_hist["t"], env_hist["r"])

    mean_reward = float(rewards.mean())
    mean_history["t"].append(step)
    mean_history["r"].append(mean_reward)
    line_mean.set_data(mean_history["t"], mean_history["r"])

    ax_mean = state["ax"]
    ax_mean.relim()
    ax_mean.autoscale_view()
    ax_env = state["ax_env"]
    ax_env.relim()
    ax_env.autoscale_view()
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(pause)
