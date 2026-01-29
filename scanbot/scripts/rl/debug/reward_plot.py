"""Mean reward plotting."""

from __future__ import annotations

from collections import deque

from . import manager
from . import plot_common


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
        plot_common.apply_qt_focus_flags(fig)

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
