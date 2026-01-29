"""Teeth coverage plotting."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from . import manager
from . import plot_common


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


def update_coverage_plot(
    env,
    state: Any,
    update_interval: int = 1,
    max_points: int = 200,
    pause: float = 0.001,
    env_ids: list[int] | None = None,
    show_legend: bool = False,
    show_summary: bool = True,
    enabled: bool = True,
) -> None:
    if not enabled:
        manager.close_plot(env, "coverage_plot")
        return
    manager.ensure_control_window(env)
    if not manager.is_enabled("coverage_plot"):
        manager.close_plot(env, "coverage_plot")
        return

    step = getattr(env, "common_step_counter", 0)
    if not plot_common.should_update(step, update_interval):
        return

    env_ids = plot_common.resolve_env_ids(env, env_ids)
    if not env_ids:
        return

    tooth_ids = [str(int(tid)) for tid in state.tracker.surface.tooth_ids]
    key = (tuple(env_ids), tuple(tooth_ids), int(max_points), bool(show_legend), bool(show_summary))

    import matplotlib.pyplot as plt

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
            env_hist: dict[str, object] = {
                "t": deque(maxlen=max_points),
                "y": {},
                "teeth_all": deque(maxlen=max_points),
            }
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

        plot_common.apply_qt_focus_flags(fig)

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
