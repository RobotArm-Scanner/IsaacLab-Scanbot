"""RL debug helpers (plots, controls, debug draw)."""

from __future__ import annotations

from .manager import ensure_control_window, is_enabled, close_plot
from .reward_plot import update_reward_plot
from .tcp_traj_plot import update_tcp_traj_plot
from .coverage_plot import update_coverage_plot
from .teeth_gum_plot import update_teeth_gum_plot
from .scanpoint_draw import draw_scanpoint_debug, clear_debug_lines

__all__ = [
    "ensure_control_window",
    "is_enabled",
    "close_plot",
    "update_reward_plot",
    "update_tcp_traj_plot",
    "update_coverage_plot",
    "update_teeth_gum_plot",
    "draw_scanpoint_debug",
    "clear_debug_lines",
]
