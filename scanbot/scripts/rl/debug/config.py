"""RL debug config helpers."""

from __future__ import annotations


def scanpoint_debug_params() -> dict:
    return {
        "debug_draw": True,
        "debug_draw_interval": 1,
    }


def reward_plot_params() -> dict:
    return {
        "reward_plot": True,
        "reward_plot_interval": 1,
        "reward_plot_max_points": 200,
        "reward_plot_pause": 0.001,
        "reward_plot_env_ids": None,
    }


def tcp_traj_plot_params() -> dict:
    return {
        "tcp_traj_plot": True,
        "tcp_traj_plot_frame": "ee_frame",
        "tcp_traj_plot_interval": 1,
        "tcp_traj_plot_max_points": 500,
        "tcp_traj_plot_pause": 0.001,
        "tcp_traj_plot_env_ids": None,
    }


def coverage_plot_params() -> dict:
    return {
        "coverage_plot": True,
        "coverage_plot_interval": 1,
        "coverage_plot_max_points": 200,
        "coverage_plot_pause": 0.001,
        "coverage_plot_env_ids": None,
        "coverage_plot_show_legend": True,
        "coverage_plot_show_summary": True,
    }


def teeth_gum_plot_params() -> dict:
    return {
        "teeth_gum_plot": True,
        "teeth_gum_plot_interval": 1,
        "teeth_gum_plot_max_points": 200,
        "teeth_gum_plot_pause": 0.001,
        "teeth_gum_plot_env_ids": None,
        "teeth_gum_plot_show_legend": True,
        "teeth_gum_plot_show_summary": True,
    }
