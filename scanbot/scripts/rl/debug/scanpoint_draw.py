"""Termination range debug drawing."""

from __future__ import annotations

import numpy as np

import isaacsim.util.debug_draw._debug_draw as omni_debug_draw

from . import manager


def _get_scanpoint_debug_state(env, max_distance: float):
    state = getattr(env, "_scanbot_scanpoint_debug_state", None)
    radius = float(max_distance)
    if state is None or state.get("radius") != radius or state.get("num_envs") != env.num_envs:
        state = {
            "radius": radius,
            "num_envs": env.num_envs,
            "draw": None,
        }
        env._scanbot_scanpoint_debug_state = state
    return state


def clear_debug_lines(env) -> None:
    state = getattr(env, "_scanbot_scanpoint_debug_state", None)
    if not state:
        return
    draw = state.get("draw")
    if draw is not None:
        draw.clear_lines()


def draw_scanpoint_debug(
    env,
    max_distance: float,
    cam_pos,
    support_pos,
    dist,
    interval: int = 1,
    enabled: bool = True,
) -> None:
    if not enabled or not manager.is_enabled("termination_draw"):
        clear_debug_lines(env)
        return

    step = getattr(env, "common_step_counter", 0)
    interval = int(interval) if interval else 1
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
