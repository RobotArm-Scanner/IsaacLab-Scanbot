"""Marker utilities that operate in world coordinates."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

import isaaclab.utils.math as math_utils
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import SPHERE_MARKER_CFG


class WorldDirectionMarker:
    def __init__(
        self,
        ui_state: dict,
        device,
        draw_interface,
        sphere_state_key: str,
        cone_state_key: str,
        sphere_prim_path: str,
        cone_prim_path: str,
        sphere_color=None,
        sphere_scale: float = 0.03,
        cone_color=None,
        arrow_len_scale: float = 0.4,
        cone_height: float = 0.0066667,
        line_color=None,
        line_thickness: float = 6.0,
    ):
        self._state = ui_state
        self._device = device
        self._draw = draw_interface
        self._sphere_state_key = sphere_state_key
        self._cone_state_key = cone_state_key
        self._sphere_prim_path = sphere_prim_path
        self._cone_prim_path = cone_prim_path
        self._sphere_scale = float(sphere_scale)
        self._arrow_len_scale = float(arrow_len_scale)
        self._cone_height = float(cone_height)
        self._line_color = list(line_color) if line_color is not None else [1.0, 0.0, 0.0, 1.0]
        self._line_thickness = float(line_thickness)
        self._sphere_color = tuple(sphere_color) if sphere_color is not None else None
        self._cone_color = tuple(cone_color) if cone_color is not None else None

    def _ensure_sphere_marker(self):
        mk = self._state.get(self._sphere_state_key)
        if mk is not None:
            return mk
        mk_cfg = SPHERE_MARKER_CFG.copy()
        mk_cfg.prim_path = self._sphere_prim_path
        if self._sphere_color is not None:
            try:
                mk_cfg.markers["sphere"].visual_material = sim_utils.PreviewSurfaceCfg(
                    diffuse_color=self._sphere_color, roughness=0.3
                )
            except Exception:
                pass
        self._state[self._sphere_state_key] = VisualizationMarkers(cfg=mk_cfg)
        return self._state[self._sphere_state_key]

    def _ensure_cone_marker(self):
        mk = self._state.get(self._cone_state_key)
        if mk is not None:
            return mk
        color = self._cone_color if self._cone_color is not None else (1.0, 0.0, 0.0)
        cone_cfg = VisualizationMarkersCfg(
            prim_path=self._cone_prim_path,
            markers={
                "arrow_head": sim_utils.ConeCfg(
                    radius=0.0016667,
                    height=self._cone_height,
                    axis="Z",
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.3),
                )
            },
        )
        self._state[self._cone_state_key] = VisualizationMarkers(cfg=cone_cfg)
        return self._state[self._cone_state_key]

    def clear(self):
        mk_s = self._state.get(self._sphere_state_key)
        if mk_s is not None:
            try:
                mk_s.set_visibility(False)
            except Exception:
                pass
        mk_c = self._state.get(self._cone_state_key)
        if mk_c is not None:
            try:
                mk_c.set_visibility(False)
            except Exception:
                pass

    def render_points(
        self,
        points_w,
        dirs_w,
        line_color=None,
        line_thickness: float | None = None,
        arrow_len: float | None = None,
        invert_direction: bool = False,
        draw_lines: bool = True,
    ):
        if points_w is None or len(points_w) == 0:
            self.clear()
            return

        pts_w = torch.tensor(points_w, dtype=torch.float32, device=self._device)
        pts_np = pts_w.detach().cpu().numpy()

        mk_s = self._ensure_sphere_marker()
        mk_s.set_visibility(True)
        scales = np.full((pts_np.shape[0], 3), self._sphere_scale, dtype=float)
        mk_s.visualize(
            translations=pts_np.astype(np.float32),
            scales=scales.astype(np.float32),
            marker_indices=[0] * pts_np.shape[0],
        )

        color_line = list(line_color) if line_color is not None else self._line_color
        thickness = float(line_thickness) if line_thickness is not None else self._line_thickness

        if dirs_w is None:
            return

        d_w = torch.tensor(dirs_w, dtype=torch.float32, device=self._device)
        if invert_direction:
            d_w = -d_w
        if arrow_len is None:
            arrow_len = float(self._arrow_len_scale * self._sphere_scale)
        shaft_start = pts_w
        shaft_end = pts_w + d_w * float(arrow_len)
        d_w_n = d_w / d_w.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-9)
        head_h = self._cone_height
        line_end = shaft_end - d_w_n * head_h

        shaft_start_np = shaft_start.detach().cpu().numpy()
        line_end_np = line_end.detach().cpu().numpy()

        starts_all = []
        ends_all = []
        colors_all = []
        thick_all = []

        for i in range(shaft_start_np.shape[0]):
            starts_all.append(shaft_start_np[i].tolist())
            ends_all.append(line_end_np[i].tolist())
            colors_all.append(color_line)
            thick_all.append(thickness)

        mk_c = self._ensure_cone_marker()
        mk_c.set_visibility(True)

        z_axis = torch.tensor([0.0, 0.0, 1.0], device=self._device, dtype=torch.float32)
        z_repeat = z_axis.view(1, 3).repeat(d_w.shape[0], 1)
        cosang = (z_repeat * d_w_n).sum(dim=-1).clamp(min=-1.0, max=1.0)
        angle = torch.acos(cosang)
        axis = torch.cross(z_repeat, d_w_n, dim=-1)
        axis_norm = axis.norm(p=2, dim=-1, keepdim=True)
        zero_mask = axis_norm.squeeze(-1) < 1e-6
        anti_mask = cosang < -0.9999
        axis = torch.where(
            zero_mask.unsqueeze(-1),
            torch.tensor([1.0, 0.0, 0.0], device=axis.device, dtype=axis.dtype).expand_as(axis),
            axis / axis_norm.clamp(min=1e-9),
        )
        angle = torch.where(zero_mask & (~anti_mask), torch.zeros_like(angle), angle)
        angle = torch.where(zero_mask & anti_mask, torch.full_like(angle, float(np.pi)), angle)
        quat_wxyz = math_utils.quat_from_angle_axis(angle, axis)

        cone_trans = shaft_end - d_w_n * (head_h * 0.5)
        mk_c.visualize(
            translations=cone_trans,
            orientations=quat_wxyz,
            marker_indices=[0] * cone_trans.shape[0],
        )

        if len(starts_all) > 0 and draw_lines and self._draw is not None:
            self._draw.draw_lines(starts_all, ends_all, colors_all, thick_all)

    def render_poses(
        self,
        positions_w,
        quats_wxyz,
        forward_axis=(0.0, 0.0, 1.0),
        invert_direction: bool = False,
        **kwargs,
    ):
        if positions_w is None or quats_wxyz is None:
            self.clear()
            return
        pos = np.asarray(positions_w, dtype=float)
        quat = np.asarray(quats_wxyz, dtype=float)
        if pos.ndim == 1:
            pos = pos.reshape(1, 3)
        if quat.ndim == 1:
            quat = quat.reshape(1, 4)
        axis = torch.tensor(forward_axis, dtype=torch.float32, device=self._device).view(1, 3)
        q_t = torch.tensor(quat, dtype=torch.float32, device=self._device)
        dirs = math_utils.quat_apply(q_t, axis.expand(q_t.shape[0], -1))
        self.render_points(pos, dirs, invert_direction=invert_direction, **kwargs)
