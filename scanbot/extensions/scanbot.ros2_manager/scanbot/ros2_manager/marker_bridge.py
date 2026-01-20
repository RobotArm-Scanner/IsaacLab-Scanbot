"""Marker subscription + rendering for scanbot.ros2_manager."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import carb

from scanbot.common.marker_util import WorldDirectionMarker
from scanbot.common import pos_util

from .config import (
    MARKER_CLEAR_TOPIC,
    MARKER_DEBUG_CUBES_TOPIC,
    MARKER_FORWARD_AXIS,
    MARKER_INVERT_DIRECTION,
    MARKER_POSE_TOPIC,
)


try:  # Isaac Sim runtime only
    import isaacsim.util.debug_draw._debug_draw as omni_debug_draw
except Exception:  # pragma: no cover
    omni_debug_draw = None

try:
    import omni.usd
    from pxr import Gf, UsdGeom
except Exception:  # pragma: no cover
    omni = None
    Gf = None
    UsdGeom = None


_BASE_FRAMES = {"base", "base_link", "robot_base"}
_WORLD_FRAMES = {"world"}


@dataclass
class _MarkerDatum:
    pos_w: np.ndarray
    quat_wxyz: np.ndarray
    color_rgba: np.ndarray


class MarkerBridge:
    def __init__(self, node, marker_msg_type, empty_msg_type) -> None:
        self._node = node
        self._marker_msg_type = marker_msg_type
        self._empty_msg_type = empty_msg_type
        self._marker_sub = None
        self._clear_sub = None
        self._debug_cube_sub = None
        self._frame_id = "base"
        self._marker_data: list[_MarkerDatum] = []
        self._clear_requested = False
        self._spawn_debug_cubes_requested = False
        self._markers_by_color: dict[tuple[float, float, float, float], WorldDirectionMarker] = {}
        self._draw = None
        self._device = "cpu"

        if self._node is not None and self._marker_msg_type is not None:
            try:
                self._marker_sub = self._node.create_subscription(
                    self._marker_msg_type, MARKER_POSE_TOPIC, self._on_marker_msg, 10
                )
            except Exception as exc:
                self._marker_sub = None
                msg = f"[scanbot.ros2_manager] Failed to create marker subscription: {exc}"
                try:
                    carb.log_warn(msg)
                except Exception:
                    pass
        elif self._marker_msg_type is None:
            msg = "[scanbot.ros2_manager] MarkerPoseArray type missing; markers topic disabled."
            try:
                carb.log_warn(msg)
            except Exception:
                pass
        if self._node is not None and self._empty_msg_type is not None:
            try:
                self._clear_sub = self._node.create_subscription(
                    self._empty_msg_type, MARKER_CLEAR_TOPIC, self._on_clear_msg, 1
                )
            except Exception:
                self._clear_sub = None
            try:
                self._debug_cube_sub = self._node.create_subscription(
                    self._empty_msg_type, MARKER_DEBUG_CUBES_TOPIC, self._on_debug_cube_msg, 1
                )
            except Exception:
                self._debug_cube_sub = None

    def shutdown(self) -> None:
        if self._node is not None:
            if self._marker_sub is not None:
                try:
                    self._node.destroy_subscription(self._marker_sub)
                except Exception:
                    pass
            if self._clear_sub is not None:
                try:
                    self._node.destroy_subscription(self._clear_sub)
                except Exception:
                    pass
            if self._debug_cube_sub is not None:
                try:
                    self._node.destroy_subscription(self._debug_cube_sub)
                except Exception:
                    pass
        self._marker_sub = None
        self._clear_sub = None
        self._debug_cube_sub = None
        self._clear_all()
        self._markers_by_color = {}
        self._marker_data = []

    def maybe_render(self, env) -> None:
        if env is None:
            return

        if self._clear_requested:
            self._clear_all()
            self._clear_requested = False
            self._marker_data = []
            return

        if self._spawn_debug_cubes_requested:
            self._spawn_debug_cubes(env)
            self._spawn_debug_cubes_requested = False

        if not self._marker_data:
            return

        if self._draw is None and omni_debug_draw is not None:
            try:
                self._draw = omni_debug_draw.acquire_debug_draw_interface()
            except Exception:
                self._draw = None

        if self._device == "cpu":
            try:
                self._device = str(env.device)
            except Exception:
                self._device = "cpu"

        data_world = self._convert_to_world(env, self._marker_data, self._frame_id)
        if not data_world:
            return

        active_colors: set[tuple[float, float, float, float]] = set()
        groups: dict[tuple[float, float, float, float], dict[str, list[np.ndarray]]] = {}

        for datum in data_world:
            color = tuple(float(c) for c in datum.color_rgba.tolist())
            active_colors.add(color)
            if color not in groups:
                groups[color] = {"pos": [], "quat": []}
            groups[color]["pos"].append(datum.pos_w)
            groups[color]["quat"].append(datum.quat_wxyz)

        # Hide markers that are no longer present.
        for color_key, marker in self._markers_by_color.items():
            if color_key not in active_colors:
                marker.clear()

        for color_key, data in groups.items():
            marker = self._markers_by_color.get(color_key)
            if marker is None:
                marker = self._build_marker(color_key)
                self._markers_by_color[color_key] = marker
            marker.render_poses(
                positions_w=np.stack(data["pos"], axis=0),
                quats_wxyz=np.stack(data["quat"], axis=0),
                forward_axis=MARKER_FORWARD_AXIS,
                invert_direction=MARKER_INVERT_DIRECTION,
                line_color=list(color_key),
                draw_lines=True,
            )

    def _on_clear_msg(self, _msg) -> None:
        self._clear_requested = True

    def _on_debug_cube_msg(self, _msg) -> None:
        self._spawn_debug_cubes_requested = True

    def _on_marker_msg(self, msg) -> None:
        try:
            frame_id = getattr(getattr(msg, "header", None), "frame_id", "") or ""
        except Exception:
            frame_id = ""
        self._frame_id = frame_id.strip() or "base"

        markers = getattr(msg, "markers", None)
        if not markers:
            self._clear_requested = True
            return

        data: list[_MarkerDatum] = []
        for entry in markers:
            try:
                pose7d = np.asarray(entry.pose7d, dtype=float).reshape(7)
            except Exception:
                continue
            pos = pose7d[:3]
            qx, qy, qz, qw = pose7d[3:].tolist()
            quat_wxyz = np.array([qw, qx, qy, qz], dtype=float)
            color_msg = getattr(entry, "color", None)
            if color_msg is None:
                color = np.array([1.0, 0.0, 0.0, 1.0], dtype=float)
            else:
                color = np.array(
                    [
                        float(getattr(color_msg, "r", 1.0)),
                        float(getattr(color_msg, "g", 0.0)),
                        float(getattr(color_msg, "b", 0.0)),
                        float(getattr(color_msg, "a", 1.0)),
                    ],
                    dtype=float,
                )
            data.append(_MarkerDatum(pos_w=pos, quat_wxyz=quat_wxyz, color_rgba=color))

        if not data:
            self._clear_requested = True
            return

        self._marker_data = data

    def _get_root_pose(self, env) -> tuple[np.ndarray | None, np.ndarray | None]:
        if env is None:
            return None, None
        try:
            robot = env.scene["robot"]
            root_pos_w = robot.data.root_pos_w[0].detach().cpu().numpy()
            root_quat_w = robot.data.root_quat_w[0].detach().cpu().numpy()
            return root_pos_w, root_quat_w
        except Exception:
            return None, None

    def _convert_to_world(
        self,
        env,
        data: list[_MarkerDatum],
        frame_id: str,
    ) -> list[_MarkerDatum]:
        if not data:
            return []
        if frame_id in _WORLD_FRAMES:
            return data
        if frame_id not in _BASE_FRAMES and frame_id != "":
            try:
                self._node.get_logger().warn(
                    f"Marker frame_id '{frame_id}' not recognized; interpreting as base frame."
                )
            except Exception:
                pass
        root_pos_w, root_quat_w = self._get_root_pose(env)
        if root_pos_w is None or root_quat_w is None:
            return []
        converted: list[_MarkerDatum] = []
        for datum in data:
            pos_w, quat_w = pos_util.base_to_world_pose(
                datum.pos_w, datum.quat_wxyz, root_pos_w, root_quat_w
            )
            converted.append(_MarkerDatum(pos_w=pos_w, quat_wxyz=quat_w, color_rgba=datum.color_rgba))
        return converted

    def _build_marker(self, color_key: tuple[float, float, float, float]) -> WorldDirectionMarker:
        rgba_255 = [max(0, min(255, int(round(c * 255)))) for c in color_key]
        color_str = "c_" + "_".join(f"{v:03d}" for v in rgba_255)
        sphere_key = f"scanbot_marker_sphere_{color_str}"
        cone_key = f"scanbot_marker_cone_{color_str}"
        sphere_path = f"/Visuals/ScanbotMarkers/{color_str}/Spheres"
        cone_path = f"/Visuals/ScanbotMarkers/{color_str}/Cones"
        rgb = (float(color_key[0]), float(color_key[1]), float(color_key[2]))
        return WorldDirectionMarker(
            ui_state={},
            device=self._device,
            draw_interface=self._draw,
            sphere_state_key=sphere_key,
            cone_state_key=cone_key,
            sphere_prim_path=sphere_path,
            cone_prim_path=cone_path,
            sphere_color=rgb,
            cone_color=rgb,
            line_color=list(color_key),
        )

    def _clear_all(self) -> None:
        if self._draw is not None:
            try:
                self._draw.clear_lines()
            except Exception:
                pass
        for marker in self._markers_by_color.values():
            try:
                marker.clear()
            except Exception:
                pass

    def _spawn_debug_cubes(self, env) -> None:
        if omni is None or UsdGeom is None or Gf is None:
            return
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        root_pos_w, _root_quat_w = self._get_root_pose(env)
        if root_pos_w is None:
            return

        def _spawn_cube(path: str, pos_w, color_rgb, size: float):
            try:
                if stage.GetPrimAtPath(path):
                    stage.RemovePrim(path)
            except Exception:
                pass
            prim = UsdGeom.Cube.Define(stage, path).GetPrim()
            cube = UsdGeom.Cube(prim)
            cube.CreateSizeAttr(float(size))
            try:
                UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*color_rgb)])
            except Exception:
                pass
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3f(*pos_w))

        _spawn_cube(
            "/Visuals/ScanbotMarkers/DebugCubeWorld",
            [0.0, 0.0, 0.05],
            (1.0, 0.0, 0.0),
            0.1,
        )
        _spawn_cube(
            "/Visuals/ScanbotMarkers/DebugCubeBase",
            [float(root_pos_w[0]), float(root_pos_w[1]), float(root_pos_w[2]) + 0.05],
            (0.0, 1.0, 0.0),
            0.1,
        )
