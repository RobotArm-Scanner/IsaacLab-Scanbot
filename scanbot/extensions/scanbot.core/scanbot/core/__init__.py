"""Scanbot core extension."""

from __future__ import annotations

import math

import omni.ext
import omni.kit.app
import omni.kit.viewport.utility as vp_utils
import omni.ui as ui
import omni.usd
from pxr import Gf, UsdGeom


DEFAULT_VIEWPORT_WINDOW_NAME = "Viewport"
CAM_PATH = "/World/default_camera"
DEFAULT_POS = Gf.Vec3f(0.34156, -3.03778, 1.98427)
DEFAULT_ROT_XYZ_DEG = Gf.Vec3f(58.41977, 0, -0.08838)
CAMERA_SUFFIX = "_camera"
CAMERA_VIEWPORT_MAX = 6
EXCLUDE_CAMERA_NAME_SUBSTR = "free_wrist"


def create_camera(
    cam_path: str = CAM_PATH, position: Gf.Vec3f = DEFAULT_POS, rotation_xyz_deg: Gf.Vec3f = DEFAULT_ROT_XYZ_DEG
) -> None:
    stage = omni.usd.get_context().get_stage()
    stage.RemovePrim(cam_path)
    prim = UsdGeom.Camera.Define(stage, cam_path).GetPrim()

    xform = UsdGeom.Xformable(prim)
    xform.AddTranslateOp().Set(position)
    xform.AddRotateXYZOp().Set(rotation_xyz_deg)


def activate_camera(cam_path: str = CAM_PATH, viewport_window_name: str = DEFAULT_VIEWPORT_WINDOW_NAME) -> None:
    vp_utils.get_viewport_from_window_name(viewport_window_name).set_active_camera(cam_path)


def find_cameras(*, suffix: str = CAMERA_SUFFIX, limit: int = CAMERA_VIEWPORT_MAX) -> list[tuple[str, str]]:
    stage = omni.usd.get_context().get_stage()
    cameras = [
        (prim.GetName(), prim.GetPath().pathString)
        for prim in stage.Traverse()
        if prim.IsA(UsdGeom.Camera) and prim.GetName().endswith(suffix) and prim.GetPath().pathString != CAM_PATH
        and EXCLUDE_CAMERA_NAME_SUBSTR not in prim.GetName()
    ]
    cameras.sort()
    return cameras[:limit]


def create_camera_viewports(cameras: list[tuple[str, str]]) -> list[str]:
    titles: list[str] = []
    for title, cam_path in cameras:
        vp = vp_utils.get_viewport_from_window_name(title) or vp_utils.create_viewport_window(title)
        if hasattr(vp, "viewport_api"):
            vp.viewport_api.set_active_camera(cam_path)
        titles.append(title)
    return titles


def _compute_columns(camera_count: int) -> int:
    if camera_count <= 0:
        return 0
    # 1: [1], 2: [1;2], 3-4: 2 columns, 5-6: 3 columns
    return min(3, (camera_count + 1) // 2)


def dock_camera_grid(camera_viewport_titles: list[str]) -> None:
    main = ui.Workspace.get_window(DEFAULT_VIEWPORT_WINDOW_NAME)
    if main is None:
        return
    windows = [ui.Workspace.get_window(title) for title in camera_viewport_titles]
    windows = [w for w in windows if w is not None]
    if not windows:
        return

    columns = _compute_columns(len(windows))
    rows = int(math.ceil(len(windows) / columns))

    # Dock the first camera to the right of the default viewport.
    windows[0].dock_in(main, ui.DockPosition.RIGHT, 0.5)

    # Dock top-row cameras left-to-right.
    for col in range(1, columns):
        if col >= len(windows):
            break
        windows[col].dock_in(windows[col - 1], ui.DockPosition.RIGHT, 0.5)

    # Dock remaining rows (max 2 rows given column rule).
    if rows > 1:
        for col in range(columns):
            idx = col + columns
            if idx >= len(windows):
                continue
            windows[idx].dock_in(windows[col], ui.DockPosition.BOTTOM, 0.5)


def set_camera_pose(
    position: Gf.Vec3f = DEFAULT_POS,
    rotation_xyz_deg: Gf.Vec3f = DEFAULT_ROT_XYZ_DEG,
    cam_path: str = CAM_PATH,
) -> None:
    stage = omni.usd.get_context().get_stage()
    ops = UsdGeom.Xformable(stage.GetPrimAtPath(cam_path)).GetOrderedXformOps()
    ops[0].Set(position)
    ops[1].Set(rotation_xyz_deg)


class Extension(omni.ext.IExt):
    def on_startup(self, ext_id: str) -> None:
        self._ext_id = ext_id
        self._configured = False
        self._camera_viewport_titles = []
        self._sub = None
        self._dock_sub = None
        create_camera()
        activate_camera()
        stream = omni.kit.app.get_app().get_update_event_stream()
        self._sub = stream.create_subscription_to_pop(self._on_update, name="scanbot.core.update")

    def on_shutdown(self) -> None:
        if self._dock_sub is not None:
            self._dock_sub.unsubscribe()
            self._dock_sub = None
        if self._sub is not None:
            self._sub.unsubscribe()
            self._sub = None
        self._configured = False
        self._ext_id = ""

    def _on_update(self, _event) -> None:
        cameras = find_cameras(limit=CAMERA_VIEWPORT_MAX)
        if not cameras:
            return
        titles = [name for name, _ in cameras]
        if titles == self._camera_viewport_titles:
            return
        if self._dock_sub is not None:
            return
        # Close viewports that are no longer present.
        for title in self._camera_viewport_titles:
            if title not in titles:
                win = ui.Workspace.get_window(title)
                if win is not None:
                    win.close()
        self._camera_viewport_titles = create_camera_viewports(cameras)
        self._dock_sub = omni.kit.app.get_app_interface().get_post_update_event_stream().create_subscription_to_pop(
            self._on_dock, name="scanbot.core.dock"
        )
        self._configured = True

    def _on_dock(self, _event) -> None:
        self._dock_sub.unsubscribe()
        self._dock_sub = None
        dock_camera_grid(self._camera_viewport_titles)
