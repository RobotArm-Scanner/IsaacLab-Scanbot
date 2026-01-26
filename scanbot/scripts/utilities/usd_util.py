"""USD helper utilities for Scanbot."""

from __future__ import annotations

import math
import os

from isaaclab.sim import schemas as schemas_utils
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.utils import clone


@clone
def _spawn_rigid_object_from_usd(
    prim_path: str,
    cfg: UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn a USD and ensure a rigid body exists at the prim root."""
    usd_path = str(cfg.usd_path)
    if "://" not in usd_path and not os.path.isfile(usd_path):
        raise FileNotFoundError(f"USD file not found at path: '{usd_path}'.")

    from isaacsim.core.utils.stage import get_current_stage
    from pxr import Gf, UsdGeom, UsdPhysics

    stage = get_current_stage()
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim.IsValid():
        root_prim = stage.DefinePrim(prim_path, "Xform")

    ref_prim_path = f"{prim_path}/asset"
    ref_prim = stage.GetPrimAtPath(ref_prim_path)
    if not ref_prim.IsValid():
        ref_prim = stage.DefinePrim(ref_prim_path, "Xform")
        ref_prim.GetReferences().AddReference(usd_path)

    asset_offset = getattr(cfg, "asset_offset", None)
    asset_orient_deg = getattr(cfg, "asset_orient_deg", None)
    if asset_offset is not None or asset_orient_deg is not None:
        ref_xformable = UsdGeom.Xformable(ref_prim)
        ref_xformable.ClearXformOpOrder()
        if asset_offset is not None:
            ox, oy, oz = (float(asset_offset[0]), float(asset_offset[1]), float(asset_offset[2]))
            if cfg.scale is not None:
                sx, sy, sz = (float(cfg.scale[0]), float(cfg.scale[1]), float(cfg.scale[2]))
                if sx != 0.0:
                    ox /= sx
                if sy != 0.0:
                    oy /= sy
                if sz != 0.0:
                    oz /= sz
            ref_xformable.AddTranslateOp().Set(Gf.Vec3d(ox, oy, oz))
        if asset_orient_deg is not None:
            rx, ry, rz = (
                float(asset_orient_deg[0]),
                float(asset_orient_deg[1]),
                float(asset_orient_deg[2]),
            )
            cos_x, sin_x = math.cos(math.radians(rx) / 2.0), math.sin(math.radians(rx) / 2.0)
            cos_y, sin_y = math.cos(math.radians(ry) / 2.0), math.sin(math.radians(ry) / 2.0)
            cos_z, sin_z = math.cos(math.radians(rz) / 2.0), math.sin(math.radians(rz) / 2.0)
            # q_total = qz ⊗ qy ⊗ qx (apply X then Y then Z).
            qw = cos_z * cos_y * cos_x + sin_z * sin_y * sin_x
            qx = cos_z * cos_y * sin_x - sin_z * sin_y * cos_x
            qy = cos_z * sin_y * cos_x + sin_z * cos_y * sin_x
            qz = sin_z * cos_y * cos_x - cos_z * sin_y * sin_x
            ref_xformable.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(
                Gf.Quatf(float(qw), Gf.Vec3f(float(qx), float(qy), float(qz)))
            )

    xformable = UsdGeom.Xformable(root_prim)
    xformable.ClearXformOpOrder()
    if translation is not None:
        tx, ty, tz = (float(translation[0]), float(translation[1]), float(translation[2]))
        xformable.AddTranslateOp().Set(Gf.Vec3d(tx, ty, tz))
    if orientation is not None:
        w, x, y, z = (float(orientation[0]), float(orientation[1]), float(orientation[2]), float(orientation[3]))
        xformable.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    if cfg.scale is not None:
        sx, sy, sz = (float(cfg.scale[0]), float(cfg.scale[1]), float(cfg.scale[2]))
        xformable.AddScaleOp().Set(Gf.Vec3d(sx, sy, sz))

    if cfg.rigid_props is not None:
        if root_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            root_prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        if ref_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            schemas_utils.modify_rigid_body_properties(ref_prim_path, cfg.rigid_props)
        else:
            schemas_utils.define_rigid_body_properties(ref_prim_path, cfg.rigid_props)
    return root_prim
