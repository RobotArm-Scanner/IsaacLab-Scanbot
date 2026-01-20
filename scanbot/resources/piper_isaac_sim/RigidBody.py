from pxr import Usd, UsdPhysics, UsdGeom
from omni.usd import get_context

stage = get_context().get_stage()
prim_path = "/World/envs/env_0/Tooth"
prim = stage.GetPrimAtPath(prim_path)
if not prim:
    raise RuntimeError("Prim not found")

# Apply rigid body API
UsdPhysics.RigidBodyAPI.Apply(prim)

# Optionally apply collider API (예: mesh collision)
UsdPhysics.CollisionAPI.Apply(prim)
