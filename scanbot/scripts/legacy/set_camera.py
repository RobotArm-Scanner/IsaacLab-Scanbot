# ================================================================
#  Isaac Sim 4.5 — 새 카메라 생성 + 뷰포트 연결 + 위치/각도 설정 + 사용자 조작 unlock
# ================================================================

from pxr import UsdGeom, Gf
import omni.usd
from omni.kit.viewport.utility import get_active_viewport

# ------------------------------------------------
# 1) 스테이지 가져오기
# ------------------------------------------------
stage = omni.usd.get_context().get_stage()

# ------------------------------------------------
# 2) 새 카메라 생성
# ------------------------------------------------
cam_path = "/World/UserCamera"

# 이미 존재한다면 삭제
old_prim = stage.GetPrimAtPath(cam_path)
if old_prim.IsValid():
    stage.RemovePrim(cam_path)

camera = UsdGeom.Camera.Define(stage, cam_path)
prim = stage.GetPrimAtPath(cam_path)

# ------------------------------------------------
# 3) Transform 설정 (원하는 초기 위치/각도)
# ------------------------------------------------
xform = UsdGeom.Xformable(prim)

t_op = xform.AddTranslateOp()
r_op = xform.AddRotateXYZOp()

# 원하는 위치 (필요시 수정)
initial_pos = Gf.Vec3f(2.0, 2.0, 2.0)
initial_rot = Gf.Vec3f(-20.0, 45.0, 0.0)

t_op.Set(initial_pos)
r_op.Set(initial_rot)

# ------------------------------------------------
# 4) 새 카메라를 뷰포트에 활성 카메라로 지정
# ------------------------------------------------
viewport = get_active_viewport()
viewport.set_active_camera(cam_path)

print("[INFO] UserCamera activated in viewport:", cam_path)
print("[INFO] Initial position set to", initial_pos)
print("[INFO] Initial rotation set to", initial_rot)

# ------------------------------------------------
# 5) 카메라 업데이트 함수 (원하는 위치/각도로 변경 가능)
# ------------------------------------------------
def move_camera(pos=None, rot=None):
    ops = UsdGeom.Xformable(stage.GetPrimAtPath(cam_path)).GetOrderedXformOps()

    if pos is not None:
        ops[0].Set(Gf.Vec3f(*pos))
        print("[INFO] Camera position updated to:", pos)

    if rot is not None:
        ops[1].Set(Gf.Vec3f(*rot))
        print("[INFO] Camera rotation updated to:", rot)


# ------------------------------------------------
# 6) 카메라 unlock — 사용자가 뷰포트에서 자유롭게 회전/이동 가능
# ------------------------------------------------
# Note: USD 카메라가 뷰포트 active camera가 되면 자동으로 조작 가능.
print("[INFO] Camera is user-interactive now (mouse navigation enabled).")

# ------------------------------------------------
# 7) 테스트로 한 번 더 이동해보기 (원하면 삭제)
# ------------------------------------------------
move_camera(pos=(5, 5, 5), rot=(-10, 30, 0))
