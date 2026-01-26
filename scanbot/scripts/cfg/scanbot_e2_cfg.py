# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Piper scanning env config (no gripper) copied locally for Scanbot use."""

import math
import os

from isaaclab.assets import AssetBaseCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.openxr.openxr_device import OpenXRDevice, OpenXRDeviceCfg
from isaaclab.devices.openxr.retargeters.manipulator.gripper_retargeter import GripperRetargeterCfg
from isaaclab.devices.openxr.retargeters.manipulator.se3_rel_retargeter import Se3RelRetargeterCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab.sensors import CameraCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import schemas as schemas_utils
from isaaclab.sim import spawners as sim_utils
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.stack import mdp
from scanbot.scripts.utilities.pos_util import quat_wxyz_from_deg_xyz
from scanbot.scripts.robots import piper_scanning_events
from scanbot.scripts.cfg.basic_env_cfg import BasicEnvCfg
from scanbot.scripts.robots.piper_no_gripper import PIPER_NO_GRIPPER_CFG


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


@configclass
class EventCfg:
    """Configuration for events."""

    init_piper_arm_pose = EventTerm(
        func=piper_scanning_events.set_default_joint_pose,
        mode="reset",
        params={
            # "default_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            # 카메라 각도 때문에 마지막 관절 값 0 유지
            "default_pose": [0.14937140047550201,
                             1.8190464973449707,
                             -0.6580374240875244,
                             0.1716867834329605,
                             -1.0519624948501587,
                             -0.08057491481304169],
        },
    )

    randomize_piper_joint_state = EventTerm(
        func=piper_scanning_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class ScanbotEnv2Cfg(BasicEnvCfg):
    """Local copy of the Piper scanning env cfg (no gripper)."""

    def __post_init__(self):
        super().__post_init__()
        self.env_name = "Scanbot-Piper-Env2"

        # Set events
        self.events = EventCfg()
        print(f"DEBUG: The Base sim.dt for {self.__class__.__name__} is: {self.sim.dt}")

        # Set Franka as robot
        # default
        # self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # for playback, sync with joint recorded by task of stack_ik_rel_env_cfg
        self.scene.robot = PIPER_NO_GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        # Add semantics to table
        self.scene.table.spawn.semantic_tags = [("class", "table")]

        # Add semantics to ground
        self.scene.plane.semantic_tags = [("class", "ground")]

        # for playback, same scale and absolute
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint[1-6]"],
            scale=1.0,
            use_default_offset=False
        )

        # Rigid body properties of each cube
        cube_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
            kinematic_enabled=True
        )

        support_properties = RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True)

        resources_root = os.path.join(
            os.environ.get("ISAACLAB_PATH", "/workspace/isaaclab"),
            "scanbot",
            "resources",
        )
        my_usd_path = os.path.join(resources_root, "teeth_models", "model1", "Mouth_open_wo_tooth_root.usd")

        print(f"[DEBUG] usd_path (final str): {my_usd_path}")

        self.scene.teeth = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Teeth",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.40 + 0.15, 0, 0.15],
                rot=tuple(quat_wxyz_from_deg_xyz((0.0, 0.0, -90.0))),
            ),
            spawn=UsdFileCfg(
                usd_path=my_usd_path,
                scale=(0.08, 0.08, 0.08),
                rigid_props=cube_properties,
                semantic_tags=[("class", "teeth")],
            ),
        )

        self.scene.teeth_support = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/TeethSupport",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.40 + 0.15, 0.0, 0.0],
                rot=tuple(quat_wxyz_from_deg_xyz((0.0, 0.0, 0.0))),
            ),
            spawn=sim_utils.CuboidCfg(
                size=(0.2, 0.2, 0.2),
                rigid_props=support_properties,
                collision_props=CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.02, 0.02, 0.02)),
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/link6",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )

        # Prefer TAA; no DLSS requirement.
        self.sim.render.antialiasing_mode = "Off"

        # Use Piper no-gripper asset under {ENV_REGEX_NS}/Robot
        self.scene.robot = PIPER_NO_GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Visual tool on link6 (non-rigid)
        tool_usd_path = os.path.join(resources_root, "tools", "case_with_scanner_colored.usd")
        self.scene.tool = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Robot/link6/tool",
            spawn=UsdFileCfg(
                usd_path=tool_usd_path,
                scale=(0.001, 0.001, 0.001),
                semantic_tags=[("class", "tool")],
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.003),
                rot=tuple(quat_wxyz_from_deg_xyz((180.0, 0.0, 90.0))),
            ),
        )

        self.scene.wrist_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/link6/tool/wrist_camera",
            update_period=0.1,
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=30.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.001, 0.3),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(-1.71854, -7.51282, -260.88225),
                rot=tuple(quat_wxyz_from_deg_xyz((90.0, 0.0, 0.0))),
            ),
        )

        # Free wrist camera used for dataset capture (disabled; keep only wrist_camera)
        # self.scene.free_wrist_camera = CameraCfg(
        #     prim_path="{ENV_REGEX_NS}/free_wrist_camera",
        #     update_period=0.1,
        #     height=480,
        #     width=640,
        #     data_types=["rgb"],
        #     spawn=sim_utils.PinholeCameraCfg(
        #         focal_length=30.0,
        #         focus_distance=400.0,
        #         horizontal_aperture=20.955,
        #         clipping_range=(0.001, 0.3),
        #     ),
        #     update_latest_camera_pose=True,
        # )

        self.scene.global_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/global_camera",
            update_period=0.1,
            height=720,
            width=1280,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.14756,
                focus_distance=400.0,
                f_stop=0.0,
                horizontal_aperture=20.955,
                vertical_aperture=15.2908,
                clipping_range=(0.01, 10.0),
                lock_camera=True,
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.35 + 0.15, 0.0, 0.45),
                rot=tuple(quat_wxyz_from_deg_xyz((0.0, 0.0, 90.0))),
                convention="opengl",
            ),
        )

        # Robot camera (disabled; keep only wrist_camera)
        # self.scene.robot_camera = CameraCfg(
        #     prim_path="{ENV_REGEX_NS}/robot_camera",
        #     update_period=0.0,
        #     height=720,
        #     width=1280,
        #     data_types=["rgb"],
        #     spawn=sim_utils.PinholeCameraCfg(
        #         focal_length=18.14756,
        #         focus_distance=400.0,
        #         f_stop=0.0,
        #         horizontal_aperture=20.955,
        #         vertical_aperture=15.2908,
        #         clipping_range=(0.01, 1_000_000.0),
        #         lock_camera=True,
        #     ),
        #     offset=CameraCfg.OffsetCfg(
        #         pos=(0.22656 + 0.15, 0.49178, 0.67233),
        #         rot=tuple(quat_wxyz_from_deg_xyz((50.0, 0.0, -180.0))),
        #         convention="opengl",
        #     ),
        # )

        # Mouth camera (disabled; keep only wrist_camera)
        # self.scene.mouth_camera = CameraCfg(
        #     prim_path="{ENV_REGEX_NS}/mouth_camera",
        #     update_period=0.0,
        #     height=720,
        #     width=1280,
        #     data_types=["rgb"],
        #     spawn=sim_utils.PinholeCameraCfg(
        #         focal_length=18.14756,
        #         focus_distance=400.0,
        #         f_stop=0.0,
        #         horizontal_aperture=20.955,
        #         vertical_aperture=15.2908,
        #         clipping_range=(0.01, 1_000_000.0),
        #         lock_camera=True,
        #     ),
        #     offset=CameraCfg.OffsetCfg(
        #         pos=(0.28186 + 0.15, 0.0, 0.2763),
        #         rot=tuple(quat_wxyz_from_deg_xyz((50.0, 0.0, -90.0))),
        #         convention="opengl",
        #     ),
        # )

        # Mouth left camera (disabled; keep only wrist_camera)
        # self.scene.mouth_left_camera = CameraCfg(
        #     prim_path="{ENV_REGEX_NS}/mouth_left_camera",
        #     update_period=0.0,
        #     height=720,
        #     width=1280,
        #     data_types=["rgb"],
        #     spawn=sim_utils.PinholeCameraCfg(
        #         focal_length=18.14756,
        #         focus_distance=400.0,
        #         f_stop=0.0,
        #         horizontal_aperture=20.955,
        #         vertical_aperture=15.2908,
        #         clipping_range=(0.01, 1_000_000.0),
        #         lock_camera=True,
        #     ),
        #     offset=CameraCfg.OffsetCfg(
        #         pos=(0.4, 0.15, 0.1763),
        #         rot=tuple(quat_wxyz_from_deg_xyz((90.0, 0.0, -135.0))),
        #         convention="opengl",
        #     ),
        # )

        # Mouth right camera (disabled; keep only wrist_camera)
        # self.scene.mouth_right_camera = CameraCfg(
        #     prim_path="{ENV_REGEX_NS}/mouth_right_camera",
        #     update_period=0.0,
        #     height=720,
        #     width=1280,
        #     data_types=["rgb"],
        #     spawn=sim_utils.PinholeCameraCfg(
        #         focal_length=18.14756,
        #         focus_distance=400.0,
        #         f_stop=0.0,
        #         horizontal_aperture=20.955,
        #         vertical_aperture=15.2908,
        #         clipping_range=(0.01, 1_000_000.0),
        #         lock_camera=True,
        #     ),
        #     offset=CameraCfg.OffsetCfg(
        #         pos=(0.4, -0.15, 0.1763),
        #         rot=tuple(quat_wxyz_from_deg_xyz((90.0, 0.0, -45.0))),
        #         convention="opengl",
        #     ),
        # )

        # Surface cameras in policy observations if needed downstream
        self.image_obs_list = [
            "wrist_camera",
            "global_camera",
        ]

        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["joint[1-6]"],
            body_name="link6",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )

        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        Se3RelRetargeterCfg(
                            bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT,
                            zero_out_xy_rotation=True,
                            use_wrist_rotation=False,
                            use_wrist_position=True,
                            delta_pos_scale_factor=10.0,
                            delta_rot_scale_factor=10.0,
                            sim_device=self.sim.device,
                        ),
                        GripperRetargeterCfg(
                            bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT,
                            sim_device=self.sim.device,
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
            }
        )


@configclass
class ScanbotEnv2M1RT1Cfg(ScanbotEnv2Cfg):
    """Scanbot env with model1 teeth rotated upward by 30 degrees."""

    def __post_init__(self):
        super().__post_init__()
        self.env_name = "Scanbot-Piper-Env2-M1RT1"
        self.scene.teeth.init_state.rot = tuple(quat_wxyz_from_deg_xyz((0.0, 30.0, -90.0)))


@configclass
class ScanbotEnv2M1RT2Cfg(ScanbotEnv2Cfg):
    """Scanbot env with model1 (more open) teeth rotated downward by 20 degrees."""

    def __post_init__(self):
        super().__post_init__()
        self.env_name = "Scanbot-Piper-Env2-M1RT2"
        resources_root = os.path.join(
            os.environ.get("ISAACLAB_PATH", "/workspace/isaaclab"),
            "scanbot",
            "resources",
        )
        self.scene.teeth.spawn.usd_path = os.path.join(
            resources_root,
            "teeth_models",
            "model1",
            "Mouth_more_open_wo_teeth_root.usd",
        )
        self.scene.teeth.init_state.rot = tuple(quat_wxyz_from_deg_xyz((-20.0, 0.0, -90.0)))


@configclass
class ScanbotEnv2M2RT1Cfg(ScanbotEnv2Cfg):
    """Scanbot env with model2 teeth rotated and offset to match 9000.usd."""

    def __post_init__(self):
        super().__post_init__()
        self.env_name = "Scanbot-Piper-Env2-M2RT1"
        resources_root = os.path.join(
            os.environ.get("ISAACLAB_PATH", "/workspace/isaaclab"),
            "scanbot",
            "resources",
        )
        self.scene.teeth.spawn.usd_path = os.path.join(
            resources_root,
            "teeth_models",
            "model2",
            "9000.usd",
        )
        self.scene.teeth.spawn.func = _spawn_rigid_object_from_usd
        self.scene.teeth.spawn.scale = (0.11, 0.11, 0.11)
        self.scene.teeth.spawn.asset_offset = (0.09785, 0.328884, 0.456378)
        self.scene.teeth.spawn.asset_orient_deg = (24.946, 0.613, 4.665)
        self.scene.teeth.init_state.pos = (0.0, -0.02, 0.01)
        self.scene.teeth.init_state.rot = tuple(quat_wxyz_from_deg_xyz((-43.9595435, 0.0, -90.0)))
