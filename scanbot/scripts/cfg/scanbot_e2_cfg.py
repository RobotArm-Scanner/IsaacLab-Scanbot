# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Piper scanning env config (no gripper) copied locally for Scanbot use."""

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
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab.sensors import CameraCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import spawners as sim_utils
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.stack import mdp
from scanbot.scripts import scanbot_mdp
from scanbot.scripts.utilities.pos_util import quat_wxyz_from_deg_xyz
from scanbot.scripts.utilities.teeth3ds_util import ensure_t3ds_usd
from scanbot.scripts.utilities.usd_util import spawn_usd_with_mesh_collision, spawn_rigid_object_from_usd
from scanbot.scripts.robots import piper_scanning_events
from scanbot.scripts.cfg.basic_env_cfg import BasicEnvCfg, ObservationsCfg
from scanbot.scripts.robots.piper_no_gripper import PIPER_NO_GRIPPER_CFG


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
class RewardsCfg:
    """Reward terms for Scanbot RL."""

    ee_delta = RewTerm(func=scanbot_mdp.ee_delta_l2, weight=-0.05, params={"ee_frame_cfg": SceneEntityCfg("ee_frame")})
    coverage_delta = RewTerm(func=scanbot_mdp.coverage_delta_reward, weight=5.0, params={})
    per_tooth_bonus = RewTerm(func=scanbot_mdp.per_tooth_coverage_bonus, weight=2.0, params={})
    total_bonus = RewTerm(func=scanbot_mdp.total_coverage_bonus, weight=10.0, params={})


@configclass
class ScanbotRLObservationsCfg(ObservationsCfg):
    """Observation config tuned for RL (flat 1D)."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        def __post_init__(self):
            super().__post_init__()
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class ScanbotE2Cfg(BasicEnvCfg):
    """Local copy of the Piper scanning env cfg (no gripper)."""
    """E2 DEFAULT TEETH = T1"""

    def __post_init__(self):
        super().__post_init__()
        self.env_name = "Scanbot-e2"

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

        self.resources_root = os.path.join(
            os.environ.get("ISAACLAB_PATH", "/workspace/isaaclab"),
            "scanbot",
            "resources",
        )
        my_usd_path = os.path.join(self.resources_root, "teeth", "t1", "Mouth_open_wo_tooth_root.usd")

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
        tool_usd_path = os.path.join(self.resources_root, "tools", "case_with_scanner_colored.usd")
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
                convention="ros",
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
class ScanbotE2T1RT1Cfg(ScanbotE2Cfg):
    """Scanbot env with t1 (more open) teeth rotated by 20 degrees around Y."""

    def __post_init__(self):
        super().__post_init__()
        self.env_name = "Scanbot-Piper-e2.t1.rt1"
        self.events.init_piper_arm_pose.params["default_pose"] = [
            -0.155398,
            1.666897,
            -0.860017,
            -0.735697,
            -0.549470,
            0.633906,
        ]
        self.scene.teeth.spawn.usd_path = os.path.join(
            self.resources_root,
            "teeth",
            "t1",
            "Mouth_more_open_wo_teeth_root.usd",
        )
        self.scene.teeth.init_state.rot = tuple(quat_wxyz_from_deg_xyz((0.0, 20.0, -90.0)))


@configclass
class ScanbotE2T2RT1Cfg(ScanbotE2Cfg):
    """Scanbot env with t2 teeth rotated and offset to match 9000.usd."""

    def __post_init__(self):
        super().__post_init__()
        self.env_name = "Scanbot-Piper-e2.t2.rt1"
        self.events.init_piper_arm_pose.params["default_pose"] = [
            -0.128291,
            1.671921,
            -0.930349,
            -0.789148,
            -0.480286,
            0.713796,
        ]
        self.scene.teeth.spawn.usd_path = os.path.join(
            self.resources_root,
            "teeth",
            "t2",
            "9000.usd",
        )
        self.scene.teeth.spawn.func = spawn_rigid_object_from_usd
        self.scene.teeth.spawn.scale = (0.11, 0.11, 0.11)
        self.scene.teeth.spawn.asset_offset = (0.09785, 0.328884, 0.456378)
        self.scene.teeth.spawn.asset_orient_deg = (24.946, 0.613, 4.665)
        self.scene.teeth.init_state.pos = (0.0, -0.02, 0.01)
        self.scene.teeth.init_state.rot = tuple(quat_wxyz_from_deg_xyz((0.0, 43.9595435, -90.0)))


@configclass
class ScanbotE2T3DSCfg(ScanbotE2Cfg):
    """Scanbot env with t3ds teeth (segmented OBJ -> USD)."""

    def __post_init__(self):
        super().__post_init__()
        self.env_name = "Scanbot-Piper-e2.t3ds"
        self.teeth_dataset_id = "A9TECAGP"
        self.scene.teeth.spawn.usd_path = ensure_t3ds_usd(self.resources_root, dataset_id=self.teeth_dataset_id)
        self.scene.teeth.spawn.func = spawn_usd_with_mesh_collision
        self.scene.teeth.spawn.scale = (0.0015, 0.0015, 0.0015)
        # Center the raw mesh bounds around the asset origin (world units, before scale is applied).
        # self.scene.teeth.spawn.asset_offset = (0.58646, -0.00021, -0.49428)
        self.scene.teeth.init_state.pos = (0.565, -0.00021, 0.28)
        self.scene.teeth.init_state.rot = tuple(quat_wxyz_from_deg_xyz((0.0, 0.0, -90.0)))


@configclass
class ScanbotE2RLT3DSCfg(ScanbotE2T3DSCfg):
    """Scanbot RL env config for t3ds (vision + proprio)."""

    def __post_init__(self):
        super().__post_init__()
        self.env_name = "Scanbot-Piper-e2.t3ds.rl"

        # RL-friendly defaults
        self.scene.num_envs = 4
        self.sim.render_interval = 1
        self.episode_length_s = 20.0
        self.actions.arm_action.scale = 0.01
        # self.sim.render.rendering_mode = "quality"
        # self.sim.render.enable_translucency = True
        # self.sim.render.enable_reflections = True
        # self.sim.render.enable_global_illumination = True
        # self.sim.render.carb_settings = {
        #     "/rtx/rendermode": "PathTracing",
        #     "/rtx/raytracing/fractionalCutoutOpacity": True,
        #     "/rtx/material/translucencyAsOpacity": True,
        #     "/rtx/translucency/maxRefractionBounces": 6,
        # }

        # Observation: do not include action history
        self.observations = ScanbotRLObservationsCfg()
        self.observations.policy.actions = None

        # Disable XR callbacks for Hydra serialization
        self.xr.anchor_rotation_custom_func = None
        self.teleop_devices = None

        # Drop global camera for RL (avoid per-env extra sensors).
        self.scene.global_camera = None

        # Depth + RGB camera for coverage and visualization
        self.scene.wrist_camera.data_types = ["rgb", "distance_to_image_plane"]
        self.scene.wrist_camera.update_period = 0.1
        self.scene.wrist_camera.height = 128
        self.scene.wrist_camera.width = 128

        # Disable image observations for now (stability first)
        self.image_obs_list = []

        # Rewards
        self.rewards = RewardsCfg()

        # Terminations (failure conditions)
        self.terminations.scanpoint_far_from_support = DoneTerm(
            func=scanbot_mdp.scanpoint_far_from_support,
            params={
                "max_distance": 0.18,
                "camera_name": "wrist_camera",
                "support_name": "teeth_support",
                "debug_draw": True,
                "debug_draw_interval": 1,
                "tcp_plot": True,
                "tcp_plot_frame": "ee_frame",
                "tcp_plot_interval": 1,
                "tcp_plot_max_points": 200,
                "tcp_plot_pause": 0.001,
                "tcp_plot_env_ids": None,
            },
        )

        self.coverage_threshold_tooth = 0.8
        self.coverage_threshold_total = 0.8
        self.coverage_params = {
            "resources_root": self.resources_root,
            "dataset_id": self.teeth_dataset_id,
            "num_samples": 20000,
            "seed": 0,
            "gum_assign_radius": 0.002,
            "coverage_radius": 0.002,
            "scale": self.scene.teeth.spawn.scale,
            "pcd_voxel_size": 0.001,
            "pcd_max_points": 60000,
            "coverage_update_every": 2,
            "camera_name": "wrist_camera",
            "data_type": "distance_to_image_plane",
            "teeth_name": "teeth",
        }
        self.rewards.coverage_delta.params = dict(self.coverage_params)
        self.rewards.per_tooth_bonus.params = dict(
            self.coverage_params,
            threshold=self.coverage_threshold_tooth,
        )
        self.rewards.total_bonus.params = dict(
            self.coverage_params,
            threshold=self.coverage_threshold_total,
        )
