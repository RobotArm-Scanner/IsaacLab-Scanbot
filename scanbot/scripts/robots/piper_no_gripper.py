import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

##
# Piper Configuration
##

PIPER_NO_GRIPPER_CFG = ArticulationCfg(
    # --- 1. 스폰(Spawn) 설정 ---
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(
            os.environ.get("ISAACLAB_PATH", "/workspace/isaaclab"),
            "scanbot",
            "resources",
            "piper_isaac_sim",
            "usd",
            "piper_no_gripper_description",
            "piper_no_gripper_description.usd",
        ),

        # 물리 속성은 Franka 예시와 유사하게 설정할 수 있습니다.
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    ),

    # --- 2. 초기 상태(Initial State) 설정 ---
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
        },
    ),

    actuators={
        "piper_arm": ImplicitActuatorCfg(
            joint_names_expr=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
            stiffness=2000.0,
            damping=300.0,
        ),

    },
)

"""Configuration for the Piper robot arm."""
