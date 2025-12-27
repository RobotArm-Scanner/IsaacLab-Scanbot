# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ROS 2 Bridge (OmniGraph) teleop: subscribe to JointState and drive Piper joints."""

import argparse
import os

from isaaclab.app import AppLauncher

# Basic CLI for teleoperation via ROS 2 Bridge graph.
parser = argparse.ArgumentParser(description="Use ROS 2 Bridge graph to mirror JointState onto Piper joints.")
parser.add_argument(
    "--task",
    type=str,
    default="Scanbot-Piper-Scene-v0",
    help="Gym task id to load (keeps the original objects).",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments to spawn.")
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="Optional step cap; 0 runs until you close the window.",
)
parser.add_argument(
    "--topic",
    type=str,
    default="/joint_states_single",
    help="JointState topic to mirror via ROS 2 Bridge.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Set sane ROS defaults if user didn't provide them.
os.environ.setdefault("ROS_DOMAIN_ID", "0")
os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
os.environ.setdefault("ROS_LOCALHOST_ONLY", "0")

# Launch Omniverse app before importing heavy modules.
app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

# Enable ROS 2 bridge extension.
from isaacsim.core.utils.extensions import enable_extension

enable_extension("omni.isaac.ros2_bridge")

# Task registration.

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import omni.graph.core as og


def build_ros_graph(robot_prim_path: str, topic: str) -> None:
    """Create an OmniGraph that subscribes to JointState and drives the articulation controller."""
    keys = og.Controller.Keys
    graph_path = "/World/ROS2PiperTeleopGraph"
    # Build graph: Tick -> SubscribeJointState -> ArticulationController.
    (graph, nodes, _, _) = og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("SubscribeJointState", "omni.isaac.ros2_bridge.ROS2SubscribeJointState"),
                ("ArticulationController", "omni.isaac.core_nodes.IsaacArticulationController"),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
                ("SubscribeJointState.outputs:execOut", "ArticulationController.inputs:execIn"),
                ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
                ("SubscribeJointState.outputs:position", "ArticulationController.inputs:positionCommand"),
            ],
        },
    )

    sub = nodes[1]
    ctrl = nodes[2]

    # Configure subscriber.
    og.Controller.attribute("inputs:topicName", sub).set(topic)
    og.Controller.attribute("inputs:queueSize", sub).set(10)

    # Configure articulation controller.
    og.Controller.attribute("inputs:targetPrim", ctrl).set(robot_prim_path)
    og.Controller.attribute("inputs:usePathJointNames", ctrl).set(True)
    og.Controller.attribute("inputs:enableJointPositionController", ctrl).set(True)

    print(f"[ROS2Graph] Built teleop graph at '{graph_path}'")
    print(f"[ROS2Graph] Target prim: {robot_prim_path}")
    print(f"[ROS2Graph] Subscribing to: {topic}")


def main() -> None:
    """Create env, build ROS2 graph, and spin."""
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
    env_cfg.terminations.time_out = None

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    robot = env.scene["robot"]
    robot_prim_path = robot.prim_path

    build_ros_graph(robot_prim_path=robot_prim_path, topic=args_cli.topic)

    step_count = 0
    try:
        while simulation_app.is_running():
            # Drive simulation; graph runs on playback ticks.
            env.sim.step(render=env.sim.has_gui() or env.sim.has_rtx_sensors())
            env.scene.update(dt=env.physics_dt)

            step_count += 1
            if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
                break
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
