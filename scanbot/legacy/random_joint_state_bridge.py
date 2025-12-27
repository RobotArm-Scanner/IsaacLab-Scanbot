# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sim runs as a ROS 2 node and publishes random JointState each sim step."""

import argparse
import os

from isaaclab.app import AppLauncher

# Basic CLI.
parser = argparse.ArgumentParser(description="Publish random JointState from sim loop.")
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
    "--action_scale",
    type=float,
    default=0.05,
    help="Stddev for random joint positions (rad) to publish.",
)
parser.add_argument(
    "--topic",
    type=str,
    default="/joint_states",
    help="Topic to publish JointState messages on.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Set sane ROS defaults if user didn't provide them (works for most setups).
os.environ.setdefault("ROS_DOMAIN_ID", "0")
os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
os.environ.setdefault("ROS_LOCALHOST_ONLY", "0")

# Launch Omniverse app before importing heavy modules.
app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

# Enable ROS 2 bridge so rclpy is available in the kit Python.
try:
    from isaacsim.core.utils.extensions import enable_extension
except ModuleNotFoundError:
    # Fallback if isaacsim namespace is not exposed; AppLauncher already loads extensions in many setups.
    enable_extension = None
if enable_extension is not None:
    enable_extension("omni.isaac.ros2_bridge")

# Task registration

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState


def main() -> None:
    """Create env and publish random joint states each sim tick."""
    rclpy.init()
    ros_node = rclpy.create_node("sim_random_joint_state_publisher")
    publisher = ros_node.create_publisher(JointState, args_cli.topic, qos_profile_sensor_data)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
    env_cfg.terminations.time_out = None

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    robot = env.scene["robot"]
    joint_names = list(robot.joint_names)
    num_joints = len(joint_names)
    scale = float(args_cli.action_scale)

    step_count = 0
    try:
        while simulation_app.is_running():
            # Random joint positions and publish.
            rand_pos = torch.randn(num_joints, device=env.device) * scale

            msg = JointState()
            msg.header.stamp = ros_node.get_clock().now().to_msg()
            msg.name = joint_names
            msg.position = rand_pos.tolist()
            publisher.publish(msg)

            # Drive the env through the normal action path so the sim stays responsive.
            with torch.inference_mode():
                actions = torch.randn((env.num_envs, env.action_manager.total_action_dim), device=env.device) * scale
                env.step(actions)

            rclpy.spin_once(ros_node, timeout_sec=0.0)

            step_count += 1
            if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
                break
    finally:
        env.close()
        ros_node.destroy_node()
        rclpy.shutdown()
        simulation_app.close()


if __name__ == "__main__":
    main()
