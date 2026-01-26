# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Move the sim robot randomly and publish its joint states to ROS 2."""

import argparse
import os

from isaaclab.app import AppLauncher

# CLI
parser = argparse.ArgumentParser(description="Publish sim joint states (randomly driven) to ROS 2.")
parser.add_argument("--task", type=str, default="Scanbot-Piper-Scene-v0", help="Gym task id to load.")
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
    help="Stddev for random actions sent to the env (controls motion amplitude).",
)
parser.add_argument(
    "--topic",
    type=str,
    default="/joint_states",
    help="Topic to publish JointState messages on.",
)
parser.add_argument(
    "--enable_topic",
    type=str,
    default="/enable_flag",
    help="Topic to publish Bool(True) once to enable motion on the real robot.",
)
parser.add_argument(
    "--frame_id",
    type=str,
    default="piper_single",
    help="JointState header.frame_id to publish.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Sensible ROS defaults; override via environment if needed.
os.environ.setdefault("ROS_DOMAIN_ID", "0")
os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
os.environ.setdefault("ROS_LOCALHOST_ONLY", "0")

# Launch Omniverse app.
app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

# Enable ROS 2 bridge (best-effort; some builds already load it).
try:
    from isaacsim.core.utils.extensions import enable_extension
except ModuleNotFoundError:
    enable_extension = None
if enable_extension is not None:
    enable_extension("omni.isaac.ros2_bridge")

# Task registration to ensure custom Gym ID is present.

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool


def main() -> None:
    """Run sim with random actions and publish joint states each step."""
    rclpy.init()
    node = rclpy.create_node("sim_to_real_joint_state_bridge")
    qos = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
    )
    pub = node.create_publisher(JointState, args_cli.topic, qos)
    enable_pub = node.create_publisher(Bool, args_cli.enable_topic, qos)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
    env_cfg.terminations.time_out = None

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    robot = env.scene["robot"]
    joint_names = list(robot.joint_names)
    scale = float(args_cli.action_scale)

    step_count = 0
    try:
        # Send enable flag once at startup.
        enable_pub.publish(Bool(data=True))

        while simulation_app.is_running():
            # Generate random action per env and step the sim.
            with torch.inference_mode():
                actions = torch.randn((env.num_envs, env.action_manager.total_action_dim), device=env.device) * scale
                env.step(actions)

            # Publish joint states from env 0.
            jp = robot.data.joint_pos[0].detach().cpu().numpy()
            try:
                jv = robot.data.joint_vel[0].detach().cpu().numpy()
            except Exception:
                jv = None
            msg = JointState()
            msg.header.stamp = node.get_clock().now().to_msg()
            msg.header.frame_id = args_cli.frame_id
            msg.name = joint_names
            msg.position = jp.tolist()
            if jv is not None and len(jv) == len(joint_names):
                msg.velocity = jv.tolist()
            msg.effort = [0.0] * len(joint_names)
            pub.publish(msg)

            rclpy.spin_once(node, timeout_sec=0.0)

            step_count += 1
            if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
                break
    finally:
        env.close()
        node.destroy_node()
        rclpy.shutdown()
        simulation_app.close()


if __name__ == "__main__":
    main()
