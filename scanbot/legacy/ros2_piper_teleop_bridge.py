# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mirror live ROS 2 JointState messages onto the Piper in Isaac Sim."""

import argparse
import os

from isaaclab.app import AppLauncher

# Basic CLI for teleoperation.
parser = argparse.ArgumentParser(description="Subscribe to ROS 2 joint states and drive the Piper in sim.")
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
    "--sim_dt",
    type=float,
    default=None,
    help="Optional physics dt override (e.g., 0.004 for ~250 Hz).",
)
parser.add_argument(
    "--decimation",
    type=int,
    default=None,
    help="Optional decimation override; 1 means one physics step per loop.",
)
parser.add_argument(
    "--render_interval",
    type=int,
    default=None,
    help="Optional render interval override; higher reduces render load.",
)
parser.add_argument(
    "--topic",
    type=str,
    default="/joint_states_single",
    help="ROS 2 JointState topic to mirror.",
)
parser.add_argument(
    "--fallback_topic",
    type=str,
    default="/joint_states_single",
    help="Secondary topic to try if no messages arrive on --topic.",
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

# Enable ROS 2 bridge so rclpy is available in the kit Python.
# try:
#     from isaacsim.core.utils.extensions import enable_extension
# except ModuleNotFoundError:
#     # Fallback if isaacsim namespace is not exposed; AppLauncher already loads extensions in many setups.
#     enable_extension = None
# if enable_extension is not None:
#     enable_extension("omni.isaac.ros2_bridge")

# Task registration.

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState


def _expand_targets(target: torch.Tensor, num_envs: int) -> torch.Tensor:
    """Return a (num_envs, num_joints) target tensor for convenience."""
    if target.dim() == 1:
        return target.unsqueeze(0).repeat(num_envs, 1)
    return target


def main() -> None:
    """Create the environment and mirror ROS 2 joint states into the sim."""
    rclpy.init()
    ros_node = rclpy.create_node("piper_joint_state_teleop")
    cb_group = ReentrantCallbackGroup()

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
    env_cfg.terminations.time_out = None
    if args_cli.sim_dt is not None:
        env_cfg.sim.dt = float(args_cli.sim_dt)
    if args_cli.decimation is not None:
        env_cfg.decimation = int(args_cli.decimation)
    if args_cli.render_interval is not None:
        env_cfg.sim.render_interval = int(args_cli.render_interval)

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    robot = env.scene["robot"]
    joint_names = list(robot.joint_names)
    name_to_index = {name: idx for idx, name in enumerate(joint_names)}

    latest_target = robot.data.joint_pos[0].clone()
    unknown_warned: set[str] = set()
    active_topic: str | None = None

    def on_joint_state(msg: JointState, *, source_topic: str) -> None:
        nonlocal latest_target, active_topic
        if not msg.name or not msg.position:
            return

        if active_topic is None:
            active_topic = source_topic
            ros_node.get_logger().info(f"Receiving JointState from '{active_topic}'")
        elif active_topic != source_topic:
            # Ignore other topics once we have a winner.
            return

        updated = latest_target.clone()
        for name, pos in zip(msg.name, msg.position):
            idx = name_to_index.get(name)
            if idx is None:
                if name not in unknown_warned:
                    ros_node.get_logger().warn(
                        f"Ignoring joint '{name}' not present in sim. Known joints: {joint_names}"
                    )
                    unknown_warned.add(name)
                continue
            updated[idx] = float(pos)

        latest_target = updated

    # Subscribe to primary and fallback topics; use the first one that delivers a message.
    ros_node.create_subscription(
        JointState,
        args_cli.topic,
        lambda msg, t=args_cli.topic: on_joint_state(msg, source_topic=t),
        qos_profile_sensor_data,
        callback_group=cb_group,
    )
    if args_cli.fallback_topic and args_cli.fallback_topic != args_cli.topic:
        ros_node.create_subscription(
            JointState,
            args_cli.fallback_topic,
            lambda msg, t=args_cli.fallback_topic: on_joint_state(msg, source_topic=t),
            qos_profile_sensor_data,
            callback_group=cb_group,
        )

    if args_cli.fallback_topic and args_cli.fallback_topic != args_cli.topic:
        fallback_str = args_cli.fallback_topic
    else:
        fallback_str = "disabled"

    ros_node.get_logger().info(
        f"Mirroring JointState messages; primary='{args_cli.topic}', fallback='{fallback_str}'. "
        f"Sim joints: {', '.join(joint_names)}"
    )

    batched_target = _expand_targets(latest_target, env.num_envs)
    step_count = 0

    try:
        while simulation_app.is_running():
            rclpy.spin_once(ros_node, timeout_sec=0.0)

            with torch.inference_mode():
                batched_target = _expand_targets(latest_target, env.num_envs)
                robot.set_joint_position_target(batched_target)

                # Push commands, step physics, and refresh scene buffers.
                env.scene.write_data_to_sim()
                env.sim.step(render=env.sim.has_gui() or env.sim.has_rtx_sensors())
                env.scene.update(dt=env.physics_dt)

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
