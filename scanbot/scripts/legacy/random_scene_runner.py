# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the Piper scene with simple random actions (no teleop)."""

import argparse

from isaaclab.app import AppLauncher

# Basic CLI.
parser = argparse.ArgumentParser(description="Run Piper scene with random actions.")
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
    help="Stddev for random actions (applied to all action dims).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Omniverse app before importing heavy modules.
app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

# Ensure our Scanbot task is registered before parse_env_cfg / gym.make.

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    """Create the environment and step with random actions."""
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
    env_cfg.terminations.time_out = None

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    action_dim = env.action_manager.total_action_dim
    scale = float(args_cli.action_scale)

    step_count = 0
    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                # Gaussian actions centered at zero.
                actions = torch.randn((env.num_envs, action_dim), device=env.device) * scale
                env.step(actions)

            step_count += 1
            if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
                break
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
