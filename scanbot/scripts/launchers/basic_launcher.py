# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load the scanning scene objects and run an empty simulation loop."""

import argparse
import time

from isaaclab.app import AppLauncher

# Basic CLI to load the existing scene and keep the loop empty.
parser = argparse.ArgumentParser(
    description="Load the scanning scene and run an empty simulation loop."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--task",
    type=str,
    default="Scanbot-Piper-Scene-v0",
    help="Gym task id to load (keeps the original objects).",
)
parser.add_argument(
    "--ext-folder",
    type=str,
    default="scanbot/scripts/extensions/",
    help="Dummy flag",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Omniverse app before importing heavy modules.
app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from scanbot.scripts import scanbot_task  # noqa: F401 - registers Scanbot-Piper-Scene-v0
from isaaclab_tasks.utils import parse_env_cfg
from isaacsim.core.utils.extensions import enable_extension
from scanbot.scripts import scanbot_context


def main() -> None:
    """Create the environment with the existing assets and drive it using queued actions."""
    enable_extension("scanbot.extension_manager")
    scanbot_context.set_app_launcher(app_launcher)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
    env_cfg.terminations.time_out = None

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()
    scanbot_context.set_env(env)

    zero_action = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)

    try:
        while simulation_app.is_running():
            action = scanbot_context.pop_action()
            if action is not None:
                # print('env.step(', action, ')')
                env.step(action)
            else:
                env.step(zero_action)

    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
