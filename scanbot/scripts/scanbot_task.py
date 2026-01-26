import gymnasium as gym

from scanbot.scripts.cfg.scanbot_e2_cfg import (
    ScanbotE2Cfg,
    ScanbotE2T1RT1Cfg,
    ScanbotE2T2RT1Cfg,
)


# Register a cleaner Gym id so we don't rely on the franka.* namespace.
gym.register(
    id="e2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ScanbotE2Cfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="e2.t1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ScanbotE2Cfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="e2.t1.rt1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ScanbotE2T1RT1Cfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="e2.t2.rt1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ScanbotE2T2RT1Cfg,
    },
    disable_env_checker=True,
)
