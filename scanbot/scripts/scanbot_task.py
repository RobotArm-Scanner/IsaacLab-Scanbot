import gymnasium as gym

from scanbot.scripts.cfg.scanbot_e2_cfg import ScanbotEnv2Cfg, ScanbotEnv2M1RT1Cfg


# Register a cleaner Gym id so we don't rely on the franka.* namespace.
gym.register(
    id="e2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ScanbotEnv2Cfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="e2m1rt1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ScanbotEnv2M1RT1Cfg,
    },
    disable_env_checker=True,
)
