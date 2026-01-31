import gymnasium as gym

from scanbot.scripts.cfg.scanbot_e2_cfg import (
    ScanbotE2Cfg,
    ScanbotE2T1RT1Cfg,
    ScanbotE2T2RT1Cfg,
    ScanbotE2T3DSCfg,
    ScanbotE2RLT3DSCfg,
)


# Register a cleaner Gym id so we don't rely on the franka.* namespace.
gym.register(
    id="e2",
    entry_point="scanbot.scripts.scanbot_entrypoints:make_manager_env",
    kwargs={
        "env_cfg_entry_point": ScanbotE2Cfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="e2.t1",
    entry_point="scanbot.scripts.scanbot_entrypoints:make_manager_env",
    kwargs={
        "env_cfg_entry_point": ScanbotE2Cfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="e2.t1.rt1",
    entry_point="scanbot.scripts.scanbot_entrypoints:make_manager_env",
    kwargs={
        "env_cfg_entry_point": ScanbotE2T1RT1Cfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="e2.t2.rt1",
    entry_point="scanbot.scripts.scanbot_entrypoints:make_manager_env",
    kwargs={
        "env_cfg_entry_point": ScanbotE2T2RT1Cfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="e2.t3ds",
    entry_point="scanbot.scripts.scanbot_entrypoints:make_manager_env",
    kwargs={
        "env_cfg_entry_point": ScanbotE2T3DSCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="e2.t3ds.rl",
    entry_point="scanbot.scripts.scanbot_entrypoints:make_manager_env",
    kwargs={
        "env_cfg_entry_point": ScanbotE2RLT3DSCfg,
        "rsl_rl_cfg_entry_point": "scanbot.scripts.rl.rsl_rl_ppo_cfg:ScanbotE2T3DSRLPPORunnerCfg",
    },
    disable_env_checker=True,
)
