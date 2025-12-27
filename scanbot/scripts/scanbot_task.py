import gymnasium as gym

from scanbot.scripts.cfg.scanbot_cfg import ScanbotEnvCfg


class ScanbotSceneCfg(ScanbotEnvCfg):
    """Alias config for the Piper scanning scene (no teleop; kept as-is)."""

    def __post_init__(self):
        super().__post_init__()
        self.env_name = "Scanbot-Piper-Scene"


# Register a cleaner Gym id so we don't rely on the franka.* namespace.
gym.register(
    id="Scanbot-Piper-Scene-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ScanbotSceneCfg,
    },
    disable_env_checker=True,
)
