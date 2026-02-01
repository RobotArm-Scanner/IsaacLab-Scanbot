"""Gym entry points for Scanbot environments.

This wrapper allows setting up Scanbot shared context without modifying upstream training scripts.
"""

from __future__ import annotations

import gymnasium as gym

from isaaclab.envs import ManagerBasedRLEnv

from scanbot.scripts import scanbot_context


class ScanbotExtensibleEnv(gym.Wrapper):
    """Gym env wrapper that integrates Scanbot shared context.

    - Registers the unwrapped env in ``scanbot_context`` after the first reset.
    - Drains queued hooks outside of ``env.step()`` to avoid re-entrancy issues.
    """

    def __init__(self, env):
        super().__init__(env)
        self._did_register = False

    @staticmethod
    def _drain_hooks() -> None:
        hook = scanbot_context.pop_hook()
        while hook is not None:
            hook()
            hook = scanbot_context.pop_hook()

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        if not self._did_register:
            scanbot_context.set_env(self.env.unwrapped)
            self._did_register = True
            print("[scanbot.scanbot_entrypoints] Registered env in scanbot_context after first reset.")
        self._drain_hooks()
        return obs, info

    def step(self, action):
        self._drain_hooks()
        return self.env.step(action)


def create_scanbot_env(*, cfg, render_mode=None, **kwargs):
    """Factory for IsaacLab gym env with Scanbot context bootstrap.

    Args are aligned with IsaacLab's gym.make(..., cfg=..., render_mode=...).
    """
    env = ManagerBasedRLEnv(cfg=cfg, render_mode=render_mode, **kwargs)
    return ScanbotExtensibleEnv(env)
