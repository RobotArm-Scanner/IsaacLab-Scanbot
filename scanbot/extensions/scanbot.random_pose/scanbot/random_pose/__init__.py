"""Random action UI for Scanbot.

Provides a small window with a `Random!` button that generates a random action
for the active gym env (from `scanbot_context`) and enqueues it for the launcher
to apply on the next update tick.
"""

from __future__ import annotations

import omni.ext
import omni.kit.app
import omni.ui as ui
import torch
from scanbot.scripts import scanbot_context


WINDOW_TITLE = "Scanbot Random Pose"


class Extension(omni.ext.IExt):
    def on_startup(self, ext_id: str) -> None:
        self._ext_id = ext_id
        self._window = None
        self._status_model = ui.SimpleStringModel("Waiting for env...")
        self._sub = None
        self._pending_action = None

        self._build_ui()

        # Subscribe to update stream to detect when env becomes available.
        app = omni.kit.app.get_app()
        stream = app.get_update_event_stream()
        self._sub = stream.create_subscription_to_pop(self._on_update, name="scanbot.random_pose.update")
        print(f"[scanbot.random_pose] Started: {ext_id}")

    def on_shutdown(self) -> None:
        self._sub.unsubscribe()
        self._sub = None
        self._window = None
        print(f"[scanbot.random_pose] Stopped: {self._ext_id}")

    def _build_ui(self) -> None:
        self._window = ui.Window(WINDOW_TITLE, width=300, height=140)
        with self._window.frame:
            with ui.VStack(spacing=8, height=0):
                ui.Label(
                    "Randomize (queued to launcher)",
                    height=22,
                    style={"font_size": 18},
                )

                ui.Button("Random!", height=24, clicked_fn=self._on_random_clicked)
                ui.Label(self._status_model.get_value_as_string(), word_wrap=True)

    def _on_random_clicked(self) -> None:
        env = scanbot_context.get_env()
        if env is None:
            self._set_status("Env not registered. Launch via basic_launcher.", error=True)
            return
        self._pending_action = self._make_random_action(env, scale=1.0)
        if self._pending_action is None:
            self._set_status("Failed to build random action.", error=True)
        else:
            self._set_status("Random action ready; will apply on next tick.")

    def _on_update(self, e) -> None:
        env = scanbot_context.get_env()
        if env is not None and self._status_model.get_value_as_string() != "Ready.":
            self._status_model.set_value("Ready.")
        if env is None:
            return

        if self._pending_action is None:
            return
        action = self._pending_action
        for _ in range(5):
            scanbot_context.enqueue_action(action)
        self._pending_action = None
        self._set_status("Random action enqueued.")

    def _make_random_action(self, env, scale: float = 1.0):
        dim = env.action_manager.total_action_dim
        shape = (env.num_envs, dim)
        device = env.device
        return (torch.rand(shape, device=device) * 2 - 1) * scale

    def _set_status(self, text: str, error: bool = False) -> None:
        self._status_model.set_value(text)
        if error:
            print(f"[scanbot.random_pose] ERROR: {text}")
        else:
            print(f"[scanbot.random_pose] {text}")
