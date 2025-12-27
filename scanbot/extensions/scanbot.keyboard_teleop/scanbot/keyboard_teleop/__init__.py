"""Keyboard teleoperation for Piper using WASDQE keys."""

from __future__ import annotations

import carb
import omni.appwindow
import omni.ext
import omni.kit.app
import torch
from scanbot.scripts import scanbot_context


class Extension(omni.ext.IExt):
    """Simple WASDQE teleop that nudges the first three action dims each frame."""

    _ext_id = ""
    _sub = None
    _input = None
    _keyboard = None
    _kb_sub_id = None
    _pressed: set[str] = set()

    def on_startup(self, ext_id: str) -> None:
        self._ext_id = ext_id
        self._sub = None
        self._input = None
        self._keyboard = None
        self._kb_sub_id = None
        self._pressed: set[str] = set()

        app = omni.kit.app.get_app()
        stream = app.get_update_event_stream()
        self._sub = stream.create_subscription_to_pop(self._on_update, name="scanbot.keyboard_teleop.update")

        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._kb_sub_id = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)

        print(f"[scanbot.keyboard_teleop] Started: {ext_id}")

    def on_shutdown(self) -> None:
        self._input.unsubscribe_to_keyboard_events(self._keyboard, self._kb_sub_id)
        self._kb_sub_id = None
        self._keyboard = None
        self._input = None
        self._sub.unsubscribe()
        self._sub = None
        self._pressed.clear()
        print(f"[scanbot.keyboard_teleop] Stopped: {self._ext_id}")

    def _on_keyboard_event(self, event, *args):
        key_input = getattr(event, "input", None)
        if isinstance(key_input, str):
            key = key_input.upper()
        else:
            name = getattr(key_input, "name", "")
            key = name.upper() if isinstance(name, str) else ""
        if not key:
            return True
        if event.type in (carb.input.KeyboardEventType.KEY_PRESS, carb.input.KeyboardEventType.KEY_REPEAT):
            # print('PRESS')
            self._pressed.add(key)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            # print('RELEASE')
            self._pressed.discard(key)

    def _on_update(self, e) -> None:
        env = scanbot_context.get_env()
        if env is None:
            return

        shape = (env.num_envs, env.action_manager.total_action_dim)
        action = torch.zeros(shape, device=env.device)
        if not self._pressed:
            return

        delta = torch.zeros_like(action)
        s = 0.2
        key_map = {
            "W": (0, +s),
            "S": (0, -s),
            "A": (1, +s),
            "D": (1, -s),
            "Q": (2, +s),
            "E": (2, -s),
        }
        for key in self._pressed:
            if key in key_map:
                idx, val = key_map[key]
                if idx < delta.shape[1]:
                    delta[:, idx] += val
        action += delta
        scanbot_context.clear_actions()
        scanbot_context.enqueue_action(action)
