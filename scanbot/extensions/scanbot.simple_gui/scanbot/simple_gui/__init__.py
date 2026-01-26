"""Minimal omni.ui extension example for Scanbot.

The window shows a slider and a button and logs activity when the user interacts
with the controls. This is intended as a starting point for custom Isaac Lab GUI
tools; drop the folder under an `--ext-folder` and enable `scanbot.simple_gui`.
"""

from __future__ import annotations

import carb
import omni.ext
import omni.kit.ui
import omni.ui as ui


WINDOW_TITLE = "Scanbot Simple GUI"
MENU_PATH = "Window/Scanbot/Simple GUI"


class Extension(omni.ext.IExt):
    """Simple UI extension that logs slider/button interactions."""

    def on_startup(self, ext_id: str) -> None:
        self._ext_id = ext_id
        self._window = None
        self._slider_model = ui.SimpleFloatModel()
        self._slider_label_model = ui.SimpleStringModel("0.25")
        self._status_model = ui.SimpleStringModel("Waiting for input...")
        self._click_count = 0
        self._menu = None

        self._slider_model.set_value(0.25)
        self._slider_model.add_value_changed_fn(self._on_slider_changed)

        carb.log_info(f"[scanbot.simple_gui] Starting extension: {ext_id}")
        self._build_ui()
        self._window.visible = False
        self._window.set_visibility_changed_fn(self._on_visibility_changed)
        self._add_menu()
        ui.Workspace.set_show_window_fn(WINDOW_TITLE, self._show_window_from_workspace)

    def on_shutdown(self) -> None:
        menu = omni.kit.ui.get_editor_menu()
        menu.remove_item(MENU_PATH)
        self._menu = None
        carb.log_info(f"[scanbot.simple_gui] Shutting down extension: {self._ext_id}")
        ui.Workspace.set_show_window_fn(WINDOW_TITLE, None)
        self._window = None

    def _build_ui(self) -> None:
        self._window = ui.Window(WINDOW_TITLE, width=360, height=240)
        with self._window.frame:
            with ui.VStack(spacing=8, height=0):
                ui.Label(
                    "Scanbot UI Examples",
                    height=22,
                    style={"font_size": 18},
                )
                ui.Label(
                    "Move the slider and click the button to see logs and status text update.",
                    word_wrap=True,
                )

                ui.Spacer(height=6)
                ui.Label("Slider value")
                ui.FloatSlider(min=0.0, max=1.0, step=0.01, model=self._slider_model)
                # Older omni.ui.Label expects only text; render model value manually.
                ui.Label(self._slider_label_model.get_value_as_string(), style={"color": ui.color.gray})

                ui.Spacer(height=6)
                ui.Button("Log current value", height=24, clicked_fn=self._on_button_clicked)
                ui.Label(self._status_model.get_value_as_string(), word_wrap=True)

    def _add_menu(self) -> None:
        self._menu = omni.kit.ui.get_editor_menu().add_item(
            MENU_PATH, self._on_menu_toggle, toggle=True, value=False
        )

    def _toggle_window(self, visible: bool):
        if visible:
            if self._window is None:
                self._build_ui()
            if self._window is None:
                return
            self._window.visible = True
            self._window.focus()
        else:
            if self._window is None:
                return
            self._window.visible = False
        self._sync_menu_with_window()

    def _on_menu_toggle(self, menu, value):
        self._toggle_window(bool(value))

    def _on_visibility_changed(self, visible):
        self._sync_menu_with_window()

    def _sync_menu_with_window(self):
        if self._menu is not None and self._window is not None:
            menu = omni.kit.ui.get_editor_menu()
            menu.set_value(MENU_PATH, self._window.visible)

    def _show_window_from_workspace(self, *_):
        self._toggle_window(True)

    def _on_slider_changed(self, model: ui.AbstractValueModel) -> None:
        value = model.get_value_as_float()
        self._slider_label_model.set_value(f"{value:.2f}")

    def _on_button_clicked(self) -> None:
        self._click_count += 1
        value = self._slider_model.get_value_as_float()
        message = f"Button clicked {self._click_count} times (slider={value:.2f})"
        carb.log_info(f"[scanbot.simple_gui] {message}")
        self._status_model.set_value(message)
