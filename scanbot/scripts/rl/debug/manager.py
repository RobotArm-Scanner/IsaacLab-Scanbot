"""Debug manager window and flag helpers."""

from __future__ import annotations

_FLAGS = {
    "reward_plot": True,
    "coverage_plot": False,
    "teeth_gum_plot": False,
    "tcp_traj_plot": False,
    "termination_draw": True,
}

_CONTROL = {
    "window": None,
    "checkboxes": None,
    "env": None,
}

_LABELS = [
    ("Reward", "reward_plot"),
    ("Teeth Coverage", "coverage_plot"),
    ("Teeth_Gum Coverage", "teeth_gum_plot"),
    ("TCP 3D Trajectory", "tcp_traj_plot"),
    ("Termination Range", "termination_draw"),
]

_PLOT_STATE_ATTRS = {
    "reward_plot": "_scanbot_reward_plot_state",
    "coverage_plot": "_scanbot_coverage_plot_state",
    "teeth_gum_plot": "_scanbot_teeth_gum_plot_state",
    "tcp_traj_plot": "_scanbot_tcp_traj_plot_state",
}


def _set_last_env(env) -> None:
    _CONTROL["env"] = env


def _set_enabled(key: str, value: bool) -> None:
    _FLAGS[key] = bool(value)


def is_enabled(key: str) -> bool:
    return bool(_FLAGS.get(key, False))


def ensure_control_window(env) -> None:
    from PyQt5 import QtCore, QtWidgets

    _set_last_env(env)
    window = _CONTROL.get("window")
    if window is not None:
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.processEvents()
        return

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    window = QtWidgets.QWidget()
    window.setWindowTitle("Scanbot Debug Manager")
    window.setAttribute(QtCore.Qt.WA_ShowWithoutActivating, True)
    window.setWindowFlag(QtCore.Qt.Tool, True)
    window.setFocusPolicy(QtCore.Qt.NoFocus)

    layout = QtWidgets.QVBoxLayout()
    layout.setContentsMargins(8, 8, 8, 8)
    title = QtWidgets.QLabel("Debug Plot Toggles")
    title.setStyleSheet("font-weight: bold;")
    layout.addWidget(title)

    checkboxes = {}
    for label, key in _LABELS:
        cb = QtWidgets.QCheckBox(label)
        cb.setChecked(is_enabled(key))

        def _make_handler(k: str):
            def _handler(state: int) -> None:
                enabled = state == QtCore.Qt.Checked
                _set_enabled(k, enabled)
                if not enabled:
                    env_ref = _CONTROL.get("env")
                    if env_ref is not None:
                        close_plot(env_ref, k)

            return _handler

        cb.stateChanged.connect(_make_handler(key))
        layout.addWidget(cb)
        checkboxes[key] = cb

    layout.addStretch(1)
    window.setLayout(layout)
    window.resize(260, 160)
    window.show()

    _CONTROL["window"] = window
    _CONTROL["checkboxes"] = checkboxes

    app.processEvents()


def close_plot(env, key: str) -> None:
    if key == "termination_draw":
        _clear_termination_draw(env)
        return
    attr = _PLOT_STATE_ATTRS.get(key)
    if attr:
        _close_plot_state(env, attr)


def _close_plot_state(env, attr: str) -> None:
    state = getattr(env, attr, None)
    if not state:
        return
    fig = state.get("fig")
    if fig is not None:
        import matplotlib.pyplot as plt

        try:
            plt.close(fig)
        except Exception:
            pass
    setattr(env, attr, None)


def _clear_termination_draw(env) -> None:
    from . import scanpoint_draw

    scanpoint_draw.clear_debug_lines(env)
