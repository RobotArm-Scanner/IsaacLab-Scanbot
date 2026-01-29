"""Shared plotting helpers."""

from __future__ import annotations


def should_update(step: int, interval: int) -> bool:
    interval = int(interval) if interval else 1
    if interval > 1 and step % interval != 0:
        return False
    return True


def resolve_env_ids(env, env_ids):
    if env_ids is None:
        return list(range(env.num_envs))
    return env_ids


def apply_qt_focus_flags(fig) -> None:
    import matplotlib

    if "Qt" not in str(matplotlib.get_backend()):
        return
    from matplotlib.backends.qt_compat import QtCore

    window = fig.canvas.manager.window
    window.setAttribute(QtCore.Qt.WA_ShowWithoutActivating, True)
    window.setWindowFlag(QtCore.Qt.Tool, True)
    window.setFocusPolicy(QtCore.Qt.NoFocus)
    window.show()
