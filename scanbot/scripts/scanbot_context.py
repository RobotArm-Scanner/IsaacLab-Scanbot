"""Shared Scanbot context for passing app/env between launcher and extensions."""

from typing import Any, Optional, List

_app_launcher: Any = None
_env: Any = None
# Simple FIFO for action tensors to be consumed by the launcher.
_action_queue: List[Any] = []


def set_app_launcher(app_launcher: Any) -> None:
    global _app_launcher
    _app_launcher = app_launcher


def get_app_launcher() -> Any:
    return _app_launcher


def set_env(env: Any) -> None:
    global _env
    _env = env


def get_env() -> Any:
    return _env


def enqueue_action(action: Any) -> None:
    """Append an action tensor to the shared queue."""
    _action_queue.append(action)


def pop_action() -> Optional[Any]:
    """Pop the oldest action from the queue, if any."""
    if not _action_queue:
        return None
    return _action_queue.pop(0)


def clear_actions() -> None:
    """Clear all queued actions."""
    _action_queue.clear()
