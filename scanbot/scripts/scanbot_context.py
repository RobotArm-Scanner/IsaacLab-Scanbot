"""Shared Scanbot context for passing app/env between launcher and extensions."""

from typing import Any, Callable, List, Optional

_app_launcher: Any = None
_env: Any = None
# Simple FIFO for action tensors to be consumed by the launcher.
_action_queue: List[Any] = []
_hook_queue: List[Callable[[], None]] = []

try:
    from scanbot.common import pos_util as _pos_util
except Exception:
    _pos_util = None

def set_app_launcher(app_launcher: Any) -> None:
    global _app_launcher
    _app_launcher = app_launcher


def get_app_launcher() -> Any:
    return _app_launcher


def set_env(env: Any) -> None:
    global _env
    _env = env
    if _pos_util is not None:
        try:
            _pos_util.configure_from_env(env)
        except Exception:
            pass


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


def enqueue_hook(func: Callable[[], None]) -> None:
    """Enqueue a hook to be executed by the launcher loop.

    Hooks are executed in the launcher loop (outside of ``env.step()``) to avoid re-entrancy issues
    if an extension triggers operations like ``env.reset()``.
    """
    _hook_queue.append(func)


def pop_hook() -> Optional[Callable[[], None]]:
    """Pop the oldest hook, if any."""
    if not _hook_queue:
        return None
    return _hook_queue.pop(0)


def clear_hook() -> None:
    """Clear all queued hooks."""
    _hook_queue.clear()
