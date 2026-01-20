"""Scanbot extension manager.

On startup:
- Enables all scanbot.* extensions under the sibling extensions folder.
- Sets up a watchdog on the extensions root and reloads an extension on change.

Reload logic: disable → sleep → enable (async, non-blocking). Watchdog-only.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List, Set

import omni.ext
import omni.kit.app
import toml
from isaacsim.core.utils import extensions as ext_utils
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


class _RootChangeHandler(FileSystemEventHandler):
    """Watches the extensions root; routes events to the owning extension id."""

    def __init__(self, extensions_root: Path, scanbot_exts: Set[str], signal_cb):
        super().__init__()
        self._root = extensions_root
        self._scanbot_exts = scanbot_exts
        self._signal_cb = signal_cb
        # Guard against reload loops caused by access/attribute events (e.g. atime updates).
        self._mtimes_ns: dict[Path, int] = {}

    def on_any_event(self, event):  # type: ignore[override]
        if getattr(event, "is_directory", False):
            return
        path = Path(getattr(event, "src_path", ""))
        try:
            rel = path.relative_to(self._root)
        except ValueError:
            return
        if "__pycache__" in rel.parts or path.suffix in {".pyc", ".pyo"}:
            return
        ext_dir = rel.parts[0] if rel.parts else ""
        if ext_dir in self._scanbot_exts:
            # Only trigger reload when file mtime actually changes.
            try:
                mtime_ns = path.stat().st_mtime_ns
            except FileNotFoundError:
                mtime_ns = -1
            except Exception:
                mtime_ns = -2

            prev = self._mtimes_ns.get(path)
            if prev is not None and prev == mtime_ns:
                return
            self._mtimes_ns[path] = mtime_ns

            print(f"[scanbot.extension_manager] FS event for {ext_dir}: {path}")
            self._signal_cb(ext_dir, str(path))


class Extension(omni.ext.IExt):
    def __init__(self):
        # Initialize to safe defaults in case async tasks outlive reloads.
        super().__init__()
        self._observer: Observer | None = None
        self._reloading: Set[str] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._cooldown_sec = 1.0
        self._ext_mgr = None
        self._extensions_root: Path | None = None

    def on_startup(self, ext_id: str):
        self._ext_id = ext_id
        self._loop = asyncio.get_event_loop()

        app = omni.kit.app.get_app()
        self._ext_mgr = app.get_extension_manager()

        # Locate extensions root and discover scanbot.* siblings.
        self_dir = Path(ext_utils.get_extension_path(ext_id))
        self._extensions_root = self_dir.parent
        # ext_id can include version suffix; exclude by directory name to avoid self-reload loops.
        scanbot_exts = set(self._discover_scanbot_exts(exclude={self_dir.name}))
        self._enable_exts(scanbot_exts)

        # Start watchdog on the root; route events to owning extension id.
        handler = _RootChangeHandler(self._extensions_root, scanbot_exts, self._request_reload)
        self._observer = Observer()
        self._observer.schedule(handler, str(self._extensions_root), recursive=True)
        self._observer.start()
        print(f"[scanbot.extension_manager] Watching root {self._extensions_root}")

        print(f"[scanbot.extension_manager] Started with extensions: {sorted(scanbot_exts)}")

    def on_shutdown(self):
        if self._observer is not None:
            self._observer.stop()
            self._observer.join()
            self._observer = None
        print("[scanbot.extension_manager] Stopped")

    def _discover_scanbot_exts(self, exclude: set[str]) -> List[str]:
        ext_ids: List[str] = []
        for ext_dir in self._extensions_root.iterdir():
            if not ext_dir.is_dir():
                continue
            ext_name = ext_dir.name
            if isinstance(ext_name, str) and ext_name.startswith("scanbot.") and ext_name not in exclude:
                ext_ids.append(ext_name)
        return ext_ids

    def _is_enabled(self, ext_dir: Path) -> bool:
        """Check scanbot.disabled flag in extension.toml (default: enabled)."""
        ext_toml = ext_dir / "extension.toml"
        if not ext_toml.is_file():
            return True
        try:
            data = toml.load(ext_toml)
            scanbot_cfg = data.get("scanbot", {})
            disabled = bool(scanbot_cfg.get("disabled", False))
            return not disabled
        except Exception:
            return True

    def _enable_exts(self, ext_ids: Set[str]):
        if self._ext_mgr is None:
            return
        for ext_id in ext_ids:
            if not self._is_enabled(self._extensions_root / ext_id):
                continue
            if not self._ext_mgr.is_extension_enabled(ext_id):
                self._ext_mgr.set_extension_enabled(ext_id, True)
                print(f"[scanbot.extension_manager] Enabled {ext_id}")

    def _request_reload(self, ext_id: str, changed_path: str):
        if self._ext_mgr is None or self._loop is None:
            return
        if ext_id in self._reloading:
            return
        self._reloading.add(ext_id)
        try:
            self._loop.call_soon_threadsafe(
                asyncio.create_task, self._reload_ext(ext_id, changed_path)
            )
        except RuntimeError:
            self._reloading.discard(ext_id)

    async def _reload_ext(self, ext_id: str, changed_path: str):
        # If this instance was reloaded/hot-swapped and state isn't ready, skip.
        if not hasattr(self, "_reloading") or not hasattr(self, "_extensions_root"):
            return
        ext_mgr = getattr(self, "_ext_mgr", None)
        if ext_mgr is None:
            return
        extensions_root = getattr(self, "_extensions_root", None)
        if extensions_root is None:
            return
        cooldown = getattr(self, "_cooldown_sec", 1.0)
        try:
            # Re-read disabled flag just before acting.
            enabled_now = self._is_enabled(extensions_root / ext_id)
            if not enabled_now:
                if ext_mgr.is_extension_enabled(ext_id):
                    ext_mgr.set_extension_enabled(ext_id, False)
                    print(f"[scanbot.extension_manager] Disabled (per extension.toml): {ext_id}")
                else:
                    print(f"[scanbot.extension_manager] Skip reload; disabled in extension.toml: {ext_id}")
                return
            if not ext_mgr.is_extension_enabled(ext_id):
                ext_mgr.set_extension_enabled(ext_id, True)
                print(f"[scanbot.extension_manager] Activated {ext_id}")
                return
            print(f"[scanbot.extension_manager] Change detected in {ext_id}: {changed_path}")
            ext_mgr.set_extension_enabled(ext_id, False)
            print(f"[scanbot.extension_manager] Deactivated {ext_id}")
            await asyncio.sleep(1.0)
            if self._is_enabled(extensions_root / ext_id):
                ext_mgr.set_extension_enabled(ext_id, True)
                print(f"[scanbot.extension_manager] Reloaded {ext_id}")
            else:
                print(f"[scanbot.extension_manager] Skipped re-enable (disabled in extension.toml): {ext_id}")
            await asyncio.sleep(cooldown)
        finally:
            reloading = getattr(self, "_reloading", None)
            if isinstance(reloading, set):
                reloading.discard(ext_id)
