"""Common utilities for Scanbot extensions."""

from __future__ import annotations

import omni.ext

from . import marker_util, pos_util

__all__ = ["marker_util", "pos_util"]


class Extension(omni.ext.IExt):
    def on_startup(self, ext_id: str) -> None:
        self._ext_id = ext_id
        print(f"[scanbot.common] Started: {ext_id}")

    def on_shutdown(self) -> None:
        print(f"[scanbot.common] Stopped: {self._ext_id}")
