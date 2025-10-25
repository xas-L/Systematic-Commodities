# src/ops/hot_reload.py
# Config hot-reload watcher
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
import time

from ..core.logging import get_logger

log = get_logger(__name__)


@dataclass
class HotReloadConfig:
    paths: list[Path]
    poll_seconds: float = 2.0


class HotReloader:
    def __init__(self, cfg: HotReloadConfig, on_change: Callable[[Path], None]):
        self.cfg = cfg
        self.on_change = on_change
        self._mtimes: dict[Path, float] = {}

    def _snapshot(self) -> None:
        for p in self.cfg.paths:
            try:
                self._mtimes[p] = p.stat().st_mtime
            except FileNotFoundError:
                self._mtimes[p] = 0.0

    def start(self, run_once: bool = False) -> None:
        self._snapshot()
        while True:
            changed = []
            for p in self.cfg.paths:
                try:
                    mt = p.stat().st_mtime
                except FileNotFoundError:
                    mt = 0.0
                if p not in self._mtimes:
                    self._mtimes[p] = mt
                if mt != self._mtimes[p]:
                    self._mtimes[p] = mt
                    changed.append(p)
            for p in changed:
                log.info("Hot reload: %s changed", p)
                try:
                    self.on_change(p)
                except Exception as e:
                    log.exception("Hot reload handler failed for %s: %s", p, e)
            if run_once:
                break
            time.sleep(max(self.cfg.poll_seconds, 0.1))


__all__ = ["HotReloader", "HotReloadConfig"]
