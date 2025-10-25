# src/ops/kill_switch.py
# Dual kill switch: soft pause (no new entries) and hard flatten (close all and halt)
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..core.logging import get_logger

log = get_logger(__name__)


@dataclass
class KillSwitchConfig:
    soft_file: str = "PAUSE.NOW"
    hard_file: str = "FLATTEN.NOW"


class KillSwitch:
    """File-based kill switch used both in backtests and paper/live.

    - Soft kill: create `PAUSE.NOW` → disable new entries but allow exits/rolls.
    - Hard kill: create `FLATTEN.NOW` → request immediate flattening and halt.
    """

    def __init__(self, root: Optional[Path] = None, cfg: Optional[KillSwitchConfig] = None):
        self.root = Path(root or Path.cwd())
        self.cfg = cfg or KillSwitchConfig()

    def _exists(self, fname: str) -> bool:
        return (self.root / fname).exists()

    def soft_engaged(self) -> bool:
        return self._exists(self.cfg.soft_file)

    def hard_engaged(self) -> bool:
        return self._exists(self.cfg.hard_file)

    def engage_soft(self) -> None:
        p = self.root / self.cfg.soft_file
        p.write_text("pause")
        log.warning("Soft kill engaged: %s", p)

    def engage_hard(self) -> None:
        p = self.root / self.cfg.hard_file
        p.write_text("flatten")
        log.error("Hard kill engaged: %s", p)

    def clear(self) -> None:
        for fname in (self.cfg.soft_file, self.cfg.hard_file):
            p = self.root / fname
            if p.exists():
                p.unlink()
                log.info("Cleared kill file: %s", p)


__all__ = ["KillSwitch", "KillSwitchConfig"]
