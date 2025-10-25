# src/ops/execution_manager.py
# Thin orchestration layer to route desired intents to the combo execution simulator (or live broker later)
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..core.logging import get_logger
from ..core.types import ComboOrder, ComboLeg, OrderStatus, ComboExecutionReport
from ..execsim.combos import ComboExecutionSimulator

log = get_logger(__name__)


@dataclass
class ExecIntent:
    symbol: str
    combo_name: str
    legs: List[Tuple[str, int]]  # (expiry_iso, ratio)
    lots: int
    limit_price: Optional[float] = None


class ExecutionManager:
    """Manages submission of combo intents to the execution backend (simulator for now).

    In paper/live this would interface with a broker adapter; for backtests it wraps the simulator.
    """

    def __init__(self, executor: ComboExecutionSimulator):
        self.exec = executor
        self._last_reports: list[ComboExecutionReport] = []

    def submit(self, intents: List[ExecIntent], md_row: Dict[str, dict], ts: Optional[datetime] = None) -> List[ComboExecutionReport]:
        """Submit a batch of intents using the provided market data row for quotes.

        md_row maps combo_name -> {combo_mid, combo_half_spread, combo_tob_size} at time ts.
        """
        ts = ts or datetime.utcnow()
        reports: list[ComboExecutionReport] = []
        for it in intents:
            md = md_row.get(it.combo_name, {})
            order = ComboOrder(
                symbol=it.symbol,
                legs=[ComboLeg(expiry=pd.to_datetime(e).date(), side=1 if r > 0 else -1, ratio=abs(r)) for e, r in it.legs],
                qty=int(it.lots),
                limit_price=it.limit_price,
            )
            rep = self.exec.simulate(order, md, now=ts)
            reports.append(rep)
        self._last_reports = reports
        return reports

    def last_reports(self) -> List[ComboExecutionReport]:
        return list(self._last_reports)


__all__ = ["ExecutionManager", "ExecIntent"]
