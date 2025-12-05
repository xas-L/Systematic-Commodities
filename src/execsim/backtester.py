# src/execsim/backtester.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.logging import get_logger
from ..core.types import ComboOrder, ComboLeg, ComboExecutionReport, OrderStatus
from ..core.utils import to_iso
from .combos import ComboExecutionSimulator

log = get_logger(__name__)

# DEFINE FUNCTIONS HERE INSTEAD OF IMPORTING
def calendar_legs_from_name(combo_name: str) -> List[Tuple[str, int]]:
    parts = combo_name.split('-')
    if len(parts) == 6:
        expiry1 = f"{parts[0]}-{parts[1]}-{parts[2]}"
        expiry2 = f"{parts[3]}-{parts[4]}-{parts[5]}"
        return [(expiry1, -1), (expiry2, 1)]
    return [(combo_name, -1), (combo_name, 1)]

def butterfly_legs_from_name(combo_name: str) -> List[Tuple[str, int]]:
    return []

@dataclass
class PortfolioState:
    positions: Dict[str, int]
    cash: float
    fees_paid: float

@dataclass
class TradeIntent:
    combo_name: str
    legs: List[Tuple[str, int]]
    lots: int

@dataclass
class BacktestConfig:
    trade_on_next_bar: bool = True
    combo_only: bool = True
    allow_legging: bool = False

class Backtester:
    def __init__(self, *, executor: ComboExecutionSimulator, cfg: Optional[BacktestConfig] = None):
        self.exec = executor
        self.cfg = cfg or BacktestConfig()

    @staticmethod
    def _legs_from_name(name: str) -> List[Tuple[str, int]]:
        parts = name.split("-")
        if len(parts) == 2:
            return calendar_legs_from_name(name)
        elif len(parts) == 3:
            return butterfly_legs_from_name(name)
        else:
            raise ValueError(f"Unrecognised combo name: {name}")

    @staticmethod
    def _mark_to_market(combo_name: str, lots: int, md_row: dict, lot_value: float) -> float:
        prev_mid = float(md_row.get("prev_mid", 0.0))
        curr_mid = float(md_row.get("curr_mid", 0.0))
        return lots * (curr_mid - prev_mid) * lot_value

    def run(
        self,
        *,
        signals: pd.DataFrame,
        quotes: Dict[str, dict],
        sizing_fn,
        sizing_cfg,
        initial_cash: float = 0.0,
    ) -> dict:
        index = signals.index
        combos = list(signals.columns)
        state = PortfolioState(positions={c: 0 for c in combos}, cash=initial_cash, fees_paid=0.0)
        pnl_path = []
        trade_log = []

        for i in range(1, len(index)):
            t_prev, t_cur = index[i - 1], index[i]
            pnl_inc = 0.0
            for c in combos:
                lots = state.positions.get(c, 0)
                if lots == 0: continue
                q = quotes[c]
                md_row = {"prev_mid": float(q["mid"].loc[t_prev]), "curr_mid": float(q["mid"].loc[t_cur])}
                pnl_inc += self._mark_to_market(c, lots, md_row, float(q["lot_value"]))
            state.cash += pnl_inc
            pnl_path.append(pnl_inc)

            last_signals = signals.loc[t_prev]
            intents = []
            for c in combos:
                sig = float(last_signals.get(c, np.nan))
                if not np.isfinite(sig): continue
                q = quotes[c]
                lot_value = float(q["lot_value"]) if np.isfinite(q.get("lot_value", np.nan)) else 1.0
                lots = sizing_fn(signal=sig, combo_price_series=q["mid"].loc[:t_prev], lot_value=lot_value, cfg=sizing_cfg)
                if lots == 0: continue
                legs = self._legs_from_name(c)
                intents.append(TradeIntent(combo_name=c, legs=legs, lots=lots))

            for ti in intents:
                q = quotes[ti.combo_name]
                md_snap = {
                    "combo_mid": float(q["mid"].loc[t_cur]),
                    "combo_half_spread": float(q["half_spread"].loc[t_cur]),
                    "combo_tob_size": int(q.get("tob_size", {}).get(t_cur, 10)),
                }
                order = ComboOrder(symbol=str(q.get("symbol", "")), legs=[ComboLeg(expiry=pd.to_datetime(x).date(), side=1 if r>0 else -1, ratio=abs(r)) for x, r in ti.legs], qty=int(ti.lots))
                rep: ComboExecutionReport = self.exec.simulate(order, md_snap)
                if rep.status in (OrderStatus.FILLED, OrderStatus.PARTIAL):
                    state.positions[ti.combo_name] = state.positions.get(ti.combo_name, 0) + rep.filled_qty
                    state.cash -= (rep.avg_fill_price or 0.0) * rep.filled_qty * float(q["lot_value"]) + rep.fees_total
                    state.fees_paid += rep.fees_total
                trade_log.append({
                    "ts": to_iso(datetime.combine(t_cur, datetime.min.time())),
                    "combo": ti.combo_name,
                    "filled": rep.filled_qty,
                    "avg_px": rep.avg_fill_price,
                    "fees": rep.fees_total,
                    "status": rep.status.value,
                })

        return {
            "pnl_path": pd.Series(pnl_path, index=index[1:]),
            "trade_log": pd.DataFrame(trade_log),
            "final_state": state,
        }
