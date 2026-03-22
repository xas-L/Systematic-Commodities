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


# Combo-name leg parsers

def calendar_legs_from_name(combo_name: str) -> List[Tuple[str, int]]:
    """Parse a calendar combo name into (expiry, ratio) legs.

    Expected format: "FAR_EXPIRY-NEAR_EXPIRY" where each expiry is ISO YYYY-MM-DD.
    Splitting by '-' produces exactly 6 parts for two ISO dates joined by one hyphen.

    Returns [(far, -1), (near, +1)] — i.e. sell the deferred, buy the prompt,
    which is the "short the roll" direction consistent with a positive carry signal
    in a contango market.
    """
    parts = combo_name.split("-")
    if len(parts) == 6:
        far  = f"{parts[0]}-{parts[1]}-{parts[2]}"
        near = f"{parts[3]}-{parts[4]}-{parts[5]}"
        return [(far, -1), (near, 1)]
    # Fallback for non-ISO formats (e.g. short codes in tests)
    if len(parts) == 2:
        return [(parts[0], -1), (parts[1], 1)]
    # Last resort: split in the middle
    mid = len(parts) // 2
    far  = "-".join(parts[:mid])
    near = "-".join(parts[mid:])
    return [(far, -1), (near, 1)]


def butterfly_legs_from_name(combo_name: str) -> List[Tuple[str, int]]:
    """Parse a butterfly combo name into (expiry, ratio) legs.

    Expected format: "NEAR~CENTER~FAR" — three ISO-date expiries separated by '~'.
    The butterfly structure is 1 * NEAR − 2 * CENTER + 1 * FAR (wings positive,
    body negative), i.e. long the wings, short the belly.

    Returns [(near, +1), (center, -2), (far, +1)].
    """
    parts = combo_name.split("~")
    if len(parts) != 3:
        log.warning(
            "butterfly_legs_from_name: expected 3 parts separated by '~', "
            "got %d from '%s'. Returning empty legs.", len(parts), combo_name
        )
        return []
    near, center, far = parts
    return [(near, 1), (center, -2), (far, 1)]


# Portfolio state & intent

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



# Backtester


class Backtester:
    def __init__(
        self,
        *,
        executor: ComboExecutionSimulator,
        cfg: Optional[BacktestConfig] = None,
    ):
        self.exec = executor
        self.cfg = cfg or BacktestConfig()

    @staticmethod
    def _legs_from_name(name: str) -> List[Tuple[str, int]]:
        """Dispatch to the correct leg-parser based on the combo name format.

        Calendar:   "YYYY-MM-DD-YYYY-MM-DD"  (two ISO dates joined by '-', 6 parts)
        Butterfly:  "YYYY-MM-DD~YYYY-MM-DD~YYYY-MM-DD"  (three ISO dates with '~')
        """
        if "~" in name:
            return butterfly_legs_from_name(name)

        parts = name.split("-")
        if len(parts) == 6:
            # Two ISO dates (YYYY-MM-DD each) joined by a single '-'
            return calendar_legs_from_name(name)

        raise ValueError(
            f"Unrecognised combo name format: '{name}'. "
            "Expected 'YYYY-MM-DD-YYYY-MM-DD' for calendars or "
            "'YYYY-MM-DD~YYYY-MM-DD~YYYY-MM-DD' for butterflies."
        )

    @staticmethod
    def _mark_to_market(
        combo_name: str, lots: int, md_row: dict, lot_value: float
    ) -> float:
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
        index  = signals.index
        combos = list(signals.columns)
        state  = PortfolioState(
            positions={c: 0 for c in combos},
            cash=initial_cash,
            fees_paid=0.0,
        )
        pnl_path: list[float] = []
        trade_log: list[dict] = []

        for i in range(1, len(index)):
            t_prev, t_cur = index[i - 1], index[i]

            # Mark existing positions to market 
            pnl_inc = 0.0
            for c in combos:
                lots = state.positions.get(c, 0)
                if lots == 0:
                    continue
                q = quotes[c]
                try:
                    prev_mid = float(q["mid"].loc[t_prev])
                    curr_mid = float(q["mid"].loc[t_cur])
                except KeyError:
                    continue
                md_row = {"prev_mid": prev_mid, "curr_mid": curr_mid}
                pnl_inc += self._mark_to_market(c, lots, md_row, float(q["lot_value"]))
            state.cash += pnl_inc
            pnl_path.append(pnl_inc)

            # Generate intents from yesterday's signal (trade_on_next_bar) 
            last_signals = signals.loc[t_prev]
            intents: list[TradeIntent] = []
            for c in combos:
                if c not in quotes:
                    continue
                sig = float(last_signals.get(c, np.nan))
                if not np.isfinite(sig):
                    continue
                q        = quotes[c]
                lot_val  = float(q["lot_value"]) if np.isfinite(float(q.get("lot_value", 1.0))) else 1.0
                price_series = q["mid"].loc[:t_prev]
                lots = sizing_fn(
                    signal=sig,
                    combo_price_series=price_series,
                    lot_value=lot_val,
                    cfg=sizing_cfg,
                )
                if lots == 0:
                    continue
                try:
                    legs = self._legs_from_name(c)
                except ValueError as exc:
                    log.warning("Skipping combo '%s': %s", c, exc)
                    continue
                intents.append(TradeIntent(combo_name=c, legs=legs, lots=lots))

            #  Simulate execution 
            for ti in intents:
                q = quotes[ti.combo_name]
                tob = q.get("tob_size", {})
                tob_val = int(tob.get(t_cur, 10)) if isinstance(tob, dict) else int(tob)
                md_snap = {
                    "combo_mid":         float(q["mid"].loc[t_cur]),
                    "combo_half_spread": float(q["half_spread"].loc[t_cur]),
                    "combo_tob_size":    tob_val,
                }
                order = ComboOrder(
                    symbol=str(q.get("symbol", "")),
                    legs=[
                        ComboLeg(
                            expiry=pd.to_datetime(x).date(),
                            side=1 if r > 0 else -1,
                            ratio=abs(r),
                        )
                        for x, r in ti.legs
                    ],
                    qty=int(ti.lots),
                )
                rep: ComboExecutionReport = self.exec.simulate(order, md_snap)
                if rep.status in (OrderStatus.FILLED, OrderStatus.PARTIAL):
                    state.positions[ti.combo_name] = (
                        state.positions.get(ti.combo_name, 0) + rep.filled_qty
                    )
                    fill_cash = (
                        (rep.avg_fill_price or 0.0) * rep.filled_qty * float(q["lot_value"])
                        + rep.fees_total
                    )
                    state.cash   -= fill_cash
                    state.fees_paid += rep.fees_total
                trade_log.append({
                    "ts":     to_iso(datetime.combine(t_cur.date() if hasattr(t_cur, "date") else t_cur,
                                                      datetime.min.time())),
                    "combo":  ti.combo_name,
                    "filled": rep.filled_qty,
                    "avg_px": rep.avg_fill_price,
                    "fees":   rep.fees_total,
                    "status": rep.status.value,
                })

        return {
            "pnl_path":    pd.Series(pnl_path, index=index[1:]),
            "trade_log":   pd.DataFrame(trade_log),
            "final_state": state,
        }