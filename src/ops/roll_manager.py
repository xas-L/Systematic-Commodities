# src/ops/roll_manager.py
# Roll planner for calendars and butterflies using exchange combo semantics
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..core.logging import get_logger
from ..data.spreads import (
    calendar_legs_from_name,
    butterfly_legs_from_name,
)

log = get_logger(__name__)


# -----------------------------
# Policy & actions
# -----------------------------
@dataclass
class RollPolicy:
    roll_window_days: int = 5
    flatten_days_before_fnd: int = 3
    flatten_days_before_ltd: int = 1
    schedule_fractions: Tuple[float, float, float] = (0.5, 0.35, 0.15)  # spread over roll window


@dataclass
class RollAction:
    symbol: str
    from_combo: str
    to_combo: Optional[str]  # None means flatten
    lots: int                # signed (positive means add/open; negative close)
    reason: str
    schedule: Tuple[float, ...]


# -----------------------------
# Helpers
# -----------------------------

def _combo_kind(name: str) -> str:
    parts = name.split("-")
    return "calendar" if len(parts) == 2 else ("butterfly" if len(parts) == 3 else "unknown")


def _exp_list(surface: pd.DataFrame) -> List[str]:
    # surface columns are expiry dates; cast to ISO strings
    return [pd.Timestamp(c).date().isoformat() for c in surface.columns]


def _next_calendar_name(name: str, exps: List[str]) -> Optional[str]:
    a, b = name[:10], name[-10:]
    if a not in exps or b not in exps:
        return None
    i, j = exps.index(a), exps.index(b)
    if i + 1 >= len(exps) or j + 1 >= len(exps):
        return None
    return f"{exps[i+1]}-{exps[j+1]}"


def _next_butterfly_name(name: str, exps: List[str]) -> Optional[str]:
    e1, e2, e3 = name.split("-")[:3]
    if e1 not in exps or e2 not in exps or e3 not in exps:
        return None
    i1, i2, i3 = exps.index(e1), exps.index(e2), exps.index(e3)
    if i3 + 1 >= len(exps):
        return None
    return f"{exps[i1+1]}-{exps[i2+1]}-{exps[i3+1]}"


def _legs_from_combo(name: str) -> List[Tuple[str, int]]:
    kind = _combo_kind(name)
    if kind == "calendar":
        return calendar_legs_from_name(name)
    if kind == "butterfly":
        return butterfly_legs_from_name(name)
    raise ValueError(f"Unrecognised combo name: {name}")


def _days_to(d0: date, d1: Optional[date]) -> Optional[int]:
    if d1 is None:
        return None
    return (d1 - d0).days


def _leg_is_in_roll_window(leg_exp: str, today: date, meta_by_exp: dict, policy: RollPolicy) -> bool:
    meta = meta_by_exp.get(leg_exp)
    exp_date = pd.to_datetime(leg_exp).date()
    dte = (exp_date - today).days
    # FND/LTD checks if available
    if meta is not None:
        d_fnd = _days_to(today, meta.get("first_notice_date"))
        d_ltd = _days_to(today, meta.get("last_trade_date"))
        if d_fnd is not None and d_fnd <= policy.flatten_days_before_fnd:
            return True
        if d_ltd is not None and d_ltd <= policy.flatten_days_before_ltd:
            return True
    # Generic roll window on expiry proximity
    return dte <= policy.roll_window_days


# -----------------------------
# Roll manager
# -----------------------------
class RollManager:
    """Plans roll actions for open combo positions to avoid FND/LTD traps.

    Inputs:
      - positions: dict combo_name -> lots (signed)
      - surface: price surface (cols=expiries) for current day; defines the tradable ladder
      - meta_df: optional DataFrame with columns [expiry, first_notice_date, last_trade_date]

    Output: list of RollAction with either (from → to) or flatten instructions.
    """

    def __init__(self, policy: Optional[RollPolicy] = None):
        self.policy = policy or RollPolicy()

    def plan(
        self,
        *,
        symbol: str,
        positions: Dict[str, int],
        surface: pd.DataFrame,
        today: date,
        meta_df: Optional[pd.DataFrame] = None,
    ) -> List[RollAction]:
        if surface.empty:
            return []
        exps = _exp_list(surface)
        # Build metadata map
        meta_by_exp: dict = {}
        if meta_df is not None and not meta_df.empty:
            m = meta_df.copy()
            m["expiry"] = pd.to_datetime(m["expiry"]).dt.date.astype(str)
            for _, row in m.iterrows():
                meta_by_exp[str(row["expiry"]) ] = {
                    "first_notice_date": pd.to_datetime(row.get("first_notice_date")).date() if pd.notna(row.get("first_notice_date")) else None,
                    "last_trade_date": pd.to_datetime(row.get("last_trade_date")).date() if pd.notna(row.get("last_trade_date")) else None,
                }

        actions: List[RollAction] = []
        for combo_name, lots in positions.items():
            if lots == 0:
                continue
            # Inspect legs
            try:
                legs = _legs_from_combo(combo_name)
            except Exception:
                log.warning("Skipping unknown combo name: %s", combo_name)
                continue
            # If any leg breaches roll window → plan roll
            needs_roll = any(_leg_is_in_roll_window(exp, today, meta_by_exp, self.policy) for exp, _ in legs)
            if not needs_roll:
                continue
            kind = _combo_kind(combo_name)
            if kind == "calendar":
                next_name = _next_calendar_name(combo_name, exps)
            elif kind == "butterfly":
                next_name = _next_butterfly_name(combo_name, exps)
            else:
                next_name = None

            if next_name is None:
                # No next combo available → flatten
                actions.append(RollAction(symbol=symbol, from_combo=combo_name, to_combo=None, lots=-lots, reason="no_next_combo", schedule=self.policy.schedule_fractions))
            else:
                # Roll: close old, open new with same signed lots
                actions.append(RollAction(symbol=symbol, from_combo=combo_name, to_combo=next_name, lots=lots, reason="roll_window", schedule=self.policy.schedule_fractions))
        return actions


# -----------------------------
# Plan → intents helper (optional)
# -----------------------------
@dataclass
class ExecIntent:
    symbol: str
    combo_name: str
    legs: List[Tuple[str, int]]
    lots: int


def actions_to_exec_intents(actions: List[RollAction]) -> List[ExecIntent]:
    intents: List[ExecIntent] = []
    for a in actions:
        if a.to_combo is None:
            # Flatten existing combo
            legs = _legs_from_combo(a.from_combo)
            intents.append(ExecIntent(symbol=a.symbol, combo_name=a.from_combo, legs=legs, lots=a.lots))
        else:
            # Close old
            legs_old = _legs_from_combo(a.from_combo)
            intents.append(ExecIntent(symbol=a.symbol, combo_name=a.from_combo, legs=legs_old, lots=-a.lots))
            # Open new
            legs_new = _legs_from_combo(a.to_combo)
            intents.append(ExecIntent(symbol=a.symbol, combo_name=a.to_combo, legs=legs_new, lots=a.lots))
    return intents


__all__ = [
    "RollPolicy",
    "RollAction",
    "RollManager",
    "actions_to_exec_intents",
]
