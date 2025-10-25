# src/data/contracts.py
# Helpers for contract chains, expiry logic, and universe selection
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from ..core.utils import month_code_to_int


@dataclass(frozen=True)
class ContractKey:
    symbol: str
    expiry: date

    def key(self) -> str:
        return f"{self.symbol}:{self.expiry.isoformat()}"


# -----------------------------
# Contract chain utilities
# -----------------------------

def available_expiries_by_date(df: pd.DataFrame) -> Dict[date, List[date]]:
    """Return mapping date -> sorted list of expiries available that day."""
    out: Dict[date, List[date]] = {}
    for d, g in df.groupby("date"):
        exps = sorted(g["expiry"].unique().tolist())
        out[d] = exps
    return out


def filter_tradeable_rows(
    df: pd.DataFrame,
    *,
    min_volume: int = 0,
    min_open_interest: int = 0,
    drop_stale_after_days: int = 10,
) -> pd.DataFrame:
    """Filter contract bars by volume, OI, and staleness.

    Stale means we have not seen a print for N days for that expiry.
    """
    if df.empty:
        return df
    df = df.sort_values(["symbol", "expiry", "date"]).copy()
    # Simple staleness: forward fill last seen date per (symbol, expiry)
    df["last_seen"] = df.groupby(["symbol", "expiry"]) ["date"].transform("max")
    df["days_since_last"] = (pd.to_datetime(df["last_seen"]) - pd.to_datetime(df["date"])) .dt.days
    ok = (
        (df["volume"].fillna(0) >= min_volume)
        & (df["open_interest"].fillna(0) >= min_open_interest)
        & (df["days_since_last"].fillna(0) <= drop_stale_after_days)
    )
    return df.loc[ok].drop(columns=["last_seen", "days_since_last"])  # type: ignore


# -----------------------------
# Month code helpers
# -----------------------------
_MONTH_INT_TO_CODE = {v: k for k, v in {
    "F": 1,  "G": 2,  "H": 3,  "J": 4,  "K": 5,  "M": 6,
    "N": 7,  "Q": 8,  "U": 9,  "V": 10, "X": 11, "Z": 12,
}.items()}


def expiry_to_code(exp: date) -> str:
    return _MONTH_INT_TO_CODE[exp.month]


def contract_code(symbol: str, exp: date) -> str:
    # CLZ5 style (one-digit year). This is for logs only.
    y = str(exp.year)[-1]
    return f"{symbol}{expiry_to_code(exp)}{y}"


# -----------------------------
# Front N contract selection per day
# -----------------------------

def select_front_n(
    df: pd.DataFrame,
    *,
    n: int,
    min_contracts: int,
) -> pd.DataFrame:
    """For each date, keep the first n expiries if at least min_contracts are listed that day."""
    if df.empty:
        return df
    out = []
    for d, g in df.groupby("date"):
        exps = sorted(g["expiry"].unique().tolist())
        if len(exps) < min_contracts:
            continue
        keep = set(exps[:n])
        out.append(g[g["expiry"].isin(keep)])
    return pd.concat(out, ignore_index=True) if out else df.iloc[0:0]


# -----------------------------
# Days-to-expiry (useful for roll yield normalisation)
# -----------------------------

def add_days_to_expiry(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    tmp = df.copy()
    tmp["dte"] = (pd.to_datetime(tmp["expiry"]) - pd.to_datetime(tmp["date"])) .dt.days
    return tmp


__all__ = [
    "ContractKey",
    "available_expiries_by_date",
    "filter_tradeable_rows",
    "expiry_to_code",
    "contract_code",
    "select_front_n",
    "add_days_to_expiry",
]
