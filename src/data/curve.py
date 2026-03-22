# src/data/curve.py
# Build point-in-time futures curves and term-structure features
from __future__ import annotations

from datetime import date
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ..core.utils import safe_div
from .contracts import filter_tradeable_rows, select_front_n, add_days_to_expiry



# Price field resolution


def _resolve_price(df: pd.DataFrame, price_field_order: list[str]) -> pd.Series:
    for f in price_field_order:
        if f in df.columns:
            s = pd.to_numeric(df[f], errors="coerce")
            if s.notna().any():
                return s
    return pd.Series(index=df.index, dtype=float)



# Curve construction


def build_curve_surface(
    contract_df: pd.DataFrame,
    *,
    min_volume: int = 0,
    min_open_interest: int = 0,
    drop_stale_after_days: int = 10,
    price_field_order: list[str] = ("settle", "last"),
    tenors_target: int = 10,
    min_contracts: int = 6,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (curve_surface, bar_subset).

    curve_surface: index=date, columns=expiry, values=resolved price
    bar_subset: the filtered subset used to build the surface
    """
    if contract_df.empty:
        return contract_df.iloc[0:0], contract_df.iloc[0:0]

    # Filter for tradeability and pick the first N expiries per day
    filt = filter_tradeable_rows(
        contract_df,
        min_volume=min_volume,
        min_open_interest=min_open_interest,
        drop_stale_after_days=drop_stale_after_days,
    )
    topn = select_front_n(filt, n=tenors_target, min_contracts=min_contracts)

    # Resolve price per row
    topn = topn.copy()
    topn["px"] = _resolve_price(topn, list(price_field_order))

    # Pivot to surface
    surface = topn.pivot_table(index="date", columns="expiry", values="px")
    surface = surface.sort_index(axis=1)  # columns ascending by expiry
    surface = surface.dropna(how="all")
    
    # CRITICAL FIX: Convert index to DatetimeIndex
    surface.index = pd.to_datetime(surface.index)

    return surface, topn



# Adjacent log spreads and roll yield


def log_adjacent_spreads(curve_surface: pd.DataFrame) -> pd.DataFrame:
    if curve_surface.empty or curve_surface.shape[1] < 2:
        return curve_surface.iloc[0:0]
    cols = list(curve_surface.columns)
    arr = np.log(curve_surface[cols].to_numpy())
    spreads = arr[:, 1:] - arr[:, :-1]
    spread_cols = [f"{cols[i]}-{cols[i-1]}" for i in range(1, len(cols))]
    # Ensure DatetimeIndex
    result = pd.DataFrame(spreads, index=curve_surface.index, columns=spread_cols)
    result.index = pd.to_datetime(result.index)
    return result


def roll_yield_adjacent(
    curve_surface: pd.DataFrame,
    *,
    as_annual: bool = True,
) -> pd.DataFrame:
    """Approximate roll yield between adjacent expiries normalised by time gap."""
    if curve_surface.empty or curve_surface.shape[1] < 2:
        return curve_surface.iloc[0:0]

    cols = list(curve_surface.columns)
    # Compute time gaps in days between expiries (constant per pair)
    exp_days = np.array([pd.Timestamp(c).to_pydatetime().date().toordinal() for c in cols], dtype=float)
    gaps_days = exp_days[1:] - exp_days[:-1]
    gaps_years = gaps_days / 365.25

    log_sp = log_adjacent_spreads(curve_surface)
    mat = log_sp.to_numpy()
    denom = gaps_years.reshape(1, -1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ry = np.divide(mat, denom)
    # Ensure DatetimeIndex
    result = pd.DataFrame(ry, index=curve_surface.index, columns=log_sp.columns)
    result.index = pd.to_datetime(result.index)
    return result



# Utility: align two surfaces on common dates/expiries


def align_surfaces(a: pd.DataFrame, b: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    common_dates = a.index.intersection(b.index)
    common_cols = a.columns.intersection(b.columns)
    return a.loc[common_dates, common_cols], b.loc[common_dates, common_cols]


__all__ = [
    "build_curve_surface",
    "log_adjacent_spreads",
    "roll_yield_adjacent",
    "align_surfaces",
]
