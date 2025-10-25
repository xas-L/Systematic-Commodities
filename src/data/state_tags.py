# src/data/state_tags.py
# Observable regime tags used for conditional performance cuts
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .contracts import add_days_to_expiry


# -----------------------------
# Core tag builders
# -----------------------------

def tag_month_of_year(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(index.month, index=index, name="month")


def tag_refinery_maintenance(index: pd.DatetimeIndex) -> pd.Series:
    """Refinery maintenance tends to cluster in Feb-Mar and Sep-Oct.
    This is a coarse tag for conditioning only.
    """
    months = index.month
    flag = (months.isin([2, 3, 9, 10])).astype(int)
    return pd.Series(flag, index=index, name="refinery_maint")


def tag_plant_harvest(index: pd.DatetimeIndex, symbol: str) -> pd.DataFrame:
    """Simple plant/harvest windows for grains.
    Corn (ZC): plant Apr-May, harvest Sep-Nov
    Soybeans (ZS): plant May-Jun, harvest Sep-Nov
    Wheat (ZW): winter wheat varies; use broad tags: plant Oct-Nov, harvest Jun-Aug
    Returns two columns: is_plant, is_harvest (0/1).
    """
    m = index.month
    plant = np.zeros(len(index), dtype=int)
    harvest = np.zeros(len(index), dtype=int)
    s = symbol.upper()
    if s == "ZC":
        plant = np.isin(m, [4, 5]).astype(int)
        harvest = np.isin(m, [9, 10, 11]).astype(int)
    elif s == "ZS":
        plant = np.isin(m, [5, 6]).astype(int)
        harvest = np.isin(m, [9, 10, 11]).astype(int)
    elif s == "ZW":
        plant = np.isin(m, [10, 11]).astype(int)
        harvest = np.isin(m, [6, 7, 8]).astype(int)
    else:
        plant[:] = 0
        harvest[:] = 0
    return pd.DataFrame({"is_plant": plant, "is_harvest": harvest}, index=index)


def tag_roll_window_from_bars(bars: pd.DataFrame, *, roll_window_days: int = 5) -> pd.Series:
    """Flag dates where the minimum days-to-expiry across active contracts is within roll window.
    bars must have columns [date, expiry]. Index can be anything.
    """
    if bars.empty:
        return pd.Series(dtype=int, name="roll_window")
    tmp = add_days_to_expiry(bars)
    g = tmp.groupby("date")["dte"].min()
    flag = (g <= roll_window_days).astype(int)
    flag.index = pd.to_datetime(flag.index)
    flag.name = "roll_window"
    return flag


def tag_storage_tightness(log_spreads: pd.DataFrame, *, first_pair_only: bool = True, z_span: int = 60, z_thresh: float = 1.0) -> pd.Series:
    """Proxy for storage tightness using the front calendar log spread.
    Tight when front spread is below -z_thresh in EWMA z units (backwardation).
    """
    if log_spreads.empty:
        return pd.Series(dtype=int, name="storage_tight")
    if first_pair_only:
        col = log_spreads.columns[0]
        s = log_spreads[col].copy()
    else:
        s = log_spreads.mean(axis=1)
    mu = s.ewm(span=z_span, adjust=False).mean()
    var = (s - mu).pow(2).ewm(span=z_span, adjust=False).mean()
    z = (s - mu) / np.sqrt(var.replace(0.0, np.nan))
    flag = (z <= -abs(z_thresh)).astype(int)
    flag.name = "storage_tight"
    return flag


def tag_shipping_stress(index: pd.DatetimeIndex, proxy_series: Optional[pd.Series] = None, *, z_thresh: float = 1.5) -> pd.Series:
    """Shipping or freight stress tag.
    If a proxy_series (e.g., Baltic Dry Index returns) is provided, flag when z-score < -z_thresh.
    Else return zeros.
    """
    if proxy_series is None or proxy_series.empty:
        return pd.Series(0, index=index, name="shipping_stress")
    s = proxy_series.reindex(index).fillna(method="ffill")
    mu = s.ewm(span=60, adjust=False).mean()
    var = (s - mu).pow(2).ewm(span=60, adjust=False).mean()
    z = (s - mu) / np.sqrt(var.replace(0.0, np.nan))
    return (z <= -abs(z_thresh)).astype(int).rename("shipping_stress")


# -----------------------------
# Aggregator
# -----------------------------

def build_state_tags(
    *,
    index: pd.DatetimeIndex,
    symbol: str,
    bars: Optional[pd.DataFrame] = None,
    log_spreads: Optional[pd.DataFrame] = None,
    shipping_proxy: Optional[pd.Series] = None,
    roll_window_days: int = 5,
) -> pd.DataFrame:
    tags = {
        "month": tag_month_of_year(index),
        "refinery_maint": tag_refinery_maintenance(index),
        "shipping_stress": tag_shipping_stress(index, shipping_proxy),
    }
    ph = tag_plant_harvest(index, symbol)
    for c in ph.columns:
        tags[c] = ph[c]
    if bars is not None:
        rw = tag_roll_window_from_bars(bars, roll_window_days=roll_window_days)
        tags["roll_window"] = rw.reindex(index, fill_value=0)
    if log_spreads is not None:
        st = tag_storage_tightness(log_spreads)
        tags["storage_tight"] = st.reindex(index, fill_value=0)

    df = pd.concat(tags.values(), axis=1)
    # Ensure integer 0/1 for flags except month
    for c in [x for x in df.columns if x != "month"]:
        df[c] = df[c].fillna(0).astype(int)
    return df


__all__ = [
    "tag_month_of_year",
    "tag_refinery_maintenance",
    "tag_plant_harvest",
    "tag_roll_window_from_bars",
    "tag_storage_tightness",
    "tag_shipping_stress",
    "build_state_tags",
]
