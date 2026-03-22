# Root-level copy for notebooks
# source is src/data/spreads.py - keep in sync.
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def calendar_prices(surface: pd.DataFrame, *, log: bool = False) -> pd.DataFrame:
    """Adjacent calendar prices.  Columns: '<far>-<near>'."""
    if surface.empty or surface.shape[1] < 2:
        return surface.iloc[0:0]
    cols = list(surface.columns)
    arr  = surface[cols].to_numpy(dtype=float)
    if log:
        arr = np.where(arr > 0, np.log(arr), np.nan)
    cal     = arr[:, 1:] - arr[:, :-1]
    cal_cols = [f"{cols[i]}-{cols[i - 1]}" for i in range(1, len(cols))]
    return pd.DataFrame(cal, index=surface.index, columns=cal_cols)


def butterfly_prices(surface: pd.DataFrame, *, log: bool = False) -> pd.DataFrame:
    """1:-2:1 butterfly prices.  Columns: '<near>~<centre>~<far>'."""
    if surface.empty or surface.shape[1] < 3:
        return surface.iloc[0:0]
    cols = list(surface.columns)
    arr  = surface[cols].to_numpy(dtype=float)
    if log:
        arr = np.where(arr > 0, np.log(arr), np.nan)
    fly      = arr[:, :-2] - 2.0 * arr[:, 1:-1] + arr[:, 2:]
    fly_cols = [
        f"{cols[i]}~{cols[i + 1]}~{cols[i + 2]}"
        for i in range(len(cols) - 2)
    ]
    return pd.DataFrame(fly, index=surface.index, columns=fly_cols)


def calendar_legs(expiries: List) -> List[Tuple[str, str]]:
    if len(expiries) < 2:
        return []
    return [(str(expiries[i]), str(expiries[i + 1])) for i in range(len(expiries) - 1)]


def butterfly_legs(expiries: List) -> List[Tuple[str, str, str]]:
    if len(expiries) < 3:
        return []
    return [
        (str(expiries[i]), str(expiries[i + 1]), str(expiries[i + 2]))
        for i in range(len(expiries) - 2)
    ]


def calendar_legs_from_name(combo_name: str) -> List[Tuple[str, int]]:
    """Parse 'FAR-NEAR' calendar name (ISO dates) → [(far, -1), (near, +1)]."""
    parts = combo_name.split("-")
    if len(parts) == 6:
        far  = f"{parts[0]}-{parts[1]}-{parts[2]}"
        near = f"{parts[3]}-{parts[4]}-{parts[5]}"
        return [(far, -1), (near, 1)]
    if len(parts) == 2:
        return [(parts[0], -1), (parts[1], 1)]
    mid = len(parts) // 2
    return [("-".join(parts[:mid]), -1), ("-".join(parts[mid:]), 1)]


def butterfly_legs_from_name(combo_name: str) -> List[Tuple[str, int]]:
    """Parse 'near~centre~far' butterfly name → [(near,+1),(centre,-2),(far,+1)]."""
    parts = combo_name.split("~")
    if len(parts) != 3:
        return []
    near, centre, far = parts
    return [(near, 1), (centre, -2), (far, 1)]


def ewma_z(df: pd.DataFrame, span: int = 60) -> pd.DataFrame:
    mu  = df.ewm(span=span, adjust=False).mean()
    var = (df - mu).pow(2).ewm(span=span, adjust=False).mean()
    return (df - mu) / np.sqrt(var)


__all__ = [
    "calendar_prices",
    "butterfly_prices",
    "calendar_legs",
    "butterfly_legs",
    "calendar_legs_from_name",
    "butterfly_legs_from_name",
    "ewma_z",
]