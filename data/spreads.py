# to build tradable calendar and butterfly panels from a curve surface
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


# Calendar (adjacent) and butterfly (1:-2:1) pricing


def calendar_prices(surface: pd.DataFrame, *, log: bool = False) -> pd.DataFrame:
    """Adjacent calendar prices from a wide curve surface (cols=expiries).

    If log=True, returns log(F_{i+1}) - log(F_{i}); else returns F_{i+1} - F_{i}.
    Columns are labelled as "<exp_{i+1}>-<exp_i>".
    """
    if surface.empty or surface.shape[1] < 2:
        return surface.iloc[0:0]
    cols = list(surface.columns)
    arr = surface[cols].to_numpy(dtype=float)
    if log:
        # Guard against non-positive values
        arr = np.where(arr > 0, np.log(arr), np.nan)
    cal = arr[:, 1:] - arr[:, :-1]
    cal_cols = [f"{cols[i]}-{cols[i-1]}" for i in range(1, len(cols))]
    return pd.DataFrame(cal, index=surface.index, columns=cal_cols)


def butterfly_prices(surface: pd.DataFrame, *, log: bool = False) -> pd.DataFrame:
    """Three-leg fly prices using 1 * i - 2 * (i+1) + 1 * (i+2).

    If log=True, uses log prices; else uses linear prices.
    Columns are labelled as "<exp_{i+2}>-2*<exp_{i+1}>+<exp_i>".
    """
    if surface.empty or surface.shape[1] < 3:
        return surface.iloc[0:0]
    cols = list(surface.columns)
    arr = surface[cols].to_numpy(dtype=float)
    if log:
        arr = np.where(arr > 0, np.log(arr), np.nan)
    fly = arr[:, 2:] - 2.0 * arr[:, 1:-1] + arr[:, :-2]
    fly_cols = [f"{cols[i+1]}_fly" for i in range(len(cols) - 2)]  # centre month label
    return pd.DataFrame(fly, index=surface.index, columns=fly_cols)



# Exchange combo mapping helpers

def calendar_legs(expiries: List[str | pd.Timestamp]) -> List[Tuple[str, str]]:
    """Return adjacent expiry pairs for exchange calendar combos."""
    if len(expiries) < 2:
        return []
    return [(str(expiries[i]), str(expiries[i + 1])) for i in range(len(expiries) - 1)]


def butterfly_legs(expiries: List[str | pd.Timestamp]) -> List[Tuple[str, str, str]]:
    """Return triplets (i, i+1, i+2) for 1:-2:1 flies."""
    if len(expiries) < 3:
        return []
    return [(str(expiries[i]), str(expiries[i + 1]), str(expiries[i + 2])) for i in range(len(expiries) - 2)]


def calendar_legs_from_name(combo_name: str) -> List[Tuple[str, int]]:
    """Convert calendar combo name to legs.
    
    Expects format like '2024-03-01-2024-02-01' 
    Returns: [(expiry1, -1), (expiry2, 1)] for calendar spread
    """
    parts = combo_name.split('-')
    
    if len(parts) == 6:  # YYYY-MM-DD-YYYY-MM-DD
        expiry1 = f"{parts[0]}-{parts[1]}-{parts[2]}"
        expiry2 = f"{parts[3]}-{parts[4]}-{parts[5]}"
        return [(expiry1, -1), (expiry2, 1)]
    elif len(parts) == 2:  # Simple two-date format
        return [(parts[0], -1), (parts[1], 1)]
    else:
        # Try to be flexible - split in middle
        mid = len(parts) // 2
        expiry1 = '-'.join(parts[:mid])
        expiry2 = '-'.join(parts[mid:])
        return [(expiry1, -1), (expiry2, 1)]


def butterfly_legs_from_name(combo_name: str) -> List[Tuple[str, int]]:
    """Convert butterfly combo name to legs (1:-2:1 structure).
    
    Placeholder - returns empty list for now.
    """
    # For basic calendar trading, butterflies not used
    return []



# normalisation

def ewma_z(df: pd.DataFrame, span: int = 60) -> pd.DataFrame:
    """Column-wise EWMA z-scores."""
    mu = df.ewm(span=span, adjust=False).mean()
    var = (df - mu).pow(2).ewm(span=span, adjust=False).mean()
    return (df - mu) / np.sqrt(var)


__all__ = [
    "calendar_prices",
    "butterfly_prices",
    "calendar_legs",
    "butterfly_legs",
    "calendar_legs_from_name",  # addition
    "butterfly_legs_from_name", # addition
    "ewma_z",
]