# src/signals/sizing.py
# Map model signals to exchange combo orders with risk-aware sizing
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.utils import clipped, safe_div


@dataclass
class SizingConfig:
    z_clip: float = 3.0
    risk_per_curve_usd: float = 100000.0
    per_trade_notional_cap_usd: float = 150000.0
    vol_target_enabled: bool = True
    vol_lookback_days: int = 60
    annual_vol_target: float = 0.10


@dataclass
class ComboSpec:
    symbol: str           # root, e.g., CL
    legs: List[Tuple[str, int]]  # list of (expiry_iso, ratio)


@dataclass
class QuoteSpec:
    combo_mid: float
    combo_half_spread: float
    lot_value: float      # tick_value or value per 1.0 price unit times multiplier


# -----------------------------
# Vol scaling
# -----------------------------

def estimate_realised_vol(series: pd.Series, lookback: int = 60) -> float:
    s = series.dropna().pct_change().tail(lookback)
    if len(s) < 2:
        return 0.0
    daily_vol = float(np.std(s, ddof=1))
    return daily_vol * np.sqrt(252.0)


# -----------------------------
# Sizing logic
# -----------------------------

def size_from_signal(
    *,
    signal: float,
    combo_price_series: pd.Series,
    lot_value: float,
    cfg: SizingConfig,
) -> int:
    """Map a single scalar signal to an integer lot size subject to risk caps and vol target.

    - signal: z-scored factor average mapped to this combo (+/-)
    - combo_price_series: historical mid prices of the combo (for vol estimate)
    - lot_value: monetary value per 1.0 price move per lot (tick_value or multiplier-adjusted)
    """
    if not np.isfinite(signal):
        return 0
    z = clipped(signal, -cfg.z_clip, cfg.z_clip)
    # Base desired notional linear in z
    base_notional = z * cfg.risk_per_curve_usd

    # Vol targeting
    if cfg.vol_target_enabled:
        ann_vol = estimate_realised_vol(combo_price_series, cfg.vol_lookback_days)
        if ann_vol > 1e-9:
            base_notional = base_notional * (cfg.annual_vol_target / ann_vol)

    # Convert notional to lots via lot_value
    lots = int(np.sign(base_notional) * np.floor(abs(base_notional) / max(lot_value, 1e-9)))

    # Per-trade cap
    max_lots = int(np.floor(cfg.per_trade_notional_cap_usd / max(lot_value, 1e-9)))
    lots = int(np.sign(lots) * min(abs(lots), max_lots))
    return lots


def map_signals_to_combos(
    *,
    signals: pd.DataFrame,
    to_combo_legs: Dict[str, List[Tuple[str, int]]],
    quotes: Dict[str, QuoteSpec],
    cfg: SizingConfig,
    price_history: Optional[Dict[str, pd.Series]] = None,
) -> List[Tuple[str, List[Tuple[str, int]], int]]:
    """Map a signal matrix to executable combo intents.

    Args:
        signals: DataFrame (index=time, columns=combo identifiers like "e2-e1" or "e3-e2-e1") of z-scored signals
        to_combo_legs: mapping combo name -> list of (expiry_iso, ratio)
        quotes: mapping combo name -> QuoteSpec (mid, half-spread, lot_value)
        cfg: sizing parameters
        price_history: optional price series per combo to estimate vol; else use constant vol proxy

    Returns:
        list of (symbol, legs, lots) representing a desired order (sign gives side)
    """
    if signals.empty:
        return []

    # Use the latest row
    last = signals.iloc[-1]
    intents: List[Tuple[str, List[Tuple[str, int]], int]] = []
    for combo_name, sig in last.dropna().items():
        legs = to_combo_legs.get(combo_name)
        q = quotes.get(combo_name)
        if legs is None or q is None:
            continue
        px_hist = price_history.get(combo_name) if price_history else pd.Series([q.combo_mid])
        lots = size_from_signal(signal=float(sig), combo_price_series=px_hist, lot_value=q.lot_value, cfg=cfg)
        if lots != 0:
            intents.append((q.__dict__.get("symbol", ""), legs, lots))
    return intents


__all__ = [
    "SizingConfig",
    "ComboSpec",
    "QuoteSpec",
    "estimate_realised_vol",
    "size_from_signal",
    "map_signals_to_combos",
]
