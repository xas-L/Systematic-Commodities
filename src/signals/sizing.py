"""
Destination: src/signals/sizing.py

Vol-targeted position sizing for combo (calendar/butterfly) strategies.

The sizing logic:
  1. Estimate realised daily vol of the combo price series (rolling window).
  2. Annualise to get dollar vol per lot: daily_vol * sqrt(252) * lot_value.
  3. Target lots = (annual_vol_target * risk_per_curve_usd) / dollar_vol_per_lot.
  4. Scale by clipped signal z-score (signal / z_clip so max lots = target at 3-sigma).
  5. Cap at per_trade_notional_cap_usd.
  6. Floor to 0 if |target| < 0.5 lots (i.e. round down small sizes).

Ensuring:
  - At zero signal → zero lots  (no position)
  - At 1-sigma signal → ~1/3 of target lots  (modest position)
  - At 3-sigma (clipped max) → full target lots

For a CL calendar spread with:
  - daily spread vol ≈ $0.20/bbl, lot_value = 1000
  - dollar vol/lot = 0.20 * sqrt(252) * 1000 ≈ $3,175/lot/year
  - risk budget = $100k, vol target = 10%
  - target lots = (0.10 * 100_000) / 3_175 ≈ 3 lots
  - At 1-sigma signal: ~1 lot, at 3-sigma: ~3 lots

Not heroic and non-zero.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class SizingConfig:
    z_clip: float = 3.0
    risk_per_curve_usd: float = 100_000.0
    per_trade_notional_cap_usd: float = 150_000.0
    vol_target_enabled: bool = True
    vol_lookback_days: int = 60
    annual_vol_target: float = 0.10
    min_vol_floor: float = 1e-4          # floor on spread vol to prevent division explosion
    fallback_lots: int = 1               # used when vol series is too short to estimate


def size_from_signal(
    signal: float,
    combo_price_series: pd.Series,
    lot_value: float,
    cfg: SizingConfig,
) -> int:
    """
    Return signed integer lot size for a single combo given a z-scored signal.

    Parameters
    ---
    signal : float
        EWMA z-scored signal (carry, momo, or blended). Typically in [-3, 3].

    combo_price_series : pd.Series
        Historical daily mid prices for this combo (used to estimate vol).
        Should be the full series up to the current bar (no look-ahead).

    lot_value : float
        Dollar value per point per lot (e.g., 1000 for CL).

    cfg : SizingConfig
        Risk and vol-target parameters.

    Returns
    ---
    int
        Signed lot size. Positive = long, negative = short, 0 = flat.
    """
    if not math.isfinite(signal):
        return 0

    #  1. Clip signal to [-z_clip, +z_clip] to prevent extreme sizing on outliers
    sig = float(np.clip(signal, -cfg.z_clip, cfg.z_clip))

    #  2. Estimate realised spread vol + annualise to get dollar vol per lot
    target_lots: float

    if cfg.vol_target_enabled and len(combo_price_series) >= max(cfg.vol_lookback_days // 2, 5):
        # Use the last `vol_lookback_days` of data to estimate daily vol
        window = combo_price_series.dropna().iloc[-cfg.vol_lookback_days:]
        daily_returns = window.diff().dropna()

        if len(daily_returns) >= 5:
            daily_vol = float(daily_returns.std(ddof=1))
        else:
            daily_vol = float(window.std(ddof=1)) if len(window) > 1 else 0.0

        # Floor vol to prevent exploding lot sizes on near-zero-vol spreads
        daily_vol = max(abs(daily_vol), cfg.min_vol_floor)

        # Dollar vol per lot per year
        annual_dollar_vol_per_lot = daily_vol * math.sqrt(252.0) * lot_value

        # ── 3. Target lots at full signal (z_clip sigma) ──────────────────
        full_target = (cfg.annual_vol_target * cfg.risk_per_curve_usd) / annual_dollar_vol_per_lot

        # ── 4. Scale by signal fraction (sig / z_clip so ±1 gives ±1/3) ──
        scaled = full_target * (sig / cfg.z_clip)
        target_lots = scaled

    else:
        # If not enough history for vol estimate then use a conservative fallback
        # of cfg.fallback_lots lots, signed by signal direction
        if abs(sig) < 0.5:
            return 0
        target_lots = math.copysign(cfg.fallback_lots, sig)

    #  5. Round toward zero (truncate) 
    # Truncating (not rounding) means we never oversize
    lots = math.trunc(target_lots)

    # Tiny residual signals become 0 (avoid 1-lot churn on noise) 
    if abs(target_lots) < 0.5:
        return 0

    #  6. Per-trade notional cap 
    if lots != 0:
        last_price = abs(float(combo_price_series.dropna().iloc[-1])) if len(combo_price_series.dropna()) > 0 else 1.0
        last_price = max(last_price, 1e-6)    # guard against zero price
        notional = abs(lots) * last_price * lot_value
        if notional > cfg.per_trade_notional_cap_usd:
            cap_lots = cfg.per_trade_notional_cap_usd / (last_price * lot_value)
            lots = math.trunc(math.copysign(cap_lots, lots))

    return int(lots)
