# src/signals/sizing.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class SizingConfig:
    z_clip: float = 3.0
    risk_per_curve_usd: float = 100000.0
    per_trade_notional_cap_usd: float = 150000.0
    vol_target_enabled: bool = True
    vol_lookback_days: int = 60
    annual_vol_target: float = 0.10


def size_from_signal(
    signal: float,
    combo_price_series: pd.Series,
    lot_value: float,
    cfg: SizingConfig,
) -> int:
    """Determine lot size from signal value.
    
    FOR TESTING: Always return at least 1 lot if |signal| > 0.1
    """
    if not np.isfinite(signal):
        return 0
    
    # Clip extreme signals
    sig_clipped = np.clip(signal, -cfg.z_clip, cfg.z_clip)
    
    # FOR TESTING: If signal has any magnitude, trade at least 1 lot
    if abs(sig_clipped) > 0.1:
        base_lots = int(np.sign(sig_clipped) * max(1, abs(sig_clipped)))
    else:
        return 0
    
    # Apply notional cap
    price = combo_price_series.iloc[-1] if len(combo_price_series) > 0 else 1.0
    notional = abs(base_lots) * abs(price) * lot_value
    if notional > cfg.per_trade_notional_cap_usd:
        base_lots = int(np.sign(base_lots) * cfg.per_trade_notional_cap_usd / (abs(price) * lot_value))
    
    return int(base_lots)
