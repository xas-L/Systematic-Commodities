# src/signals/attribution.py
# P&L attribution by factor with residualisation (Gram–Schmidt style)
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Utilities
# -----------------------------

def _safe_nan_to_num(a: np.ndarray) -> np.ndarray:
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def residualise_columns(df: pd.DataFrame, order: List[str]) -> pd.DataFrame:
    """Orthogonalise columns in the supplied order, returning residuals.

    The first column remains as-is. Each subsequent column has a linear regression on prior columns removed.
    """
    if df.empty or not order:
        return df.copy()
    X = df[order].to_numpy(dtype=float)
    X = _safe_nan_to_num(X)
    R = np.zeros_like(X)
    for j in range(X.shape[1]):
        y = X[:, j]
        if j == 0:
            R[:, j] = y
        else:
            A = R[:, :j]
            # Solve least squares A b = y, subtract projection
            try:
                b, *_ = np.linalg.lstsq(A, y, rcond=None)
                R[:, j] = y - A @ b
            except Exception:
                R[:, j] = y
    return pd.DataFrame(R, index=df.index, columns=order)


# -----------------------------
# Attribution
# -----------------------------

def pnl_attribution(
    *,
    factor_pnls: Dict[str, pd.Series],
    order: List[str],
) -> pd.DataFrame:
    """Return per-factor P&L (residualised) and summary stats.

    Args:
        factor_pnls: mapping factor_name -> P&L series (aligned index)
        order: attribution order (e.g., ["carry", "momo", "seas", "pca", "coint"]).

    Returns:
        DataFrame with columns [bp_per_trade, ic, hit_rate, sharpe, t_stat, net_bp] per factor
        Note: this function computes residual P&L; net and cost application should be handled upstream.
    """
    # Align and combine
    df = pd.concat(factor_pnls.values(), axis=1)
    df.columns = list(factor_pnls.keys())
    df = df.reindex(columns=order)
    # Residualise in order
    R = residualise_columns(df, order)

    # Compute basic stats per column
    stats = []
    for col in R.columns:
        pnl = R[col].fillna(0.0)
        # bp/trade proxy: mean absolute pnl sign times 10000; you should replace with true per-trade stats in backtester
        trades = (pnl != 0.0).sum()
        bp_per_trade = 10000.0 * (pnl.sum() / max(trades, 1))
        ic = pnl.corr(np.sign(pnl)) if pnl.std(ddof=1) > 0 else 0.0
        hit_rate = float((pnl > 0).mean())
        mean = pnl.mean()
        std = pnl.std(ddof=1)
        sharpe = mean / std * np.sqrt(252.0) if std > 0 else 0.0
        t_stat = mean / (std / np.sqrt(max(len(pnl), 1))) if std > 0 else 0.0
        stats.append([bp_per_trade, ic, hit_rate, sharpe, t_stat, pnl.sum() * 10000.0])

    out = pd.DataFrame(stats, index=R.columns,
                       columns=["bp_per_trade", "ic", "hit_rate", "sharpe", "t_stat", "net_bp"])
    return out


__all__ = [
    "residualise_columns",
    "pnl_attribution",
]
