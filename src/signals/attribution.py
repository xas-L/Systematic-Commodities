"""
Destination: src/signals/attribution.py

P&L attribution by factor with residualisation (Gram-Schmidt style).

Two public functions:
  - residualise_columns: orthogonalise a set of P&L series in a given order
  - pnl_attribution:     produce a stats table per factor

IC definition (corrected from original):
  Information Coefficient = Spearman rank correlation of signal[t] vs return[t+1].
  The original code computed corr(pnl, sign(pnl)) which is tautological.
  Here we require signals to be passed alongside pnl to compute real IC.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


#  Gram-Schmidt residualisation 

def _safe_nan_to_num(a: np.ndarray) -> np.ndarray:
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def residualise_columns(df: pd.DataFrame, order: List[str]) -> pd.DataFrame:
    """
    Orthogonalise columns in the supplied order, returning residuals.

    The first column is unchanged. Each subsequent column has the linear
    projection onto all prior columns subtracted.  This is a sequential
    Gram-Schmidt decomposition of the P&L panel so that each factor's
    contribution is measured net of everything that came before it in `order`.
    """
    if df.empty or not order:
        return df.copy()

    cols = [c for c in order if c in df.columns]
    if not cols:
        return df.copy()

    X = df[cols].to_numpy(dtype=float)
    X = _safe_nan_to_num(X)
    R = np.zeros_like(X)

    for j in range(X.shape[1]):
        y = X[:, j]
        if j == 0:
            R[:, j] = y
        else:
            A = R[:, :j]
            try:
                b, *_ = np.linalg.lstsq(A, y, rcond=None)
                R[:, j] = y - A @ b
            except Exception:
                R[:, j] = y

    return pd.DataFrame(R, index=df.index, columns=cols)


#  Attribution table 

def _ic_spearman(signal: Optional[pd.Series], pnl: pd.Series) -> float:
    """
    Spearman rank IC: corr(signal[t], pnl[t+1]).

    pnl[t+1] is the realised P&L of the position entered on signal[t],
    which is exactly what the backtester computes (T+1 execution).
    So we align signal[t] vs pnl[t] directly (the shift is already baked in).

    Returns NaN if there is insufficient data.
    """
    if signal is None or signal.empty or pnl.empty:
        return float("nan")

    aligned = pd.concat([signal.rename("sig"), pnl.rename("pnl")], axis=1).dropna()
    if len(aligned) < 10:
        return float("nan")

    rho, _ = spearmanr(aligned["sig"], aligned["pnl"])
    return float(rho) if math.isfinite(rho) else float("nan")


def pnl_attribution(
    *,
    factor_pnls: Dict[str, pd.Series],
    order: List[str],
    factor_signals: Optional[Dict[str, pd.Series]] = None,
) -> pd.DataFrame:
    """
    Return per-factor P&L stats table after Gram-Schmidt residualisation.

    Parameters
    ---
    factor_pnls : dict
        factor_name → daily P&L series (aligned on the same DatetimeIndex).
    order : list of str
        Attribution order, e.g. ["carry", "momo", "seasonality"].
        The first factor absorbs the most variance; later factors are net.
    factor_signals : dict, optional
        factor_name → signal series used to generate pnl (for true IC).
        If omitted, IC is reported as NaN.

    Returns
    ---
    DataFrame with columns:
        total_pnl   - cumulative P&L for this factor (residualised)
        daily_mean  - mean daily P&L
        daily_std   - std of daily P&L
        sharpe      - annualised Sharpe (daily mean / daily std * sqrt(252))
        t_stat      - t-statistic of mean P&L (mean / (std / sqrt(N)))
        hit_rate    - fraction of days with positive P&L
        ic          - Spearman rank IC (signal vs realised P&L)
    """
    if not factor_pnls:
        return pd.DataFrame()

    # Align all series on a common index (inner join → only days where all factors traded)
    df = pd.DataFrame(factor_pnls).reindex(columns=order)

    # Residualise
    R = residualise_columns(df, order)

    rows = []
    for col in R.columns:
        pnl = R[col].fillna(0.0)
        n   = len(pnl)
        mu  = pnl.mean()
        std = pnl.std(ddof=1) if n > 1 else 0.0

        sharpe  = (mu / std * np.sqrt(252.0)) if std > 1e-12 else 0.0
        t_stat  = (mu / (std / np.sqrt(max(n, 1)))) if std > 1e-12 else 0.0
        hit     = float((pnl > 0).mean())

        # IC: use provided signal if available
        sig = factor_signals.get(col) if factor_signals else None
        ic  = _ic_spearman(sig, pnl)

        rows.append({
            "factor":     col,
            "total_pnl":  pnl.sum(),
            "daily_mean": mu,
            "daily_std":  std,
            "sharpe":     sharpe,
            "t_stat":     t_stat,
            "hit_rate":   hit,
            "ic":         ic,
        })

    return pd.DataFrame(rows).set_index("factor")


__all__ = [
    "residualise_columns",
    "pnl_attribution",
]
