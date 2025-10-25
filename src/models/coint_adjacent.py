# src/models/coint_adjacent.py
# Engle–Granger cointegration checks on adjacent log spreads with simple p-value signals
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from ..core.types import HealthReport
from ..core.health import nan_fraction


@dataclass
class CointParams:
    lookback: int = 250
    pval_entry: float = 0.05


class CointAdjSpreads:
    """Compute rolling Engle–Granger cointegration p-values for adjacent log spreads.

    transform(X): returns DataFrame with p-values per adjacent pair
    signal(Z): returns (pval_entry - pval) so that lower p-values create positive signal
    health: NaN fraction and minimal variance (on 1 - pval)
    """

    def __init__(self, lookback: int = 250, pval_entry: float = 0.05):
        self.params = CointParams(lookback=lookback, pval_entry=pval_entry)
        self._fit_columns: Optional[list[str]] = None

    def fit(self, X: pd.DataFrame, **_):
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("CointAdjSpreads expects X with DatetimeIndex")
        self._fit_columns = list(X.columns)

    def _adjacent_pairs(self, cols: list[str]) -> list[tuple[str, str]]:
        pairs = []
        for i in range(len(cols) - 1):
            a, b = cols[i], cols[i + 1]
            pairs.append((a, b))
        return pairs

    def transform(self, X: pd.DataFrame, **_) -> pd.DataFrame:
        if self._fit_columns is None:
            raise RuntimeError("CointAdjSpreads not fitted")
        X2 = X.reindex(columns=self._fit_columns).copy()
        cols = list(X2.columns)
        pairs = self._adjacent_pairs(cols)
        out = pd.DataFrame(index=X2.index)
        lb = self.params.lookback
        for a, b in pairs:
            # rolling p-values: compute at each t using window [t-lb+1, t]
            pvals = [np.nan] * len(X2)
            s1 = X2[a].to_numpy()
            s2 = X2[b].to_numpy()
            for t in range(lb - 1, len(X2)):
                x1 = s1[t - lb + 1 : t + 1]
                x2 = s2[t - lb + 1 : t + 1]
                if np.any(~np.isfinite(x1)) or np.any(~np.isfinite(x2)):
                    continue
                try:
                    pval = coint(x1, x2, trend="c")[1]
                except Exception:
                    pval = np.nan
                pvals[t] = pval
            out[f"coint_p_{a}_{b}"] = pvals
        return out

    def signal(self, Z: pd.DataFrame, **_) -> pd.DataFrame:
        # Positive when pval below entry threshold, scaled by distance to threshold
        return (self.params.pval_entry - Z).rename(columns=lambda c: f"sig_{c}")

    def health(self, Z: pd.DataFrame, **_) -> HealthReport:
        metrics = {"nan_frac": nan_fraction(Z)}
        # use variance of (1 - p) as a crude activity proxy
        var_min = float((1.0 - Z).var(numeric_only=True).min()) if not Z.empty else 0.0
        metrics["var_min"] = var_min
        ok = (metrics["nan_frac"] <= 0.20) and (var_min >= 1e-8)
        return HealthReport(component="coint_adjacent", ok=ok, metrics=metrics)


__all__ = ["CointAdjSpreads", "CointParams"]
