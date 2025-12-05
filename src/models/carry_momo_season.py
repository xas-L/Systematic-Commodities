# src/models/carry_momo_season.py
# Carry, momentum, and deterministic seasonality features with adaptive z-scored signals
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..core.types import HealthReport
from ..core.health import nan_fraction
from ..data.seasonality import seasonality_features


@dataclass
class CMSParams:
    k_momo: int = 20
    ewma_span: int = 60
    seasonality: bool = True
    seas_t_min: float = 2.0


class CarryMomentumSeasonality:
    """Feature builder on top of log-adjacent spreads (and optionally roll-yield/flies)."""

    def __init__(self, k_momo: int = 20, ewma_span: int = 60, seasonality: bool = True, seas_t_min: float = 2.0):
        self.params = CMSParams(k_momo=k_momo, ewma_span=ewma_span, seasonality=seasonality, seas_t_min=seas_t_min)
        self._fit_columns: Optional[list[str]] = None

    # -------------------------
    # Helpers
    # -------------------------
    @staticmethod
    def _ewma_z_df(df: pd.DataFrame, span: int) -> pd.DataFrame:
        def z(s: pd.Series) -> pd.Series:
            mu = s.ewm(span=span, adjust=False).mean()
            var = (s - mu).pow(2).ewm(span=span, adjust=False).mean()
            return (s - mu) / np.sqrt(var.replace(0.0, np.nan))
        return df.apply(z)

    def _seasonal_resid(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.params.seasonality:
            return X.copy()
        return seasonality_features(X, t_min=self.params.seas_t_min)

    # -------------------------
    # API
    # -------------------------
    def fit(self, X: pd.DataFrame, **_):
        # Accept any index, convert to DatetimeIndex if needed
        if not isinstance(X.index, pd.DatetimeIndex):
            X = X.copy()
            X.index = pd.to_datetime(X.index)
        self._fit_columns = list(X.columns)

    def transform(self, X: pd.DataFrame, **_) -> pd.DataFrame:
        if self._fit_columns is None:
            raise RuntimeError("CarryMomentumSeasonality not fitted")
        X2 = X.reindex(columns=self._fit_columns).copy()
        # Carry proxy: level (residualised for seasonality if enabled)
        carry = self._seasonal_resid(X2).add_prefix("carry_")
        # Momentum: k-day change on spreads
        momo = X2.diff(self.params.k_momo).add_prefix(f"momo{self.params.k_momo}_")
        # Combine
        Z = pd.concat([carry, momo], axis=1)
        return Z

    def signal(self, Z: pd.DataFrame, **_) -> pd.DataFrame:
        return self._ewma_z_df(Z, span=self.params.ewma_span)

    def health(self, Z: pd.DataFrame, **_) -> HealthReport:
        metrics = {"nan_frac": nan_fraction(Z)}
        var_min = float(Z.var(numeric_only=True).min()) if not Z.empty else 0.0
        metrics["var_min"] = var_min
        ok = (metrics["nan_frac"] <= 0.05) and (var_min >= 1e-8)
        return HealthReport(component="carry_momo_season", ok=ok, metrics=metrics)


__all__ = ["CarryMomentumSeasonality", "CMSParams"]
