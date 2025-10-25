# src/models/nelson_siegel.py
# Optional Nelson–Siegel factorisation for futures curves (no-arb fit not enforced)
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from ..core.types import HealthReport
from ..core.health import nan_fraction


@dataclass
class NSParams:
    lambda_decay: float = 0.060  # shape
    ewma_span: int = 60          # for adaptive z of factor values
    min_tenors: int = 5


class NelsonSiegelFactor:
    """Fit Nelson–Siegel loadings to each cross-section and extract level/slope/curvature.

    y(τ) = β0 + β1 * ((1 - e^{-λτ})/(λτ)) + β2 * ((1 - e^{-λτ})/(λτ) - e^{-λτ})

    - We treat τ as time to expiry in years for each expiry column
    - For stability: fix λ (lambda_decay) and solve β via least squares per day
    - Signals are EWMA z-scored β_t
    - Health: NaN fraction and min variance
    """

    def __init__(self, lambda_decay: float = 0.060, ewma_span: int = 60, min_tenors: int = 5):
        self.params = NSParams(lambda_decay=lambda_decay, ewma_span=ewma_span, min_tenors=min_tenors)
        self._taus: Optional[np.ndarray] = None  # time-to-expiry grid in years
        self._exp_cols: Optional[list[pd.Timestamp]] = None

    # -------------------------
    # Helpers
    # -------------------------
    def _design(self, tau: np.ndarray, lam: float) -> np.ndarray:
        # Build NS loadings matrix for given tau and lambda
        with np.errstate(divide="ignore", invalid="ignore"):
            x = tau * lam
            # Avoid division by zero at tau=0 via series expansion: limit is 1
            f1 = np.where(x == 0, 1.0, (1 - np.exp(-x)) / x)
            f2 = f1 - np.exp(-x)
        # Columns: [1, f1, f2]
        return np.column_stack([np.ones_like(tau), f1, f2])

    def _taus_from_cols(self, cols: list[pd.Timestamp]) -> np.ndarray:
        # Convert expiry timestamps to relative years from the first column
        # We approximate τ by (expiry - first_expiry).days / 365.25
        base = cols[0]
        return np.array([(c - base).days / 365.25 for c in cols], dtype=float)

    def _solve_beta(self, y: np.ndarray, A: np.ndarray) -> np.ndarray:
        # Linear least squares β = argmin ||Aβ - y||
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        return beta

    @staticmethod
    def _ewma_z_df(df: pd.DataFrame, span: int) -> pd.DataFrame:
        def z(s: pd.Series) -> pd.Series:
            mu = s.ewm(span=span, adjust=False).mean()
            var = (s - mu).pow(2).ewm(span=span, adjust=False).mean()
            return (s - mu) / np.sqrt(var.replace(0.0, np.nan))
        return df.apply(z)

    # -------------------------
    # API
    # -------------------------
    def fit(self, X: pd.DataFrame, **_):
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("NelsonSiegelFactor expects X with DatetimeIndex")
        if X.shape[1] < self.params.min_tenors:
            raise ValueError("Not enough tenors to fit Nelson–Siegel")
        # Store tau grid from columns
        cols = [pd.Timestamp(c) for c in X.columns]
        self._exp_cols = cols
        self._taus = self._taus_from_cols(cols)

    def transform(self, X: pd.DataFrame, **_) -> pd.DataFrame:
        if self._taus is None or self._exp_cols is None:
            raise RuntimeError("NelsonSiegelFactor not fitted")
        # Align columns to the original expiries; drop others
        cols_ts = [pd.Timestamp(c) for c in X.columns]
        common = sorted(set(self._exp_cols).intersection(cols_ts))
        if len(common) < self.params.min_tenors:
            raise ValueError("Insufficient overlap of expiries for NS transform")
        X2 = X[common].copy()
        tau_idx = [self._exp_cols.index(c) for c in common]
        tau = self._taus[tau_idx]
        A = self._design(tau, self.params.lambda_decay)
        # Solve β per row (date)
        betas = []
        for _, row in X2.iterrows():
            y = row.values.astype(float)
            if np.any(~np.isfinite(y)):
                betas.append([np.nan, np.nan, np.nan])
                continue
            beta = self._solve_beta(y, A)
            betas.append(beta.tolist())
        Z = pd.DataFrame(betas, index=X2.index, columns=["ns_level", "ns_slope", "ns_curve"])
        return Z

    def signal(self, Z: pd.DataFrame, **_) -> pd.DataFrame:
        return self._ewma_z_df(Z, span=self.params.ewma_span)

    def health(self, Z: pd.DataFrame, **_) -> HealthReport:
        metrics = {"nan_frac": nan_fraction(Z)}
        var_min = float(Z.var(numeric_only=True).min()) if not Z.empty else 0.0
        metrics["var_min"] = var_min
        ok = (metrics["nan_frac"] <= 0.05) and (var_min >= 1e-8)
        return HealthReport(component="nelson_siegel", ok=ok, metrics=metrics)


__all__ = ["NelsonSiegelFactor", "NSParams"]
