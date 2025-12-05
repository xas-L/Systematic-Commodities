# src/models/pca.py
# PCA factor model over log-adjacent spreads (or any wide feature panel)
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from ..core.types import HealthReport
from ..core.health import nan_fraction, corr_drift


@dataclass
class PCAParams:
    n_components: int = 3
    standardize: bool = True   # z-score columns on the fit window
    center_only: bool = False  # if True, subtract mean but do not scale
    ewma_span: int = 60        # for adaptive z-scoring of factor values in signal()
    min_samples: int = 20     # minimal rows to fit


class PCAFactor:
    """Principal Components on a panel X (rows=time, cols=features).

    - Stores feature means/stds from the fit window (for reproducible transform)
    - Aligns transform input to the original feature set
    - Produces factor scores Z and EWMA z-scored signals S
    - Health metrics: NaN fraction, min variance, correlation-drift vs fit reference
    """

    def __init__(self, n_components: int = 3, ewma_span: int = 60,
                 standardize: bool = True, center_only: bool = False):
        self.params = PCAParams(
            n_components=n_components,
            standardize=standardize,
            center_only=center_only,
            ewma_span=ewma_span,
        )
        self._pca: Optional[PCA] = None
        self._feat_cols: Optional[list[str]] = None
        self._mu: Optional[pd.Series] = None
        self._sigma: Optional[pd.Series] = None
        self._refZ: Optional[pd.DataFrame] = None  # factor panel on fit window (for drift)

    # -------------------------
    # Internal helpers
    # -------------------------
    def _fit_scaler(self, X: pd.DataFrame) -> None:
        mu = X.mean(axis=0)
        if self.params.standardize and not self.params.center_only:
            sigma = X.std(axis=0).replace(0.0, np.nan)
            sigma = sigma.fillna(1.0)
        else:
            sigma = pd.Series(1.0, index=X.columns)
        self._mu, self._sigma = mu, sigma

    def _apply_scaler(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self._mu is not None and self._sigma is not None
        Xc = X.subtract(self._mu, axis=1)
        if self.params.standardize and not self.params.center_only:
            Xc = Xc.divide(self._sigma, axis=1)
        return Xc

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self._feat_cols is not None and self._mu is not None
        # Ensure columns match the fit set; missing -> fill with mean; extras -> drop
        X2 = X.reindex(columns=self._feat_cols)
        # Fill missing with training mean
        for c in X2.columns:
            if X2[c].isna().all():
                X2[c] = self._mu[c]
        return X2

    @staticmethod
    def _ewma_z_df(Z: pd.DataFrame, span: int) -> pd.DataFrame:
        def _z(s: pd.Series) -> pd.Series:
            mu = s.ewm(span=span, adjust=False).mean()
            var = (s - mu).pow(2).ewm(span=span, adjust=False).mean()
            return (s - mu) / np.sqrt(var.replace(0.0, np.nan))
        return Z.apply(_z)

    # -------------------------
    # Public API
    # -------------------------
    def fit(self, X: pd.DataFrame, **_) -> None:
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("PCAFactor expects X with DatetimeIndex")
        # Drop rows with any NaNs for the fit window
        X_fit = X.dropna(axis=0, how="any")
        if len(X_fit) < self.params.min_samples:
            raise ValueError(f"Insufficient samples to fit PCA: {len(X_fit)} < {self.params.min_samples}")
        self._feat_cols = list(X_fit.columns)
        self._fit_scaler(X_fit)
        Xs = self._apply_scaler(X_fit)
        pca = PCA(n_components=min(self.params.n_components, Xs.shape[1]))
        pca.fit(Xs.values)
        self._pca = pca
        # Keep reference factor scores on the fit window for drift checks
        Z_ref = pd.DataFrame(pca.transform(Xs.values), index=X_fit.index,
                             columns=[f"pc{i+1}" for i in range(pca.n_components_)])
        self._refZ = Z_ref

    def transform(self, X: pd.DataFrame, **_) -> pd.DataFrame:
        if self._pca is None or self._feat_cols is None:
            raise RuntimeError("PCAFactor not fitted")
        X_aligned = self._align_features(X)
        X_aligned = X_aligned.fillna(method="ffill")
        Xs = self._apply_scaler(X_aligned)
        Z = pd.DataFrame(self._pca.transform(Xs.values), index=X.index,
                         columns=[f"pc{i+1}" for i in range(self._pca.n_components_)])
        return Z

    def signal(self, Z: pd.DataFrame, **_) -> pd.DataFrame:
        # EWMA z-scored factor values (adaptive)
        return self._ewma_z_df(Z, span=self.params.ewma_span)

    def health(self, Z: pd.DataFrame, **_) -> HealthReport:
        # Simple metrics: overall NaN fraction, min variance, corr drift vs reference
        metrics = {"nan_frac": nan_fraction(Z)}
        var_min = float(Z.var(numeric_only=True).min()) if not Z.empty else 0.0
        metrics["var_min"] = var_min
        drift = float("nan")
        if self._refZ is not None and not Z.empty:
            # Align columns just in case
            cols = [c for c in Z.columns if c in self._refZ.columns]
            if cols:
                drift = corr_drift(Z[cols], self._refZ[cols])
        metrics["corr_drift"] = drift
        ok = True
        if metrics["nan_frac"] > 0.05:
            ok = False
        if var_min < 1e-8:
            ok = False
        return HealthReport(component="pca", ok=ok, metrics=metrics)

    # -------------------------
    # Extras
    # -------------------------
    @property
    def explained_variance_ratio_(self) -> Optional[np.ndarray]:
        return None if self._pca is None else self._pca.explained_variance_ratio_

    def loadings(self) -> pd.DataFrame:
        if self._pca is None or self._feat_cols is None:
            raise RuntimeError("PCAFactor not fitted")
        comps = self._pca.components_  # shape (n_components, n_features)
        return pd.DataFrame(comps, index=[f"pc{i+1}" for i in range(self._pca.n_components_)], columns=self._feat_cols)


__all__ = ["PCAFactor", "PCAParams"]
