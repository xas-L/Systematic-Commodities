# src/models/hub.py
# Pluggable factor-model hub (no DL). Coordinates fit/transform/signal/health across models.
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import pandas as pd

from ..core.types import HealthReport


# -----------------------------
# Base interface
# -----------------------------
class FactorModel(ABC):
    """Interface for all factor models.

    X: panel with rows = dates (DatetimeIndex), cols = features (e.g., log-adjacent spreads)
    transform(X) -> Z: model-specific factor panel (e.g., PCs, carry/momo/seas features)
    signal(Z) -> S: tradable signal panel (z-scored; same shape as Z or mapped keys)
    health(Z) -> HealthReport: basic variance/NaN/drift checks (model-defined)
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, **kwargs) -> None:  # pragma: no cover - interface only
        raise NotImplementedError

    @abstractmethod
    def transform(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def signal(self, Z: pd.DataFrame, **kwargs) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def health(self, Z: pd.DataFrame, **kwargs) -> HealthReport:  # pragma: no cover
        raise NotImplementedError


@dataclass
class HubConfig:
    order: list[str]
    weights: Optional[dict[str, float]] = None  # for blended output


class FactorModelHub:
    """Container that orchestrates a set of FactorModel implementations.

    Typical usage:
        hub = FactorModelHub(models={"pca": PCAFactor(...), "cms": CarryMomentumSeasonality(...)}, order=["cms","pca"])
        hub.fit(X_train)
        S = hub.blended_signal(X_test)
    """

    def __init__(self, models: Dict[str, FactorModel], order: list[str], weights: Optional[dict[str, float]] = None):
        self.models = dict(models)
        self.order = list(order)
        self.weights = dict(weights) if weights is not None else None
        # validate
        for name in self.order:
            if name not in self.models:
                raise KeyError(f"Model '{name}' not provided in models dict")

    # -------------------------
    # Lifecycle
    # -------------------------
    def fit(self, X: pd.DataFrame, **kwargs) -> None:
        for name in self.order:
            self.models[name].fit(X, **kwargs)

    def transform_map(self, X: pd.DataFrame, **kwargs) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for name in self.order:
            out[name] = self.models[name].transform(X, **kwargs)
        return out

    def signals_map(self, X: pd.DataFrame, **kwargs) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for name in self.order:
            Z = self.models[name].transform(X, **kwargs)
            out[name] = self.models[name].signal(Z, **kwargs)
        return out

    def health_report(self, X: pd.DataFrame, **kwargs) -> dict[str, HealthReport]:
        rep: dict[str, HealthReport] = {}
        for name in self.order:
            Z = self.models[name].transform(X, **kwargs)
            rep[name] = self.models[name].health(Z, **kwargs)
        return rep

    # -------------------------
    # Blending & utilities
    # -------------------------
    def blended_signal(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Weighted sum of model signals aligned on columns; NaNs treated as 0.

        If no weights supplied, equal-weight across models in self.order.
        """
        smap = self.signals_map(X, **kwargs)
        # union of columns
        all_cols = sorted(set().union(*[df.columns for df in smap.values()])) if smap else []
        if not all_cols:
            return X.iloc[0:0]
        if self.weights is None:
            w = {name: 1.0 for name in self.order}
        else:
            w = {name: float(self.weights.get(name, 0.0)) for name in self.order}
        total_w = sum(w.values()) or 1.0
        # accumulate
        acc = None
        for name in self.order:
            S = smap[name].reindex(columns=all_cols).fillna(0.0)
            weight = w.get(name, 0.0) / total_w
            acc = S * weight if acc is None else acc.add(S * weight, fill_value=0.0)
        return acc if acc is not None else X.iloc[0:0]


# -----------------------------
# Factory from settings.yaml
# -----------------------------

def build_from_settings(models_cfg: dict) -> FactorModelHub:
    """Instantiate enabled models from a settings.yaml 'models' section.

    Expected shape:
      models:
        hub_order: [carry_momo_season, pca, coint_adjacent]
        pca: {enabled: true, n_components: 3, ewma_span: 60}
        nelson_siegel: {enabled: false, ...}
        carry_momo_season: {enabled: true, k_momentum_days: 20, ewma_span: 60, seasonality: {enabled: true, significance_min_t: 2.0}}
        coint_adjacent: {enabled: true, lookback_days: 250, pval_entry: 0.05}
    """
    order = list(models_cfg.get("hub_order", []))
    models: dict[str, FactorModel] = {}

    # PCA
    pca_cfg = models_cfg.get("pca", {})
    if pca_cfg.get("enabled", False):
        from .pca import PCAFactor  # local import to avoid hard dependency during scaffolding
        models["pca"] = PCAFactor(
            n_components=int(pca_cfg.get("n_components", 3)),
            ewma_span=int(pca_cfg.get("ewma_span", 60)),
        )

    # Nelson–Siegel (optional)
    ns_cfg = models_cfg.get("nelson_siegel", {})
    if ns_cfg.get("enabled", False):
        try:
            from .nelson_siegel import NelsonSiegelFactor  # pragma: no cover (optional)
            models["nelson_siegel"] = NelsonSiegelFactor(
                lambda_decay=float(ns_cfg.get("lambda_decay", 0.06))
            )
        except Exception as _:
            pass  # keep optional

    # Carry + Momentum + Seasonality
    cms_cfg = models_cfg.get("carry_momo_season", {})
    if cms_cfg.get("enabled", False):
        from .carry_momo_season import CarryMomentumSeasonality
        models["carry_momo_season"] = CarryMomentumSeasonality(
            k_momo=int(cms_cfg.get("k_momentum_days", 20)),
            ewma_span=int(cms_cfg.get("ewma_span", 60)),
            seasonality=cms_cfg.get("seasonality", {}).get("enabled", True),
            seas_t_min=float(cms_cfg.get("seasonality", {}).get("significance_min_t", 2.0)),
        )

    # Cointegration on adjacent spreads
    coint_cfg = models_cfg.get("coint_adjacent", {})
    if coint_cfg.get("enabled", False):
        from .coint_adjacent import CointAdjSpreads
        models["coint_adjacent"] = CointAdjSpreads(
            lookback=int(coint_cfg.get("lookback_days", 250)),
            pval_entry=float(coint_cfg.get("pval_entry", 0.05)),
        )

    # Default order if not supplied
    if not order:
        order = [k for k in ["carry_momo_season", "pca", "coint_adjacent", "nelson_siegel"] if k in models]

    return FactorModelHub(models=models, order=order)


__all__ = [
    "FactorModel",
    "HubConfig",
    "FactorModelHub",
    "build_from_settings",
]
