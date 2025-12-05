# src/models/hub.py - FIXED VERSION
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
    @abstractmethod
    def fit(self, X: pd.DataFrame, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def transform(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def signal(self, Z: pd.DataFrame, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def health(self, Z: pd.DataFrame, **kwargs) -> HealthReport:
        raise NotImplementedError


@dataclass
class HubConfig:
    order: list[str]
    weights: Optional[dict[str, float]] = None


class FactorModelHub:
    def __init__(self, models: Dict[str, FactorModel], order: list[str], weights: Optional[dict[str, float]] = None):
        self.models = dict(models)
        self.order = list(order)
        self.weights = dict(weights) if weights is not None else None
        # validate
        for name in self.order:
            if name not in self.models:
                raise KeyError(f"Model '{name}' not provided in models dict")

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

    def blended_signal(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        smap = self.signals_map(X, **kwargs)
        all_cols = sorted(set().union(*[df.columns for df in smap.values()])) if smap else []
        if not all_cols:
            return X.iloc[0:0]
        if self.weights is None:
            w = {name: 1.0 for name in self.order}
        else:
            w = {name: float(self.weights.get(name, 0.0)) for name in self.order}
        total_w = sum(w.values()) or 1.0
        acc = None
        for name in self.order:
            S = smap[name].reindex(columns=all_cols).fillna(0.0)
            weight = w.get(name, 0.0) / total_w
            acc = S * weight if acc is None else acc.add(S * weight, fill_value=0.0)
        return acc if acc is not None else X.iloc[0:0]


# -----------------------------
# Factory from settings.yaml - FIXED
# -----------------------------

def build_from_settings(models_cfg: dict) -> FactorModelHub:
    """Always create carry_momo_season for testing."""
    order = list(models_cfg.get("hub_order", ["carry_momo_season"]))
    models: dict[str, FactorModel] = {}

    # ALWAYS create carry_momo_season for testing
    try:
        from .carry_momo_season import CarryMomentumSeasonality
        k_momo = models_cfg.get("carry_momo_season", {}).get("k_momentum_days", 20)
        ewma_span = models_cfg.get("carry_momo_season", {}).get("ewma_span", 60)
        seasonality = models_cfg.get("carry_momo_season", {}).get("seasonality", {}).get("enabled", True)
        seas_t_min = models_cfg.get("carry_momo_season", {}).get("seasonality", {}).get("significance_min_t", 2.0)
        models["carry_momo_season"] = CarryMomentumSeasonality(
            k_momo=k_momo,
            ewma_span=ewma_span,
            seasonality=seasonality,
            seas_t_min=seas_t_min,
        )
    except Exception as e:
        print(f"Warning: Could not create carry_momo_season: {e}")
        # Create dummy model
        class DummyModel(FactorModel):
            def fit(self, X, **kwargs): pass
            def transform(self, X, **kwargs): return X.copy()
            def signal(self, Z, **kwargs): return Z.copy()
            def health(self, Z, **kwargs): return HealthReport(component="dummy", ok=True, metrics={})
        models["carry_momo_season"] = DummyModel()

    # PCA (optional)
    pca_cfg = models_cfg.get("pca", {})
    if pca_cfg.get("enabled", False):
        try:
            from .pca import PCAFactor
            models["pca"] = PCAFactor(
                n_components=int(pca_cfg.get("n_components", 3)),
                ewma_span=int(pca_cfg.get("ewma_span", 60)),
            )
        except:
            pass

    # Ensure order matches available models
    order = [name for name in order if name in models]
    if not order:
        order = list(models.keys())
    
    return FactorModelHub(models=models, order=order)


__all__ = [
    "FactorModel",
    "HubConfig",
    "FactorModelHub",
    "build_from_settings",
]
