# src/models/hub.py
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from ..core.types import HealthReport

log = logging.getLogger(__name__)



# Base interface

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
    def __init__(
        self,
        models: Dict[str, FactorModel],
        order: list[str],
        weights: Optional[dict[str, float]] = None,
    ):
        self.models = dict(models)
        self.order = list(order)
        self.weights = dict(weights) if weights is not None else None
        for name in self.order:
            if name not in self.models:
                raise KeyError(f"Model '{name}' not provided in models dict")

    def fit(self, X: pd.DataFrame, **kwargs) -> None:
        for name in self.order:
            self.models[name].fit(X, **kwargs)

    def transform_map(self, X: pd.DataFrame, **kwargs) -> dict[str, pd.DataFrame]:
        return {name: self.models[name].transform(X, **kwargs) for name in self.order}

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


# Factory from settings.yaml
# Respects the enabled: true/false flag per model.
# Falls back to carry_momo_season if nothing is explicitly enabled (with a warning lol)
# so that run_walkforward.py always has at least one signal source.


def build_from_settings(models_cfg: dict) -> FactorModelHub:
    """Build a FactorModelHub from the models section of settings.yaml.

    Each model is only instantiated if its 'enabled' key is True.
    If no model is enabled, carry_momo_season is added as a fallback with a
    visible warning so the pipeline never silently produces zero signals.
    """
    hub_order_cfg: list[str] = list(models_cfg.get("hub_order", ["carry_momo_season"]))
    models: dict[str, FactorModel] = {}

    #  Carry / Momentum / Seasonality 
    cms_cfg = models_cfg.get("carry_momo_season", {})
    if cms_cfg.get("enabled", False):
        try:
            from .carry_momo_season import CarryMomentumSeasonality
            models["carry_momo_season"] = CarryMomentumSeasonality(
                k_momo=int(cms_cfg.get("k_momentum_days", 20)),
                ewma_span=int(cms_cfg.get("ewma_span", 60)),
                seasonality=bool(cms_cfg.get("seasonality", {}).get("enabled", True)),
                seas_t_min=float(
                    cms_cfg.get("seasonality", {}).get("significance_min_t", 2.0)
                ),
            )
            log.info("Model loaded: carry_momo_season")
        except Exception as exc:
            log.warning("Could not load carry_momo_season: %s", exc)

    #  PCA 
    pca_cfg = models_cfg.get("pca", {})
    if pca_cfg.get("enabled", False):
        try:
            from .pca import PCAFactor
            models["pca"] = PCAFactor(
                n_components=int(pca_cfg.get("n_components", 3)),
                ewma_span=int(pca_cfg.get("ewma_span", 60)),
            )
            log.info("Model loaded: pca")
        except Exception as exc:
            log.warning("Could not load pca: %s", exc)

    #  Nelson–Siegel 
    ns_cfg = models_cfg.get("nelson_siegel", {})
    if ns_cfg.get("enabled", False):
        try:
            from .nelson_siegel import NelsonSiegelFactor
            models["nelson_siegel"] = NelsonSiegelFactor(
                lambda_decay=float(ns_cfg.get("lambda_decay", 0.060)),
                ewma_span=int(ns_cfg.get("ewma_span", 60)),
            )
            log.info("Model loaded: nelson_siegel")
        except Exception as exc:
            log.warning("Could not load nelson_siegel: %s", exc)

    #  Cointegration 
    coint_cfg = models_cfg.get("coint_adjacent", {})
    if coint_cfg.get("enabled", False):
        try:
            from .coint_adjacent import CointAdjSpreads
            models["coint_adjacent"] = CointAdjSpreads(
                lookback=int(coint_cfg.get("lookback_days", 250)),
                pval_entry=float(coint_cfg.get("pval_entry", 0.05)),
            )
            log.info("Model loaded: coint_adjacent")
        except Exception as exc:
            log.warning("Could not load coint_adjacent: %s", exc)

    #  Fallback: if nothing was enabled, warn and load CMS unconditionally 
    if not models:
        log.warning(
            "No models have 'enabled: true' in settings.yaml. "
            "Loading carry_momo_season as fallback. "
            "Set 'carry_momo_season.enabled: true' to silence this warning."
        )
        try:
            from .carry_momo_season import CarryMomentumSeasonality
            models["carry_momo_season"] = CarryMomentumSeasonality()
        except Exception as exc:
            raise RuntimeError(
                "Cannot build FactorModelHub: no models enabled and "
                f"carry_momo_season fallback also failed: {exc}"
            ) from exc

    # Keep only models that were actually loaded, preserving hub_order_cfg ordering.
    order = [name for name in hub_order_cfg if name in models]
    if not order:
        order = list(models.keys())

    return FactorModelHub(models=models, order=order)


__all__ = [
    "FactorModel",
    "HubConfig",
    "FactorModelHub",
    "build_from_settings",
]