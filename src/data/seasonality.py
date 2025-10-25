# src/data/seasonality.py
# Month-of-year fixed effects for spreads, with simple significance filtering
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class SeasonalityConfig:
    min_obs_per_month: int = 20
    winsor_pct: float = 0.01           # winsorise tails before measuring effects
    require_tstat: float = 2.0         # suppress effects with |t| < threshold


@dataclass
class SeasonalityModel:
    effects: pd.DataFrame              # columns=spread names, index=1..12 months
    tstats: pd.DataFrame               # same shape as effects

    def deseasonalise(self, spreads: pd.DataFrame) -> pd.DataFrame:
        """Subtract month-of-year effects when significant; leave others unchanged."""
        if spreads.empty:
            return spreads
        out = spreads.copy()
        months = spreads.index.to_period("M").month
        for col in spreads.columns:
            eff = self.effects[col]
            t = self.tstats[col]
            significant = t.abs() >= 2.0
            # map each row month to effect if significant else 0
            adj = months.map(lambda m: eff.loc[m] if bool(significant.loc[m]) else 0.0)
            out[col] = spreads[col] - adj.values
        return out


def _winsorise(s: pd.Series, p: float) -> pd.Series:
    if p <= 0:
        return s
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lower=lo, upper=hi)


def fit_month_effects(spreads: pd.DataFrame, cfg: SeasonalityConfig) -> SeasonalityModel:
    """Estimate month-of-year means and simple t-stats per spread column.

    t = mean(m) / (std(m)/sqrt(n_m)) computed per month independently.
    """
    if spreads.empty:
        return SeasonalityModel(effects=pd.DataFrame(), tstats=pd.DataFrame())

    # winsorise column-wise to reduce the influence of outliers
    W = spreads.apply(lambda s: _winsorise(s.dropna(), cfg.winsor_pct).reindex_like(s))
    months = W.index.to_period("M").month

    eff = pd.DataFrame(index=range(1, 13), columns=spreads.columns, dtype=float)
    tst = eff.copy()

    for col in spreads.columns:
        grp = W[col].groupby(months)
        mu = grp.mean()
        sd = grp.std(ddof=1)
        n = grp.count().astype(float)
        # avoid divide-by-zero
        t = mu / (sd / np.sqrt(n))
        # enforce minimum observations per month
        mu[n < cfg.min_obs_per_month] = 0.0
        t[n < cfg.min_obs_per_month] = 0.0
        # suppress small effects by zeroing them (optional; leave to model.deseasonalise otherwise)
        mu[t.abs() < cfg.require_tstat] = 0.0
        eff[col] = mu
        tst[col] = t

    eff = eff.fillna(0.0)
    tst = tst.fillna(0.0)
    return SeasonalityModel(effects=eff, tstats=tst)


__all__ = [
    "SeasonalityConfig",
    "SeasonalityModel",
    "fit_month_effects",
]
