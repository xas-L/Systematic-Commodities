# src/core/health.py
# Model and ops health checks used by the engine and risk layer.
# Keep the API small and explainable.
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Iterable

import numpy as np
import pandas as pd

from .types import HealthReport


# -----------------------------
# Small stats helpers
# -----------------------------

def _ewma(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def ewma_z(series: pd.Series, span: int) -> pd.Series:
    mu = _ewma(series, span)
    var = (series - mu).pow(2).ewm(span=span, adjust=False).mean()
    return (series - mu) / np.sqrt(var.replace(0.0, np.nan))


def nan_fraction(df: pd.DataFrame) -> float:
    if len(df) == 0:
        return 1.0
    return float(df.isna().values.mean())


def fro_norm(a: np.ndarray) -> float:
    return float(np.sqrt(np.square(a).sum()))


# -----------------------------
# Drift metrics
# -----------------------------

def corr_drift(cur: pd.DataFrame, ref: pd.DataFrame) -> float:
    """Frobenius norm of correlation matrix difference (reference vs current)."""
    if cur.shape[1] == 0 or ref.shape[1] == 0:
        return float("nan")
    c1 = np.nan_to_num(np.corrcoef(np.nan_to_num(cur.values, nan=0.0).T))
    c0 = np.nan_to_num(np.corrcoef(np.nan_to_num(ref.values, nan=0.0).T))
    return fro_norm(c1 - c0)


def var_floor_breach(df: pd.DataFrame, min_var: float) -> bool:
    if df.shape[1] == 0:
        return True
    return bool((df.var(numeric_only=True) < min_var).any())


# -----------------------------
# Configs
# -----------------------------
@dataclass
class ModelHealthConfig:
    name: str
    span: int = 60                 # EWMA span for z and variance tracking
    min_var: float = 1e-8          # minimal variance to consider factor alive
    max_nan_frac: float = 0.05     # 5% NaN across panel tolerated
    max_corr_drift: float = 0.5    # tune from risk_limits.yaml (factor_drift_threshold)


@dataclass
class SlippageGuardConfig:
    tolerance_bp: float = 50.0     # live - backtest tolerance before action
    persistence_bars: int = 5


@dataclass
class DrawdownGuardConfig:
    lookback_bars: int = 60
    sigma_threshold: float = 3.0   # 3-sigma drawdown -> breach


# -----------------------------
# Monitors
# -----------------------------
class ModelHealthMonitor:
    """Monitors factor panel for NaNs, variance floor, and correlation drift against a reference."""

    def __init__(self, cfg: ModelHealthConfig):
        self.cfg = cfg
        self._ref: Optional[pd.DataFrame] = None

    def set_reference(self, Z_ref: pd.DataFrame) -> None:
        # store a compact reference window (dropna to avoid bias)
        self._ref = Z_ref.dropna().copy()

    def assess(self, Z_cur: pd.DataFrame) -> HealthReport:
        metrics = {}
        ok = True

        nf = nan_fraction(Z_cur)
        metrics["nan_frac"] = nf
        if nf > self.cfg.max_nan_frac:
            ok = False

        var_min = float(Z_cur.var(numeric_only=True).min()) if not Z_cur.empty else 0.0
        metrics["var_min"] = var_min
        if var_min < self.cfg.min_var:
            ok = False

        drift = float("nan")
        if self._ref is not None and not Z_cur.empty:
            drift = corr_drift(Z_cur, self._ref)
            metrics["corr_drift"] = drift
            if np.isfinite(drift) and drift > self.cfg.max_corr_drift:
                ok = False
        else:
            metrics["corr_drift"] = float("nan")

        return HealthReport(component=self.cfg.name, ok=ok, metrics=metrics)


class SlippageGuard:
    """Compares rolling live slippage vs backtest baseline and raises a breach when persistent."""

    def __init__(self, cfg: SlippageGuardConfig):
        self.cfg = cfg
        self._hist_live: list[float] = []  # bp per combo
        self._hist_bt: list[float] = []

    def update(self, live_bp: float, backtest_bp: float) -> Optional[HealthReport]:
        self._hist_live.append(live_bp)
        self._hist_bt.append(backtest_bp)
        # keep last N bars
        n = max(self.cfg.persistence_bars, 1)
        self._hist_live = self._hist_live[-n:]
        self._hist_bt = self._hist_bt[-n:]
        if len(self._hist_live) < n:
            return None
        gap = float(np.mean(self._hist_live) - np.mean(self._hist_bt))
        ok = gap <= self.cfg.tolerance_bp
        return HealthReport(
            component="slippage_guard",
            ok=ok,
            metrics={"gap_bp": gap, "tolerance_bp": self.cfg.tolerance_bp},
            message=None if ok else "Live slippage exceeds backtest tolerance",
        )


class DrawdownGuard:
    """Detects large drawdowns relative to recent sigma of P&L."""

    def __init__(self, cfg: DrawdownGuardConfig):
        self.cfg = cfg
        self._pnl_path: list[float] = []  # cumulative P&L (net)

    def update(self, pnl_increment: float) -> Optional[HealthReport]:
        # accumulate
        next_val = (self._pnl_path[-1] + pnl_increment) if self._pnl_path else pnl_increment
        self._pnl_path.append(next_val)
        n = self.cfg.lookback_bars
        if len(self._pnl_path) < n:
            return None
        window = np.array(self._pnl_path[-n:], dtype=float)
        peak = float(np.max(window))
        trough = float(np.min(window[window.argmax():], initial=window[-1])) if len(window) > 1 else window[-1]
        dd = peak - window[-1]
        sigma = float(np.std(np.diff(window))) if len(window) > 1 else 0.0
        sigma = max(sigma, 1e-9)
        dd_sigma = dd / sigma
        ok = dd_sigma <= self.cfg.sigma_threshold
        return HealthReport(
            component="drawdown_guard",
            ok=ok,
            metrics={"dd": dd, "sigma": sigma, "dd_sigma": dd_sigma, "threshold": self.cfg.sigma_threshold},
            message=None if ok else "Drawdown exceeds sigma threshold",
        )


__all__ = [
    "ModelHealthConfig",
    "SlippageGuardConfig",
    "DrawdownGuardConfig",
    "ModelHealthMonitor",
    "SlippageGuard",
    "DrawdownGuard",
    "ewma_z",
    "nan_fraction",
    "corr_drift",
]
