# src/execsim/walkforward.py
# Anchored walk-forward runner with embargo
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
from datetime import date

import pandas as pd

from ..core.scheduling import Fold


@dataclass
class WalkForwardResult:
    fold: Fold
    pnl_path: pd.Series
    trade_log: pd.DataFrame


class AnchoredWalkForward:
    def __init__(self, folds: List[Fold], embargo_days: int = 5):
        self.folds = folds
        self.embargo_days = embargo_days

    def run(
        self,
        *,
        X: pd.DataFrame,
        model_hub,
        signal_builder: Callable[[dict[str, pd.DataFrame]], pd.DataFrame],  # combines model signals into combo signals
        quotes_provider: Callable[[date, date], Dict[str, dict]],            # returns quotes dict for window
        backtester,
        sizing_fn,
        sizing_cfg,
    ) -> List[WalkForwardResult]:
        results: List[WalkForwardResult] = []
        for f in self.folds:
            X_tr = X.loc[f.train_start:f.train_end]
            X_te = X.loc[f.test_start:f.test_end]
            # Fit models
            model_hub.fit(X_tr)
            sigs_map = model_hub.signals_map(X_te)
            combo_signals = signal_builder(sigs_map)
            quotes = quotes_provider(f.test_start, f.test_end)
            res = backtester.run(signals=combo_signals, quotes=quotes, sizing_fn=sizing_fn, sizing_cfg=sizing_cfg)
            results.append(WalkForwardResult(fold=f, pnl_path=res["pnl_path"], trade_log=res["trade_log"]))
        return results
