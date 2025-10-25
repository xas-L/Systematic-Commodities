# src/ops/engine.py
# Orchestration engine for batch backtests and (later) live modes
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import pandas as pd

from ..core.logging import configure_logging, get_logger
from ..core.utils import project_root, load_settings, load_fees_slippage, load_risk_limits, ensure_dir
from ..core.scheduling import Fold
from ..data.loader import load_contract_bars, validate_bars
from ..data.curve import build_curve_surface, log_adjacent_spreads
from ..models.hub import build_from_settings, FactorModelHub
from ..signals.sizing import SizingConfig, size_from_signal
from ..execsim.cost_model import YamlFeeModel, YamlSlippageModel
from ..execsim.combos import ComboExecutionSimulator
from ..execsim.backtester import Backtester, BacktestConfig
from ..execsim.walkforward import AnchoredWalkForward

log = get_logger(__name__)


@dataclass
class EngineConfig:
    symbol: str
    out_dir: Path


class Engine:
    """High-level facade that wires settings → data → models → backtests.

    This is the batch/offline engine used by scripts/run_walkforward.py.
    """

    def __init__(self, *, root: Optional[Path] = None):
        self.root = project_root(root)
        self.settings = load_settings(self.root)
        self.fees_cfg = load_fees_slippage(self.root)
        self.risk_cfg = load_risk_limits(self.root)

        # logging
        log_level = self.settings.get("ops", {}).get("logging", {}).get("level", "INFO")
        rotate_mb = int(self.settings.get("ops", {}).get("logging", {}).get("rotate_mb", 50))
        configure_logging(level=log_level, log_dir=self.root / "logs", rotate_mb=rotate_mb)

    # -------------------------
    # Data pipe
    # -------------------------
    def _load_surface(self, symbol: str) -> pd.DataFrame:
        bars = load_contract_bars(symbol, self.root)
        ok, errs = validate_bars(bars)
        if not ok:
            log.warning("Validation issues: %s", errs)
        ccfg = self.settings.get("curve_construction", {})
        uni = self.settings.get("universe", {}).get("curves", {})
        surface, _ = build_curve_surface(
            bars,
            min_volume=int(ccfg.get("filter", {}).get("min_volume", 0)),
            min_open_interest=int(ccfg.get("filter", {}).get("min_open_interest", 0)),
            drop_stale_after_days=int(ccfg.get("drop_stale_after_days", 10)),
            price_field_order=list(ccfg.get("price_field_order", ["settle", "last"])),
            tenors_target=int(ccfg.get("tenors_target", 10)),
            min_contracts=int(uni.get(symbol, {}).get("min_contracts", 6)),
        )
        return surface

    def _calendar_quotes(self, surface: pd.DataFrame, product: str) -> dict[str, dict]:
        # Same as in scripts/run_walkforward.py but encapsulated here
        if surface.shape[1] < 2:
            return {}
        cols = list(surface.columns)
        diffs = surface[cols].diff(axis=1).iloc[:, 1:].copy()
        names = [f"{cols[i]}-{cols[i-1]}" for i in range(1, len(cols))]
        diffs.columns = names
        prod = self.fees_cfg.get("products", {}).get(product, {})
        min_inc = float(prod.get("combo", {}).get("min_price_increment", prod.get("tick_size", 0.01)))
        typical_half_ticks = float(prod.get("combo", {}).get("typical_half_spread_ticks", 1.0))
        half_spread = typical_half_ticks * min_inc
        lot_value = float(prod.get("multiplier", 1.0))
        quotes: dict[str, dict] = {}
        for c in diffs.columns:
            quotes[c] = {
                "mid": diffs[c],
                "half_spread": pd.Series(half_spread, index=diffs.index),
                "lot_value": lot_value,
                "symbol": product,
                "tob_size": {ts: 10 for ts in diffs.index},
            }
        return quotes

    # -------------------------
    # Public API
    # -------------------------
    def run_walkforward(self, symbol: str, out_dir: Optional[Path] = None) -> dict:
        surface = self._load_surface(symbol)
        if surface.empty or surface.shape[1] < 2:
            raise RuntimeError("Insufficient surface depth to build calendars")
        X = log_adjacent_spreads(surface)

        # Model hub
        hub: FactorModelHub = build_from_settings(self.settings.get("models", {}))

        # Folds
        folds_cfg = self.settings.get("walkforward", {}).get("folds", [])
        folds = [
            Fold(
                train_start=pd.to_datetime(f["train_start"]).date(),
                train_end=pd.to_datetime(f["train_end"]).date(),
                test_start=pd.to_datetime(f["test_start"]).date(),
                test_end=pd.to_datetime(f["test_end"]).date(),
            )
            for f in folds_cfg
        ]
        if not folds:
            idx = surface.index
            first_test = idx[min(len(idx) - 1, int(5 * 252))]
            folds = [Fold(train_start=idx[0].date(), train_end=first_test.date(), test_start=first_test.date(), test_end=idx[-1].date())]

        # Quote provider closure
        def quotes_provider(d0: pd.Timestamp, d1: pd.Timestamp) -> dict[str, dict]:
            sv = surface.loc[d0:d1]
            return self._calendar_quotes(sv, product=symbol)

        # Sizing config
        saz = self.settings.get("signal_and_sizing", {})
        scfg = SizingConfig(
            z_clip=float(saz.get("z_clip", 3.0)),
            risk_per_curve_usd=float(saz.get("risk_per_curve_usd", 100000.0)),
            per_trade_notional_cap_usd=float(self.risk_cfg.get("sizing_policy", {}).get("per_trade_notional_cap_usd", 150000.0)),
            vol_target_enabled=bool(saz.get("vol_target", {}).get("enabled", True)),
            vol_lookback_days=int(saz.get("vol_target", {}).get("lookback_days", 60)),
            annual_vol_target=float(saz.get("vol_target", {}).get("annual_vol_target", 0.10)),
        )

        # Fee/Slippage models and executor
        cost_profile = self.settings.get("execution_sim", {}).get("cost_profile", "conservative")
        slip_profile = self.settings.get("execution_sim", {}).get("slippage_profile", "conservative")
        fee_model = YamlFeeModel(product=symbol, profile_name=cost_profile, root=self.root)
        slip_model = YamlSlippageModel(product=symbol, profile_name=slip_profile, root=self.root)
        executor = ComboExecutionSimulator(fee_model=fee_model, slip_model=slip_model)

        backtester = Backtester(executor=executor, cfg=BacktestConfig())
        wf = AnchoredWalkForward(folds=folds, embargo_days=int(self.settings.get("walkforward", {}).get("embargo_days", 5)))

        def signal_builder(smap: dict[str, pd.DataFrame]) -> pd.DataFrame:
            # same as scripts/run_walkforward.py: average carry & momo per combo
            if "carry_momo_season" not in smap:
                return next(iter(smap.values())) if smap else pd.DataFrame()
            S = smap["carry_momo_season"].copy()
            buckets: dict[str, list[str]] = {}
            for c in S.columns:
                if c.startswith("carry_"):
                    base = c[len("carry_"):]
                    buckets.setdefault(base, []).append(c)
                elif c.startswith("momo"):
                    _, base = c.split("_", 1)
                    buckets.setdefault(base, []).append(c)
            out = {base: S[members].mean(axis=1) for base, members in buckets.items()}
            return pd.DataFrame(out, index=S.index).sort_index(axis=1)

        results = wf.run(
            X=X,
            model_hub=hub,
            signal_builder=signal_builder,
            quotes_provider=quotes_provider,
            backtester=backtester,
            sizing_fn=size_from_signal,
            sizing_cfg=scfg,
        )

        # Collate
        pnl_all = pd.concat([r.pnl_path for r in results], axis=0).sort_index()
        trades_all = pd.concat([r.trade_log for r in results], axis=0).reset_index(drop=True)

        out_dir = ensure_dir(self.root / "reports" / "pm_brief" / symbol if out_dir is None else out_dir)
        pnl_all.to_csv(out_dir / "pnl_path.csv")
        trades_all.to_csv(out_dir / "trade_log.csv", index=False)
        log.info("Engine wrote %s and %s", out_dir / "pnl_path.csv", out_dir / "trade_log.csv")
        return {"pnl_path": pnl_all, "trade_log": trades_all}
