# scripts/run_walkforward.py
# First-pass end-to-end: build curve, generate combo quotes, fit model hub, and run anchored walk-forward
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import date, datetime

import numpy as np
import pandas as pd

# Make repo root importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Local imports from src/
from src.core.utils import load_settings, load_fees_slippage, load_risk_limits, ensure_dir
from src.core.logging import configure_logging, get_logger
from src.core.scheduling import Fold
from src.data.loader import load_contract_bars, load_contract_meta, validate_bars
from src.data.curve import build_curve_surface, log_adjacent_spreads
from src.data.spreads import calendar_spreads
from src.models.hub import build_from_settings
from src.signals.sizing import SizingConfig, size_from_signal
from src.execsim.cost_model import YamlFeeModel, YamlSlippageModel
from src.execsim.combos import ComboExecutionSimulator
from src.execsim.backtester import Backtester, BacktestConfig
from src.execsim.walkforward import AnchoredWalkForward

# Minor patch: ensure pandas is visible inside backtester if it expects it
import src.execsim.backtester as _bt
_bt.pd = pd  # type: ignore

log = get_logger(__name__)


def _build_folds_from_settings(settings: dict) -> list[Fold]:
    folds_cfg = settings.get("walkforward", {}).get("folds", [])
    folds: list[Fold] = []
    for f in folds_cfg:
        folds.append(
            Fold(
                train_start=pd.to_datetime(f["train_start"]).date(),
                train_end=pd.to_datetime(f["train_end"]).date(),
                test_start=pd.to_datetime(f["test_start"]).date(),
                test_end=pd.to_datetime(f["test_end"]).date(),
            )
        )
    return folds


def _simple_calendar_diff(surface: pd.DataFrame) -> pd.DataFrame:
    """Simple calendar (far - near) differences from price surface for quotes."""
    if surface.shape[1] < 2:
        return surface.iloc[0:0]
    cols = list(surface.columns)
    arr = surface[cols].to_numpy()
    diffs = arr[:, 1:] - arr[:, :-1]
    names = [f"{cols[i]}-{cols[i-1]}" for i in range(1, len(cols))]
    return pd.DataFrame(diffs, index=surface.index, columns=names)


def _calendar_quotes(surface: pd.DataFrame, product: str, fees_cfg: dict) -> dict[str, dict]:
    """Build quotes dict for calendars: mid series (simple diff), half-spread series, lot_value, symbol."""
    pdif = _simple_calendar_diff(surface)
    prod = fees_cfg.get("products", {}).get(product, {})
    min_inc = float(prod.get("combo", {}).get("min_price_increment", prod.get("tick_size", 0.01)))
    typical_half_ticks = float(prod.get("combo", {}).get("typical_half_spread_ticks", 1.0))
    half_spread = typical_half_ticks * min_inc
    lot_value = float(prod.get("multiplier", 1.0))  # $ P&L per 1.0 price move per lot

    quotes: dict[str, dict] = {}
    for c in pdif.columns:
        quotes[c] = {
            "mid": pdif[c],
            "half_spread": pd.Series(half_spread, index=pdif.index),
            "lot_value": lot_value,
            "symbol": product,
            "tob_size": {ts: 10 for ts in pdif.index},  # coarse constant depth
        }
    return quotes


def _cms_combo_signal_builder(smap: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine Carry/Momo/Season signals into per-combo signals.
    Strategy: for each base combo name K, average available z-signal columns: carry_K and momo*_K.
    PCA and coint are ignored here (used in attribution separately).
    """
    if "carry_momo_season" not in smap:
        # fall back to PCA if it happened to return per-feature signals (unlikely)
        if not smap:
            return pd.DataFrame()
        any_df = next(iter(smap.values()))
        return any_df.copy()

    S = smap["carry_momo_season"].copy()
    cols = list(S.columns)
    # Collect base names
    buckets: dict[str, list[str]] = {}
    for c in cols:
        if c.startswith("carry_"):
            base = c[len("carry_"):]
            buckets.setdefault(base, []).append(c)
        elif c.startswith("momo"):
            # momo20_<base>
            _, base = c.split("_", 1)
            buckets.setdefault(base, []).append(c)
    out = {}
    for base, members in buckets.items():
        out[base] = S[members].mean(axis=1)
    return pd.DataFrame(out, index=S.index).sort_index(axis=1)


def main():
    parser = argparse.ArgumentParser(description="Run anchored walk-forward for a single curve")
    parser.add_argument("--symbol", default="CL", help="Root symbol (e.g., CL, NG, ZC)")
    args = parser.parse_args()

    settings = load_settings(ROOT)
    fees_cfg = load_fees_slippage(ROOT)
    risk_cfg = load_risk_limits(ROOT)

    # Logging
    log_level = settings.get("ops", {}).get("logging", {}).get("level", "INFO")
    rotate_mb = int(settings.get("ops", {}).get("logging", {}).get("rotate_mb", 50))
    configure_logging(level=log_level, log_dir=ROOT / "logs", rotate_mb=rotate_mb)

    symbol = args.symbol.upper()
    uni = settings.get("universe", {}).get("curves", {})
    if symbol not in uni:
        raise SystemExit(f"Symbol {symbol} not in universe settings")

    # Load raw contract bars
    log.info(f"Loading raw bars for {symbol}…")
    bars = load_contract_bars(symbol, ROOT)
    ok, errs = validate_bars(bars)
    if not ok:
        log.warning("Validation issues: %s", errs)

    ccfg = settings.get("curve_construction", {})
    surface, used = build_curve_surface(
        bars,
        min_volume=int(ccfg.get("filter", {}).get("min_volume", 0)),
        min_open_interest=int(ccfg.get("filter", {}).get("min_open_interest", 0)),
        drop_stale_after_days=int(ccfg.get("drop_stale_after_days", 10)),
        price_field_order=list(ccfg.get("price_field_order", ["settle", "last"])),
        tenors_target=int(ccfg.get("tenors_target", 10)),
        min_contracts=int(uni[symbol].get("min_contracts", 6)),
    )
    if surface.empty or surface.shape[1] < 2:
        raise SystemExit("Insufficient surface depth to build calendars")

    # Feature panel X: log-adjacent spreads
    X = log_adjacent_spreads(surface)

    # Model hub from settings
    models_cfg = settings.get("models", {})
    hub = build_from_settings(models_cfg)

    # Folds
    folds = _build_folds_from_settings(settings)
    if not folds:
        # Simple default: last 3 years, 6-month steps after 5-year warmup
        idx = surface.index
        first_test = idx[min(len(idx)-1, int(5*252))]
        folds = [
            Fold(train_start=idx[0].date(), train_end=first_test.date(), test_start=first_test.date(), test_end=idx[-1].date())
        ]

    # Quotes provider closure
    def quotes_provider(d0: date, d1: date) -> dict[str, dict]:
        sv = surface.loc[pd.to_datetime(d0):pd.to_datetime(d1)]
        return _calendar_quotes(sv, product=symbol, fees_cfg=fees_cfg)

    # Sizing config
    saz = settings.get("signal_and_sizing", {})
    scfg = SizingConfig(
        z_clip=float(saz.get("z_clip", 3.0)),
        risk_per_curve_usd=float(saz.get("risk_per_curve_usd", 100000.0)),
        per_trade_notional_cap_usd=float(risk_cfg.get("sizing_policy", {}).get("per_trade_notional_cap_usd", 150000.0)),
        vol_target_enabled=bool(saz.get("vol_target", {}).get("enabled", True)),
        vol_lookback_days=int(saz.get("vol_target", {}).get("lookback_days", 60)),
        annual_vol_target=float(saz.get("vol_target", {}).get("annual_vol_target", 0.10)),
    )

    # Executor (combo simulator) with YAML-driven fees & slippage
    cost_profile = settings.get("execution_sim", {}).get("cost_profile", "conservative")
    slip_profile = settings.get("execution_sim", {}).get("slippage_profile", "conservative")
    fee_model = YamlFeeModel(product=symbol, profile_name=cost_profile, root=ROOT)
    slip_model = YamlSlippageModel(product=symbol, profile_name=slip_profile, root=ROOT)
    executor = ComboExecutionSimulator(fee_model=fee_model, slip_model=slip_model)

    backtester = Backtester(executor=executor, cfg=BacktestConfig())
    wf = AnchoredWalkForward(folds=folds, embargo_days=int(settings.get("walkforward", {}).get("embargo_days", 5)))

    # Run
    log.info("Running %d folds for %s…", len(folds), symbol)
    results = wf.run(
        X=X,
        model_hub=hub,
        signal_builder=_cms_combo_signal_builder,
        quotes_provider=quotes_provider,
        backtester=backtester,
        sizing_fn=size_from_signal,
        sizing_cfg=scfg,
    )

    # Collate results
    pnl_all = pd.concat([r.pnl_path for r in results], axis=0).sort_index()
    trades_all = pd.concat([r.trade_log for r in results], axis=0).reset_index(drop=True)

    # Summary
    mu = pnl_all.mean()
    sd = pnl_all.std(ddof=1) if len(pnl_all) > 1 else 0.0
    sharpe = (mu / sd * np.sqrt(252.0)) if sd > 0 else 0.0
    total = pnl_all.sum()
    log.info("Summary — Total: $%.0f | Daily μ: $%.2f | Daily σ: $%.2f | Sharpe: %.2f", total, mu, sd, sharpe)

    # Persist
    out_dir = ensure_dir(ROOT / "reports" / "pm_brief" / symbol)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    pnl_all.to_csv(out_dir / f"pnl_path_{ts}.csv")
    trades_all.to_csv(out_dir / f"trade_log_{ts}.csv", index=False)
    log.info("Wrote %s and %s", out_dir / f"pnl_path_{ts}.csv", out_dir / f"trade_log_{ts}.csv")


if __name__ == "__main__":
    main()
