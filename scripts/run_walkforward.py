"""
scripts/run_walkforward.py  — LOCALLY FIXED VERSION (finally lol)

Key fixes:
  1. Signal-to-quote alignment: combo name format is now consistent
     (both derived from the same surface column string representation).
  2. Sizing uses the real vol-targeted function, not the stub.
  3. Quote tob_size stores integers keyed by Timestamp, matching backtester lookup.
  4. Fold fallback auto-generates folds from the data date range if settings.yaml
     folds don't span the available data.
  5. Debug logging shows matched/unmatched combos so you can see if alignment fails.

Usage:
    python scripts/run_walkforward.py --symbol CL
    python scripts/run_walkforward.py --symbol CL --start-year 2018 --n-folds 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.utils import load_settings, load_fees_slippage, load_risk_limits, ensure_dir
from src.core.logging import configure_logging, get_logger
from src.core.scheduling import Fold, generate_anchored_folds
from src.data.loader import load_contract_bars, validate_bars
from src.data.curve import build_curve_surface, log_adjacent_spreads
from src.models.hub import build_from_settings
from src.signals.sizing import SizingConfig, size_from_signal
from src.execsim.cost_model import YamlFeeModel, YamlSlippageModel
from src.execsim.combos import ComboExecutionSimulator
from src.execsim.backtester import Backtester, BacktestConfig
from src.execsim.walkforward import AnchoredWalkForward

log = get_logger(__name__)


#  Quote builder 

def build_quotes(surface: pd.DataFrame, symbol: str, fees_cfg: dict) -> dict[str, dict]:
    """
    Build quotes dict for all adjacent calendar combos on the surface.
    Key format: f"{far_expiry}-{near_expiry}" where expiry values are
    the exact same string representation as the surface column index.
    This guarantees alignment with signal column names from the CMS model.
    """
    if surface.shape[1] < 2:
        return {}

    prod       = fees_cfg.get("products", {}).get(symbol, {})
    min_inc    = float(prod.get("combo", {}).get("min_price_increment", 0.01))
    half_ticks = float(prod.get("combo", {}).get("typical_half_spread_ticks", 1.0))
    half_sp    = half_ticks * min_inc
    lot_val    = float(prod.get("multiplier", 1000.0))

    cols   = list(surface.columns)
    arr    = surface.to_numpy(dtype=float)
    # Calendar spread = far - near
    diffs  = arr[:, 1:] - arr[:, :-1]

    quotes: dict[str, dict] = {}
    for i in range(len(cols) - 1):
        # CRITICAL: key must match what CMS model produces for its carry_ / momo_ columns.
        # CMS model calls log_adjacent_spreads which does f"{cols[i]}-{cols[i-1]}"
        # i.e.  far  is cols[i+1],  near is cols[i].  Direction: far-near (positive in contango).
        near, far = cols[i], cols[i + 1]
        key = f"{far}-{near}"           # matches log_adjacent_spreads column naming

        mid_series = pd.Series(diffs[:, i], index=surface.index, name=key)
        quotes[key] = {
            "mid":         mid_series,
            "half_spread": pd.Series(half_sp, index=surface.index),
            "lot_value":   lot_val,
            "symbol":      symbol,
            # tob_size: dict keyed by Timestamp (what backtester.run() expects)
            "tob_size":    {ts: 10 for ts in surface.index},
        }
    return quotes


#  Signal builder 

def cms_signal_builder(sigs_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Average carry and momentum sub-signals per combo base name.

    CMS model produces columns like:
      carry_2024-01-20-2023-12-20
      momo20_2024-01-20-2023-12-20

    We strip the prefix and average per base, producing one signal per combo.
    """
    if "carry_momo_season" not in sigs_map:
        return pd.DataFrame()

    S   = sigs_map["carry_momo_season"].copy()
    out: dict[str, pd.Series] = {}

    for col in S.columns:
        if col.startswith("carry_"):
            base = col[len("carry_"):]
        elif col.startswith("momo"):
            # momo20_<base> — split on first underscore only
            parts = col.split("_", 1)
            if len(parts) < 2:
                continue
            base = parts[1]
        else:
            continue

        if base not in out:
            out[base] = S[col]
        else:
            # Average with existing (carry and momo get equal weight)
            out[base] = (out[base] + S[col]) / 2.0

    if not out:
        return pd.DataFrame()

    result = pd.DataFrame(out, index=S.index)
    log.info("Signal builder produced %d combo signals", len(result.columns))
    return result


# Fold generation helper

def make_folds(surface: pd.DataFrame, settings: dict, n_folds: int) -> list[Fold]:
    """
    Try to load folds from settings.yaml first.
    If those folds don't overlap the data, auto-generate anchored folds.
    """
    idx         = surface.index
    data_start  = idx[0].date()
    data_end    = idx[-1].date()

    # Try settings folds first
    folds_cfg = settings.get("walkforward", {}).get("folds", [])
    valid_folds = []
    for f in folds_cfg:
        fold = Fold(
            train_start=pd.to_datetime(f["train_start"]).date(),
            train_end=pd.to_datetime(f["train_end"]).date(),
            test_start=pd.to_datetime(f["test_start"]).date(),
            test_end=pd.to_datetime(f["test_end"]).date(),
        )
        # Only keep folds that have data
        if fold.train_start <= data_end and fold.test_end >= data_start:
            valid_folds.append(fold)

    if valid_folds:
        log.info("Using %d folds from settings.yaml", len(valid_folds))
        return valid_folds

    # Auto-generate: use first 70% for training, then roll 6-month test windows
    log.info("No valid folds in settings.yaml — auto-generating anchored folds from data")
    cutoff_days = int(0.70 * len(idx))
    first_test  = idx[cutoff_days].date()

    folds = generate_anchored_folds(
        train_start=data_start,
        first_test_start=first_test,
        last_date=data_end,
        test_months=6,
        step_months=6,
        embargo_bdays=5,
    )

    if not folds:
        # Absolute fallback: one fold
        mid = idx[len(idx) // 2].date()
        folds = [Fold(train_start=data_start, train_end=mid, test_start=mid, test_end=data_end)]

    log.info("Auto-generated %d anchored folds", len(folds))
    for f in folds:
        log.info("  Fold: train %s→%s | test %s→%s", f.train_start, f.train_end, f.test_start, f.test_end)

    return folds[:n_folds]


#  Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",     default="CL")
    parser.add_argument("--n-folds",    type=int, default=999, help="Max folds to run (default: all)")
    parser.add_argument("--min-vol",    type=int, default=0,   help="Min volume filter (0=off)")
    parser.add_argument("--min-oi",     type=int, default=0,   help="Min open interest filter")
    args = parser.parse_args()

    settings  = load_settings(ROOT)
    fees_cfg  = load_fees_slippage(ROOT)
    risk_cfg  = load_risk_limits(ROOT)

    configure_logging(level="INFO", log_dir=ROOT / "logs")
    symbol = args.symbol.upper()

    # ── Load and validate raw bars ────────────────────────────────────────
    log.info("Loading raw bars for %s ...", symbol)
    bars   = load_contract_bars(symbol, ROOT)
    ok, errs = validate_bars(bars)
    if not ok:
        for e in errs:
            log.warning("  %s", e)

    log.info("  %d rows | %d contracts | %s → %s",
             len(bars), bars["expiry"].nunique(), bars["date"].min(), bars["date"].max())

    #  Build curve surface + filtering - basis for everything else, do early so we know what data we have to work with
    ccfg = settings.get("curve_construction", {})
    uni  = settings.get("universe", {}).get("curves", {})

    surface, _ = build_curve_surface(
        bars,
        min_volume=args.min_vol,
        min_open_interest=args.min_oi,
        drop_stale_after_days=int(ccfg.get("drop_stale_after_days", 10)),
        price_field_order=list(ccfg.get("price_field_order", ["settle", "last"])),
        tenors_target=int(ccfg.get("tenors_target", 10)),
        min_contracts=2,                   # relaxed for real data
    )
    if surface.empty or surface.shape[1] < 2:
        raise SystemExit("Insufficient surface depth — need ≥2 expiries per day")

    log.info("Surface: %d dates x %d expiries", *surface.shape)

    # Log spreads (feature panel)
    X = log_adjacent_spreads(surface)
    X.index = pd.to_datetime(X.index)
    log.info("Feature panel (log spreads): %d x %d  [%s → %s]",
             *X.shape, X.index[0].date(), X.index[-1].date())

    #  Quotes 
    all_quotes = build_quotes(surface, symbol, fees_cfg)
    log.info("Quotes: %d calendar combos", len(all_quotes))

    #  Model hub 
    hub = build_from_settings(settings.get("models", {}))

    #  Folds 
    folds = make_folds(surface, settings, args.n_folds)

    #  Sizing config 
    saz  = settings.get("signal_and_sizing", {})
    scfg = SizingConfig(
        z_clip=float(saz.get("z_clip", 3.0)),
        risk_per_curve_usd=float(saz.get("risk_per_curve_usd", 100_000.0)),
        per_trade_notional_cap_usd=float(
            risk_cfg.get("sizing_policy", {}).get("per_trade_notional_cap_usd", 150_000.0)
        ),
        vol_target_enabled=bool(saz.get("vol_target", {}).get("enabled", True)),
        vol_lookback_days=int(saz.get("vol_target", {}).get("lookback_days", 60)),
        annual_vol_target=float(saz.get("vol_target", {}).get("annual_vol_target", 0.10)),
    )

    #  Executor to simulate trades with costs + slippage. Both come from settings.yaml, instantitated per combo
    cost_pr = settings.get("execution_sim", {}).get("cost_profile", "conservative")
    slip_pr = settings.get("execution_sim", {}).get("slippage_profile", "conservative")
    executor = ComboExecutionSimulator(
        fee_model=YamlFeeModel(symbol, cost_pr, ROOT),
        slip_model=YamlSlippageModel(symbol, slip_pr, ROOT),
    )

    backtester = Backtester(executor=executor, cfg=BacktestConfig())

    #  Quote provider (closure) - provides quotes for a given date range, used by backtester.run() per fold
    def quotes_provider(d0: date, d1: date) -> dict[str, dict]:
        d0_ts, d1_ts = pd.Timestamp(d0), pd.Timestamp(d1)
        sv   = surface.loc[d0_ts:d1_ts]
        q    = build_quotes(sv, symbol, fees_cfg)
        return q

    #  Run walk-forward backtest
    wf = AnchoredWalkForward(
        folds=folds,
        embargo_days=int(settings.get("walkforward", {}).get("embargo_days", 5)),
    )

    log.info("Running %d folds ...", len(folds))
    results = wf.run(
        X=X,
        model_hub=hub,
        signal_builder=cms_signal_builder,
        quotes_provider=quotes_provider,
        backtester=backtester,
        sizing_fn=size_from_signal,
        sizing_cfg=scfg,
    )

    #  Result collation 
    pnl_parts   = [r.pnl_path   for r in results if not r.pnl_path.empty]
    trade_parts = [r.trade_log  for r in results if not r.trade_log.empty]

    pnl_all    = pd.concat(pnl_parts,   axis=0).sort_index() if pnl_parts   else pd.Series(dtype=float)
    trades_all = pd.concat(trade_parts, axis=0).reset_index(drop=True) if trade_parts else pd.DataFrame()

    #  Summary 
    if not pnl_all.empty:
        mu     = pnl_all.mean()
        sd     = pnl_all.std(ddof=1) if len(pnl_all) > 1 else 0.0
        sharpe = (mu / sd * np.sqrt(252.0)) if sd > 0 else 0.0
        total  = pnl_all.sum()
        dd_peak = (pnl_all.cumsum() - pnl_all.cumsum().cummax()).min()

        print("\n" + "="*50)
        print(f"  WALK-FORWARD SUMMARY — {symbol}")
        print("="*50)
        print(f"  Total P&L (net of costs):  ${total:>12,.0f}")
        print(f"  Annualised Sharpe:          {sharpe:>8.2f}")
        print(f"  Daily μ:                   ${mu:>10,.2f}")
        print(f"  Daily σ:                   ${sd:>10,.2f}")
        print(f"  Max Drawdown:              ${dd_peak:>10,.0f}")
        print(f"  Trading days:              {len(pnl_all):>8,}")
        print(f"  Total trades:              {len(trades_all):>8,}")
        print("="*50 + "\n")
    else:
        log.warning("No P&L generated across all folds.")
        log.warning("Check alignment with:  python scripts/validate_pipeline.py --symbol %s", symbol)

    # ── Persist ───────────────────────────────────────────────────────────
    from datetime import datetime
    out_dir = ensure_dir(ROOT / "reports" / "pm_brief" / symbol)
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")

    pnl_all.to_csv(out_dir / f"pnl_path_{ts}.csv")
    trades_all.to_csv(out_dir / f"trade_log_{ts}.csv", index=False)
    log.info("Wrote outputs to %s", out_dir)


if __name__ == "__main__":
    main()
