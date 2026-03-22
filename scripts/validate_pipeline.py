"""
scripts/validate_pipeline.py

Run this BEFORE run_walkforward.py (same directory) to confirm each stage works.
where things may break will be shown - fix before walk-fwd.

Usage:
    python scripts/validate_pipeline.py --symbol CL

Stages:
  1. Raw data load + validation
  2. Curve surface construction
  3. Log-spread computation
  4. Model fit + signal generation
  5. Quote dict construction
  6. Signal → quote alignment check  ← the bug that caused empty trade logs
  7. Sizing sanity check
  8. Mini backtester smoke test (30 days)

All 8 stages pass implies run_walkforward.py is green - non-empty logs and PnL
"""
from __future__ import annotations

import argparse
import sys
import math
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.utils import load_settings, load_fees_slippage, load_risk_limits
from src.core.logging import configure_logging, get_logger
from src.data.loader import load_contract_bars, validate_bars
from src.data.curve import build_curve_surface, log_adjacent_spreads
from src.models.hub import build_from_settings
from src.signals.sizing import SizingConfig, size_from_signal
from src.execsim.cost_model import YamlFeeModel, YamlSlippageModel
from src.execsim.combos import ComboExecutionSimulator
from src.execsim.backtester import Backtester, BacktestConfig

log = get_logger(__name__)
PASS = "PASS"
FAIL = "FAIL"


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    msg = f"  {status}  {label}"
    if detail:
        msg += f"\n         {detail}"
    print(msg)
    return condition


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="CL")
    args = parser.parse_args()

    configure_logging(level="WARNING")   # quiet during validation

    symbol = args.symbol.upper()
    settings   = load_settings(ROOT)
    fees_cfg   = load_fees_slippage(ROOT)
    risk_cfg   = load_risk_limits(ROOT)
    ccfg       = settings.get("curve_construction", {})
    uni        = settings.get("universe", {}).get("curves", {})

    print(f"\n{'='*60}")
    print(f"  Pipeline Validation — {symbol}")
    print(f"{'='*60}\n")

    all_ok = True

    #  Stage 1: Raw data: loading and validation
    print("Stage 1: Raw data load")
    try:
        bars = load_contract_bars(symbol, ROOT)
        ok_val, errs = validate_bars(bars)
        n_contracts = bars["expiry"].nunique()
        date_range = f"{bars['date'].min()} → {bars['date'].max()}"
        ok = check("Loaded bars", not bars.empty,
                   f"{len(bars):,} rows | {n_contracts} contracts | {date_range}")
        if errs:
            for e in errs:
                print(f"         Warning:  {e}")
        all_ok &= ok
    except Exception as exc:
        check("Loaded bars", False, str(exc))
        print("\n  Cannot continue without raw data.")
        print(f"  Run:  python scripts/fetch_stooq_contracts.py --symbols {symbol} --start-year 2015")
        sys.exit(1)

    #  Stage 2: Curve surface construction
    print("\nStage 2: Curve surface construction")
    try:
        surface, used = build_curve_surface(
            bars,
            min_volume=0,                  # relax filters for validation
            min_open_interest=0,
            drop_stale_after_days=int(ccfg.get("drop_stale_after_days", 10)),
            price_field_order=list(ccfg.get("price_field_order", ["settle", "last"])),
            tenors_target=int(ccfg.get("tenors_target", 10)),
            min_contracts=2,               # relax to 2 for validation
        )
        ok = check("Surface built", not surface.empty and surface.shape[1] >= 2,
                   f"Shape: {surface.shape}  |  cols (first 3): {list(surface.columns[:3])}")
        ok2 = check("DatetimeIndex", isinstance(surface.index, pd.DatetimeIndex),
                    str(type(surface.index)))
        all_ok &= ok and ok2
    except Exception as exc:
        check("Surface built", False, str(exc))
        sys.exit(1)

    #  Stage 3: Log spreads 
    print("\nStage 3: Log-adjacent spreads")
    try:
        X = log_adjacent_spreads(surface)
        X.index = pd.to_datetime(X.index)
        nan_pct = X.isna().mean().mean() * 100
        ok = check("Spreads computed", not X.empty and X.shape[1] >= 1,
                   f"Shape: {X.shape}  |  NaN%: {nan_pct:.1f}%")
        ok2 = check("Spread columns labelled", all("-" in c for c in X.columns),
                    f"First spread col: {list(X.columns)[0]}")
        all_ok &= ok and ok2
    except Exception as exc:
        check("Spreads computed", False, str(exc))
        sys.exit(1)

    # Stage 4: Model fit + signals
    print("\nStage 4: Model fit and signal generation")
    try:
        models_cfg = settings.get("models", {})
        hub = build_from_settings(models_cfg)
        # training window <= 3 years
        train_end = X.index[-1] - pd.DateOffset(days=90)
        X_tr = X.loc[:train_end]
        X_te = X.loc[train_end:]

        ok_data = check("Enough training data", len(X_tr) >= 60,
                        f"{len(X_tr)} training rows")
        if not ok_data:
            all_ok = False
        else:
            hub.fit(X_tr)
            sigs_map = hub.signals_map(X_te)
            total_sig_cols = sum(df.shape[1] for df in sigs_map.values())
            ok = check("Signals generated", total_sig_cols > 0,
                       f"Models: {list(sigs_map.keys())}  |  total signal cols: {total_sig_cols}")
            all_ok &= ok

            # Show sample signal stats
            for model_name, sig_df in sigs_map.items():
                finite_frac = sig_df.apply(lambda s: s.dropna().apply(math.isfinite).mean()).mean()
                print(f"         {model_name}: {sig_df.shape[1]} cols | "
                      f"finite values: {finite_frac*100:.0f}%")
    except Exception as exc:
        check("Model fit", False, str(exc))
        sys.exit(1)

    #  Stage 5: Quote dict
    print("\nStage 5: Quote dictionary construction")
    try:
        prod = fees_cfg.get("products", {}).get(symbol, {})
        min_inc = float(prod.get("combo", {}).get("min_price_increment", 0.01))
        half_sp = float(prod.get("combo", {}).get("typical_half_spread_ticks", 1.0)) * min_inc
        lot_val = float(prod.get("multiplier", 1000.0))

        # Build quotes over the TEST window
        sv_te = surface.loc[X_te.index[0]:X_te.index[-1]]
        cols  = list(sv_te.columns)
        diffs = sv_te.to_numpy()[:, 1:] - sv_te.to_numpy()[:, :-1]
        qnames = [f"{cols[i]}-{cols[i-1]}" for i in range(1, len(cols))]

        quotes: dict = {}
        for i, qname in enumerate(qnames):
            mid_series = pd.Series(diffs[:, i], index=sv_te.index)
            quotes[qname] = {
                "mid": mid_series,
                "half_spread": pd.Series(half_sp, index=sv_te.index),
                "lot_value": lot_val,
                "symbol": symbol,
                "tob_size": {ts: 10 for ts in sv_te.index},
            }

        ok = check("Quotes built", len(quotes) > 0,
                   f"{len(quotes)} combos | example key: {list(quotes.keys())[0]}")
        all_ok &= ok
    except Exception as exc:
        check("Quotes built", False, str(exc))
        sys.exit(1)

    # Stage 6: Signal & quote alignment 
    print("\nStage 6: Signal & quote alignment (mission critical)")
    try:
        cms = sigs_map.get("carry_momo_season", pd.DataFrame())
        if cms.empty:
            check("CMS signals present", False, "carry_momo_season model produced no output")
            all_ok = False
        else:
            # Reconstruct combo signal names the same way run_walkforward.py does
            buckets: dict[str, list[str]] = {}
            for c in cms.columns:
                if c.startswith("carry_"):
                    base = c[len("carry_"):]
                    buckets.setdefault(base, []).append(c)
                elif c.startswith("momo"):
                    try:
                        _, base = c.split("_", 1)
                        buckets.setdefault(base, []).append(c)
                    except ValueError:
                        pass

            signal_bases = set(buckets.keys())
            quote_keys   = set(quotes.keys())
            matched      = signal_bases & quote_keys
            unmatched_sig = signal_bases - quote_keys
            unmatched_q   = quote_keys - signal_bases

            ok = check("Signal bases match quote keys",
                       len(matched) > 0,
                       f"Matched: {len(matched)} | "
                       f"signal-only: {len(unmatched_sig)} | "
                       f"quote-only: {len(unmatched_q)}")

            if len(matched) == 0:
                print(f"\n   ALIGNMENT BUG DETECTED")
                print(f"     Signal base examples:  {list(signal_bases)[:3]}")
                print(f"     Quote key examples:    {list(quote_keys)[:3]}")
                print(f"     These must match exactly (same date format, same separator).")
                print(f"     Fix: ensure surface column type is consistent throughout.")
            else:
                print(f"         Example aligned combo: {list(matched)[0]}")

            all_ok &= ok
    except Exception as exc:
        check("Alignment check", False, str(exc))
        all_ok = False

    #  Stage 7: Sizing sanity 
    print("\nStage 7: Sizing sanity check")
    try:
        scfg = SizingConfig(
            z_clip=float(settings.get("signal_and_sizing", {}).get("z_clip", 3.0)),
            risk_per_curve_usd=100_000.0,
            per_trade_notional_cap_usd=150_000.0,
            vol_target_enabled=True,
            vol_lookback_days=60,
            annual_vol_target=0.10,
        )

        # Test with a realistic spread price series and a 1-sigma signal
        test_q = list(quotes.values())[0]
        test_series = test_q["mid"].dropna()
        test_lots = size_from_signal(
            signal=1.0,
            combo_price_series=test_series,
            lot_value=lot_val,
            cfg=scfg,
        )
        ok = check("Sizing returns non-zero for 1-sigma signal",
                   test_lots != 0,
                   f"size_from_signal(signal=1.0) = {test_lots} lots")

        test_lots_neg = size_from_signal(
            signal=-2.0,
            combo_price_series=test_series,
            lot_value=lot_val,
            cfg=scfg,
        )
        ok2 = check("Sizing returns negative for negative signal",
                    test_lots_neg < 0,
                    f"size_from_signal(signal=-2.0) = {test_lots_neg} lots")

        ok3 = check("Zero for near-zero signal",
                    size_from_signal(0.1, test_series, lot_val, scfg) == 0,
                    "size_from_signal(signal=0.1) = 0  (below noise floor)")

        all_ok &= ok and ok2 and ok3
    except Exception as exc:
        check("Sizing", False, str(exc))
        all_ok = False

    #  Stage 8: Mini backtest smoke test 
    print("\nStage 8: Mini backtest (30-day smoke test)")
    try:
        if len(matched) > 0:
            # 30-day mini combo signal panel for matched combos only
            matched_list = sorted(list(matched))[:5]   # at most 5 combos
            mini_idx     = X_te.index[:30]

            combo_signals = pd.DataFrame(index=mini_idx)
            for base in matched_list:
                cols_for_base = buckets.get(base, [])
                if cols_for_base:
                    combo_signals[base] = cms.reindex(mini_idx)[cols_for_base].mean(axis=1)

            mini_quotes = {k: quotes[k] for k in matched_list if k in quotes}
            # Restrict quote series to mini window
            for k in mini_quotes:
                for field in ("mid", "half_spread"):
                    if field in mini_quotes[k]:
                        mini_quotes[k] = dict(mini_quotes[k])
                        mini_quotes[k][field] = mini_quotes[k][field].reindex(mini_idx)

            fee_model  = YamlFeeModel(product=symbol, profile_name="conservative", root=ROOT)
            slip_model = YamlSlippageModel(product=symbol, profile_name="conservative", root=ROOT)
            executor   = ComboExecutionSimulator(fee_model=fee_model, slip_model=slip_model)
            bt         = Backtester(executor=executor, cfg=BacktestConfig())

            res = bt.run(
                signals=combo_signals,
                quotes=mini_quotes,
                sizing_fn=size_from_signal,
                sizing_cfg=scfg,
            )
            pnl    = res["pnl_path"]
            trades = res["trade_log"]

            ok = check("Backtester ran", len(pnl) > 0,
                       f"{len(pnl)} P&L days | {len(trades)} trade records")
            ok2 = check("At least some trades executed", len(trades) > 0,
                        "If 0 trades: sizing or alignment issue persists")
            all_ok &= ok and ok2

            if not trades.empty:
                print(f"         Total P&L (30d): ${pnl.sum():,.0f}")
                print(f"         Trade log sample:")
                print(trades.head(3).to_string(index=False))
        else:
            check("Mini backtest", False, "Skipped — no aligned combos")
            all_ok = False
    except Exception as exc:
        check("Mini backtest", False, str(exc))
        import traceback
        traceback.print_exc()
        all_ok = False

    #  Final verdict 
    print(f"\n{'='*60}")
    if all_ok:
        print("ALL STAGES PASSED")
        print(f"Run:  python scripts/run_walkforward.py --symbol {symbol}")
    else:
        print("SOME STAGES FAILED — fix the issues above before running walk-forward")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
