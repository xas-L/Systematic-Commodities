# scripts/gen_pm_brief.py
# Generate PM-facing brief tables from walk-forward outputs
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.utils import ensure_dir
from src.core.logging import configure_logging, get_logger

log = get_logger(__name__)


def _summary_table(pnl: pd.Series) -> pd.DataFrame:
    mu = pnl.mean()
    sd = pnl.std(ddof=1)
    sharpe = (mu / sd * np.sqrt(252.0)) if sd > 0 else 0.0
    out = pd.DataFrame({
        "Total $": [pnl.sum()],
        "Daily μ": [mu],
        "Daily σ": [sd],
        "Sharpe": [sharpe],
        "Days": [len(pnl)],
    })
    return out


def main():
    p = argparse.ArgumentParser(description="Generate PM brief tables from reports/pm_brief/<SYMBOL>/ outputs")
    p.add_argument("--symbol", default="CL")
    args = p.parse_args()

    configure_logging(level="INFO", log_dir=ROOT / "logs")
    sym = args.symbol.upper()

    rep_dir = ROOT / "reports" / "pm_brief" / sym
    pnl_files = sorted(rep_dir.glob("pnl_path*.csv")) or [rep_dir / "pnl_path.csv"]
    trade_files = sorted(rep_dir.glob("trade_log*.csv")) or [rep_dir / "trade_log.csv"]

    # Load most recent
    pnl = pd.read_csv(pnl_files[-1], index_col=0, parse_dates=True, squeeze=True)
    pnl = pnl.iloc[:, 0] if isinstance(pnl, pd.DataFrame) else pnl
    trades = pd.read_csv(trade_files[-1])

    # Summary
    summary = _summary_table(pnl)

    # Trades by combo
    by_combo = trades.groupby("combo").agg(
        fills=("filled", "sum"),
        avg_price=("avg_px", "mean"),
        fees=("fees", "sum"),
        n_trades=("combo", "count"),
    ).sort_values("n_trades", ascending=False)

    # Save
    out_dir = ensure_dir(rep_dir)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    summary.to_csv(out_dir / f"pm_summary_{ts}.csv", index=False)
    by_combo.to_csv(out_dir / f"pm_trades_{ts}.csv")
    log.info("Wrote %s and %s", out_dir / f"pm_summary_{ts}.csv", out_dir / f"pm_trades_{ts}.csv")


if __name__ == "__main__":
    main()
