# scripts/build_curve_snapshot.py
# Build and persist a point-in-time curve surface + log-spreads snapshot
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.utils import ensure_dir, load_settings, load_fees_slippage
from src.core.logging import configure_logging, get_logger
from src.data.loader import load_contract_bars, write_curated_snapshot
from src.data.curve import build_curve_surface, log_adjacent_spreads

log = get_logger(__name__)


def main():
    p = argparse.ArgumentParser(description="Build curve surface snapshot and save Parquet")
    p.add_argument("--symbol", default="CL")
    p.add_argument("--snapshot", default=None, help="Snapshot name (default: AUTO timestamp)")
    args = p.parse_args()

    settings = load_settings(ROOT)
    log_level = settings.get("ops", {}).get("logging", {}).get("level", "INFO")
    configure_logging(level=log_level, log_dir=ROOT / "logs")

    symbol = args.symbol.upper()

    # Load raw
    bars = load_contract_bars(symbol, ROOT)

    # Build surface
    ccfg = settings.get("curve_construction", {})
    uni = settings.get("universe", {}).get("curves", {})
    surface, used = build_curve_surface(
        bars,
        min_volume=int(ccfg.get("filter", {}).get("min_volume", 0)),
        min_open_interest=int(ccfg.get("filter", {}).get("min_open_interest", 0)),
        drop_stale_after_days=int(ccfg.get("drop_stale_after_days", 10)),
        price_field_order=list(ccfg.get("price_field_order", ["settle", "last"])),
        tenors_target=int(ccfg.get("tenors_target", 10)),
        min_contracts=int(uni.get(symbol, {}).get("min_contracts", 6)),
    )

    # Compute log spreads
    logsp = log_adjacent_spreads(surface)

    # Save
    snap_name = args.snapshot or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = ensure_dir(ROOT / "data" / "curated" / symbol)
    surface.to_parquet(out_dir / f"surface_{snap_name}.parquet")
    logsp.to_parquet(out_dir / f"logsp_{snap_name}.parquet")
    log.info("Wrote %s and %s", out_dir / f"surface_{snap_name}.parquet", out_dir / f"logsp_{snap_name}.parquet")


if __name__ == "__main__":
    main()
