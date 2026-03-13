"""
scripts/fetch_stooq_contracts.py

Fetching individual futures contract daily settle data from Stooq.com.
DataBento alternative - free + no API key + no registration > 2010-present for CL/NG.

Stooq URL pattern note:
  https://stooq.com/q/d/l/?s=clf24.f&i=d
  = CL (Crude), F month code (January), 24 (2024), daily bars

Saves to data/raw/<SYMBOL>/ in the exact schema that src/data/loader.py expects.

Usage:
  python scripts/fetch_stooq_contracts.py --symbols CL NG --start-year 2015
  python scripts/fetch_stooq_contracts.py --symbols CL --start-year 2018  # quicker for first run

Runtime: ~10-20 min for CL+NG 2015-present due to polite rate limiting.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from io import StringIO
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.logging import configure_logging, get_logger
from src.core.utils import ensure_dir

log = get_logger(__name__)

# ── Month code helpers ──────────────────────────────────────────────────────
MONTH_CODES = list("FGHJKMNQUVXZ")          # Jan-Dec
MONTH_CODE_TO_INT = {c: i + 1 for i, c in enumerate(MONTH_CODES)}

# ── Universe definition ─────────────────────────────────────────────────────
# Only ALL-month products here; quarterly metals (GC, SI) need a different
# month list – add them after you've got CL/NG working.
UNIVERSE: dict[str, list[str]] = {
    "CL": list("FGHJKMNQUVXZ"),   # WTI Crude – every month
    "NG": list("FGHJKMNQUVXZ"),   # Henry Hub – every month
    # "GC": list("GJMQVZ"),       # Gold – Feb,Apr,Jun,Aug,Oct,Dec
    # "RB": list("FGHJKMNQUVXZ"), # RBOB Gasoline – every month
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def stooq_ticker(symbol: str, month_code: str, year_4: int) -> str:
    """e.g. CL + F + 2024 → clf24.f"""
    y2 = year_4 % 100
    return f"{symbol.lower()}{month_code.lower()}{y2:02d}.f"


def expiry_approx(month_code: str, year_4: int) -> dt.date:
    """
    Approx. expiry date used as a KEY to distinguish contracts.
    CL expires ~3 business days before the 25th of the month prior to
    delivery. We use the 20th of the month prior as a conservative placeholder.
    The exact FND/LTD should come from data/meta/<SYMBOL>_meta.csv later;
    for backtesting purposes this placeholder is sufficient.
    """
    delivery_month = MONTH_CODE_TO_INT[month_code]
    if delivery_month == 1:
        exp_month, exp_year = 12, year_4 - 1
    else:
        exp_month, exp_year = delivery_month - 1, year_4
    try:
        return dt.date(exp_year, exp_month, 20)
    except ValueError:
        return dt.date(exp_year, exp_month, 28)


def fetch_one_contract(
    symbol: str,
    month_code: str,
    year_4: int,
    session: requests.Session,
) -> pd.DataFrame:
    """Download a single contract from Stooq and return a normalised DataFrame."""
    ticker = stooq_ticker(symbol, month_code, year_4)
    url = f"https://stooq.com/q/d/l/?s={ticker}&i=d"

    try:
        resp = session.get(url, timeout=20)
    except Exception as exc:
        log.debug("Network error for %s: %s", ticker, exc)
        return pd.DataFrame()

    if resp.status_code != 200:
        log.debug("HTTP %s for %s", resp.status_code, ticker)
        return pd.DataFrame()

    text = resp.text.strip()
    # Stooq returns "No data" or a single header line when contract doesn't exist
    if not text or len(text.splitlines()) < 3 or "No data" in text:
        return pd.DataFrame()

    try:
        raw = pd.read_csv(StringIO(text))
    except Exception as exc:
        log.debug("CSV parse error for %s: %s", ticker, exc)
        return pd.DataFrame()

    if raw.empty or "Date" not in raw.columns:
        return pd.DataFrame()

    expiry = expiry_approx(month_code, year_4)

    # Map to the canonical schema that src/data/loader.py expects
    df = pd.DataFrame()
    df["date"]         = pd.to_datetime(raw["Date"], errors="coerce").dt.date
    df["symbol"]       = symbol
    df["expiry"]       = expiry
    # Stooq "Close" = official daily settle (for futures); fall back to Open
    close_col = next((c for c in raw.columns if c.lower() in ("close", "settle")), None)
    df["settle"]       = pd.to_numeric(raw[close_col], errors="coerce") if close_col else float("nan")
    df["last"]         = df["settle"]      # stooq doesn't separate last from close
    df["bid"]          = float("nan")
    df["ask"]          = float("nan")
    vol_col = next((c for c in raw.columns if c.lower() == "volume"), None)
    df["volume"]       = pd.to_numeric(raw[vol_col], errors="coerce").fillna(0).astype(int) if vol_col else 0
    df["open_interest"] = 0               # stooq doesn't publish OI

    df = df.dropna(subset=["date", "settle"])
    df = df[df["settle"] > 0]
    df = df.drop_duplicates(subset=["date"])
    return df.reset_index(drop=True)


#  Main

def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch individual futures contracts from Stooq")
    parser.add_argument("--symbols", nargs="+", default=["CL"], help="Root symbols, e.g. CL NG")
    parser.add_argument("--start-year", type=int, default=2015, help="First delivery year to fetch")
    parser.add_argument("--end-year",   type=int, default=dt.date.today().year + 1)
    parser.add_argument("--sleep",      type=float, default=0.4, help="Seconds between requests")
    args = parser.parse_args()

    configure_logging(level="INFO", log_dir=ROOT / "logs")

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (academic research)"})

    for symbol in [s.upper() for s in args.symbols]:
        if symbol not in UNIVERSE:
            log.warning("Symbol %s not in UNIVERSE dict – skipping", symbol)
            continue

        months   = UNIVERSE[symbol]
        out_dir  = ensure_dir(ROOT / "data" / "raw" / symbol)
        total_rows = 0
        contract_count = 0

        log.info("=== %s: fetching %d–%d ===", symbol, args.start_year, args.end_year)

        for year in range(args.start_year, args.end_year + 1):
            for mc in months:
                delivery_month = MONTH_CODE_TO_INT[mc]
                delivery_date  = dt.date(year, delivery_month, 1)

                # Skip contracts that expire before our start window or too far in the future
                approx_exp = expiry_approx(mc, year)
                if approx_exp < dt.date(args.start_year - 1, 1, 1):
                    continue
                if approx_exp > dt.date.today() + dt.timedelta(days=400):
                    continue

                log.info("  fetching %s%s%02d ...", symbol, mc, year % 100)
                df = fetch_one_contract(symbol, mc, year, session)

                if not df.empty:
                    fname = f"{symbol}{mc}{year % 100:02d}.csv"
                    df.to_csv(out_dir / fname, index=False)
                    total_rows += len(df)
                    contract_count += 1
                    log.info("    ✓ %d rows  [%s → %s]", len(df), df["date"].min(), df["date"].max())
                else:
                    log.debug("    – no data")

                time.sleep(args.sleep)

        log.info(
            "%s complete: %d contracts, %d total rows → %s",
            symbol, contract_count, total_rows, out_dir,
        )


if __name__ == "__main__":
    main()
