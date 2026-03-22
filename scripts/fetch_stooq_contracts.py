"""
scripts/fetch_stooq_contracts.py

Fetching individual futures contract daily settle data from Stooq.com.
Free, no API key, covers CL/NG/RB/HO from ~2010 and metals from ~2000.

Stooq URL pattern:
  https://stooq.com/q/d/l/?s=clf24.f&i=d
  = CL (Crude), F month code (January), 24 (2024), daily bars

Saves to data/raw/<SYMBOL>/ in the exact schema that src/data/loader.py expects.

Usage:
  # Energy only (quickest first run, ~10-20 min):
  python scripts/fetch_stooq_contracts.py --symbols CL NG --start-year 2015

  # Full universe fetch (~45-60 min with polite rate limiting):
  python scripts/fetch_stooq_contracts.py --symbols CL NG RB HO GC SI HG --start-year 2015

Note on metals expiry approximation:
  COMEX Gold (GC) officially expires on the last business day of the contract
  month; Silver (SI) on the last business day prior to the delivery period.
  The expiry_approx() function uses the 20th of the prior month as a
  conservative placeholder — sufficient for backtesting.  To get exact FND/LTD
  dates, populate data/meta/contracts/<SYMBOL>_meta.csv from the CME website.
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

#  Month code helpers 
MONTH_CODES = list("FGHJKMNQUVXZ")          # Jan-Dec
MONTH_CODE_TO_INT = {c: i + 1 for i, c in enumerate(MONTH_CODES)}

#  Universe definition 
# Month cycles must match settings.yaml universe.curves.<SYMBOL>.month_cycle.
# ALL-month products: every calendar month is a listed contract.
# Quarterly / bi-monthly: only the delivery months listed trade actively.
UNIVERSE: dict[str, list[str]] = {
    #  Energy (NYMEX)  all months
    "CL": list("FGHJKMNQUVXZ"),   # WTI Crude Oil
    "NG": list("FGHJKMNQUVXZ"),   # Henry Hub Natural Gas
    "RB": list("FGHJKMNQUVXZ"),   # RBOB Gasoline
    "HO": list("FGHJKMNQUVXZ"),   # NY Harbor Heating Oil / ULSD

    #  Metals (COMEX)  specific delivery months only
    # Gold: Feb(G), Apr(J), Jun(M), Aug(Q), Oct(V), Dec(Z)
    "GC": list("GJMQVZ"),
    # Silver: Jan(F), Mar(H), May(K), Jul(N), Sep(U), Dec(Z)
    "SI": list("FHKNUZ"),
    # Copper: Mar(H), May(K), Jul(N), Sep(U), Dec(Z)
    "HG": list("HKNUZ"),
}


#  Helpers 

def stooq_ticker(symbol: str, month_code: str, year_4: int) -> str:
    """e.g. CL + F + 2024 → clf24.f"""
    y2 = year_4 % 100
    return f"{symbol.lower()}{month_code.lower()}{y2:02d}.f"


def expiry_approx(month_code: str, year_4: int) -> dt.date:
    """Approximate contract expiry used as a unique key for each contract.

    For energy (CL/NG): expires ~3 business days before the 25th of the month
    PRIOR to the delivery month.  We use the 20th of that prior month.

    For metals (GC/SI/HG): COMEX expires on the last business day of the
    contract month; we approximate with the 28th (conservative placeholder).
    The actual FND/LTD should come from data/meta/contracts/<SYM>_meta.csv
    once populated from the CME website.
    """
    delivery_month = MONTH_CODE_TO_INT[month_code]
    if delivery_month == 1:
        exp_month, exp_year = 12, year_4 - 1
    else:
        exp_month, exp_year = delivery_month - 1, year_4

    # Metals expire within the contract month; energy expires prior month.
    # Use 28th as a safe upper bound for metals, 20th for energy.
    day = 28 if month_code in set("GJMQVZ") | set("FHKNUZ") else 20
    try:
        return dt.date(exp_year, exp_month, day)
    except ValueError:
        return dt.date(exp_year, exp_month, 20)


def fetch_one_contract(
    symbol: str,
    month_code: str,
    year_4: int,
    session: requests.Session,
) -> pd.DataFrame:
    """Download a single contract from Stooq and return a normalised DataFrame."""
    ticker = stooq_ticker(symbol, month_code, year_4)
    url    = f"https://stooq.com/q/d/l/?s={ticker}&i=d"

    try:
        resp = session.get(url, timeout=20)
    except Exception as exc:
        log.debug("Network error for %s: %s", ticker, exc)
        return pd.DataFrame()

    if resp.status_code != 200:
        log.debug("HTTP %s for %s", resp.status_code, ticker)
        return pd.DataFrame()

    text = resp.text.strip()
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

    df = pd.DataFrame()
    df["date"]   = pd.to_datetime(raw["Date"], errors="coerce").dt.date
    df["symbol"] = symbol
    df["expiry"] = expiry

    close_col = next(
        (c for c in raw.columns if c.lower() in ("close", "settle")), None
    )
    df["settle"] = (
        pd.to_numeric(raw[close_col], errors="coerce") if close_col else float("nan")
    )
    df["last"]   = df["settle"]
    df["bid"]    = float("nan")
    df["ask"]    = float("nan")

    vol_col = next((c for c in raw.columns if c.lower() == "volume"), None)
    df["volume"] = (
        pd.to_numeric(raw[vol_col], errors="coerce").fillna(0).astype(int)
        if vol_col else 0
    )
    df["open_interest"] = 0   # Stooq does not publish OI

    df = df.dropna(subset=["date", "settle"])
    df = df[df["settle"] > 0]
    df = df.drop_duplicates(subset=["date"])
    return df.reset_index(drop=True)


#  Main filtering and front N selection logic is in src/data/contracts.py, not here.

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch individual futures contracts from Stooq"
    )
    parser.add_argument(
        "--symbols", nargs="+", default=["CL"],
        help="Root symbols to fetch, e.g. CL NG GC SI HG RB HO",
    )
    parser.add_argument(
        "--start-year", type=int, default=2015,
        help="First delivery year to include (default: 2015)",
    )
    parser.add_argument(
        "--end-year", type=int, default=dt.date.today().year + 1,
        help="Last delivery year to include (default: current year + 1)",
    )
    parser.add_argument(
        "--sleep", type=float, default=0.4,
        help="Seconds between requests — keep ≥0.3 to be polite to Stooq",
    )
    args = parser.parse_args()

    configure_logging(level="INFO", log_dir=ROOT / "logs")

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (academic research)"})

    for symbol in [s.upper() for s in args.symbols]:
        if symbol not in UNIVERSE:
            log.warning(
                "Symbol %s not in UNIVERSE dict — skipping. "
                "Add it to UNIVERSE with the correct month list first.", symbol
            )
            continue

        months        = UNIVERSE[symbol]
        out_dir       = ensure_dir(ROOT / "data" / "raw" / symbol)
        total_rows    = 0
        contract_count = 0

        log.info(
            "=== %s: fetching delivery years %d–%d (%d months/year) ===",
            symbol, args.start_year, args.end_year, len(months),
        )

        for year in range(args.start_year, args.end_year + 1):
            for mc in months:
                approx_exp = expiry_approx(mc, year)

                # Skip contracts that expired well before our window
                if approx_exp < dt.date(args.start_year - 1, 1, 1):
                    continue
                # Skip contracts too far in the future (no data yet)
                if approx_exp > dt.date.today() + dt.timedelta(days=400):
                    continue

                log.info("  fetching %s%s%02d ...", symbol, mc, year % 100)
                df = fetch_one_contract(symbol, mc, year, session)

                if not df.empty:
                    fname = f"{symbol}{mc}{year % 100:02d}.csv"
                    df.to_csv(out_dir / fname, index=False)
                    total_rows    += len(df)
                    contract_count += 1
                    log.info(
                        "    ✓ %d rows  [%s → %s]",
                        len(df), df["date"].min(), df["date"].max(),
                    )
                else:
                    log.debug("    – no data for %s%s%02d", symbol, mc, year % 100)

                time.sleep(args.sleep)

        log.info(
            "%s complete: %d contracts, %d total rows → %s",
            symbol, contract_count, total_rows, out_dir,
        )


if __name__ == "__main__":
    main()