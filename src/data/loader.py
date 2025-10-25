# src/data/loader.py
# IO layer for raw/curated data, metadata, and trading calendars
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

import pandas as pd

from ..core.utils import ensure_dir, project_root

REQUIRED_BAR_COLS = {
    "date", "symbol", "expiry", "settle", "last", "bid", "ask", "volume", "open_interest"
}


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parq", ".parquet"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def _normalise_bars(df: pd.DataFrame) -> pd.DataFrame:
    # Standard column set and types
    cols = {
        "Date": "date",
        "TradingDate": "date",
        "Symbol": "symbol",
        "Root": "symbol",
        "Expiry": "expiry",
        "Expiration": "expiry",
        "Settle": "settle",
        "Close": "settle",
        "Last": "last",
        "Bid": "bid",
        "Ask": "ask",
        "Volume": "volume",
        "OpenInterest": "open_interest",
        "Open Interest": "open_interest",
    }
    df = df.rename(columns={k: v for k, v in cols.items() if k in df.columns})
    # Lowercase
    df.columns = [c.lower() for c in df.columns]
    # Ensure all expected columns exist
    for c in REQUIRED_BAR_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    # Types
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
    num_cols = ["settle", "last", "bid", "ask"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("Int64")
    df["open_interest"] = pd.to_numeric(df["open_interest"], errors="coerce").astype("Int64")
    df["symbol"] = df["symbol"].astype(str)
    # Drop rows with no symbol or expiry
    df = df.dropna(subset=["symbol", "expiry"]).reset_index(drop=True)
    return df[list(REQUIRED_BAR_COLS)]


def load_contract_bars(symbol: str, root: Optional[Path] = None) -> pd.DataFrame:
    """Load all raw bars for a root symbol from data/raw/<symbol>/.

    Supports CSV and Parquet. Returns a normalised DataFrame with the standard columns.
    """
    root = project_root(root)
    folder = root / "data" / "raw" / symbol
    if not folder.exists():
        raise FileNotFoundError(f"No raw data folder for {symbol}: {folder}")
    frames: list[pd.DataFrame] = []
    for p in sorted(folder.glob("**/*")):
        if p.suffix.lower() not in {".csv", ".parq", ".parquet"}:
            continue
        df = _read_any(p)
        frames.append(_normalise_bars(df))
    if not frames:
        raise FileNotFoundError(f"No data files found under {folder}")
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["date", "expiry"])  # stable order
    return out


def load_contract_meta(symbol: str, root: Optional[Path] = None) -> pd.DataFrame:
    """Load per-expiry metadata from data/meta/contracts/<symbol>_meta.csv if present.

    Expected columns: expiry, first_notice_date, last_trade_date, tick_size, multiplier, currency, month_code, year
    Missing columns are filled with NA. Dates are parsed to Python date.
    """
    root = project_root(root)
    path = root / "data" / "meta" / "contracts" / f"{symbol}_meta.csv"
    if not path.exists():
        # Return empty frame with the schema so joins work downstream
        cols = [
            "expiry", "first_notice_date", "last_trade_date", "tick_size", "multiplier", "currency", "month_code", "year"
        ]
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path)
    df = df.rename(columns={c: c.lower() for c in df.columns})
    for c in ["expiry", "first_notice_date", "last_trade_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.date
    for c in ["tick_size", "multiplier"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "currency" in df.columns:
        df["currency"] = df["currency"].fillna("USD")
    return df


def load_holidays(exchange: str, root: Optional[Path] = None) -> set:
    """Load exchange holiday dates from config/calendars/<exchange>_holidays.csv.
    Returns a set of Python date objects.
    """
    root = project_root(root)
    path = root / "config" / "calendars" / f"{exchange.lower()}_holidays.csv"
    if not path.exists():
        # Allow missing calendars during prototyping
        return set()
    s = pd.read_csv(path, header=None)[0]
    return set(pd.to_datetime(s, errors="coerce").dt.date.dropna().tolist())


def write_curated_snapshot(symbol: str, df: pd.DataFrame, snapshot_name: str, root: Optional[Path] = None) -> Path:
    """Persist a curated snapshot under data/curated/<symbol>/<snapshot_name>.parquet"""
    root = project_root(root)
    folder = ensure_dir(root / "data" / "curated" / symbol)
    path = folder / f"{snapshot_name}.parquet"
    df.to_parquet(path, index=True)
    return path


def validate_bars(df: pd.DataFrame) -> Tuple[bool, list[str]]:
    """Basic invariants: columns present, dates sorted, no duplicate (date, expiry)."""
    errors: list[str] = []
    missing = REQUIRED_BAR_COLS - set(df.columns)
    if missing:
        errors.append(f"Missing columns: {sorted(missing)}")
    # date monotonic is not required globally, but duplicates per (date, expiry) are not allowed
    idx = df.set_index(["date", "expiry"]).index
    dup = idx.duplicated()
    if dup.any():
        errors.append("Duplicate (date, expiry) rows detected")
    ok = len(errors) == 0
    return ok, errors
