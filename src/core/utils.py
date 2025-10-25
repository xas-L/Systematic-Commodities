# src/core/utils.py
# Small, sharp utilities used across the codebase. Keep side-effect free.
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional
from datetime import datetime, date, timedelta
import hashlib
import json

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    yaml = None


# -----------------------------
# Filesystem helpers
# -----------------------------

def ensure_dir(p: Path | str) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def project_root(start: Optional[Path] = None) -> Path:
    """Find repo root by looking for 'config/settings.yaml'."""
    here = Path(start or Path.cwd()).resolve()
    for parent in [here, *here.parents]:
        if (parent / "config" / "settings.yaml").exists():
            return parent
    return here  # fallback


# -----------------------------
# YAML config loading
# -----------------------------

def load_yaml(path: Path | str) -> dict:
    if yaml is None:
        raise RuntimeError("pyyaml is required to load YAML configs")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing YAML file: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_settings(root: Optional[Path] = None) -> dict:
    root = project_root(root)
    return load_yaml(root / "config" / "settings.yaml")


def load_fees_slippage(root: Optional[Path] = None) -> dict:
    root = project_root(root)
    return load_yaml(root / "config" / "fees_slippage.yaml")


def load_risk_limits(root: Optional[Path] = None) -> dict:
    root = project_root(root)
    return load_yaml(root / "config" / "risk_limits.yaml")


# -----------------------------
# Time utilities
# -----------------------------

def utc_now() -> datetime:
    return datetime.utcnow()


def to_iso(dt: datetime | date) -> str:
    if isinstance(dt, datetime):
        return dt.replace(microsecond=0).isoformat() + "Z"
    return dt.isoformat()


def add_business_days(d: date, n: int, holidays: set[date] | None = None) -> date:
    """Very simple business-day adder. For exchange-accurate calendars, use pandas_market_calendars.
    Here we exclude weekends and the provided holiday set.
    """
    holidays = holidays or set()
    step = 1 if n >= 0 else -1
    remaining = abs(n)
    cur = d
    while remaining != 0:
        cur = cur + timedelta(days=step)
        if cur.weekday() < 5 and cur not in holidays:
            remaining -= 1
    return cur


# -----------------------------
# Math helpers
# -----------------------------

def clipped(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def safe_div(num: float, den: float, default: float = 0.0) -> float:
    return default if den == 0 else num / den


# -----------------------------
# Futures month codes
# -----------------------------
_MONTH_CODE_TO_INT = {
    "F": 1,  "G": 2,  "H": 3,  "J": 4,  "K": 5,  "M": 6,
    "N": 7,  "Q": 8,  "U": 9,  "V": 10, "X": 11, "Z": 12,
}


def month_code_to_int(code: str) -> int:
    code = code.upper()
    if code not in _MONTH_CODE_TO_INT:
        raise ValueError(f"Invalid month code: {code}")
    return _MONTH_CODE_TO_INT[code]


# -----------------------------
# Hashing
# -----------------------------

def sha1_of(obj: Any) -> str:
    """Stable SHA1 for small objects (dicts/lists converted to JSON)."""
    if isinstance(obj, (dict, list)):
        payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    elif isinstance(obj, (str, bytes)):
        payload = obj if isinstance(obj, bytes) else obj.encode()
    else:
        payload = str(obj).encode()
    return hashlib.sha1(payload).hexdigest()


# -----------------------------
# Simple timing / profiling
# -----------------------------
class time_block:
    """Context manager for ad-hoc timing.

    with time_block("curve.build") as tb:
        build_curve()
    print(tb.ms)
    """

    def __init__(self, name: str = "block"):
        self.name = name
        self._t0: Optional[datetime] = None
        self.ms: float = 0.0

    def __enter__(self):
        self._t0 = datetime.utcnow()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._t0 is not None:
            self.ms = (datetime.utcnow() - self._t0).total_seconds() * 1000.0
        return False  # do not suppress exceptions
