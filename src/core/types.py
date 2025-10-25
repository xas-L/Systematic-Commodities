# src/core/types.py
# Central type definitions shared across the codebase.
# Keep light and dependency-minimal (pydantic only) so other modules import cheaply.
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Literal, TypedDict, Any
from datetime import date, datetime

from pydantic import BaseModel, Field, field_validator


# -----------------------------
# Enums & literals
# -----------------------------
class Side(int, Enum):
    BUY = 1
    SELL = -1


class OrderTIF(str, Enum):
    DAY = "DAY"
    GTC = "GTC"


class StrategyKind(str, Enum):
    CALENDAR = "calendar"
    BUTTERFLY = "butterfly"


class Sector(str, Enum):
    ENERGY = "ENERGY"
    METALS = "METALS"
    GRAINS = "GRAINS"
    OILSEEDS = "OILSEEDS"
    OTHER = "OTHER"


# -----------------------------
# Contract specification & metadata
# -----------------------------
class ContractSpec(BaseModel):
    symbol: str
    exchange: str
    tick_size: float
    multiplier: float
    currency: str = "USD"

    @property
    def tick_value(self) -> float:
        return self.tick_size * self.multiplier


class ContractMeta(BaseModel):
    """Point-in-time metadata that can change across vintages (FND/LTD etc.)."""

    symbol: str
    expiry: date
    first_notice_date: Optional[date] = None
    last_trade_date: date
    month_code: Optional[str] = None
    year: Optional[int] = None
    spec: ContractSpec

    @field_validator("month_code")
    @classmethod
    def _upper(cls, v: Optional[str]) -> Optional[str]:
        return v.upper() if isinstance(v, str) else v

    @property
    def contract_code(self) -> str:
        """Human-friendly code like CLZ5 if month_code/year are known, else ISO date."""
        if self.month_code and self.year:
            y = str(self.year)[-1]
            return f"{self.symbol}{self.month_code}{y}"
        return f"{self.symbol}_{self.expiry.isoformat()}"


class ContractBar(BaseModel):
    """One day (or bar) of data for a specific contract expiry."""

    date: date
    symbol: str
    expiry: date
    settle: Optional[float] = None
    last: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = Field(default=None, alias="openInterest")

    model_config = dict(populate_by_name=True)


# -----------------------------
# Exchange combo order primitives (used both in exec simulator and live harness)
# -----------------------------
class ComboLeg(BaseModel):
    expiry: date
    side: Side
    ratio: int = 1


class ComboOrder(BaseModel):
    symbol: str  # root, e.g., CL
    legs: list[ComboLeg]
    qty: int
    limit_price: Optional[float] = None
    tif: OrderTIF = OrderTIF.DAY
    strategy: StrategyKind = StrategyKind.CALENDAR
    client_order_id: Optional[str] = None
    ts_created: datetime = Field(default_factory=datetime.utcnow)


class OrderStatus(str, Enum):
    NEW = "NEW"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class Fill(BaseModel):
    ts: datetime
    price: float
    qty: int
    fees: float = 0.0


class ComboExecutionReport(BaseModel):
    order: ComboOrder
    status: OrderStatus
    filled_qty: int
    avg_fill_price: Optional[float] = None
    fees_total: float = 0.0
    remaining_qty: int = 0
    message: Optional[str] = None


# -----------------------------
# Market data snapshots (typed dicts keep it lightweight)
# -----------------------------
class ComboTopOfBook(TypedDict, total=False):
    combo_mid: float
    combo_bid: float
    combo_ask: float
    combo_half_spread: float
    combo_tob_size: int


class LegTopOfBook(TypedDict, total=False):
    bid: float
    ask: float
    size: int


class MarketDataSnapshot(TypedDict, total=False):
    symbol: str
    ts: datetime
    combo: ComboTopOfBook
    legs: dict[str, LegTopOfBook]  # key by ISO expiry string


# -----------------------------
# Risk & breach signalling
# -----------------------------
class RiskBreach(BaseModel):
    ts: datetime
    curve: str
    code: str               # e.g., "DD_SIGMA", "SLIPPAGE_HARD"
    magnitude: float
    action: str             # e.g., "PAUSE", "REDUCE_50", "FLATTEN"
    details: dict[str, Any] = {}


# -----------------------------
# Generic health report (used by models and ops)
# -----------------------------
class HealthReport(BaseModel):
    component: str
    ok: bool
    metrics: dict[str, float] = {}
    message: Optional[str] = None


# Lightweight timer (dataclass for zero dep)
@dataclass
class TimerMark:
    name: str
    t0: datetime

    @classmethod
    def start(cls, name: str) -> "TimerMark":
        return cls(name=name, t0=datetime.utcnow())

    def elapsed_ms(self) -> float:
        return (datetime.utcnow() - self.t0).total_seconds() * 1000.0
