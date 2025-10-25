# src/ops/order_status_manager.py
# Tracks lifecycle of combo orders and provides a simple queryable state for ops/risk
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from ..core.types import ComboOrder, ComboExecutionReport, OrderStatus


@dataclass
class OrderRecord:
    order: ComboOrder
    status: OrderStatus
    filled_qty: int = 0
    avg_fill_price: Optional[float] = None
    fees_total: float = 0.0
    ts_last: datetime = field(default_factory=datetime.utcnow)
    message: Optional[str] = None


class OrderStatusManager:
    """In-memory store of the most recent status for each client_order_id.

    For backtests we rely on simulator reports which include the original order.
    In paper/live this would be keyed by broker order IDs and updated via callbacks.
    """

    def __init__(self):
        self._store: Dict[str, OrderRecord] = {}

    def upsert(self, report: ComboExecutionReport) -> None:
        oid = report.order.client_order_id or f"{report.order.symbol}:{report.order.ts_created.timestamp()}"
        rec = self._store.get(oid)
        if rec is None:
            rec = OrderRecord(order=report.order, status=report.status)
            self._store[oid] = rec
        # Update fields
        rec.status = report.status
        rec.filled_qty = report.filled_qty
        rec.avg_fill_price = report.avg_fill_price
        rec.fees_total = report.fees_total
        rec.ts_last = datetime.utcnow()
        rec.message = report.message

    def bulk_upsert(self, reports: List[ComboExecutionReport]) -> None:
        for r in reports:
            self.upsert(r)

    def get(self, client_order_id: str) -> Optional[OrderRecord]:
        return self._store.get(client_order_id)

    def all(self) -> List[OrderRecord]:
        return list(self._store.values())

    def outstanding(self) -> List[OrderRecord]:
        return [r for r in self._store.values() if r.status in {OrderStatus.NEW, OrderStatus.PARTIAL}]

    def filled(self) -> List[OrderRecord]:
        return [r for r in self._store.values() if r.status == OrderStatus.FILLED]

    def cancelled(self) -> List[OrderRecord]:
        return [r for r in self._store.values() if r.status in {OrderStatus.CANCELLED, OrderStatus.EXPIRED, OrderStatus.REJECTED}]


__all__ = ["OrderStatusManager", "OrderRecord"]
