# src/execsim/combos.py
# Exchange-combo execution simulator: partial fills, fees, slippage, and timeouts
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np

from ..core.types import ComboOrder, ComboExecutionReport, OrderStatus



# Fee & slippage callables (injected)

class FeeModel:
    def __call__(self, order: ComboOrder, fill_qty: int) -> float:  # pragma: no cover - interface
        return 0.0


class SlippageModel:
    def __call__(self, order: ComboOrder, md_snapshot: dict, fill_qty: int) -> float:  # pragma: no cover - interface
        return 0.0



# Simulator

@dataclass
class SimConfig:
    timeout_seconds: int = 5
    queue_fraction: float = 0.30  # fraction of TOB we assume we get on first attempt


class ComboExecutionSimulator:
    def __init__(self, fee_model: FeeModel, slip_model: SlippageModel, cfg: Optional[SimConfig] = None):
        self.fee_model = fee_model
        self.slip_model = slip_model
        self.cfg = cfg or SimConfig()

    def simulate(
        self,
        order: ComboOrder,
        md_snapshot: dict,
        now: Optional[datetime] = None,
    ) -> ComboExecutionReport:
        """Simulate immediate-or-timeout behaviour against a combo TOB.

        md_snapshot expects keys: {"combo_mid", "combo_half_spread", "combo_tob_size"}
        We fill up to queue_fraction * TOB size. No legging by default.
        """
        now = now or datetime.utcnow()
        tob_size = int(md_snapshot.get("combo_tob_size", 0))
        half_spread = float(md_snapshot.get("combo_half_spread", 0.0))

        if tob_size <= 0:
            return ComboExecutionReport(order=order, status=OrderStatus.EXPIRED, filled_qty=0, remaining_qty=order.qty,
                                        message="No TOB size available")

        # Determine fill quantity
        qty = int(np.sign(order.qty) * min(abs(order.qty), max(1, int(self.cfg.queue_fraction * tob_size))))

        # Slippage: assume we cross half-spread plus any modelled penalty
        slip = float(self.slip_model(order, md_snapshot, abs(qty)))
        fees = float(self.fee_model(order, abs(qty)))

        avg_px = (order.limit_price if order.limit_price is not None else float(md_snapshot.get("combo_mid", 0.0)))
        # Crossing half-spread depends on side; approximate as adverse move by +half_spread * sign
        side = np.sign(qty) or 1
        avg_px = avg_px + side * (half_spread + slip)

        status = OrderStatus.FILLED if abs(qty) == abs(order.qty) else OrderStatus.PARTIAL
        left = order.qty - qty
        rep = ComboExecutionReport(
            order=order,
            status=status,
            filled_qty=qty,
            avg_fill_price=avg_px,
            fees_total=fees,
            remaining_qty=left,
            message=None,
        )
        return rep


__all__ = [
    "FeeModel",
    "SlippageModel",
    "SimConfig",
    "ComboExecutionSimulator",
]
