# src/execsim/cost_model.py
# Fee and slippage models parameterised by fees_slippage.yaml
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..core.utils import load_fees_slippage
from ..core.types import ComboOrder
from .combos import FeeModel, SlippageModel


@dataclass
class Profile:
    combo_half_spread_ticks: float
    legging_penalty_ticks: float
    illiquidity_extra_ticks: float
    partial_fill_ratio: float
    fee_round_turn_per_lot: float


class YamlFeeModel(FeeModel):
    def __init__(self, product: str, profile_name: str = "conservative", root: Optional[str] = None):
        cfg = load_fees_slippage(root)
        self.profile = cfg["profiles"][profile_name]
        self.product = product

    def __call__(self, order: ComboOrder, fill_qty: int) -> float:
        # Round-turn fee per lot times abs(fill_qty)
        per_lot = float(self.profile.get("fee_round_turn_per_lot", 2.50))
        return abs(fill_qty) * per_lot


class YamlSlippageModel(SlippageModel):
    def __init__(self, product: str, profile_name: str = "conservative", root: Optional[str] = None):
        cfg = load_fees_slippage(root)
        self.profile = cfg["profiles"][profile_name]
        self.product = product
        # Pull product-specific min increment if needed
        self.min_inc = float(cfg.get("products", {}).get(product, {}).get("combo", {}).get("min_price_increment", 0.01))

    def __call__(self, order: ComboOrder, md_snapshot: dict, fill_qty: int) -> float:
        # Half-spread base plus penalties for legging/illiquidity
        half_spread_ticks = float(self.profile.get("combo_half_spread_ticks", 1.0))
        illiq = float(self.profile.get("illiquidity_extra_ticks", 1.0)) if int(md_snapshot.get("combo_tob_size", 0)) < 10 else 0.0
        leg_pen = 0.0  # no legging in base simulator
        total_ticks = half_spread_ticks + illiq + leg_pen
        return total_ticks * self.min_inc


__all__ = [
    "YamlFeeModel",
    "YamlSlippageModel",
]
