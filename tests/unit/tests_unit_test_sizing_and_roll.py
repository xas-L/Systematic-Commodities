import pandas as pd
from datetime import date

from src.signals.sizing import SizingConfig, size_from_signal
from src.ops.roll_manager import RollManager, RollPolicy


def test_size_from_signal_caps():
    cfg = SizingConfig(
        z_clip=2.0,
        risk_per_curve_usd=100000,
        per_trade_notional_cap_usd=50000,
        vol_target_enabled=False,
    )
    # With lot_value 1000, max lots = 50
    lots = size_from_signal(
        signal=10.0,
        combo_price_series=pd.Series([1, 1.1, 1.2]),
        lot_value=1000.0,
        cfg=cfg,
    )
    assert abs(lots) == 50


def test_roll_manager_calendar_flatten_if_no_next():
    # Two expiries only → next calendar not available
    surface = pd.DataFrame(
        {
            date(2020, 3, 1): [50, 51],
            date(2020, 4, 1): [51, 52],
        },
        index=pd.bdate_range("2020-01-01", periods=2),
    )
    positions = {"2020-03-01-2020-04-01": 3}
    rm = RollManager(policy=RollPolicy(roll_window_days=100))
    actions = rm.plan(symbol="CL", positions=positions, surface=surface, today=date(2020, 1, 15))
    assert actions and actions[0].to_combo is None
