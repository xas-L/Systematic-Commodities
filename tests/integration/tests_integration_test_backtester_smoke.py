import pandas as pd
import numpy as np

from src.execsim.combos import ComboExecutionSimulator
from src.execsim.backtester import Backtester


class _ZeroFee:
    def __call__(self, order, fill_qty):
        return 0.0


class _ConstSlip:
    def __call__(self, order, md, fill_qty):
        return 0.0


def test_backtester_runs_with_minimal_inputs():
    # Two combos, simple mid series
    idx = pd.bdate_range("2020-01-01", periods=30)
    mids_a = pd.Series(np.linspace(0.0, 1.0, len(idx)), index=idx)
    mids_b = pd.Series(np.linspace(1.0, 0.0, len(idx)), index=idx)
    signals = pd.DataFrame({"2020-02-01-2020-01-01": 1.0, "2020-03-01-2020-02-01": -1.0}, index=idx)
    quotes = {
        "2020-02-01-2020-01-01": {
            "mid": mids_a,
            "half_spread": pd.Series(0.01, index=idx),
            "lot_value": 1000.0,
            "symbol": "CL",
            "tob_size": {t: 10 for t in idx},
        },
        "2020-03-01-2020-02-01": {
            "mid": mids_b,
            "half_spread": pd.Series(0.01, index=idx),
            "lot_value": 1000.0,
            "symbol": "CL",
            "tob_size": {t: 10 for t in idx},
        },
    }

    def sizing_fn(signal, combo_price_series, lot_value, cfg):
        return int(np.sign(signal))  # ±1 lot

    class _Cfg:
        pass

    execsim = ComboExecutionSimulator(fee_model=_ZeroFee(), slip_model=_ConstSlip())
    bt = Backtester(executor=execsim)
    res = bt.run(signals=signals, quotes=quotes, sizing_fn=sizing_fn, sizing_cfg=_Cfg())
    assert len(res["pnl_path"]) == len(idx) - 1
    assert not res["trade_log"].empty
