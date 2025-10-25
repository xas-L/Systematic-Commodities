import pandas as pd
import numpy as np

from src.data.curve import build_curve_surface, log_adjacent_spreads
from src.data.spreads import calendar_spreads, butterflies_from_log


def _toy_bars():
    # 6 business days, 4 expiries
    dates = pd.bdate_range("2020-01-01", periods=6).date
    exps = [
        pd.Timestamp("2020-03-01").date(),
        pd.Timestamp("2020-04-01").date(),
        pd.Timestamp("2020-05-01").date(),
        pd.Timestamp("2020-06-01").date(),
    ]
    rows = []
    for d in dates:
        base = 50 + (pd.Timestamp(d) - pd.Timestamp(dates[0])).days * 0.1
        for k, e in enumerate(exps):
            rows.append(
                {
                    "date": d,
                    "symbol": "CL",
                    "expiry": e,
                    "settle": base + k * 1.0,
                    "last": np.nan,
                    "bid": np.nan,
                    "ask": np.nan,
                    "volume": 1000,
                    "open_interest": 2000,
                }
            )
    return pd.DataFrame(rows)


def test_build_curve_surface_and_spreads():
    bars = _toy_bars()
    surface, used = build_curve_surface(bars, min_contracts=3, tenors_target=4)
    assert surface.shape[1] == 4
    S = log_adjacent_spreads(surface)
    assert S.shape[1] == 3
    C = calendar_spreads(surface)
    assert C.equals(S)
    F = butterflies_from_log(surface)
    assert F.shape[1] == 2
