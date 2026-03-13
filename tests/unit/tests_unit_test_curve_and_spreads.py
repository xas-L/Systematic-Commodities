"""
Unit tests for curve construction and spread pricing.

Destination: tests/unit/tests_unit_test_curve_and_spreads.py

NOTE: Uses calendar_prices / butterfly_prices (the actual function names).
The original test referenced calendar_spreads / butterflies_from_log which
do not exist — this is the corrected version.
"""
import numpy as np
import pandas as pd

from src.data.curve import build_curve_surface, log_adjacent_spreads
from src.data.spreads import calendar_prices, butterfly_prices


def _toy_bars():
    """6 business days, 4 expiries, clean synthetic data with no NaNs."""
    dates = pd.bdate_range("2020-01-01", periods=6).date
    exps = [
        pd.Timestamp("2020-03-20").date(),
        pd.Timestamp("2020-04-20").date(),
        pd.Timestamp("2020-05-20").date(),
        pd.Timestamp("2020-06-20").date(),
    ]
    rows = []
    for d in dates:
        base = 50.0 + (pd.Timestamp(d) - pd.Timestamp(dates[0])).days * 0.1
        for k, e in enumerate(exps):
            rows.append(
                {
                    "date": d,
                    "symbol": "CL",
                    "expiry": e,
                    "settle": round(base + k * 1.0, 4),
                    "last": float("nan"),
                    "bid": float("nan"),
                    "ask": float("nan"),
                    "volume": 5000,
                    "open_interest": 20000,
                }
            )
    return pd.DataFrame(rows)


def test_build_curve_surface_shape():
    """Surface should have 4 expiry columns and same row count as unique dates."""
    bars = _toy_bars()
    surface, used = build_curve_surface(bars, min_contracts=3, tenors_target=4)
    assert isinstance(surface, pd.DataFrame), "surface must be a DataFrame"
    assert surface.shape[1] == 4, f"expected 4 expiry columns, got {surface.shape[1]}"
    assert len(surface) == 6, f"expected 6 date rows, got {len(surface)}"
    assert isinstance(surface.index, pd.DatetimeIndex), "surface index must be DatetimeIndex"


def test_log_adjacent_spreads_shape():
    """N expiries → N-1 adjacent log spreads."""
    bars = _toy_bars()
    surface, _ = build_curve_surface(bars, min_contracts=3, tenors_target=4)
    S = log_adjacent_spreads(surface)
    assert S.shape[1] == 3, f"expected 3 spread columns, got {S.shape[1]}"
    assert S.shape[0] == surface.shape[0]
    # Log spreads should be finite for a clean contango surface
    assert S.notna().all().all(), "expected no NaNs in clean log spreads"


def test_calendar_prices_matches_log_spreads():
    """calendar_prices(log=True) should match log_adjacent_spreads."""
    bars = _toy_bars()
    surface, _ = build_curve_surface(bars, min_contracts=3, tenors_target=4)
    log_sp = log_adjacent_spreads(surface)
    cal_log = calendar_prices(surface, log=True)
    # Same shape
    assert cal_log.shape == log_sp.shape, "shape mismatch between calendar_prices and log_adjacent_spreads"
    # Values should be numerically close
    np.testing.assert_allclose(
        cal_log.values, log_sp.values, rtol=1e-6, atol=1e-10,
        err_msg="calendar_prices(log=True) disagrees with log_adjacent_spreads",
    )


def test_butterfly_prices_shape():
    """N expiries → N-2 butterfly prices."""
    bars = _toy_bars()
    surface, _ = build_curve_surface(bars, min_contracts=3, tenors_target=4)
    F = butterfly_prices(surface, log=False)
    assert F.shape[1] == 2, f"expected 2 fly columns, got {F.shape[1]}"


def test_calendar_prices_sign_contango():
    """In a contango surface (far > near), linear calendar spreads should be positive."""
    bars = _toy_bars()
    surface, _ = build_curve_surface(bars, min_contracts=3, tenors_target=4)
    cal = calendar_prices(surface, log=False)
    # Every spread should be positive (contango) 
    assert (cal > 0).all().all(), "expected positive calendar spreads in contango surface"


def test_butterfly_prices_value():
    """Butterfly 1:-2:1 should be approximately zero for a linearly sloped curve."""
    # Create a perfectly linear term structure per day
    dates = pd.bdate_range("2020-01-01", periods=3).date
    exps = [
        pd.Timestamp("2020-03-20").date(),
        pd.Timestamp("2020-04-20").date(),
        pd.Timestamp("2020-05-20").date(),
    ]
    rows = []
    for d in dates:
        for k, e in enumerate(exps):
            rows.append({
                "date": d, "symbol": "CL", "expiry": e,
                "settle": 50.0 + k * 2.0,   # perfectly linear 
                "last": float("nan"), "bid": float("nan"), "ask": float("nan"),
                "volume": 5000, "open_interest": 20000,
            })
    bars = pd.DataFrame(rows)
    surface, _ = build_curve_surface(bars, min_contracts=3, tenors_target=3)
    F = butterfly_prices(surface, log=False)
    # Linear term structure → fly = 0 
    np.testing.assert_allclose(F.values, 0.0, atol=1e-8,
                               err_msg="butterfly should be ~0 for linear term structure")
