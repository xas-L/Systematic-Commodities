# src/core/scheduling.py
# Fold generation for anchored walk-forward and basic halt windows.
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Iterable, List, Optional

import calendar as _cal

from .utils import add_business_days


# -----------------------------
# Walk-forward folds
# -----------------------------
@dataclass(frozen=True)
class Fold:
    train_start: date
    train_end: date
    test_start: date
    test_end: date

    def as_dict(self) -> dict:
        return {
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
        }


def _month_add(d: date, months: int) -> date:
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    day = min(d.day, _cal.monthrange(y, m)[1])
    return date(y, m, day)


def generate_anchored_folds(
    *,
    train_start: date,
    first_test_start: date,
    last_date: date,
    test_months: int = 6,
    step_months: int = 6,
    embargo_bdays: int = 5,
) -> List[Fold]:
    """Anchored folds: training start is fixed; training end grows; test windows roll.

    Example: train_start=2006-01-01, first_test_start=2012-01-01, test_months=6
    -> (2012-01-01 .. 2012-06-30), then step by 6m.
    """
    folds: List[Fold] = []
    test_start = first_test_start
    while test_start <= last_date:
        test_end = _month_add(test_start, test_months) - timedelta(days=1)
        test_end = min(test_end, last_date)
        train_end = add_business_days(test_start, -embargo_bdays)
        folds.append(
            Fold(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        # advance
        test_start = _month_add(test_start, step_months)
        if test_start > last_date:
            break
    return folds


# -----------------------------
# News / halt windows
# -----------------------------
@dataclass(frozen=True)
class WeeklyWindow:
    weekday: int         # 0=Mon ... 6=Sun
    time_utc: time
    window_min: int

    @staticmethod
    def parse(weekday: str, time_utc: str, window_min: int) -> "WeeklyWindow":
        wd_map = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4, "SAT": 5, "SUN": 6}
        hh, mm = [int(x) for x in time_utc.split(":", 1)]
        return WeeklyWindow(weekday=wd_map[weekday.upper()], time_utc=time(hh, mm), window_min=window_min)


@dataclass
class HaltPolicy:
    inventory_releases: dict[str, WeeklyWindow]
    action: str  # "pause_new_orders_only" or "halt_all"

    @staticmethod
    def from_dict(d: dict) -> "HaltPolicy":
        inv = {}
        for k, v in d.get("inventory_releases", {}).items():
            inv[k] = WeeklyWindow.parse(v["weekday"], v["time_utc"], int(v["window_min"]))
        return HaltPolicy(inventory_releases=inv, action=d.get("action", "pause_new_orders_only"))


def is_in_weekly_window(ts_utc: datetime, ww: WeeklyWindow) -> bool:
    if ts_utc.weekday() != ww.weekday:
        return False
    center = datetime.combine(ts_utc.date(), ww.time_utc)
    delta = abs((ts_utc - center).total_seconds()) / 60.0
    return delta <= ww.window_min


def should_halt_for_news(ts_utc: datetime, policy: HaltPolicy) -> bool:
    for ww in policy.inventory_releases.values():
        if is_in_weekly_window(ts_utc, ww):
            return True
    return False


__all__ = [
    "Fold",
    "generate_anchored_folds",
    "WeeklyWindow",
    "HaltPolicy",
    "is_in_weekly_window",
    "should_halt_for_news",
]
