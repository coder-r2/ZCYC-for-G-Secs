"""Bond mathematics: coupon schedules, 30/360 accrued interest, price <-> YTM.

Conventions locked at kickoff (see docs/assumptions.md):
  * Semiannual coupons, schedule generated backwards from the maturity date.
  * Accrued interest on 30/360 (bond basis), 180-day semiannual period.
  * Semiannual compounding: DF = (1 + z/2) ** (-2t).
  * Time to cash flow: actual days / 365 from settlement ("act365", headline),
    with half-year-period counting ("halfyear") available for the ablation grid.

Units: every rate (coupon, ytm, zero rate) is a PERCENT per annum, e.g. 7.26
means 7.26%. Every price is per 100 face value.
"""

from __future__ import annotations

import calendar
from datetime import date, timedelta

import pandas as pd
from scipy.optimize import brentq

COUPONS_PER_YEAR = 2
MONTHS_PER_PERIOD = 12 // COUPONS_PER_YEAR
DAYS_PER_PERIOD_30_360 = 360 // COUPONS_PER_YEAR
TIME_CONVENTIONS = ("act365", "halfyear")


def to_date(value) -> date:
    """Coerce str / datetime / pandas Timestamp / date into a plain date."""
    if isinstance(value, date) and not hasattr(value, "hour"):
        return value
    return pd.Timestamp(value).date()


def settlement_date(valuation_date, lag_days: int = 1) -> date:
    """T+1 settlement, rolled forward over Sat/Sun.

    Public holidays are not handled -- we have no NSE/RBI holiday calendar in
    the data we pulled, and a holiday shifts every cash flow time by a day or
    two, far below the bps-level differences we are comparing. Stated as a
    simplification in docs/assumptions.md.
    """
    d = to_date(valuation_date)
    moved = 0
    while moved < lag_days:
        d += timedelta(days=1)
        if d.weekday() < 5:
            moved += 1
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def _add_months(anchor: date, months: int) -> date:
    """Shift a date by whole months, clamping to the end of the target month."""
    total = anchor.month - 1 + months
    year = anchor.year + total // 12
    month = total % 12 + 1
    day = min(anchor.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def days_30_360(start: date, end: date) -> int:
    """Day count on the 30/360 US (bond basis) convention."""
    d1, d2 = start.day, end.day
    if d1 == 31:
        d1 = 30
    if d2 == 31 and d1 == 30:
        d2 = 30
    return 360 * (end.year - start.year) + 30 * (end.month - start.month) + (d2 - d1)


def coupon_dates(maturity, settle) -> list[date]:
    """Remaining coupon dates (strictly after settlement), earliest first.

    Every date is measured back from maturity rather than chained one period at
    a time, so a 31st-of-the-month maturity keeps landing on the 31st instead of
    being dragged to the 30th by a single February.
    """
    maturity, settle = to_date(maturity), to_date(settle)
    if maturity <= settle:
        raise ValueError(f"maturity {maturity} is not after settlement {settle}")
    dates, k = [], 0
    while True:
        d = _add_months(maturity, -MONTHS_PER_PERIOD * k)
        if d <= settle:
            break
        dates.append(d)
        k += 1
    dates.reverse()
    return dates


def previous_coupon_date(maturity, settle) -> date:
    """Start of the coupon period containing settlement."""
    return _add_months(to_date(maturity), -MONTHS_PER_PERIOD * len(coupon_dates(maturity, settle)))


def accrued_30_360(coupon: float, maturity, settle) -> float:
    """Accrued interest per 100 face, 30/360 over a 180-day semiannual period."""
    settle = to_date(settle)
    prev = previous_coupon_date(maturity, settle)
    return (coupon / COUPONS_PER_YEAR) * days_30_360(prev, settle) / DAYS_PER_PERIOD_30_360


def period_fraction_remaining(maturity, settle) -> float:
    """Fraction of the current coupon period still to run, on 30/360."""
    settle = to_date(settle)
    dates = coupon_dates(maturity, settle)
    prev = _add_months(to_date(maturity), -MONTHS_PER_PERIOD * len(dates))
    return days_30_360(settle, dates[0]) / DAYS_PER_PERIOD_30_360


def cashflows(coupon: float, maturity, settle, time_convention: str = "act365") -> pd.DataFrame:
    """Remaining cash flows per 100 face: DataFrame[date, t, cf].

    ``t`` is the time to the cash flow in years, measured under
    ``time_convention``:
      * "act365"   -- actual days / 365 from settlement (headline choice).
      * "halfyear" -- (i + w) / 2 for the i-th remaining flow, where w is the
        30/360 fraction of the current coupon period still to run. This reduces
        to the textbook 0.5, 1.0, 1.5, ... when settlement falls on a coupon
        date, and is the alternative tested in the ablation grid.
    """
    if time_convention not in TIME_CONVENTIONS:
        raise ValueError(f"time_convention must be one of {TIME_CONVENTIONS}")
    settle = to_date(settle)
    dates = coupon_dates(maturity, settle)

    amounts = [coupon / COUPONS_PER_YEAR] * len(dates)
    amounts[-1] += 100.0

    if time_convention == "act365":
        times = [(d - settle).days / 365.0 for d in dates]
    else:
        w = period_fraction_remaining(maturity, settle)
        times = [(i + w) / COUPONS_PER_YEAR for i in range(len(dates))]

    return pd.DataFrame({"date": pd.to_datetime(dates), "t": times, "cf": amounts})


def discount_factor(rate: float, t: float) -> float:
    """Semiannually compounded discount factor for a percent rate."""
    return (1.0 + rate / (100.0 * COUPONS_PER_YEAR)) ** (-COUPONS_PER_YEAR * t)


def annualise(rate_semi: float) -> float:
    """Semiannual -> effective annual, both in percent."""
    return ((1.0 + rate_semi / (100.0 * COUPONS_PER_YEAR)) ** COUPONS_PER_YEAR - 1.0) * 100.0


def deannualise(rate_annual: float) -> float:
    """Effective annual -> semiannual, both in percent."""
    return ((1.0 + rate_annual / 100.0) ** (1.0 / COUPONS_PER_YEAR) - 1.0) * 100.0 * COUPONS_PER_YEAR


def dirty_price(coupon: float, maturity, settle, ytm: float, time_convention: str = "act365") -> float:
    """Present value of the remaining cash flows at a flat YTM, per 100 face."""
    cf = cashflows(coupon, maturity, settle, time_convention)
    return float((cf["cf"] * [discount_factor(ytm, t) for t in cf["t"]]).sum())


def clean_price(coupon: float, maturity, settle, ytm: float, time_convention: str = "act365") -> float:
    """Dirty price less 30/360 accrued interest, per 100 face."""
    return dirty_price(coupon, maturity, settle, ytm, time_convention) - accrued_30_360(coupon, maturity, settle)


def ytm_from_price(
    coupon: float,
    maturity,
    settle,
    price: float,
    price_type: str = "clean",
    time_convention: str = "act365",
) -> float:
    """Invert price -> YTM (percent) by bisection on [-50%, 200%]."""
    if price_type not in ("clean", "dirty"):
        raise ValueError("price_type must be 'clean' or 'dirty'")
    price_fn = clean_price if price_type == "clean" else dirty_price

    def objective(y: float) -> float:
        return price_fn(coupon, maturity, settle, y, time_convention) - price

    lo, hi = -50.0, 200.0
    if objective(lo) * objective(hi) > 0:
        raise ValueError(f"no YTM in [{lo}, {hi}] reprices {price} for {coupon}% maturing {maturity}")
    return float(brentq(objective, lo, hi, xtol=1e-12, rtol=1e-14, maxiter=200))
