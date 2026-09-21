"""Shared fixtures for the curve-engine tests.

The curve tests run against a synthetic market: a known smooth zero curve, with
a set of bonds priced exactly off it. That makes every assertion checkable
without waiting on the real FBIL download -- if the engine cannot recover a
curve it generated the prices from, it will not recover the real one either.
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.bond import accrued_30_360, cashflows  # noqa: E402

SETTLE = date(2026, 9, 21)
BOND_YEARS = [1.2, 1.6, 2.1, 2.7, 3.4, 4.2, 5.1, 6.3, 7.5, 8.8,
              10.2, 11.6, 13.1, 15.5, 19.0, 24.0, 28.0, 32.0, 38.0]


def true_zero(t):
    """The synthetic 'market' curve: smooth, upward sloping, 5.2% -> 7.3%."""
    return 5.2 + 2.1 * (1.0 - np.exp(-np.asarray(t, dtype=float) / 4.5))


def true_df(t):
    return (1.0 + true_zero(t) / 200.0) ** (-2.0 * np.asarray(t, dtype=float))


@pytest.fixture(scope="session")
def settle():
    return SETTLE


@pytest.fixture(scope="session")
def synthetic_bonds():
    """Bonds in the frozen schema, priced exactly off ``true_zero``."""
    rows = []
    for yrs in BOND_YEARS:
        maturity = SETTLE + timedelta(days=int(365 * yrs))
        coupon = round(float(true_zero(yrs)), 2)
        cf = cashflows(coupon, maturity, SETTLE)
        dirty = float((cf["cf"] * true_df(cf["t"].to_numpy())).sum())
        rows.append(
            {
                "isin": f"INSYN{yrs}",
                "desc": f"{coupon}% GS {maturity:%Y}",
                "coupon": coupon,
                "maturity": maturity,
                "price": dirty - accrued_30_360(coupon, maturity, SETTLE),
                "ytm": coupon,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def synthetic_tbills():
    tenors = {"7D": 7 / 365, "182D": 0.5, "364D": 1.0}
    return pd.DataFrame(
        {"tenor": list(tenors), "rate": [float(true_zero(t)) for t in tenors.values()]}
    )


@pytest.fixture(scope="session")
def fitted_curve(synthetic_bonds, synthetic_tbills):
    from src.curve import build_zcyc

    return build_zcyc(synthetic_bonds, synthetic_tbills, SETTLE)


def real_data_settle() -> pd.Timestamp:
    """The settlement date matching whatever's actually in data/clean/ right
    now, read from SOURCE.md rather than hardcoded.

    data/clean/*.csv's own values (prices, YTMs, book values) are anchored to
    a specific valuation date, and a fixed constant here would go stale the
    next time the data is refreshed -- this already happened twice
    independently (a real-data curve test and the SDL pricing test each had
    their own stale hardcoded settle before real FBIL data landed). Shared
    here so it can't drift out of sync between test files again.
    """
    import re

    text = (REPO_ROOT / "data" / "clean" / "SOURCE.md").read_text()
    match = re.search(r"Settlement date \(T\+1\): \*\*([\d-]+)\*\*", text)
    if not match:
        raise ValueError("could not find the settlement date in data/clean/SOURCE.md")
    return pd.Timestamp(match.group(1))
