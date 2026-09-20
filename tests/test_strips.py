"""Tests for src/strips.py -- Person C's strip engine.

Covers the two frozen textbook fixtures from SPEC.md / technical-schema.md:
Annex 4 (single-parent normalization) and Annex 3 (combined coupon STRIP
across two parents). These are the correctness gate for C1 before any real
(non-fixture) pricing work in C2.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.strips import strip_and_price  # noqa: E402

SETTLE = pd.Timestamp("2010-03-17")  # the guideline's own illustrative settle date

# ---------------------------------------------------------------------------
# Annex 4 (SPEC.md Sec 5.1 / technical-schema.md Sec 6.3): 12.30% 2016,
# face=100, book_value=120.00. (period, date, cashflow, zcyc %, pv, normalized)
# ---------------------------------------------------------------------------
ANNEX4_ROWS = [
    (1, "2010-07-02", 6.15, 4.0683, 6.0274, 5.6564),
    (2, "2011-01-02", 6.15, 4.6948, 5.8711, 5.5098),
    (3, "2011-07-02", 6.15, 5.3212, 5.6841, 5.3343),
    (4, "2012-01-02", 6.15, 5.6128, 5.5055, 5.1666),
    (5, "2012-07-02", 6.15, 5.9044, 5.3174, 4.9901),
    (6, "2013-01-02", 6.15, 6.1339, 5.1305, 4.8147),
    (7, "2013-07-02", 6.15, 6.3633, 4.9392, 4.6352),
    (8, "2014-01-02", 6.15, 6.4744, 4.7663, 4.4730),
    (9, "2014-07-02", 6.15, 6.5855, 4.5946, 4.3118),
    (10, "2015-01-02", 6.15, 6.7227, 4.4187, 4.1467),
    (11, "2015-07-02", 6.15, 6.8599, 4.2439, 3.9827),
    (12, "2016-01-02", 6.15, 6.9971, 4.0707, 3.8201),
    (13, "2016-07-02", 106.15, 7.1343, 67.3029, 63.1606),
]
ANNEX4_TOTAL_PV = 127.87
ANNEX4_NORMALIZED_TOTAL = 120.00
ANNEX4_FACTOR = 0.9385


def _annex4_df_func(t_years: float) -> float:
    """Explicit lookup against the fixture's own printed per-period ZCYC
    rates -- not an interpolated curve. Under half_year convention,
    t_years = period_index * 0.5, so period_index = round(t_years * 2)."""
    period_index = round(t_years * 2)
    rate_percent = ANNEX4_ROWS[period_index - 1][3]
    rate = rate_percent / 100.0
    return (1.0 + rate / 2.0) ** (-period_index)


def test_annex4_single_parent_normalization():
    parent = {"coupon": 12.30, "maturity": pd.Timestamp("2016-07-02")}
    result = strip_and_price(
        parent,
        face=100,
        settle=SETTLE,
        df_func=_annex4_df_func,
        book_value=120.00,
        market_value=10_000_000,  # far above book_value -- book_value is the binding constraint here
        time_convention="half_year",
    )

    # 13 coupon rows + 1 principal row, all sharing the same underlying
    # 13 dates (principal falls on the last coupon date).
    assert len(result) == 14
    assert (result["type"] == "coupon").sum() == 13
    assert (result["type"] == "principal").sum() == 1

    total_pv = result["pv"].sum()
    factor = min(120.00, 10_000_000) / total_pv
    normalized_total = result["normalized_value"].sum()

    assert total_pv == pytest.approx(ANNEX4_TOTAL_PV, abs=0.01)
    assert factor == pytest.approx(ANNEX4_FACTOR, abs=0.0001)
    assert normalized_total == pytest.approx(ANNEX4_NORMALIZED_TOTAL, abs=0.01)

    # Per-period checks for the 12 pure-coupon periods (1-12): cashflow, pv,
    # and normalized_value should match the fixture table row for row.
    coupon_rows = result[result["type"] == "coupon"].sort_values("t_years").reset_index(drop=True)
    for i in range(12):
        _, _, cashflow, _, pv, normalized = ANNEX4_ROWS[i]
        assert coupon_rows.loc[i, "cashflow"] == pytest.approx(cashflow, abs=0.01)
        assert coupon_rows.loc[i, "pv"] == pytest.approx(pv, abs=0.01)
        assert coupon_rows.loc[i, "normalized_value"] == pytest.approx(normalized, abs=0.01)

    # Period 13 is split across a coupon row (6.15) + a principal row (100)
    # in our output -- combined, they must match the fixture's single
    # period-13 line (106.15 cashflow, 67.3029 pv, 63.1606 normalized).
    period13_cashflow, _, period13_cf, _, period13_pv, period13_norm = ANNEX4_ROWS[12]
    last_coupon = coupon_rows.iloc[12]
    principal = result[result["type"] == "principal"].iloc[0]
    assert last_coupon["cashflow"] + principal["cashflow"] == pytest.approx(period13_cf, abs=0.01)
    assert last_coupon["pv"] + principal["pv"] == pytest.approx(period13_pv, abs=0.01)
    assert last_coupon["normalized_value"] + principal["normalized_value"] == pytest.approx(
        period13_norm, abs=0.01
    )


# ---------------------------------------------------------------------------
# Annex 3 (SPEC.md Sec 5.2 / technical-schema.md Sec 6.4): 9.39% 2011
# stripped at 5cr + 12.30% 2016 stripped at 10cr, settle 17-Mar-2010. Pure
# cashflow-generation check -- no discounting/normalization involved, so a
# trivial (undiscounted) df_func is used.
# ---------------------------------------------------------------------------


def _flat_df_func(_t_years: float) -> float:
    return 1.0


def test_annex3_combined_coupon_strip():
    parent_a = {"coupon": 9.39, "maturity": pd.Timestamp("2011-07-02")}
    strips_a = strip_and_price(
        parent_a,
        face=50_000_000,
        settle=SETTLE,
        df_func=_flat_df_func,
        book_value=50_000_000,
        market_value=50_000_000,
        time_convention="half_year",
    )

    parent_b = {"coupon": 12.30, "maturity": pd.Timestamp("2016-07-02")}
    strips_b = strip_and_price(
        parent_b,
        face=100_000_000,
        settle=SETTLE,
        df_func=_flat_df_func,
        book_value=100_000_000,
        market_value=100_000_000,
        time_convention="half_year",
    )

    shared_date = pd.Timestamp("2010-07-02")
    a_row = strips_a[(strips_a["type"] == "coupon") & (strips_a["cashflow_date"] == shared_date)]
    b_row = strips_b[(strips_b["type"] == "coupon") & (strips_b["cashflow_date"] == shared_date)]
    assert len(a_row) == 1 and len(b_row) == 1
    assert a_row["cashflow"].iloc[0] == pytest.approx(2_347_500, abs=1)
    assert b_row["cashflow"].iloc[0] == pytest.approx(6_150_000, abs=1)

    # Both parents' first remaining coupon lands on the same date, so they
    # share the same date-only strip_id and can be summed.
    assert a_row["strip_id"].iloc[0] == b_row["strip_id"].iloc[0] == "CS-2010-07-02"

    combined = pd.concat([strips_a, strips_b], ignore_index=True)
    combined_strip = combined[combined["strip_id"] == "CS-2010-07-02"]
    assert len(combined_strip) == 2
    assert combined_strip["cashflow"].sum() == pytest.approx(8_497_500, abs=1)
