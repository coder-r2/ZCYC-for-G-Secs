"""Tests for C2 -- bonus SDL pricing (SPEC.md Section 2/6).

Reads data/clean/sdl_bond.csv and data/clean/fbil_sdl_zcyc.csv, prices both
book-value scenarios against the published SDL ZCYC curve, writes
outputs/strips_table_scenario{1,2}.csv, and checks the two things SPEC.md
makes the Definition of Done for C2: normalized values sum to
min(book_value, market_value), and the two scenarios actually differ in
which value binds.

NOTE: sdl_bond.csv / fbil_sdl_zcyc.csv are STUB data as of this test -- A
hasn't pushed the real FBIL download yet (see C_status.md dependency watch
and divergence D7). Swap the two CSVs for A's real files once available;
nothing here should need to change since both stubs match the frozen
schema exactly (technical-schema.md Sec 2.3/2.4).
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.bond import accrued_30_360  # noqa: E402
from src.strips import price_sdl  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "clean"
OUTPUTS_DIR = REPO_ROOT / "outputs"
SETTLE = pd.Timestamp("2026-09-21")  # STUB valuation date -- see C_status.md; matches B's own test settle


def _load_sdl_bond_row():
    sdl_bond = pd.read_csv(DATA_DIR / "sdl_bond.csv")
    assert len(sdl_bond) == 1, "expected exactly one bonus SDL row"
    return sdl_bond.iloc[0]


def _load_sdl_zcyc():
    return pd.read_csv(DATA_DIR / "fbil_sdl_zcyc.csv")


def _market_value(row, settle) -> float:
    """Recomputed independently of price_sdl(), so this test doesn't just
    check the function against itself."""
    coupon = float(row["coupon"])
    maturity = pd.Timestamp(row["maturity"])
    face = float(row["face_stripped"])
    dirty_per_100 = float(row["market_price"]) + accrued_30_360(coupon, maturity, settle)
    return dirty_per_100 * (face / 100.0)


def test_sdl_eligibility_residual_maturity_within_14y():
    row = _load_sdl_bond_row()
    residual_years = (pd.Timestamp(row["maturity"]) - SETTLE).days / 365.0
    assert residual_years <= 14.0


def test_c2_sdl_pricing_scenarios():
    row = _load_sdl_bond_row()
    sdl_zcyc = _load_sdl_zcyc()

    scenario1 = price_sdl(row, sdl_zcyc, SETTLE, book_value_col="book_value_scenario_1")
    scenario2 = price_sdl(row, sdl_zcyc, SETTLE, book_value_col="book_value_scenario_2")

    OUTPUTS_DIR.mkdir(exist_ok=True)
    scenario1.to_csv(OUTPUTS_DIR / "strips_table_scenario1.csv", index=False)
    scenario2.to_csv(OUTPUTS_DIR / "strips_table_scenario2.csv", index=False)

    market_value = _market_value(row, SETTLE)
    bv1 = float(row["book_value_scenario_1"])
    bv2 = float(row["book_value_scenario_2"])

    # The stub data is deliberately set up so scenario 1 is MV-bound and
    # scenario 2 is BV-bound -- that's the whole point of running two
    # scenarios (SPEC.md Sec 2, C2).
    assert bv1 > market_value
    assert bv2 < market_value

    assert scenario1["normalized_value"].sum() == pytest.approx(min(bv1, market_value), rel=1e-9)
    assert scenario2["normalized_value"].sum() == pytest.approx(min(bv2, market_value), rel=1e-9)
    assert min(bv1, market_value) != min(bv2, market_value)

    # Every row's price_per_100 should be positive and finite -- a cheap
    # sanity check that df_func / normalization didn't produce garbage.
    for scenario in (scenario1, scenario2):
        assert (scenario["price_per_100"] > 0).all()
        assert scenario["price_per_100"].notna().all()
