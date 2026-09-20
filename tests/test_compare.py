"""Tests for A2 -- compare() against FBIL's published curve.

The comparison can be tested to the last decimal without the curve engine
existing, because FBIL's own published grid is a perfectly good stand-in for
"our" curve:

  * compared against itself, every difference must be exactly zero;
  * shifted by a known amount, every difference must be exactly that amount in
    bps, which pins the direction and the scale of the conversion.

Those two cases fix the arithmetic completely. The remaining tests cover the
contract's edges: the 1:1 join it insists on, the segment boundaries, the
absolute-value convention for max_diff_bps, and empty segments.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.compare import (  # noqa: E402
    COMPARISON_TENORS,
    METRICS_COLUMNS,
    OUTPUT_FILES,
    SUMMARY_KEYS,
    compare,
    comparison_table,
    write_comparison_outputs,
)

CLEAN_DIR = REPO_ROOT / "data" / "clean"


@pytest.fixture(scope="module")
def fbil_grid():
    return pd.read_csv(CLEAN_DIR / "fbil_zcyc.csv")


def as_ours(fbil: pd.DataFrame, shift_pct: float = 0.0) -> pd.DataFrame:
    """FBIL's grid dressed up as ours, optionally shifted in percentage points."""
    ours = fbil.rename(columns={"zcy_semi": "zero_semi", "zcy_annual": "zero_annual"}).copy()
    for col in ("zero_semi", "zero_annual", "par_semi", "par_annual"):
        ours[col] = ours[col] + shift_pct
    return ours


# ----------------------------------------------------- the two anchoring cases

def test_fbil_against_itself_is_exactly_zero(fbil_grid):
    metrics_df, summary = compare(as_ours(fbil_grid), fbil_grid)

    assert (metrics_df["diff_bps"] == 0.0).all()
    for key in SUMMARY_KEYS:
        assert summary[key] == {"rmse_bps": 0.0, "mae_bps": 0.0, "max_diff_bps": 0.0}


def test_a_known_shift_converts_to_the_right_number_of_bps(fbil_grid):
    """+0.01 percentage points everywhere is +1.00 bp everywhere, not -1 or +100."""
    metrics_df, summary = compare(as_ours(fbil_grid, shift_pct=0.01), fbil_grid)

    assert np.allclose(metrics_df["diff_bps"], 1.0)
    for key in SUMMARY_KEYS:
        assert summary[key]["rmse_bps"] == pytest.approx(1.0)
        assert summary[key]["mae_bps"] == pytest.approx(1.0)
        assert summary[key]["max_diff_bps"] == pytest.approx(1.0)


def test_the_sign_says_which_curve_is_higher(fbil_grid):
    """diff_bps is ours minus FBIL, so a curve below FBIL's is negative."""
    metrics_df, _ = compare(as_ours(fbil_grid, shift_pct=-0.05), fbil_grid)
    assert np.allclose(metrics_df["diff_bps"], -5.0)


# -------------------------------------------------------------------- contract

def test_metrics_columns_are_exactly_the_frozen_list(fbil_grid):
    metrics_df, _ = compare(as_ours(fbil_grid), fbil_grid)
    assert list(metrics_df.columns) == METRICS_COLUMNS


def test_summary_has_exactly_the_frozen_keys(fbil_grid):
    _, summary = compare(as_ours(fbil_grid), fbil_grid)
    assert set(summary) == set(SUMMARY_KEYS)
    for key in SUMMARY_KEYS:
        assert set(summary[key]) == {"rmse_bps", "mae_bps", "max_diff_bps"}


def test_one_row_per_fbil_tenor(fbil_grid):
    metrics_df, _ = compare(as_ours(fbil_grid), fbil_grid)
    assert len(metrics_df) == len(fbil_grid)


def test_the_callers_frame_is_never_mutated(fbil_grid):
    ours = as_ours(fbil_grid)
    before = ours.copy(deep=True)
    compare(ours, fbil_grid)
    pd.testing.assert_frame_equal(ours, before)


def test_the_curve_engines_own_column_names_are_accepted(fbil_grid):
    """Curve.grid() emits zcy_semi, not the zero_semi the contract documents."""
    metrics_df, _ = compare(fbil_grid.copy(), fbil_grid)
    assert (metrics_df["diff_bps"] == 0.0).all()


def test_a_grid_mismatch_raises_rather_than_falling_back_to_nearest_match(fbil_grid):
    ours = as_ours(fbil_grid).drop(index=[3, 9]).reset_index(drop=True)
    with pytest.raises(ValueError, match="1:1"):
        compare(ours, fbil_grid)


def test_float_tenors_still_join(fbil_grid):
    """0.25 * 3 is not exactly 0.75 in binary; the join must survive that."""
    ours = as_ours(fbil_grid)
    ours["tenor_years"] = np.arange(1, len(ours) + 1) * 0.25
    metrics_df, _ = compare(ours, fbil_grid)
    assert len(metrics_df) == len(fbil_grid)


# -------------------------------------------------------------------- segments

@pytest.mark.parametrize(
    "tenor,expected",
    [(0.25, "<=1y"), (1.0, "<=1y"), (1.25, "1-14y"), (14.0, "1-14y"), (14.25, ">14y"), (40.0, ">14y")],
)
def test_segment_boundaries_are_inclusive_at_the_top(fbil_grid, tenor, expected):
    metrics_df, _ = compare(as_ours(fbil_grid), fbil_grid)
    row = metrics_df.loc[metrics_df["tenor_years"] == tenor]
    assert row["segment"].iloc[0] == expected


def test_segments_partition_the_grid(fbil_grid):
    metrics_df, _ = compare(as_ours(fbil_grid), fbil_grid)
    counts = metrics_df["segment"].value_counts()
    assert set(counts.index) == set(SUMMARY_KEYS[1:])
    assert counts.sum() == len(metrics_df)


def test_an_empty_segment_gives_nan_and_does_not_crash(fbil_grid):
    """A curve stopping at 10y has nothing to report above 14y."""
    short = fbil_grid[fbil_grid["tenor_years"] <= 10.0].reset_index(drop=True)
    _, summary = compare(as_ours(short), short)
    assert np.isnan(summary[">14y"]["rmse_bps"])
    assert np.isnan(summary[">14y"]["max_diff_bps"])
    assert summary["overall"]["rmse_bps"] == 0.0


# ------------------------------------------------------- max_diff_bps is ABSOLUTE

def test_max_diff_bps_is_the_maximum_absolute_difference(fbil_grid):
    """A curve that crosses FBIL's: the largest error is the negative one.

    A signed maximum would report +10 bps here and quietly hide a 25 bps miss,
    which is exactly the number that would end up misread on a slide.
    """
    ours = as_ours(fbil_grid)
    ours.loc[0, "zero_semi"] += 0.10    # +10 bps
    ours.loc[1, "zero_semi"] -= 0.25    # -25 bps

    metrics_df, summary = compare(ours, fbil_grid)
    assert metrics_df.loc[0, "diff_bps"] == pytest.approx(10.0)
    assert metrics_df.loc[1, "diff_bps"] == pytest.approx(-25.0)
    assert summary["overall"]["max_diff_bps"] == pytest.approx(25.0)


def test_rmse_and_mae_are_the_textbook_formulas(fbil_grid):
    ours = as_ours(fbil_grid)
    ours.loc[0, "zero_semi"] += 0.03
    ours.loc[1, "zero_semi"] -= 0.04

    metrics_df, summary = compare(ours, fbil_grid)
    d = metrics_df["diff_bps"].to_numpy()
    assert summary["overall"]["rmse_bps"] == pytest.approx(float(np.sqrt(np.mean(d**2))))
    assert summary["overall"]["mae_bps"] == pytest.approx(float(np.mean(np.abs(d))))


# --------------------------------------------------------------------- outputs

def test_comparison_table_is_the_twelve_tenor_subset(fbil_grid):
    metrics_df, _ = compare(as_ours(fbil_grid), fbil_grid)
    table = comparison_table(metrics_df)
    assert list(table.columns) == METRICS_COLUMNS
    assert sorted(table["tenor_years"]) == sorted(float(t) for t in COMPARISON_TENORS)


def test_write_comparison_outputs_writes_all_four_frozen_names(fbil_grid, tmp_path):
    metrics_df, summary = write_comparison_outputs(as_ours(fbil_grid), fbil_grid, tmp_path, provenance="SAMPLE")

    for name in OUTPUT_FILES:
        assert (tmp_path / name).exists(), f"{name} was not written"
        assert (tmp_path / name).stat().st_size > 0

    written = json.loads((tmp_path / "comparison_summary.json").read_text())
    assert written == summary
    assert len(pd.read_csv(tmp_path / "comparison_table.csv")) == len(COMPARISON_TENORS)
