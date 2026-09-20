"""B2 -- input selection: 90-day thinning, long-end buckets, T-bill points."""

from datetime import date, timedelta

import pandas as pd
import pytest

from src.selection import parse_tenor, select_bonds, select_inputs, select_tbills

SETTLE = date(2026, 9, 21)


def _bond(isin, years, coupon=7.0, **extra):
    return {
        "isin": isin,
        "desc": f"{coupon}% GS",
        "coupon": coupon,
        "maturity": SETTLE + timedelta(days=int(365 * years)),
        "price": 100.0,
        "ytm": coupon,
        **extra,
    }


class TestTenorParsing:
    @pytest.mark.parametrize(
        "value, expected",
        [(0.5, 0.5), ("6M", 0.5), ("12M", 1.0), ("182D", 182 / 365),
         ("7D", 7 / 365), ("364 days", 364 / 365), ("2W", 14 / 365)],
    )
    def test_parses(self, value, expected):
        assert parse_tenor(value) == pytest.approx(expected)

    def test_rejects_nonsense(self):
        with pytest.raises(ValueError):
            parse_tenor("next Tuesday")


class TestTbillSelection:
    def test_picks_three_points(self):
        tbills = pd.DataFrame({"tenor": ["7D", "14D", "91D", "182D", "364D"],
                               "rate": [5.4, 5.45, 5.6, 5.9, 6.1]})
        chosen = select_tbills(tbills)
        assert list(chosen["tenor"]) == ["7D", "182D", "364D"]

    def test_nearest_match_when_exact_tenors_are_missing(self):
        tbills = pd.DataFrame({"tenor": ["8D", "90D", "370D"], "rate": [5.4, 5.8, 6.1]})
        chosen = select_tbills(tbills)
        assert list(chosen["tenor"]) == ["8D", "90D", "370D"]

    def test_no_duplicate_rows_when_the_list_is_short(self):
        tbills = pd.DataFrame({"tenor": ["91D"], "rate": [5.6]})
        assert len(select_tbills(tbills)) == 1

    def test_empty_input_is_tolerated(self):
        assert len(select_tbills(pd.DataFrame(columns=["tenor", "rate"]))) == 0


class TestMidSegmentThinning:
    def test_keeps_bonds_more_than_90_days_apart(self):
        bonds = pd.DataFrame([_bond("A", 2.0), _bond("B", 2.1), _bond("C", 3.0)])
        selected, _ = select_bonds(bonds, SETTLE)
        # B matures ~36 days after A, so it is dropped; C is a year later.
        assert list(selected["isin"]) == ["A", "C"]

    def test_thinning_can_be_switched_off_for_the_ablation(self):
        bonds = pd.DataFrame([_bond("A", 2.0), _bond("B", 2.1), _bond("C", 3.0)])
        selected, _ = select_bonds(bonds, SETTLE, thin_90d=False)
        assert list(selected["isin"]) == ["A", "B", "C"]

    def test_gap_is_measured_against_the_last_kept_bond(self):
        # Three bonds 60 days apart: A kept, B dropped, C is 120 days after A
        # so it is kept -- the clock does not reset on the dropped bond.
        bonds = pd.DataFrame([
            {**_bond("A", 2.0)},
            {"isin": "B", "desc": "x", "coupon": 7.0,
             "maturity": SETTLE + timedelta(days=int(365 * 2) + 60), "price": 100.0, "ytm": 7.0},
            {"isin": "C", "desc": "x", "coupon": 7.0,
             "maturity": SETTLE + timedelta(days=int(365 * 2) + 120), "price": 100.0, "ytm": 7.0},
        ])
        selected, _ = select_bonds(bonds, SETTLE)
        assert list(selected["isin"]) == ["A", "C"]


class TestSegmentBoundaries:
    def test_sub_one_year_bonds_are_left_to_the_tbills(self):
        bonds = pd.DataFrame([_bond("SHORT", 0.4), _bond("MID", 5.0)])
        selected, report = select_bonds(bonds, SETTLE)
        assert list(selected["isin"]) == ["MID"]
        assert "T-bills" in report.loc[report["isin"] == "SHORT", "reason"].iloc[0]

    def test_matured_bonds_are_dropped(self):
        bonds = pd.DataFrame([_bond("DEAD", -1.0), _bond("MID", 5.0)])
        selected, _ = select_bonds(bonds, SETTLE)
        assert list(selected["isin"]) == ["MID"]

    def test_report_covers_every_input_bond(self):
        bonds = pd.DataFrame([_bond(f"B{i}", 1.0 + i * 0.3) for i in range(12)])
        selected, report = select_bonds(bonds, SETTLE)
        assert len(report) == len(bonds)
        assert report["kept"].sum() == len(selected)


class TestLongEndBuckets:
    def test_one_isin_per_bucket(self):
        bonds = pd.DataFrame([_bond(f"L{y}", y) for y in (15, 16, 17, 19, 23, 27, 31, 36)])
        selected, _ = select_bonds(bonds, SETTLE)
        # 15/16/17 all fall in the 14-18 bucket; only one survives.
        kept_long = [i for i in selected["isin"] if i.startswith("L")]
        assert sorted(kept_long) == ["L15", "L19", "L23", "L27", "L31", "L36"]

    def test_terminal_isin_is_always_kept(self):
        bonds = pd.DataFrame([_bond(f"L{y}", y) for y in (15, 36, 39)])
        selected, report = select_bonds(bonds, SETTLE)
        assert "L39" in list(selected["isin"])
        assert "terminal" in report.loc[report["isin"] == "L39", "reason"].iloc[0]

    def test_more_liquid_bond_wins_the_bucket(self):
        # TERM sits in a bucket of its own so the terminal-ISIN rule cannot
        # rescue the loser of the 14-18y contest.
        bonds = pd.DataFrame([
            _bond("ILLIQ", 15.0, volume=10.0),
            _bond("LIQUID", 16.5, volume=9_000.0),
            _bond("TERM", 36.0, volume=5.0),
            _bond("MID", 5.0, volume=50.0),
        ])
        selected, _ = select_bonds(bonds, SETTLE)
        assert "LIQUID" in list(selected["isin"])
        assert "ILLIQ" not in list(selected["isin"])

    def test_earlier_maturity_wins_when_there_is_no_volume_column(self):
        bonds = pd.DataFrame([
            _bond("EARLY", 15.0), _bond("LATE", 16.5),
            _bond("TERM", 36.0), _bond("MID", 5.0),
        ])
        selected, _ = select_bonds(bonds, SETTLE)
        assert "EARLY" in list(selected["isin"])
        assert "LATE" not in list(selected["isin"])

    def test_terminal_rule_overrides_the_bucket_rule(self):
        # With nothing beyond it, the longest-dated bond is kept even though
        # another ISIN already represents its bucket.
        bonds = pd.DataFrame([_bond("EARLY", 15.0), _bond("LATE", 16.5), _bond("MID", 5.0)])
        selected, _ = select_bonds(bonds, SETTLE)
        assert {"EARLY", "LATE"} <= set(selected["isin"])


class TestSelectInputs:
    def test_returns_bonds_tbills_and_report(self, synthetic_bonds, synthetic_tbills):
        bonds_sel, tbills_sel, report = select_inputs(synthetic_bonds, synthetic_tbills, SETTLE)
        assert len(bonds_sel) > 0
        assert len(tbills_sel) == 3
        assert len(report) == len(synthetic_bonds)
        assert bonds_sel["residual_years"].is_monotonic_increasing
