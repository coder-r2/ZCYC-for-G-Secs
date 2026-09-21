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


class TestAgainstConceptNoteIllustration:
    """The worked example from the Concept Note on Revised G-Sec Valuation
    Methodology (analysis dated 20 Nov 2019), right-hand column: which ISINs
    survive once securities redeeming within 90 days of one another are grouped.

    The document does not publish the volume*trades figures behind its picks, so
    the liquidity column here is set to agree with its stated winners. What the
    test pins is the grouping -- how many nodes survive and which windows they
    fall in -- which is the part the document fully determines.
    """

    ROWS = [
        ("7.37", "2023-04-16", True), ("7.16", "2023-05-20", False), ("6.17", "2023-06-12", False),
        ("8.83", "2023-11-25", False), ("7.68", "2023-12-15", False), ("7.32", "2024-01-28", True),
        ("7.35", "2024-06-22", True), ("8.40", "2024-07-28", False), ("6.18", "2024-11-04", True),
        ("9.15", "2024-11-14", False), ("8.20", "2025-09-24", True),
    ]
    ANALYSIS_DATE = "2019-11-20"

    def _frame(self, with_volume: bool):
        return pd.DataFrame([
            {"isin": f"IN{i}", "desc": f"{c}% GS", "coupon": float(c),
             "maturity": pd.Timestamp(d), "price": 100.0, "ytm": 6.0,
             **({"volume": 900.0 if sel else 100.0, "trades": 5} if with_volume else {})}
            for i, (c, d, sel) in enumerate(self.ROWS)
        ])

    def test_number_of_surviving_nodes_matches(self):
        _, report = select_bonds(self._frame(with_volume=False), self.ANALYSIS_DATE)
        assert report["kept"].sum() == sum(sel for _, _, sel in self.ROWS) == 5

    def test_every_row_matches_when_liquidity_favours_the_documented_picks(self):
        _, report = select_bonds(self._frame(with_volume=True), self.ANALYSIS_DATE)
        kept = dict(zip(report["isin"], report["kept"]))
        for i, (_, date, expected) in enumerate(self.ROWS):
            assert bool(kept[f"IN{i}"]) is expected, f"IN{i} ({date})"

    def test_selected_nodes_are_more_than_90_days_apart(self):
        selected, _ = select_bonds(self._frame(with_volume=True), self.ANALYSIS_DATE)
        gaps = selected.sort_values("maturity")["maturity"].diff().dt.days.dropna()
        assert (gaps > 90).all()

    def test_one_year_exactly_is_excluded_not_included(self):
        # "securities having less than or equal to 1-year residual maturity will
        # not be used as model input" -- the boundary is inclusive.
        bonds = pd.DataFrame([
            {"isin": "EXACT1Y", "desc": "x", "coupon": 7.0,
             "maturity": pd.Timestamp("2027-09-21"), "price": 100.0, "ytm": 7.0},
            {"isin": "MID", "desc": "x", "coupon": 7.0,
             "maturity": pd.Timestamp("2031-09-21"), "price": 100.0, "ytm": 7.0},
        ])
        selected, report = select_bonds(bonds, "2026-09-21")
        assert list(selected["isin"]) == ["MID"]
        assert "1y or less" in report.loc[report["isin"] == "EXACT1Y", "reason"].iloc[0]
