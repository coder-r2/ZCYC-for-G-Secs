"""B3 -- the curve engine: solver, spline, and the auto-checks.

The five tests the plan puts on B's side of the line are all here:
input bonds reprice, the spline is C2 at every knot, discount factors fall
monotonically, the par curve reprices a par bond at 100, and (in test_bond.py)
price <-> YTM round-trips.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.bond import annualise, cashflows, accrued_30_360
from src.curve import Curve, CurveConfig, ablation_grid, build_zcyc

from conftest import true_zero

REPO_ROOT = Path(__file__).resolve().parents[1]
TOL = 1e-6


class TestRepricing:
    """Every input instrument must come back to its own market price."""

    def test_inputs_reprice_inside_tolerance(self, fitted_curve):
        assert fitted_curve.diagnostics["max_abs_reprice_error"] < TOL

    def test_every_single_instrument_reprices(self, fitted_curve):
        errors = fitted_curve.diagnostics["reprice_errors"]
        assert len(errors) == fitted_curve.diagnostics["n_instruments"]
        assert (errors["abs_error"] < TOL).all(), errors[errors["abs_error"] >= TOL].to_string()

    def test_global_solve_converged(self, fitted_curve):
        assert fitted_curve.diagnostics["solver"] == "global"
        assert fitted_curve.diagnostics["converged"]

    def test_recovers_the_curve_it_was_priced_from(self, fitted_curve):
        # The synthetic bonds were priced off true_zero, so the fit should
        # reproduce it across the whole grid, not merely reprice the inputs.
        grid = fitted_curve.grid()
        error_bps = np.abs(grid["zcy_semi"].to_numpy() - true_zero(grid["tenor_years"].to_numpy())) * 100
        assert error_bps.max() < 2.0


class TestAutoChecks:
    """The three checks that must hold on every run, with no manual inspection."""

    def test_all_checks_pass(self, fitted_curve):
        checks = fitted_curve.checks()
        assert checks["passed"].all(), checks.to_string(index=False)

    def test_assert_checks_is_silent_on_a_good_curve(self, fitted_curve):
        fitted_curve.assert_checks()

    def test_second_derivative_is_continuous_at_every_knot(self, fitted_curve):
        assert fitted_curve._c2_discontinuity() < 1e-9

    def test_discount_factors_are_strictly_decreasing(self, fitted_curve):
        grid = np.arange(0.25, 40.25, 0.25)
        assert (np.diff(fitted_curve.df(grid)) < 0).all()

    def test_assert_checks_raises_on_a_broken_curve(self):
        # An inverted, deeply negative curve makes discount factors rise.
        broken = Curve([1.0, 5.0, 10.0, 20.0], [5.0, -2.0, -8.0, -12.0])
        with pytest.raises(AssertionError):
            broken.assert_checks()


class TestCurveEvaluation:
    def test_discount_factor_matches_the_zero_rate(self, fitted_curve):
        for t in (0.5, 1.0, 7.0, 23.0):
            z = fitted_curve.zero(t)
            assert fitted_curve.df(t) == pytest.approx((1 + z / 200) ** (-2 * t))

    def test_df_at_zero_is_one(self, fitted_curve):
        assert fitted_curve.df(0.0) == pytest.approx(1.0)

    def test_scalar_and_vector_calls_agree(self, fitted_curve):
        tenors = np.array([1.0, 5.0, 15.0])
        assert fitted_curve.zero(tenors) == pytest.approx([fitted_curve.zero(t) for t in tenors])

    def test_rates_are_flat_beyond_the_last_node(self, fitted_curve):
        last = fitted_curve.node_t[-1]
        assert fitted_curve.zero(last + 5) == pytest.approx(fitted_curve.zero(last))
        assert fitted_curve.zero(200.0) == pytest.approx(fitted_curve.zero(last))

    def test_nodes_are_hit_exactly(self, fitted_curve):
        if fitted_curve.cfg.spline_variable == "zero":
            assert fitted_curve.zero(fitted_curve.node_t) == pytest.approx(fitted_curve.node_z)

    def test_rejects_unsorted_or_degenerate_nodes(self):
        with pytest.raises(ValueError):
            Curve([5.0, 1.0, 10.0], [6.0, 6.5, 7.0])
        with pytest.raises(ValueError):
            Curve([1.0], [6.0])


class TestParCurve:
    """par(T) = 2(1 - DF(T)) / sum(DF) -- must reprice a par bond at 100."""

    @pytest.mark.parametrize("T", [1.0, 2.0, 5.0, 10.0, 14.0, 30.0, 40.0])
    def test_par_bond_prices_at_exactly_100(self, fitted_curve, T):
        coupon = fitted_curve.par(T)
        dates = np.arange(0.5, T + 0.25, 0.5)
        price = float(np.sum((coupon / 2) * fitted_curve.df(dates)) + 100 * fitted_curve.df(T))
        assert price == pytest.approx(100.0, abs=1e-8)

    def test_par_sits_near_the_zero_rate_on_a_gently_sloping_curve(self, fitted_curve):
        for T in (2.0, 10.0, 30.0):
            assert abs(fitted_curve.par(T) - fitted_curve.zero(T)) < 0.75

    def test_par_needs_a_positive_tenor(self, fitted_curve):
        with pytest.raises(ValueError):
            fitted_curve.par(0.0)


class TestOutputGrid:
    def test_matches_the_frozen_fbil_schema(self, fitted_curve):
        grid = fitted_curve.grid()
        assert list(grid.columns) == ["tenor_years", "zcy_semi", "zcy_annual", "par_semi", "par_annual"]

    def test_covers_0_25_to_40_in_quarter_steps(self, fitted_curve):
        grid = fitted_curve.grid()
        assert len(grid) == 160
        assert grid["tenor_years"].iloc[0] == pytest.approx(0.25)
        assert grid["tenor_years"].iloc[-1] == pytest.approx(40.0)
        assert np.diff(grid["tenor_years"]) == pytest.approx(0.25)

    def test_annualised_columns_are_consistent(self, fitted_curve):
        grid = fitted_curve.grid()
        assert grid["zcy_annual"].to_numpy() == pytest.approx([annualise(z) for z in grid["zcy_semi"]])
        assert (grid["zcy_annual"] > grid["zcy_semi"]).all()

    def test_no_missing_values_anywhere(self, fitted_curve):
        assert not fitted_curve.grid().isna().to_numpy().any()

    def test_nodes_table_reports_the_fitted_degrees_of_freedom(self, fitted_curve):
        nodes = fitted_curve.nodes()
        assert len(nodes) == fitted_curve.diagnostics["n_instruments"]
        assert (nodes["df"] > 0).all()


class TestSolverFallback:
    """The documented fallback: sequential bootstrap via brentq."""

    def test_sequential_solver_reprices_to_its_own_looser_tolerance(self, synthetic_bonds, synthetic_tbills, settle):
        # A spline is global, so fitting node i shifts the segments behind it.
        # The sequential bootstrap therefore lands near 1e-3 per 100 face, not
        # at the 1e-6 the simultaneous solve reaches. Pinned here so the gap is
        # a measured, disclosed property rather than a surprise on the day.
        curve = build_zcyc(synthetic_bonds, synthetic_tbills, settle, CurveConfig(solver="sequential"))
        assert curve.diagnostics["solver"] == "sequential"
        assert curve.diagnostics["max_abs_reprice_error"] < 1e-2

    def test_global_solve_is_the_more_accurate_of_the_two(self, fitted_curve, synthetic_bonds, synthetic_tbills, settle):
        sequential = build_zcyc(synthetic_bonds, synthetic_tbills, settle, CurveConfig(solver="sequential"))
        assert fitted_curve.diagnostics["max_abs_reprice_error"] < sequential.diagnostics["max_abs_reprice_error"]

    def test_auto_keeps_the_better_solve_rather_than_the_fallback(self, synthetic_bonds, synthetic_tbills, settle):
        # With an unreachable tolerance the auto path tries both and must keep
        # the global result, since the fallback is more robust but less exact.
        cfg = CurveConfig(solver="auto", reprice_tol=1e-30)
        curve = build_zcyc(synthetic_bonds, synthetic_tbills, settle, cfg)
        assert curve.diagnostics["solver"] == "global"
        assert "no better" in curve.diagnostics["message"]

    def test_both_solvers_land_on_the_same_curve(self, fitted_curve, synthetic_bonds, synthetic_tbills, settle):
        sequential = build_zcyc(synthetic_bonds, synthetic_tbills, settle, CurveConfig(solver="sequential"))
        tenors = np.arange(1.0, 35.0, 1.0)
        assert sequential.zero(tenors) == pytest.approx(fitted_curve.zero(tenors), abs=0.02)


class TestAblationGrid:
    """A3 drives these 16 configurations; every one of them has to work."""

    def test_grid_has_sixteen_distinct_configurations(self):
        configs = ablation_grid()
        assert len(configs) == 16
        assert len({c.label() for c in configs}) == 16

    @pytest.mark.parametrize("cfg", ablation_grid(), ids=lambda c: c.label())
    def test_every_configuration_fits_and_passes_its_checks(self, cfg, synthetic_bonds, synthetic_tbills, settle):
        curve = build_zcyc(synthetic_bonds, synthetic_tbills, settle, cfg)
        assert curve.diagnostics["max_abs_reprice_error"] < TOL
        curve.assert_checks()

    def test_switches_actually_change_the_curve(self, synthetic_bonds, synthetic_tbills, settle):
        base = build_zcyc(synthetic_bonds, synthetic_tbills, settle, CurveConfig())
        variants = {
            "spline_variable": CurveConfig(spline_variable="logdf"),
            "boundary": CurveConfig(boundary="not-a-knot"),
            "time_convention": CurveConfig(time_convention="halfyear"),
        }
        tenors = np.arange(0.5, 40.0, 0.5)
        for name, cfg in variants.items():
            other = build_zcyc(synthetic_bonds, synthetic_tbills, settle, cfg)
            # Exact comparison: this asserts the switch is wired through, not
            # that it moves the curve a lot. On a smooth synthetic market the
            # spline-variable switch shifts the curve by well under a basis
            # point -- a result for the ablation table, not a failure.
            assert not np.array_equal(base.zero(tenors), other.zero(tenors)), f"{name} changed nothing"

    def test_thinning_switch_changes_the_node_count(self, synthetic_bonds, synthetic_tbills, settle):
        thinned = build_zcyc(synthetic_bonds, synthetic_tbills, settle, CurveConfig(thin_90d=True))
        full = build_zcyc(synthetic_bonds, synthetic_tbills, settle, CurveConfig(thin_90d=False))
        assert full.diagnostics["n_bonds"] >= thinned.diagnostics["n_bonds"]

    def test_rejects_an_unknown_switch_value(self):
        with pytest.raises(ValueError):
            CurveConfig(spline_variable="quadratic")
        with pytest.raises(ValueError):
            CurveConfig(boundary="clamped")


class TestInputHandling:
    def test_ytm_is_used_when_the_price_is_missing(self, synthetic_bonds, synthetic_tbills, settle):
        bonds = synthetic_bonds.copy()
        bonds.loc[bonds.index[:3], "price"] = np.nan
        curve = build_zcyc(bonds, synthetic_tbills, settle)
        assert curve.diagnostics["max_abs_reprice_error"] < TOL

    def test_a_bond_with_neither_price_nor_ytm_is_rejected(self, synthetic_bonds, synthetic_tbills, settle):
        bonds = synthetic_bonds.copy()
        bonds.loc[bonds.index[0], ["price", "ytm"]] = np.nan
        with pytest.raises(ValueError, match="neither a price nor a YTM"):
            build_zcyc(bonds, synthetic_tbills, settle)

    def test_too_few_instruments_is_rejected(self, settle):
        bonds = pd.DataFrame([{
            "isin": "ONLY", "desc": "x", "coupon": 7.0,
            "maturity": pd.Timestamp("2036-09-21"), "price": 100.0, "ytm": 7.0,
        }])
        with pytest.raises(ValueError, match="at least two"):
            build_zcyc(bonds, pd.DataFrame(columns=["tenor", "rate"]), settle)

    def test_annual_basis_tbills_are_converted(self, synthetic_bonds, synthetic_tbills, settle):
        annual = synthetic_tbills.copy()
        annual["rate"] = [annualise(r) for r in annual["rate"]]
        curve = build_zcyc(synthetic_bonds, annual, settle, CurveConfig(tbill_rate_basis="annual"))
        base = build_zcyc(synthetic_bonds, synthetic_tbills, settle)
        assert curve.zero(0.5) == pytest.approx(base.zero(0.5), abs=1e-6)

    def test_diagnostics_carry_the_selection_report(self, fitted_curve):
        report = fitted_curve.diagnostics["selection_report"]
        assert {"isin", "segment", "kept", "reason"} <= set(report.columns)


@pytest.mark.skipif(
    not (REPO_ROOT / "data" / "clean" / "bonds.csv").exists(),
    reason="waiting on A's cleaned FBIL data",
)
class TestAgainstRealData:
    """B1's second acceptance test: our dirty price must agree with FBIL's
    published price for at least 5 bonds. Runs automatically once A lands
    data/clean/bonds.csv -- skipped until then."""

    def test_dirty_price_agrees_with_fbil_for_five_bonds(self, settle):
        bonds = pd.read_csv(REPO_ROOT / "data" / "clean" / "bonds.csv", parse_dates=["maturity"])
        sample = bonds.dropna(subset=["price", "ytm"]).head(5)
        assert len(sample) >= 5, "need at least 5 priced bonds in data/clean/bonds.csv"
        for _, row in sample.iterrows():
            cf = cashflows(row["coupon"], row["maturity"], settle)
            ours = float((cf["cf"] * [(1 + row["ytm"] / 200) ** (-2 * t) for t in cf["t"]]).sum())
            theirs = row["price"] + accrued_30_360(row["coupon"], row["maturity"], settle)
            assert ours == pytest.approx(theirs, abs=0.05), row.get("isin")
