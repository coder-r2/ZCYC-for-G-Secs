"""B1 -- bond utilities: coupon schedule, 30/360 accrued, price <-> YTM."""

from datetime import date

import pytest

from src.bond import (
    accrued_30_360,
    annualise,
    cashflows,
    clean_price,
    coupon_dates,
    days_30_360,
    deannualise,
    dirty_price,
    previous_coupon_date,
    settlement_date,
    ytm_from_price,
)

SETTLE = date(2026, 9, 21)


class TestCouponSchedule:
    def test_counts_back_from_maturity(self):
        dates = coupon_dates(maturity=date(2033, 1, 6), settle=SETTLE)
        assert dates[0] == date(2027, 1, 6)
        assert dates[-1] == date(2033, 1, 6)
        assert len(dates) == 13

    def test_every_date_is_after_settlement(self):
        for d in coupon_dates(date(2030, 3, 15), SETTLE):
            assert d > SETTLE

    def test_end_of_month_maturity_stays_on_the_31st(self):
        # Measured back from maturity each time, so a single February cannot
        # drag the whole schedule onto the 30th.
        dates = coupon_dates(date(2031, 1, 31), SETTLE)
        assert all(d.day == 31 for d in dates if d.month == 1)
        assert all(d.day in (30, 31) for d in dates)

    def test_maturity_at_or_before_settlement_is_rejected(self):
        with pytest.raises(ValueError):
            coupon_dates(SETTLE, SETTLE)

    def test_final_cashflow_returns_principal(self):
        cf = cashflows(7.26, date(2033, 1, 6), SETTLE)
        assert cf["cf"].iloc[-1] == pytest.approx(100 + 7.26 / 2)
        assert cf["cf"].iloc[0] == pytest.approx(7.26 / 2)

    def test_times_are_increasing_and_positive(self):
        for convention in ("act365", "halfyear"):
            cf = cashflows(7.26, date(2033, 1, 6), SETTLE, convention)
            assert (cf["t"] > 0).all()
            assert cf["t"].is_monotonic_increasing

    def test_halfyear_convention_is_whole_periods_on_a_coupon_date(self):
        # Settling exactly on a coupon date, the flows land on 0.5, 1.0, 1.5...
        cf = cashflows(7.00, date(2032, 1, 6), settle=date(2027, 1, 6), time_convention="halfyear")
        assert list(cf["t"]) == pytest.approx([0.5 * (i + 1) for i in range(10)])

    def test_unknown_time_convention_is_rejected(self):
        with pytest.raises(ValueError):
            cashflows(7.0, date(2030, 1, 1), SETTLE, "act360")


class TestAccruedInterest:
    def test_zero_on_a_coupon_date(self):
        assert accrued_30_360(7.26, date(2033, 1, 6), date(2027, 1, 6)) == pytest.approx(0.0)

    def test_half_a_coupon_after_three_months(self):
        # 6 Jan -> 6 Apr is 90 days on 30/360, half of the 180-day period.
        accrued = accrued_30_360(8.00, date(2033, 1, 6), date(2027, 4, 6))
        assert accrued == pytest.approx(0.5 * 8.00 / 2)

    def test_approaches_a_full_coupon_just_before_the_next_date(self):
        accrued = accrued_30_360(8.00, date(2033, 1, 6), date(2027, 7, 5))
        assert accrued == pytest.approx(8.00 / 2 * 179 / 180)

    def test_previous_coupon_date_brackets_settlement(self):
        prev = previous_coupon_date(date(2033, 1, 6), SETTLE)
        assert prev <= SETTLE < coupon_dates(date(2033, 1, 6), SETTLE)[0]

    @pytest.mark.parametrize(
        "start, end, expected",
        [
            (date(2026, 1, 1), date(2026, 7, 1), 180),
            (date(2026, 1, 31), date(2026, 7, 31), 180),
            (date(2026, 1, 1), date(2027, 1, 1), 360),
            (date(2026, 1, 15), date(2026, 2, 15), 30),
        ],
    )
    def test_day_count(self, start, end, expected):
        assert days_30_360(start, end) == expected


class TestPriceYieldRoundTrip:
    """The headline B1 test: price -> YTM -> price back to 1e-6."""

    @pytest.mark.parametrize("coupon", [0.0, 5.15, 7.26, 9.40])
    @pytest.mark.parametrize("maturity", [date(2028, 5, 15), date(2033, 1, 6), date(2056, 6, 30)])
    @pytest.mark.parametrize("ytm", [3.5, 6.95, 11.0])
    def test_clean_price_round_trips(self, coupon, maturity, ytm):
        price = clean_price(coupon, maturity, SETTLE, ytm)
        assert ytm_from_price(coupon, maturity, SETTLE, price) == pytest.approx(ytm, abs=1e-6)

    def test_dirty_price_round_trips(self):
        price = dirty_price(7.26, date(2033, 1, 6), SETTLE, 6.95)
        recovered = ytm_from_price(7.26, date(2033, 1, 6), SETTLE, price, price_type="dirty")
        assert recovered == pytest.approx(6.95, abs=1e-6)

    def test_dirty_less_accrued_is_clean(self):
        dirty = dirty_price(7.26, date(2033, 1, 6), SETTLE, 6.95)
        clean = clean_price(7.26, date(2033, 1, 6), SETTLE, 6.95)
        assert dirty - clean == pytest.approx(accrued_30_360(7.26, date(2033, 1, 6), SETTLE))

    def test_par_bond_prices_at_par_under_half_year_counting(self):
        # Coupon == yield, settling on a coupon date: with the flows on exact
        # half-year periods this is 100 by construction.
        price = clean_price(7.00, date(2033, 1, 6), date(2027, 1, 6), 7.00, "halfyear")
        assert price == pytest.approx(100.0)

    def test_actual_365_shifts_the_par_bond_off_100(self):
        # Same bond on actual/365: the periods are 181/184 days rather than
        # exactly half a year, so it prices a few basis points away from par.
        # This gap is the whole point of the time-convention ablation -- it is a
        # real modelling difference, not a bug.
        price = clean_price(7.00, date(2033, 1, 6), date(2027, 1, 6), 7.00, "act365")
        assert price != pytest.approx(100.0, abs=1e-6)
        assert abs(price - 100.0) < 0.05

    def test_price_falls_as_yield_rises(self):
        prices = [clean_price(7.26, date(2036, 1, 6), SETTLE, y) for y in (5.0, 6.0, 7.0, 8.0)]
        assert prices == sorted(prices, reverse=True)

    def test_unreachable_price_is_rejected(self):
        with pytest.raises(ValueError):
            ytm_from_price(7.26, date(2033, 1, 6), SETTLE, price=10_000.0)


class TestCompounding:
    def test_annualise_round_trips(self):
        assert deannualise(annualise(7.26)) == pytest.approx(7.26)

    def test_effective_annual_exceeds_semiannual(self):
        assert annualise(7.26) > 7.26


class TestSettlement:
    def test_t_plus_one_on_a_weekday(self):
        assert settlement_date(date(2026, 9, 21)) == date(2026, 9, 22)  # Mon -> Tue

    def test_rolls_over_the_weekend(self):
        assert settlement_date(date(2026, 9, 18)) == date(2026, 9, 21)  # Fri -> Mon
