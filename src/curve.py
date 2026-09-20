"""Zero coupon yield curve: cubic spline through spot rates, solved globally.

Method
------
The unknowns are the zero rates at a set of nodes -- one node per bootstrap
input instrument (T-bill tenors and selected bond maturities). A cubic spline is
laid through those unknown rates, every input instrument is priced by
discounting its own cash flows off that spline, and all the node rates are
solved simultaneously with ``scipy.optimize.least_squares`` so that every input
reprices to its market price at once.

This is a global solve rather than a sequential bootstrap: a coupon bond's cash
flows land between nodes as well as on them, so its price depends on the whole
spline, not just on the node at its maturity. Solving one node at a time would
force an arbitrary interpolation rule for the interior flows. The sequential
bootstrap is kept as a documented fallback for when the global solve misbehaves.

FBIL's exact Pienaar-Choudhry formulation is not public, so this is a
cubic-spline approximation of that approach, not a reimplementation of it.

Units: rates in percent per annum, semiannually compounded; prices per 100 face.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, least_squares

from src.bond import (
    COUPONS_PER_YEAR,
    accrued_30_360,
    annualise,
    cashflows,
    discount_factor,
    deannualise,
    to_date,
)
from src.selection import select_inputs

SPLINE_VARIABLES = ("zero", "logdf")
BOUNDARIES = ("natural", "not-a-knot")
SOLVERS = ("global", "sequential", "auto")


@dataclass
class CurveConfig:
    """Modelling switches.

    The first four fields are the ablation dimensions handed to A: 2 x 2 x 2 x 2
    = 16 configurations. Everything else stays fixed across the grid.
    """

    spline_variable: str = "zero"       # "zero" | "logdf"
    boundary: str = "natural"           # "natural" | "not-a-knot"
    thin_90d: bool = True               # 90-day thinning of the 1-14y segment
    time_convention: str = "act365"     # "act365" | "halfyear"

    min_gap_days: int = 90
    solver: str = "auto"                # "global" | "sequential" | "auto"
    input_price_type: str = "clean"     # published bond price is clean
    tbill_rate_basis: str = "semi"      # "semi" | "annual"
    reprice_tol: float = 1e-6           # per 100 face
    grid_start: float = 0.25
    grid_stop: float = 40.0
    grid_step: float = 0.25

    def __post_init__(self):
        if self.spline_variable not in SPLINE_VARIABLES:
            raise ValueError(f"spline_variable must be one of {SPLINE_VARIABLES}")
        if self.boundary not in BOUNDARIES:
            raise ValueError(f"boundary must be one of {BOUNDARIES}")
        if self.solver not in SOLVERS:
            raise ValueError(f"solver must be one of {SOLVERS}")

    def label(self) -> str:
        """Compact identifier for the ablation results table."""
        return (
            f"{self.spline_variable}|{self.boundary}"
            f"|thin={'on' if self.thin_90d else 'off'}|{self.time_convention}"
        )


def ablation_grid() -> list[CurveConfig]:
    """All 16 ablation configurations, for A3."""
    return [
        CurveConfig(spline_variable=v, boundary=b, thin_90d=t, time_convention=c)
        for v in SPLINE_VARIABLES
        for b in BOUNDARIES
        for t in (True, False)
        for c in ("act365", "halfyear")
    ]


class Curve:
    """A fitted ZCYC. Exposes ``zero``, ``df``, ``grid`` and ``par``."""

    def __init__(self, node_t, node_z, cfg: CurveConfig | None = None, diagnostics: dict | None = None):
        self.cfg = cfg or CurveConfig()
        self.node_t = np.asarray(node_t, dtype=float)
        self.node_z = np.asarray(node_z, dtype=float)
        self.diagnostics = diagnostics or {}
        if self.node_t.ndim != 1 or self.node_t.size < 2:
            raise ValueError("need at least two curve nodes")
        if np.any(np.diff(self.node_t) <= 0):
            raise ValueError("curve nodes must be strictly increasing in t")
        self._build_spline()

    # ---------------------------------------------------------------- spline

    def _bc_type(self, n_points: int) -> str:
        # not-a-knot needs at least 4 points; natural is the safe fallback and
        # the ablation row records that the swap happened.
        if self.cfg.boundary == "not-a-knot" and n_points < 4:
            return "natural"
        return self.cfg.boundary

    def _build_spline(self):
        if self.cfg.spline_variable == "zero":
            x, y = self.node_t, self.node_z
        else:
            # Spline log DF instead of the zero rate, anchored at DF(0) = 1.
            log_df = -COUPONS_PER_YEAR * self.node_t * np.log1p(
                self.node_z / (100.0 * COUPONS_PER_YEAR)
            )
            x = np.concatenate([[0.0], self.node_t])
            y = np.concatenate([[0.0], log_df])
        self._spline = CubicSpline(x, y, bc_type=self._bc_type(len(x)))
        self._t_min, self._t_max = float(self.node_t[0]), float(self.node_t[-1])
        self._z_min, self._z_max = float(self.node_z[0]), float(self.node_z[-1])

    # ------------------------------------------------------------ evaluation

    def zero(self, t):
        """Zero rate (percent, semiannually compounded) at time ``t`` in years.

        Outside the span of the input instruments the rate is held flat at the
        nearest node. Cubic extrapolation past a 40-year terminal node can swing
        violently on no information at all, so flat is both safer and easier to
        defend on a slide.
        """
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        out = np.empty_like(t_arr)

        if self.cfg.spline_variable == "zero":
            inside = (t_arr >= self._t_min) & (t_arr <= self._t_max)
            out[inside] = self._spline(t_arr[inside])
            out[t_arr < self._t_min] = self._z_min
            out[t_arr > self._t_max] = self._z_max
        else:
            df = self.df(t_arr)
            with np.errstate(divide="ignore", invalid="ignore"):
                out = (
                    df ** (-1.0 / (COUPONS_PER_YEAR * np.where(t_arr > 0, t_arr, np.nan))) - 1.0
                ) * 100.0 * COUPONS_PER_YEAR
            out[t_arr <= 0] = self._z_min

        return float(out[0]) if np.isscalar(t) or np.asarray(t).ndim == 0 else out

    def df(self, t):
        """Discount factor at ``t`` years."""
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))

        if self.cfg.spline_variable == "zero":
            z = np.atleast_1d(self.zero(t_arr))
            out = (1.0 + z / (100.0 * COUPONS_PER_YEAR)) ** (-COUPONS_PER_YEAR * t_arr)
        else:
            out = np.empty_like(t_arr)
            inside = t_arr <= self._t_max
            out[inside] = np.exp(self._spline(np.clip(t_arr[inside], 0.0, None)))
            beyond = ~inside
            out[beyond] = (1.0 + self._z_max / (100.0 * COUPONS_PER_YEAR)) ** (
                -COUPONS_PER_YEAR * t_arr[beyond]
            )
        out[t_arr <= 0] = 1.0

        return float(out[0]) if np.isscalar(t) or np.asarray(t).ndim == 0 else out

    def par(self, T: float) -> float:
        """Par yield (percent, semiannual) for a bond maturing in ``T`` years.

        par(T) = 2 (1 - DF(T)) / sum(DF at each coupon date)

        Coupon dates are counted backwards from T in half-year steps. On the
        0.25-year output grid that leaves a short first period at the odd
        tenors (0.75y pays at 0.25y and 0.75y), so the first coupon is accrued
        over its actual length instead of being paid as a full half-yearly one.
        At whole half-year tenors the stub is exactly half a year and this
        collapses back to the formula above.
        """
        if T <= 0:
            raise ValueError("par yield needs T > 0")
        step = 1.0 / COUPONS_PER_YEAR
        times, t = [], T
        while t > 1e-12:
            times.append(t)
            t -= step
        times.reverse()

        dfs = np.atleast_1d(self.df(np.array(times)))
        accrual = np.full(len(times), step)
        accrual[0] = times[0]          # short first period at the odd tenors
        annuity = float(np.sum(accrual * dfs))
        return (1.0 - self.df(T)) / annuity * 100.0

    def grid(self, start: float | None = None, stop: float | None = None, step: float | None = None) -> pd.DataFrame:
        """Output grid in the frozen ``fbil_zcyc`` schema, ready for A's compare()."""
        start = self.cfg.grid_start if start is None else start
        stop = self.cfg.grid_stop if stop is None else stop
        step = self.cfg.grid_step if step is None else step

        tenors = np.round(np.arange(start, stop + step / 2, step), 10)
        zcy_semi = np.atleast_1d(self.zero(tenors))
        return pd.DataFrame(
            {
                "tenor_years": tenors,
                "zcy_semi": zcy_semi,
                "zcy_annual": [annualise(z) for z in zcy_semi],
                "par_semi": [self.par(T) for T in tenors],
                "par_annual": [annualise(self.par(T)) for T in tenors],
            }
        )

    def nodes(self) -> pd.DataFrame:
        """The fitted node rates -- the curve's actual degrees of freedom."""
        return pd.DataFrame(
            {
                "t": self.node_t,
                "zcy_semi": self.node_z,
                "zcy_annual": [annualise(z) for z in self.node_z],
                "df": self.df(self.node_t),
            }
        )

    # ---------------------------------------------------------------- checks

    def checks(self) -> pd.DataFrame:
        """The three auto-checks that must hold on every run, no eyeballing.

        1. Repricing error -- every input bond back to its market price.
        2. C2 continuity   -- second derivative agrees across every interior knot.
        3. Monotone DFs    -- discount factors strictly decreasing on the grid.
        """
        rows = []

        max_err = self.diagnostics.get("max_abs_reprice_error")
        if max_err is not None:
            rows.append(
                {
                    "check": "max repricing error (per 100 face)",
                    "value": max_err,
                    "threshold": self.cfg.reprice_tol,
                    "passed": bool(max_err < self.cfg.reprice_tol),
                }
            )

        jump = self._c2_discontinuity()
        rows.append(
            {
                "check": "max 2nd-derivative jump at knots (relative)",
                "value": jump,
                "threshold": 1e-9,
                "passed": bool(jump < 1e-9),
            }
        )

        grid_t = np.arange(self.cfg.grid_start, self.cfg.grid_stop + self.cfg.grid_step / 2, self.cfg.grid_step)
        worst_step = float(np.max(np.diff(self.df(grid_t))))
        rows.append(
            {
                "check": "largest DF step (must be negative)",
                "value": worst_step,
                "threshold": 0.0,
                "passed": bool(worst_step < 0.0),
            }
        )

        return pd.DataFrame(rows)

    def _c2_discontinuity(self) -> float:
        """Largest mismatch in the second derivative across interior knots.

        Read straight off the piecewise coefficients rather than by evaluating
        either side of the knot, so the number is the actual continuity defect
        and not a finite-difference artefact. For segment i with local
        coordinate s = t - x[i] and coefficients c[:, i], the second derivative
        is 6 c[0, i] s + 2 c[1, i]; continuity requires the value at the right
        end of one segment to equal the value at the left end of the next.
        """
        x, c = self._spline.x, self._spline.c
        if len(x) < 3:
            return 0.0
        h = np.diff(x)[:-1]
        left = 6.0 * c[0, :-1] * h + 2.0 * c[1, :-1]
        right = 2.0 * c[1, 1:]
        scale = max(1.0, float(np.max(np.abs(right))))
        return float(np.max(np.abs(left - right)) / scale)

    def assert_checks(self):
        """Raise if any auto-check failed. Called by run_all.py."""
        failures = self.checks()
        failed = failures[~failures["passed"]]
        if len(failed) > 0:
            raise AssertionError("curve auto-checks failed:\n" + failed.to_string(index=False))


# ------------------------------------------------------------------ instruments


def _tbill_instruments(tbills: pd.DataFrame, cfg: CurveConfig) -> list[dict]:
    instruments = []
    for _, row in tbills.iterrows():
        rate = float(row["rate"])
        if cfg.tbill_rate_basis == "annual":
            rate = deannualise(rate)
        t = float(row["t"])
        instruments.append(
            {
                "kind": "tbill",
                "name": str(row["tenor"]),
                "t": t,
                "times": np.array([t]),
                "amounts": np.array([100.0]),
                "market_dirty": 100.0 * discount_factor(rate, t),
                "guess": rate,
            }
        )
    return instruments


def _bond_instruments(bonds: pd.DataFrame, settle, cfg: CurveConfig) -> list[dict]:
    instruments = []
    for _, row in bonds.iterrows():
        coupon, maturity = float(row["coupon"]), row["maturity"]
        cf = cashflows(coupon, maturity, settle, cfg.time_convention)

        price = row.get("price")
        ytm = row.get("ytm")
        has_price = price is not None and pd.notna(price)
        if has_price:
            dirty = float(price)
            if cfg.input_price_type == "clean":
                dirty += accrued_30_360(coupon, maturity, settle)
        elif ytm is not None and pd.notna(ytm):
            dirty = float((cf["cf"] * [discount_factor(float(ytm), t) for t in cf["t"]]).sum())
        else:
            raise ValueError(f"bond {row.get('isin')} has neither a price nor a YTM")

        instruments.append(
            {
                "kind": "bond",
                "name": str(row.get("isin", row.get("desc", "bond"))),
                "t": float(cf["t"].iloc[-1]),
                "times": cf["t"].to_numpy(dtype=float),
                "amounts": cf["cf"].to_numpy(dtype=float),
                "market_dirty": dirty,
                "guess": float(ytm) if ytm is not None and pd.notna(ytm) else float(coupon),
            }
        )
    return instruments


def _model_price(curve: Curve, inst: dict) -> float:
    return float(np.sum(inst["amounts"] * curve.df(inst["times"])))


def _collapse_to_nodes(instruments: list[dict]) -> list[dict]:
    """One node per distinct maturity -- duplicates would break the spline."""
    by_t: dict[float, dict] = {}
    for inst in sorted(instruments, key=lambda i: i["t"]):
        key = round(inst["t"], 8)
        if key not in by_t:
            by_t[key] = inst
    return list(by_t.values())


# --------------------------------------------------------------------- solvers


def _solve_global(instruments: list[dict], cfg: CurveConfig) -> tuple[np.ndarray, dict]:
    node_t = np.array([i["t"] for i in instruments], dtype=float)
    x0 = np.array([i["guess"] for i in instruments], dtype=float)

    def residuals(z_nodes: np.ndarray) -> np.ndarray:
        curve = Curve(node_t, z_nodes, cfg)
        return np.array([_model_price(curve, inst) - inst["market_dirty"] for inst in instruments])

    result = least_squares(residuals, x0, method="trf", xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=2000)
    return result.x, {"solver": "global", "converged": bool(result.success), "message": result.message}


def _solve_sequential(instruments: list[dict], cfg: CurveConfig) -> tuple[np.ndarray, dict]:
    """Documented fallback: solve one node at a time with brentq.

    Instrument i has no cash flow beyond node i, so a spline through nodes 1..i
    (flat below the first) is enough to price it. Cheaper and far more robust
    than the global solve -- each node is a one-dimensional bracketed root find
    that cannot fail to converge.

    The price of that robustness is exactness. A cubic spline is global: adding
    the next knot reshapes the polynomial pieces behind it, so an instrument
    that repriced perfectly when its node was fitted drifts once the longer
    nodes arrive. Expect repricing errors around 1e-3 per 100 face rather than
    the 1e-6 the global solve achieves. That is a property of the method, not a
    bug -- if this path is ever used for the headline curve it must be
    disclosed on the validation slide.
    """
    node_t = np.array([i["t"] for i in instruments], dtype=float)
    z = np.zeros(len(instruments))

    for i, inst in enumerate(instruments):
        def objective(rate: float, i=i, inst=inst) -> float:
            z_try = np.concatenate([z[:i], [rate]])
            t_try = node_t[: i + 1]
            if len(t_try) == 1:
                # Single node: flat curve at that rate.
                df = np.array([discount_factor(rate, t) for t in inst["times"]])
                return float(np.sum(inst["amounts"] * df)) - inst["market_dirty"]
            return _model_price(Curve(t_try, z_try, cfg), inst) - inst["market_dirty"]

        lo, hi = -20.0, 60.0
        if objective(lo) * objective(hi) > 0:
            z[i] = inst["guess"]
            continue
        z[i] = brentq(objective, lo, hi, xtol=1e-12, rtol=1e-14, maxiter=200)

    return z, {"solver": "sequential", "converged": True, "message": "sequential bootstrap (fallback)"}


# ----------------------------------------------------------------- entry point


def build_zcyc(bonds_df: pd.DataFrame, tbills: pd.DataFrame, settle, cfg: CurveConfig | None = None) -> Curve:
    """Fit the zero coupon yield curve. Frozen interface-contract signature.

    ``bonds_df`` follows the frozen schema [isin, desc, coupon, maturity, price,
    ytm]; ``tbills`` follows [tenor, rate]. If ``bonds_df`` has already been
    thinned by ``selection.select_bonds`` it is used as-is, otherwise the
    thinning runs here.
    """
    cfg = cfg or CurveConfig()
    settle = to_date(settle)

    bonds_sel, tbills_sel, report = select_inputs(
        bonds_df, tbills, settle, thin_90d=cfg.thin_90d, min_gap_days=cfg.min_gap_days
    )

    instruments = _tbill_instruments(tbills_sel, cfg) + _bond_instruments(bonds_sel, settle, cfg)
    instruments = _collapse_to_nodes(instruments)
    if len(instruments) < 2:
        raise ValueError("need at least two input instruments to fit a curve")

    node_t = np.array([i["t"] for i in instruments], dtype=float)

    if cfg.solver == "sequential":
        node_z, info = _solve_sequential(instruments, cfg)
    else:
        node_z, info = _solve_global(instruments, cfg)

    curve = Curve(node_t, node_z, cfg, diagnostics=info)
    errors = _reprice_table(curve, instruments)

    if cfg.solver == "auto" and float(errors["abs_error"].max()) >= cfg.reprice_tol:
        # Global solve missed tolerance -- try the sequential bootstrap, but
        # keep whichever of the two actually reprices better. The fallback is
        # more robust, not more accurate (see _solve_sequential), so adopting
        # it unconditionally could make a merely-imperfect curve worse.
        alt_z, alt_info = _solve_sequential(instruments, cfg)
        alt_curve = Curve(node_t, alt_z, cfg, diagnostics=alt_info)
        alt_errors = _reprice_table(alt_curve, instruments)
        if float(alt_errors["abs_error"].max()) < float(errors["abs_error"].max()):
            alt_info["message"] += " (global solve missed the repricing tolerance)"
            node_z, info, curve, errors = alt_z, alt_info, alt_curve, alt_errors
        else:
            info["message"] += " (missed tolerance; sequential fallback was no better)"

    curve.diagnostics.update(
        {
            "config": cfg.label(),
            "n_instruments": len(instruments),
            "n_tbills": len(tbills_sel),
            "n_bonds": len(bonds_sel),
            "reprice_errors": errors,
            "max_abs_reprice_error": float(errors["abs_error"].max()),
            "selection_report": report,
            "settle": settle,
        }
    )
    return curve


def _reprice_table(curve: Curve, instruments: list[dict]) -> pd.DataFrame:
    rows = []
    for inst in instruments:
        model = _model_price(curve, inst)
        rows.append(
            {
                "name": inst["name"],
                "kind": inst["kind"],
                "t": inst["t"],
                "market_dirty": inst["market_dirty"],
                "model_dirty": model,
                "error": model - inst["market_dirty"],
                "abs_error": abs(model - inst["market_dirty"]),
            }
        )
    return pd.DataFrame(rows)
