# Locked Assumptions

Fill in as each decision is confirmed at kickoff. This file is a submission deliverable — keep it accurate, it's what a grader reads to understand our modeling choices.

- [ ] Valuation date (D)
- [ ] Input data source (published YTMs/prices, trade-level assumed unavailable)
- [ ] Settlement convention
- [ ] Accrued interest convention
- [ ] Compounding convention
- [ ] Time-to-cashflow convention (headline vs ablation alternative)
- [ ] Spline variable + boundary condition (headline vs ablation alternatives)
- [ ] Output grid range/step
- [ ] Bonus discounting curve (SDL ZCYC) and how it's applied
- [ ] Bonus SDL selection criteria
- [ ] Parent value convention used in STRIP normalisation
- [ ] Known simplifications vs the official FBIL/RBI methodology (state each one explicitly)

---

## Curve engine (B) — locked

Appended by B; A and C own their own sections. These were fixed **before** any
comparison against FBIL's published curve was run, per the honesty rule.

| Decision | Choice | Note |
|---|---|---|
| Settlement | T+1, rolled over weekends | No public-holiday calendar — see simplifications |
| Coupon frequency | Semiannual, schedule counted backwards from maturity | Measured from maturity each period, so an end-of-month maturity keeps its day |
| Accrued interest | 30/360 (bond basis), 180-day semiannual period | |
| Compounding | Semiannual: `DF = (1 + z/2)^(-2t)`; annualised = `(1 + z/2)^2 - 1` | |
| Time to cash flow | Actual days / 365 from settlement | Half-year-period counting is the ablation alternative |
| Rate units | Percent per annum throughout; prices per 100 face | |
| Short end (≤ 1y) | T-bills at 7D / 6M / 12M; bonds under 1y dropped | Nearest available tenor if FBIL quotes 91D/182D/364D instead |
| Mid segment (1–14y) | Greedy 90-day minimum spacing between kept bonds | Thinning on/off is an ablation switch |
| Long end (> 14y) | One ISIN per bucket (14–18, 18–22, 22–26, 26–30, 30–34, 34+), plus the longest-dated ISIN | More liquid bond wins a bucket, else the earlier maturity |
| Spline | `scipy.CubicSpline` on zero rates, natural boundary | Log-DF and not-a-knot are ablation alternatives |
| Solver | All node rates solved simultaneously (`least_squares`) so every input reprices at once | Sequential bootstrap kept as a disclosed fallback |
| Extrapolation | Flat beyond the longest input bond, and below the shortest | Cubic extrapolation past 40y is unstable on no information |
| Par yield | `par(T) = 2(1 - DF(T)) / sum(DF)`, first coupon accrued over a short stub at odd quarter-year tenors | Collapses to the standard formula at whole half-years |
| Output grid | 0.25 to 40 years, step 0.25 (160 points), semiannual + annualised + par | Matches the frozen `fbil_zcyc` schema |

**Auto-checks enforced on every run** (the pipeline raises and stops if any fail):
max repricing error < 1e-6 per 100 face; second derivative continuous at every
knot; discount factors strictly decreasing across the whole output grid.

**Known simplifications vs the official methodology**

1. FBIL's Pienaar–Choudhry formulation is not public, so this is a cubic-spline
   **approximation of** that approach, not a reimplementation of it.
2. No public-holiday calendar; T+1 settlement rolls over weekends only.
3. Flat extrapolation beyond the longest input bond — the far end is asserted
   rather than fitted.
4. Published YTMs of non-traded ISINs already embed FBIL's own model, so the
   inputs are partly circular with the curve we compare against.
5. The sequential fallback solver reprices to roughly 1e-3 rather than 1e-6; if
   it is ever the path taken, that must be disclosed on the validation slide.

---

## Data, comparison and ablation (A) — locked

Appended by A. Covers `data_loader.py`, `compare.py` and `ablation.py` only;
B's and C's sections above stand as written.

| Decision | Choice | Note |
|---|---|---|
| Valuation date D | **2026-09-18** (Friday) | Settlement 2026-09-21, T+1 per B's convention |
| Rate units in `data/clean/` | Percent throughout; `6.05` means 6.05% | The loader never divides a rate by 100 — conversion happens at point of use |
| Dates in `data/clean/` | ISO `YYYY-MM-DD` strings | Raw files are parsed `dayfirst=True` (FBIL quotes 14/05/2031) |
| Money in `data/clean/` | Absolute rupees; `price` and `market_price` per 100 face | |
| Raw file identification | By role keyword in the filename, not by position in the directory | A real download always takes priority over a `*_SAMPLE` file of the same role |
| Column identification | Alias table, never column position | An unmatched required column raises; nothing is silently dropped or renamed |
| Bad input handling | Every validation **raises**; none warn | Duplicate ISINs, matured bonds, rates already divided by 100, gaps in the published grid |
| Published grid check | Step and endpoints asserted, **row count is not** | 0.25→40.00 in 0.25 steps is 160 points; the contract's "159 rows" does not match its own step and endpoints |
| T-bill tenors | Nearest published bill's rate, carried onto the nominal tenor (7/365, 0.5, 1.0) | FBIL quotes 91D/182D/364D on some dates. Worst case this mis-times the 12M point by one day |
| `bonds.csv` `volume` | Nullable, but the column always exists | `selection.py` has a documented NaN fallback; a missing *column* would raise |
| Bonus SDL selection | Longest-dated SDL with residual maturity in (1, 14] years from settlement | The published SDL ZCYC stops at 14y, so a longer parent has no curve to discount off. Longest ⇒ most Coupon STRIPS to show |
| `face_stripped` | ₹5,00,00,000 (5 crore), validated as a whole multiple of ₹1 crore | |
| SDL book-value scenarios | Scenario 1 = market value × 1.03, scenario 2 = market value × 0.97 | **Not market data.** Nothing publishes a book value; these are set to straddle market value so `min(book, market)` can be shown binding each way |
| `compare()` units | No unit conversion — both grids are already percent | The only `×100` converts percentage points to basis points |
| `compare()` grid join | Inner join on `tenor_years` rounded to 2dp, asserted 1:1 | Rounding is a tolerance on the join key, not a nearest-match join; a genuine mismatch raises |
| `diff_bps` sign | Ours **minus** FBIL; positive means our curve is above FBIL's | |
| **`max_diff_bps`** | **Maximum ABSOLUTE difference** | The contract left the sign open. Our curve crosses FBIL's, so a signed maximum would report the smaller of the two errors as the headline |
| Empty comparison segment | `NaN`, not an error | A curve that stops at 10y has nothing to report above 14y — a fact about the data, not a failure |
| Ablation failures | Row kept with `rmse_bps = NaN`, sorted last | `build_zcyc` raises by design when its auto-checks fail; dropping the row would break the contract's 16-row table |
| Ablation status | Reported as **sensitivity**, never as model selection | The headline curve keeps the defaults locked above, chosen before any comparison was run. A better-scoring config is disclosed on the slide; it does not become the headline |

**Provenance — the one thing to check before submitting**

`data/clean/SOURCE.md` is written by the loader on every run and records which
raw file produced each clean CSV. It must read `Provenance: **FBIL**` in the
submitted work. While it reads `SAMPLE`, the numbers come from generated
stand-in files, the comparison charts carry a "SAMPLE DATA — NOT FBIL PUBLISHED
DATA" watermark, and nothing produced from them may be quoted as a result.

**Known simplifications vs the official methodology (A's stages)**

1. We feed FBIL's **published per-ISIN YTMs** into the model. FBIL builds its
   own curve from trade-level VWAY under Level 1/2/3 input rules that need the
   trade tape. The published YTM of a non-traded ISIN is itself *model YTM +
   adjustment factor*, so our inputs are **partly circular** with the curve we
   are comparing against. This is the headline caveat of the whole comparison.
2. Node selection is our own 90-day greedy scan plus long-end buckets, not
   FBIL's weekly grouping by volume × number of trades.
3. T-bill points use the nominal 7/365, 0.5 and 1.0 tenors even when the rate
   came from a 91D/182D/364D bill.
4. The two SDL book values are constructed to straddle market value (see above),
   not observed.
