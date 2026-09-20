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
