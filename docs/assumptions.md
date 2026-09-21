# Locked Assumptions

Fill in as each decision is confirmed at kickoff. This file is a submission deliverable — keep it accurate, it's what a grader reads to understand our modeling choices.

- [x] Valuation date (D) — 2026-09-11, see A's section
- [x] Input data source (published YTMs/prices, trade-level assumed unavailable) — A's section
- [x] Settlement convention — T+1, rolled over weekends; B's section
- [x] Accrued interest convention — 30/360 bond basis; B's section
- [x] Compounding convention — semiannual; B's section
- [x] Time-to-cashflow convention (headline vs ablation alternative) — act365 headline, halfyear ablation; B's section
- [x] Spline variable + boundary condition (headline vs ablation alternatives) — zero/natural headline; B's section
- [x] Output grid range/step — 0.25 to 50 years, step 0.25; B's section
- [x] Bonus discounting curve (SDL ZCYC) and how it's applied — published `fbil_sdl_zcyc.csv`, cubic-spline; C's section
- [x] Bonus SDL selection criteria — longest-dated SDL with residual maturity in (1, 14]; A's section
- [x] Parent value convention used in STRIP normalisation — `min(book_value, market_value)`; A's and C's sections
- [x] Known simplifications vs the official FBIL/RBI methodology (state each one explicitly) — listed at the end of each of the three sections below

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
| Output grid | 0.25 to 50 years, step 0.25 (200 points), semiannual + annualised + par | Matches the grid FBIL actually publishes — see "Grid range" under A's section |

**Auto-checks enforced on every run** (the pipeline raises and stops if any fail):
max repricing error < 1e-6 per 100 face; second derivative continuous at every
knot; discount factors strictly decreasing across the whole output grid.

**Known simplifications vs the official methodology**

1. FBIL's Pienaar–Choudhry formulation is not public, so this is a cubic-spline
   **approximation of** that approach, not a reimplementation of it.
2. No public-holiday calendar; T+1 settlement rolls over weekends only.
3. Flat extrapolation beyond the longest input bond — the far end is asserted
   rather than fitted. On the 2026-09-11 file the longest G-Sec matures in
   2076, so the 50y grid point is inside the fitted range, not extrapolated.
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
| Valuation date D | **2026-09-11** (Friday) | Settlement 2026-09-14, T+1 per B's convention. Newest day on which all five FBIL publications were in the free public archive when the data was pulled |
| Rate units in `data/clean/` | Percent throughout; `6.05` means 6.05% | The loader never divides a rate by 100 — conversion happens at point of use |
| Dates in `data/clean/` | ISO `YYYY-MM-DD` strings | Raw files are parsed `dayfirst=True` (FBIL quotes 14/05/2031) |
| Money in `data/clean/` | Absolute rupees; `price` and `market_price` per 100 face | |
| Raw file identification | By role keyword in the filename, not by position in the directory | A real download always takes priority over a `*_SAMPLE` file of the same role |
| Column identification | Alias table, never column position | An unmatched required column raises; nothing is silently dropped or renamed |
| Bad input handling | Every validation **raises**; none warn | Duplicate ISINs, matured bonds, rates already divided by 100, gaps in the published grid |
| Published grid check | Step and endpoints asserted, **row count is not** | Both are read off FBIL's files into `ZCYC_GRID` / `SDL_ZCYC_GRID`; the contract's "159 rows to 40y" matched neither its own arithmetic nor the publication |
| T-bill tenors | Nearest published bill's rate, carried onto the nominal tenor (7/365, 0.5, 1.0) | FBIL quotes 91D/182D/364D on some dates. Worst case this mis-times the 12M point by one day |
| `bonds.csv` `volume` | Nullable, but the column always exists | `selection.py` has a documented NaN fallback; a missing *column* would raise |
| Bonus SDL selection | Longest-dated SDL with residual maturity in (1, 14] years from settlement | The published SDL ZCYC stops at 14y, so a longer parent has no curve to discount off. Longest ⇒ most Coupon STRIPS to show |
| `face_stripped` | ₹5,00,00,000 (5 crore), validated as a whole multiple of ₹1 crore | |
| SDL book-value scenarios | Scenario 1 = **face value** (real, published). Scenario 2 = 0.97 × market value (`SCENARIO_2_BOOK_FRACTION`) | Book value is the holder's own carrying value — portfolio-, price- and depreciation-specific — and nobody publishes it. Scenario 1 uses face, which is real; this SDL is sub-par so `min()` picks market value and **scenario 1's STRIP prices are FBIL data end to end**. Scenario 2 is the **only assumed number in the pipeline**, needed because `min()`'s book-bound branch is otherwise unreachable. The loader asserts face > market value |
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
4. **Exactly one number in this project is assumed rather than published:**
   `book_value_scenario_2`. Book value is an accounting carrying value
   specific to the holder and is not published by anyone. Scenario 1 avoids
   the problem by using face value, so the scenario-1 STRIP table — the one to
   quote — contains no assumed input. Scenario 2 exists only to show the
   book-bound branch of RBI's `min(book, market)` rule and must be presented
   as an illustration of the rule, never as a result.

---

## STRIPS and bonus SDL (C) — locked

Appended by C. Covers `strips.py` only; A's and B's sections above stand as written.

| Decision | Choice | Note |
|---|---|---|
| STRIP construction | One Coupon STRIP per remaining coupon date, plus one Principal STRIP | Per RBI's 2010 Stripping Guidelines para 15.2 |
| Normalisation | `factor = min(book_value, market_value) / sum(PV of all STRIPS)`; every STRIP's PV scaled by that one factor | Per RBI para 15.2 |
| Time convention | `act365` for all real pricing (bonus SDL); `"half_year"` (integer semiannual periods, with underscore) only to reproduce the Annex 3/4 textbook fixtures | `curve.py`'s own ablation convention is spelled `"halfyear"` (no underscore) — a different literal for a different function, not a typo |
| Bonus discounting curve | `fbil_sdl_zcyc.csv` (FBIL's own published SDL ZCYC), cubic-spline interpolated on `zcy_semi`, flat beyond the published 0.25–14y grid | The *published* SDL curve, not the headline G-Sec curve — permitted by Stripping Guidelines para 13 ("if traded zero-coupon rates are not available, use FIMMDA/FBIL's published yields instead") |
| Coupon dates for the SDL | The SDL's own actual semiannual coupon dates | RBI's guideline is written for GoI securities on the 2-Jan/2-Jul cycle; applying it to a state-issued SDL at all is this team's own extrapolation — the guidelines don't explicitly cover SDLs |
| `strip_and_price()`'s `coupon` input | **Percent** (e.g. `7.47`), not decimal | The interface contract originally labelled this decimal; its own worked examples (Annex 3/4) only reproduce to tolerance with percent input — corrected 2026-09-21 |
| SDL choice, book-value scenarios | Not C's decision — A's loader picks the bonus SDL and computes both `book_value_scenario_1`/`_2`; `strips.py` just applies `min(book_value, market_value)` to whatever the file provides | See A's section above for how the two scenarios are derived |

**Known simplifications vs the official guidelines**

1. RBI's Stripping Guidelines (2010) are written for Government of India dated
   securities on the 2 Jan/2 Jul coupon cycle. Applying them to a State
   Development Loan at all is this team's own extrapolation — the guidelines
   don't explicitly cover SDLs. The SDL's actual coupon dates are used rather
   than forcing the GoI cycle.
2. HTM/AFS/HFT bank-accounting treatment (Stripping Guidelines paras 15.3–15.4)
   is out of scope — a banking-book nuance, irrelevant to valuation-only work.
3. No STRIPS ISIN/nomenclature generation and no reconstitution demo — the
   guidelines cover both, neither is needed to price a single stripping
   exercise.

---

## Data acquisition — where the numbers actually come from

Pulled 2026-09-21 by `src/fetch_fbil.py` from FBIL's public archive at
`https://www.fbil.org.in/wasdm`, the same API the public website uses. The free
public tier lags the live benchmark by about a week; **2026-09-11** was the
newest business day on which all five publications we need were available.
Nothing here is behind a subscription and nothing was scraped from a logged-in
session.

| Clean CSV | FBIL publication | Workbook / sheet |
|---|---|---|
| `bonds.csv` | FBIL GOI Prices | `gsec_11092026.xlsx`, sheet `G-Sec` |
| `fbil_zcyc.csv` (zero half) | FBIL GOI STRIPS and ZCYC | `strips_11092026.xlsx`, sheet `ZCYC` |
| `fbil_zcyc.csv` (par half) | FBIL GOI Prices | `gsec_11092026.xlsx`, sheet `Par Yield` |
| `fbil_sdl_zcyc.csv` | FBIL SDL ZCYC | `sdlzcyc_11092026.xlsx`, sheet `SDL_ZCYC` |
| `sdl_bond.csv` | FBIL SDL/SGS Prices | `sdl_11092026.xlsx`, sheet `SDL` |
| `tbills.csv` | FBIL T-bill rates | `/wasdm/tbill/fetchfiltered` JSON (`/tbill/download` returns HTTP 500 on the public tier) |

Every byte FBIL served is kept unmodified in `data/raw/fbil_source/` under
FBIL's own filenames. The role-named files in `data/raw/` are copies of those,
except for two that are reshaped and **never recalculated**: the G-Sec ZCYC csv
joins the two published halves of one curve on tenor, and the T-bill csv writes
the published JSON out as a table.

**What the real files changed, relative to what the interface contract assumed**

| Assumed | Actually published | Consequence |
|---|---|---|
| G-Sec ZCYC runs 0.25→40.00 | 0.25→**50.00**, 200 points | `ZCYC_GRID` and `Curve.grid_stop` both moved to 50.0 — `compare()` requires a 1:1 join, so our grid has to match FBIL's |
| Zero and par curves ship together | Published in **two different workbooks** | The fetcher joins them on tenor before the loader sees them |
| SDL file carries a par curve | FBIL publishes **no SDL par curve** | `par_semi`/`par_annual` exist but are **empty** in `fbil_sdl_zcyc.csv`. They are not filled with par yields implied by the published zeros: that would be our derivation printed in a file the rest of the pipeline reads as "what FBIL said". Nothing consumes them — `compare()` works off `zcy_semi` |
| Bond file carries volume / trade counts | Only ISIN, coupon, maturity, price, YTM, two remark columns | `bonds.csv.volume` is entirely NaN, so `selection.py` falls back to earliest-maturity for every long-end bucket. Our node set is therefore **not** liquidity-weighted the way FBIL's is |
| — | Remark column flags **5 FRBs** | Dropped in `parse_bonds` (`FLOATER_REMARKS`): a floating-rate bond's quoted YTM is not a fixed-coupon YTM and cannot price off a nominal zero curve. 124 published rows → **119 bonds** |
| — | Remark column flags **26 "Input Point"** ISINs | This is FBIL's own node set, published. We do not use it (our selection stays as locked above), but it is available in `data/raw/fbil_source/gsec_11092026.xlsx` if the team wants a node-selection comparison on the slide |

**Refreshing the data.** `python -m src.fetch_fbil` takes the newest common
date, `--date YYYY-MM-DD` takes a specific one, `--list` shows what is
available. Re-run `build_clean_data`, `src.compare` and `src.ablation` with the
same date afterwards.
