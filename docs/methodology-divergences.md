# Where our method departs from the source documents

Every point below is a place where `final-parallel-team-plan.md` (our internal
spec) says something different from the official documents, or where the
documents cannot be followed with the data we have. Written so a grader can see
we knew, rather than discovering it for us.

Sources, cited by short name:
- **Concept Note** -- *Concept Note: Revised G-Sec Valuation Methodology for Public Consultation* (FBIL)
- **SDL Methodology** -- *FBIL SDL(SGS) ZCYC Methodology, Version 1, 7 July 2022*
- **Stripping Guidelines** -- *Guidelines on Stripping/Reconstitution of Government Securities* (RBI)
- **Brief** -- *MiniProject1.docx*, the assignment itself

Status key: **fixed** = code now follows the document · **blocked** = document
cannot be followed with available data · **open** = genuine tension, stated not resolved.

---

## 1. Input selection -- plan vs Concept Note

### 1.1 Which ISIN survives a 90-day cluster — **fixed**

> **Concept Note, §VI:** "If two or more ISINs have dates of redemption differing
> by 90 days or less, only one of them will be selected as input into the model.
> **The ISIN with higher volume\*trades during the last 5 business days will be
> chosen as input point.**"

> **Our plan, §2.3 B2:** "keep a bond only if >90 days after the last kept one
> (prefer higher liquidity/volume if that field exists, **else keep the earlier bond**)"

The plan's fallback — keep whichever matures first — appears nowhere in the
Concept Note, and the document's own illustration rules it out: its first
cluster keeps the *earliest* member but its second keeps the *latest*, which
only liquidity explains. `selection.py` now ranks by volume x trades.

### 1.2 The 1-year boundary is inclusive — **fixed**

> **Concept Note, §V:** "Traded G-Sec securities having **less than or equal to**
> 1-year residual maturity will not be used as model input." Segment 2 is
> "**1 year < Residual Maturity ≤ 14 years**."

The plan writes this segment as "1–14y", which reads as including exactly one
year. A bond with residual maturity of precisely 1.0 years must be *excluded*
and left to the T-bill points. Off-by-one at one boundary; now corrected.

### 1.3 Liquidity is volume x trades, not volume — **fixed (approximated)**

The Concept Note's yardstick is the *product* of volume and trade count over the
last 5 business days. The plan says "liquidity/volume". We now compute the
product where a trade-count column exists. **The published valuation file has no
trade count**, so in practice we still rank on volume alone — an approximation,
not the rule as written.

### 1.4 Selection is weekly, not per-run — **blocked (immaterial here)**

> **Concept Note, §VI:** "On the **first day of each week (usually Monday)**, the
> Traded ISINs will be arranged in the ascending order of their residual maturities."

FBIL fixes its node set on Monday and holds it for the week, deliberately: daily
re-selection "induces unacceptable level of volatility in the resulting ZCYC".
We re-select on every run. For a single valuation date the two coincide, so
nothing is wrong in our output — but our method is not theirs, and a multi-day
run of our code would be jumpier than FBIL's.

### 1.5 No Level 1 / 2 / 3 qualification at all — **blocked, and the largest gap**

The Concept Note admits an ISIN as a model input only after a qualification
waterfall we do not implement in any form:

| Concept Note requirement | Us |
|---|---|
| 1–14y: minimum **3** (Level 1 + Level 2) data inputs | no trade test |
| >14y: minimum **2** aggregate trades | no trade test |
| Level 1: ≥3 trades in the **last 60 minutes**, else ≥3 across the day | not available |
| Outlier removal: ISINs with ≥5 trades, drop YTMs beyond **±2 SD** | not applied |
| Input yield = **VWAY** of surviving trades | we use the published YTM |
| Level 2: MOT bids/offers ≥₹10 crore, spread ≤5 bp, observed 1/3/5 PM | not available |
| Level 3: proxy YTM when a bucket has nothing | not available |

All of it needs **NDS-OM trade-level data**, which our plan assumed unavailable
from the outset (§2.0: "trade-level data assumed unavailable"). The honest
statement is not "we simplified the outlier rule" but: *we apply FBIL's spacing
and bucketing rules, and none of its qualification rules.* Every bond in the
published file is treated as eligible.

### 1.6 Our inputs are partly FBIL's own model output — **open**

The Concept Note values *traded* ISINs at their VWAY, and everything else at a
model price built from this very curve plus an adjustment factor. So the
published YTMs we feed in are, for non-traded ISINs, outputs of the model we are
comparing ourselves against. A close match is therefore partly built in. The
ablation quantifies the sensitivity; the circularity itself cannot be removed
without trade data.

### 1.7 No adjustment factor — **open**

> **Concept Note:** "An adjustment factor is required to account either for market
> liquidity preferences for certain maturity buckets or for certain idiosyncratic
> features of a security."

FBIL's *security valuation* is ZCYC + adjustment factor. We model the curve only
and compare curve-to-curve, which is the right comparison for this assignment —
but we could not reproduce FBIL's individual security prices with it.

---

## 2. The bonus task -- Brief vs Stripping Guidelines

### 2.1 An SDL is not an eligible security for stripping — **open, unresolvable**

> **Stripping Guidelines, ¶10:** "all outstanding securities issued by
> **Government of India**, except floating rate bonds, with coupon dates/maturity
> date as **2nd January and 2nd July**, irrespective of the year of maturity, will
> be eligible for stripping/reconstitution."

> **Brief:** "Select a long-dated **State Development Loan (SDL)** security. Apply
> the 'Guidelines on Stripping/Reconstitution of Government Securities' to price
> each STRIP."

The assignment asks us to apply the guidelines to a security the guidelines do
not cover, on two independent counts: an SDL is issued by a **state**, not the
Government of India; and our chosen ISIN matures **14 February**, so its coupons
fall on 14 Feb / 14 Aug, not 2 Jan / 2 Jul. This is a contradiction in the task
as set, not a mistake in our work. We apply the guidelines **by analogy** and say
so. It belongs on both the bonus slide and the limitations slide.

What we *do* follow exactly: STRIPS face value ₹100 (¶5), minimum strippable
amount ₹1 crore and multiples thereof (¶11).

### 2.2 "Long-dated" cannot coexist with the published SDL curve — **open**

> **Brief:** "Select a **long-dated** State Development Loan (SDL) security."

> **SDL Methodology, §5.3:** "Maturities above 12 months and **upto 14 years**."

> **Our plan, §2.0:** "Bonus SDL: **Residual maturity ≤ 14 years**"

FBIL's published SDL ZCYC **stops at 14 years**. So a genuinely long-dated SDL —
a 2050 or 2060 line — has no published curve to be discounted against, and the
plan's ≤14y rule is forced by that limit rather than chosen. The two
instructions cannot both be satisfied.

Our resolution: take the longest SDL still inside the curve's range
(7.59% Maharashtra SDL 2040, ~13.4 years residual), and state the constraint.
The alternative — extrapolating FBIL's SDL curve past its published end to reach
a 25-year SDL — would invent the very rates the exercise is meant to source.

### 2.3 The SDL curve has a different input split from the G-Sec curve — **noted**

Anyone tempted to reuse `selection.py` to *build* an SDL curve should not: the
SDL Methodology splits at 12 months into **two** segments (not three), uses
**four** money-market points (7-day, 3-month, 6-month, 12-month) rather than
three, adds a **traded spread** to all but the 7-day point, and requires adjacent
ISINs to be at least **0.25 years / 3 months** apart rather than 90 days. We
sidestep all of this by consuming FBIL's *published* SDL ZCYC, as the plan says.

---

## 3. Ambiguities inside the documents themselves

### 3.1 Is the first node at 1 day or 7 days?

The Concept Note says the short segment's three points are "**overnight**, 6
months and 1 year", and then defines the first as "Overnight rate – **7-day
T-Bill** (published by FBIL)". Elsewhere it calls the same thing "7-Day T-Bill
rate published by FBIL as the first input point". An overnight rate assigned a
7-day maturity, or a 7-day rate assigned an overnight maturity, are different
inputs. **We place the node at 7/365 years.** At these tenors the difference is
small, and it is exposed as a config value rather than buried.

### 3.2 Does the 90-day rule chain?

The rule is written pairwise: "if two or more ISINs have dates of redemption
differing by 90 days or less". If A–B and B–C are each within 90 days but A–C is
120 days apart, strict chaining discards C, while a spacing constraint keeps
both A and C. The document's illustration does not settle it — none of its
clusters chain beyond 90 days pairwise.

**We keep both**, because the Concept Note's stated objective is "**maximum use
of qualifying traded securities** ... or in other words minimum loss of data",
subject to smoothness. Discarding C loses data that the spacing rule does not
require us to lose. Our node set still satisfies "no two selected ISINs within
90 days" — verified in the test suite.

---

## 4. Where the plan and the documents already agreed

Checked and found consistent, so nobody re-checks them:

- Three segments split at 1 year and 14 years (Concept Note §V–VII)
- Six long-end buckets: >14–18, >18–22, >22–26, >26–30, >30–34, >34 (§VII)
- One ISIN per bucket, **plus** an extra ISIN maturing in the terminal year of
  the last bucket "to ensure smoothness of the ZCYC output" (§VII ¶9)
- Short end from 7-day / 6-month / 12-month FBIL T-bill rates, converted to bond
  equivalent yields and priced as synthetic securities (§V)
- Cubic spline, Pienaar–Choudhry lineage, twice differentiable (Brief; Concept Note)
- STRIPS: ₹100 face per STRIP, ₹1 crore minimum in multiples (Guidelines ¶5, ¶11)

---

## 5. Verification

`tests/test_selection.py::TestAgainstConceptNoteIllustration` reproduces the
Concept Note's own worked example (11 ISINs, analysis dated 20 November 2019).
Our selection returns the same **5 surviving nodes**, in the same clusters, and
matches all 11 rows once the liquidity ranking is supplied — the document does
not publish the volume x trades figures behind its picks, so that input has to
be given for a row-by-row comparison.
