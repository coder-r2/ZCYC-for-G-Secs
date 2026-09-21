"""Choice of bootstrap input instruments.

The published G-Sec list carries ~100 ISINs, far more than the curve needs.
Splining through all of them would chase the noise in thinly-traded lines, so
we thin the list down to a well-spaced set of nodes, per the rules locked at
kickoff:

  * <= 1 year   -- covered by T-bills (7D / 6M / 12M), bonds in this segment dropped.
  * 1 - 14 years -- sort by maturity, greedily keep a bond only if it matures more
    than 90 days after the last one kept.
  * > 14 years  -- one ISIN per maturity bucket (14-18, 18-22, 22-26, 26-30,
    30-34, 34+), plus the longest-dated ISIN on the list so the far end of the
    curve is pinned by a real instrument rather than by extrapolation.

Where a rule leaves a choice between several bonds, the more liquid one wins if
the data carries a volume/liquidity column, otherwise the earlier maturity does.

Schemas (frozen at kickoff):
    bonds  [isin, desc, coupon, maturity, price, ytm]
    tbills [tenor, rate]
"""

from __future__ import annotations

import re

import pandas as pd

from src.bond import to_date

SHORT_END_YEARS = 1.0
LONG_END_YEARS = 14.0
MIN_GAP_DAYS = 90
LONG_BUCKETS = ((14, 18), (18, 22), (22, 26), (26, 30), (30, 34), (34, 1000))
TBILL_TARGETS = (7 / 365, 0.5, 1.0)
VOLUME_COLUMNS = ("volume", "traded_volume", "turnover", "liquidity")
TRADES_COLUMNS = ("trades", "no_of_trades", "nos_trades", "num_trades", "n_trades")


def residual_years(maturity, settle) -> float:
    """Residual maturity in years, actual days / 365 from settlement."""
    return (to_date(maturity) - to_date(settle)).days / 365.0


def parse_tenor(value) -> float:
    """T-bill tenor -> years. Accepts 0.5, '6M', '182D', '12 M', '364 days'."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip().lower()
    match = re.match(r"^([0-9]*\.?[0-9]+)\s*([a-z]*)$", text)
    if not match:
        raise ValueError(f"cannot parse T-bill tenor {value!r}")
    number, unit = float(match.group(1)), match.group(2)
    if unit.startswith("d"):
        return number / 365.0
    if unit.startswith("w"):
        return number * 7 / 365.0
    if unit.startswith("m"):
        return number / 12.0
    return number


def _first_column(df: pd.DataFrame, names) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def selection_metric(df: pd.DataFrame) -> pd.Series | None:
    """FBIL's liquidity yardstick: volume x number of trades.

    The Concept Note picks the surviving ISIN of a 90-day group by "higher
    volume*trades during the last 5 business days", and falls back to the same
    product on the previous day for the long-end buckets. The published
    valuation file carries a traded-volume column but no trade count, so when
    the count is absent we rank on volume alone -- a documented approximation,
    not the rule as written.
    """
    volume = _first_column(df, VOLUME_COLUMNS)
    trades = _first_column(df, TRADES_COLUMNS)
    if volume is None and trades is None:
        return None
    if volume is None:
        return df[trades].astype(float)
    if trades is None:
        return df[volume].astype(float)
    return df[volume].astype(float) * df[trades].astype(float)


def _pick_one(group: pd.DataFrame, metric: pd.Series | None) -> pd.Series:
    """Highest volume*trades in the group, falling back to earliest maturity."""
    if metric is not None:
        scores = metric.reindex(group.index)
        if scores.notna().any():
            ranked = group.assign(_score=scores).sort_values(
                ["_score", "residual_years"], ascending=[False, True]
            )
            return group.loc[ranked.index[0]]
    return group.sort_values("residual_years").iloc[0]


def select_tbills(tbills: pd.DataFrame, targets=TBILL_TARGETS) -> pd.DataFrame:
    """Nearest available T-bill to each target tenor, deduplicated.

    Nearest-match rather than exact-match because FBIL quotes 91D/182D/364D
    bills on some dates and 7D/6M/12M on others; taking the closest point keeps
    the short end anchored either way.
    """
    if tbills is None or len(tbills) == 0:
        return pd.DataFrame(columns=["tenor", "rate", "t"])

    out = tbills.copy()
    out["t"] = [parse_tenor(v) for v in out["tenor"]]

    chosen = []
    for target in targets:
        idx = (out["t"] - target).abs().idxmin()
        if idx not in chosen:
            chosen.append(idx)
    return out.loc[chosen].sort_values("t").reset_index(drop=True)


def select_bonds(
    bonds: pd.DataFrame,
    settle,
    thin_90d: bool = True,
    min_gap_days: int = MIN_GAP_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Thin the bond list to curve nodes.

    Returns ``(selected, report)``. The report keeps one row per input bond with
    the segment it fell in and why it was kept or dropped -- it goes straight
    onto a slide and makes the thinning auditable rather than a black box.
    """
    work = bonds.copy()
    work["residual_years"] = [residual_years(m, settle) for m in work["maturity"]]
    work["maturity_date"] = [to_date(m) for m in work["maturity"]]
    work = work.sort_values("residual_years").reset_index(drop=True)

    metric = selection_metric(work)
    keep, reason, segment = {}, {}, {}

    for i, row in work.iterrows():
        rm = row["residual_years"]
        if rm <= 0:
            segment[i], keep[i], reason[i] = "matured", False, "matured on or before settlement"
        elif rm <= SHORT_END_YEARS:
            # "Traded G-Sec securities having less than or equal to 1-year
            # residual maturity will not be used as model input" -- the second
            # segment starts strictly above 1 year.
            segment[i], keep[i], reason[i] = "short", False, "1y or less, covered by T-bills"
        elif rm <= LONG_END_YEARS:
            segment[i] = "mid"
        else:
            segment[i] = "long"

    # 1-14y: "If two or more ISINs have dates of redemption differing by 90
    # days or less, only one of them will be selected as input into the model.
    # The ISIN with higher volume*trades during the last 5 business days will be
    # chosen." The Concept Note's two stated objectives are a minimum spacing
    # between nodes AND "maximum use of qualifying traded securities ... minimum
    # loss of data", so we take the most liquid ISIN of each 90-day window as a
    # node, then resume scanning from the first bond more than 90 days after it.
    # That keeps every node >90 days apart without discarding a bond merely for
    # being chained to one through an intermediate ISIN.
    mid = work[[segment[i] == "mid" for i in work.index]].sort_values("maturity_date")
    if not thin_90d:
        for i in mid.index:
            keep[i], reason[i] = True, "mid segment, 90-day grouping disabled"
    else:
        remaining = list(mid.index)
        while remaining:
            anchor = remaining[0]
            anchor_date = mid.loc[anchor, "maturity_date"]
            window = [
                i for i in remaining
                if (mid.loc[i, "maturity_date"] - anchor_date).days <= min_gap_days
            ]
            winner = _pick_one(mid.loc[window], metric).name if len(window) > 1 else anchor
            keep[winner] = True
            reason[winner] = (
                "mid segment, no other ISIN within 90 days"
                if len(window) == 1
                else f"mid segment, most liquid of {len(window)} ISINs within {min_gap_days}d"
            )
            for i in window:
                if i != winner:
                    keep[i] = False
                    reason[i] = f"mid segment, within {min_gap_days}d of a more liquid ISIN"
            winner_date = mid.loc[winner, "maturity_date"]
            remaining = [
                i for i in remaining
                if i not in window
                and (mid.loc[i, "maturity_date"] - winner_date).days > min_gap_days
            ]

    # >14y: one per bucket, plus the terminal ISIN.
    long = work[[segment[i] == "long" for i in work.index]]
    for i in long.index:
        keep[i], reason[i] = False, "long segment, bucket filled by another ISIN"
    for lo, hi in LONG_BUCKETS:
        bucket = long[(long["residual_years"] > lo) & (long["residual_years"] <= hi)]
        if len(bucket) == 0:
            continue
        pick = _pick_one(bucket, metric)
        keep[pick.name], reason[pick.name] = True, f"long segment, representative of {lo}-{hi}y bucket"
    if len(long) > 0:
        terminal = long.sort_values("residual_years").index[-1]
        keep[terminal], reason[terminal] = True, "long segment, terminal (longest-dated) ISIN"

    work["segment"] = [segment[i] for i in work.index]
    work["kept"] = [keep.get(i, False) for i in work.index]
    work["reason"] = [reason.get(i, "") for i in work.index]

    report_cols = [c for c in ("isin", "desc", "coupon", "maturity", "residual_years", "segment", "kept", "reason") if c in work.columns]
    selected = work[work["kept"]].drop(columns=["maturity_date", "kept", "reason"]).reset_index(drop=True)
    return selected, work[report_cols].reset_index(drop=True)


def select_inputs(
    bonds: pd.DataFrame,
    tbills: pd.DataFrame,
    settle,
    thin_90d: bool = True,
    min_gap_days: int = MIN_GAP_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Full input set for the bootstrap: ``(bonds_sel, tbills_sel, report)``."""
    bonds_sel, report = select_bonds(bonds, settle, thin_90d=thin_90d, min_gap_days=min_gap_days)
    tbills_sel = select_tbills(tbills)
    return bonds_sel, tbills_sel, report
