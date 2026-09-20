"""Our bootstrapped ZCYC against FBIL's published one.

``compare()`` is the frozen interface: two grids in, a per-tenor difference
table and a summary dict out. Everything else in this module is presentation --
the two charts, the 12-tenor table and the summary json that Person 4 puts on
the comparison slide.

Units: both inputs are already in PERCENT, so there is no unit conversion
anywhere in this file. The single multiplication by 100 converts a difference in
percentage points into basis points, which is a different thing entirely.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS_COLUMNS = ["tenor_years", "our_zcy_semi", "fbil_zcy_semi", "diff_bps", "segment"]
SUMMARY_KEYS = ("overall", "<=1y", "1-14y", ">14y")
SEGMENT_BOUNDS = (1.0, 14.0)

# The tenors that go on the slide. The full grid is 160 rows; nobody reads 160
# rows off a slide, and these are the points the curve is quoted at anyway.
COMPARISON_TENORS = (0.25, 0.5, 1, 2, 3, 5, 7, 10, 14, 20, 30, 40)

# Both grids sit on the same 0.25 ladder, but 0.25 * 3 is not exactly 0.75 in
# binary, so the join key is rounded first. This is a tolerance on the key, not
# a nearest-match join -- an actual mismatch in the grids still raises.
JOIN_DECIMALS = 2

OUTPUT_FILES = (
    "comparison_overlay.png",
    "comparison_diff_bps.png",
    "comparison_table.csv",
    "comparison_summary.json",
)


def _segment(t: float) -> str:
    """Which reporting segment a tenor falls in.

    The boundaries are not arbitrary: below 1y the curve is pinned by T-bills
    rather than bonds, and above 14y FBIL's own inputs thin out to a handful of
    trades. Splitting the error this way separates "our short end is built
    differently" from "nobody traded a 37-year bond that day".
    """
    short, long = SEGMENT_BOUNDS
    if t <= short:
        return "<=1y"
    if t <= long:
        return "1-14y"
    return ">14y"


def _normalise_ours(our_grid_df: pd.DataFrame) -> pd.DataFrame:
    """Rename our grid's zero-rate columns onto FBIL's names, on a copy.

    The interface contract documents ``Curve.grid()`` as emitting ``zero_semi``
    / ``zero_annual``; the curve engine actually emits ``zcy_semi`` /
    ``zcy_annual`` so its output already matches the published file. Both
    spellings are accepted here rather than making the caller care, and the
    caller's frame is never mutated.
    """
    out = our_grid_df.copy()
    out = out.rename(columns={"zero_semi": "zcy_semi", "zero_annual": "zcy_annual"})
    missing = [c for c in ("tenor_years", "zcy_semi") if c not in out.columns]
    if missing:
        raise ValueError(
            f"our_grid_df is missing {missing}; expected the Curve.grid() schema "
            "[tenor_years, zero_semi|zcy_semi, zero_annual|zcy_annual, par_semi, par_annual]"
        )
    return out


def compare(our_grid_df: pd.DataFrame, fbil_grid_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Compare our zero curve against FBIL's published one.

    Both frames are in percent and on the same 0.25-year grid.

    Returns ``(metrics_df, summary)``:

    * ``metrics_df`` -- one row per grid tenor, columns
      ``[tenor_years, our_zcy_semi, fbil_zcy_semi, diff_bps, segment]``, where
      ``diff_bps = (ours - FBIL) * 100``. Positive means our curve is above
      FBIL's at that tenor.
    * ``summary`` -- ``rmse_bps``, ``mae_bps`` and ``max_diff_bps`` for the
      whole curve and for each of the three segments. ``max_diff_bps`` is the
      maximum **absolute** difference: the contract leaves the sign convention
      open, and a signed maximum on a curve that crosses FBIL's would report
      the smaller of the two errors as the headline.
    """
    ours = _normalise_ours(our_grid_df)
    fbil = fbil_grid_df.copy()
    if "zcy_semi" not in fbil.columns:
        raise ValueError(f"fbil_grid_df is missing 'zcy_semi'; got {list(fbil.columns)}")

    ours["tenor_years"] = ours["tenor_years"].round(JOIN_DECIMALS)
    fbil["tenor_years"] = fbil["tenor_years"].round(JOIN_DECIMALS)

    merged = ours.merge(fbil, on="tenor_years", how="inner", suffixes=("_our", "_fbil"))
    if not len(merged) == len(ours) == len(fbil):
        raise ValueError(
            f"grids do not align 1:1: ours has {len(ours)} rows, FBIL's {len(fbil)}, "
            f"the join {len(merged)}. Both are meant to be the same 0.25-year ladder, so a "
            "mismatch is a bug in whichever grid is wrong -- not something to paper over with "
            "a nearest-match join."
        )

    metrics_df = pd.DataFrame(
        {
            "tenor_years": merged["tenor_years"],
            "our_zcy_semi": merged["zcy_semi_our"],
            "fbil_zcy_semi": merged["zcy_semi_fbil"],
        }
    )
    # Percentage points -> basis points. The one legitimate *100 in this file.
    metrics_df["diff_bps"] = (metrics_df["our_zcy_semi"] - metrics_df["fbil_zcy_semi"]) * 100.0
    metrics_df["segment"] = [_segment(float(t)) for t in metrics_df["tenor_years"]]
    metrics_df = metrics_df[METRICS_COLUMNS]

    summary = {"overall": _metrics(metrics_df["diff_bps"])}
    for key in SUMMARY_KEYS[1:]:
        summary[key] = _metrics(metrics_df.loc[metrics_df["segment"] == key, "diff_bps"])
    return metrics_df, summary


def _metrics(diff_bps: pd.Series) -> dict:
    """RMSE, MAE and max absolute difference, all in bps.

    An empty segment yields NaN rather than raising: a curve that stops at 30y
    has nothing above 14y to report, and that is a fact about the data, not a
    failure of the comparison.
    """
    values = np.asarray(diff_bps, dtype=float)
    if values.size == 0:
        return {"rmse_bps": float("nan"), "mae_bps": float("nan"), "max_diff_bps": float("nan")}
    return {
        "rmse_bps": float(np.sqrt(np.mean(values**2))),
        "mae_bps": float(np.mean(np.abs(values))),
        "max_diff_bps": float(np.max(np.abs(values))),
    }


# ------------------------------------------------------------------- reporting

def comparison_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """The 12-tenor subset of ``metrics_df`` that goes on the slide."""
    wanted = [round(float(t), JOIN_DECIMALS) for t in COMPARISON_TENORS]
    subset = metrics_df[metrics_df["tenor_years"].round(JOIN_DECIMALS).isin(wanted)]
    return subset.reset_index(drop=True)


def _watermark(fig, provenance: str) -> None:
    """Stamp a chart built from anything other than a real FBIL download.

    A chart is the one artefact that travels on its own -- it ends up on a
    slide with no path, no README and nobody to ask. If the numbers behind it
    are not market data, the chart has to say so itself.
    """
    if provenance == "FBIL":
        return
    fig.text(
        0.5,
        0.5,
        f"{provenance} DATA\nNOT FBIL PUBLISHED DATA",
        fontsize=30,
        color="crimson",
        alpha=0.18,
        ha="center",
        va="center",
        rotation=25,
        fontweight="bold",
        zorder=10,
    )


def plot_overlay(our_grid_df: pd.DataFrame, fbil_grid_df: pd.DataFrame, path, provenance: str = "FBIL"):
    """Ours vs FBIL, semiannual and annualised, one panel each."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ours, fbil = _normalise_ours(our_grid_df), fbil_grid_df
    pairs = [("zcy_semi", "zero-coupon yield, % (semiannual)")]
    if "zcy_annual" in ours.columns and "zcy_annual" in fbil.columns:
        pairs.append(("zcy_annual", "zero-coupon yield, % (annualised)"))

    fig, axes = plt.subplots(1, len(pairs), figsize=(6.2 * len(pairs), 4.6), squeeze=False)
    for ax, (col, ylabel) in zip(axes[0], pairs):
        ax.plot(ours["tenor_years"], ours[col], label="Ours (cubic spline bootstrap)", linewidth=1.8)
        ax.plot(fbil["tenor_years"], fbil[col], label="FBIL published", linewidth=1.8, linestyle="--")
        ax.set_xlabel("tenor, years")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
    fig.suptitle("G-Sec zero-coupon yield curve: ours vs FBIL published")
    _watermark(fig, provenance)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_diff_bps(metrics_df: pd.DataFrame, path, provenance: str = "FBIL"):
    """Difference in bps against tenor, with the segment boundaries marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(metrics_df["tenor_years"], metrics_df["diff_bps"], linewidth=1.8, color="#1f77b4")
    ax.axhline(0.0, color="black", linewidth=1.0)
    for bound in SEGMENT_BOUNDS:
        ax.axvline(bound, color="grey", linestyle=":", linewidth=1.2)

    # Segment labels sit just inside the top of the axes, in axis-fraction
    # coordinates, so they follow the y-limits instead of colliding with the title.
    for label, x in zip(SUMMARY_KEYS[1:], (0.5, 7.5, 27.0)):
        ax.text(
            x, 0.97, label,
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=9, color="grey",
        )

    ax.set_xlabel("tenor, years")
    ax.set_ylabel("difference, bps (ours minus FBIL)")
    ax.set_title("Difference from FBIL's published zero curve")
    ax.grid(alpha=0.3)
    _watermark(fig, provenance)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_comparison_outputs(
    our_grid_df: pd.DataFrame,
    fbil_grid_df: pd.DataFrame,
    outputs_dir: str | Path = "outputs",
    provenance: str = "FBIL",
) -> tuple[pd.DataFrame, dict]:
    """Run ``compare()`` and write all four frozen-name output files."""
    outputs = Path(outputs_dir)
    outputs.mkdir(parents=True, exist_ok=True)

    metrics_df, summary = compare(our_grid_df, fbil_grid_df)
    plot_overlay(our_grid_df, fbil_grid_df, outputs / "comparison_overlay.png", provenance)
    plot_diff_bps(metrics_df, outputs / "comparison_diff_bps.png", provenance)
    comparison_table(metrics_df).to_csv(outputs / "comparison_table.csv", index=False)
    with open(outputs / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return metrics_df, summary


def format_summary(summary: dict) -> str:
    """The summary as a table, for the terminal and for the slide notes."""
    header = f"{'segment':<10}{'rmse_bps':>12}{'mae_bps':>12}{'max_abs_bps':>14}"
    lines = [header, "-" * len(header)]
    for key in SUMMARY_KEYS:
        m = summary[key]
        lines.append(f"{key:<10}{m['rmse_bps']:>12.2f}{m['mae_bps']:>12.2f}{m['max_diff_bps']:>14.2f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Build the headline curve and write the four comparison outputs.

    run_all.py writes comparison_table.csv and comparison_summary.json as part
    of the pipeline but does not produce the two charts, so this stays runnable
    on its own.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Compare our ZCYC against FBIL's published curve.")
    parser.add_argument("--date", required=True, help="Valuation date, YYYY-MM-DD")
    parser.add_argument("--clean-dir", default="data/clean")
    parser.add_argument("--outputs-dir", default="outputs")
    args = parser.parse_args(argv)

    from src.bond import settlement_date
    from src.curve import CurveConfig, build_zcyc
    from src.data_loader import clean_data_provenance

    clean = Path(args.clean_dir)
    bonds = pd.read_csv(clean / "bonds.csv")
    tbills = pd.read_csv(clean / "tbills.csv")
    fbil = pd.read_csv(clean / "fbil_zcyc.csv")

    settle = settlement_date(args.date)
    curve = build_zcyc(bonds, tbills, settle, CurveConfig())
    curve.assert_checks()

    provenance = clean_data_provenance(clean)
    _, summary = write_comparison_outputs(curve.grid(), fbil, args.outputs_dir, provenance)

    print(f"valuation date {args.date} (settle {settle}), data provenance: {provenance}\n")
    print(format_summary(summary))
    print("\nwrote " + ", ".join(f"{args.outputs_dir}/{f}" for f in OUTPUT_FILES))
    if provenance != "FBIL":
        print(f"\nWARNING: built from {provenance} data -- the charts are watermarked and no number here is a result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
