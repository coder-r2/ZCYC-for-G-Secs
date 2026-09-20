"""Single command chaining the whole pipeline: data -> curve -> comparison -> STRIPS.

    python run_all.py --date YYYY-MM-DD

Pipeline (SPEC.md Sec 7 / technical-schema.md Sec 0):

    data/clean/*.csv (A, produced ahead of time by data_loader.py)
        -> selection.py + curve.py (B)  -> Curve object
        -> compare.py (A)               -> our curve vs FBIL's published curve
        -> strips.py (C)                -> STRIP prices for the bonus SDL

data_loader.py has no frozen call signature (technical-schema.md only
freezes the *schemas* it must produce, Sec 2) -- treated here as a
precondition, not a step this script calls: this script starts by reading
the already-clean CSVs, same as every other stage reads its predecessor's
output.

As of this writing A hasn't pushed data_loader.py or compare.py yet, so the
comparison stage below raises a clear ModuleNotFoundError naming exactly
what's missing -- that's an accurate reflection of today's state, not a bug
to work around here (PLAN.md Phase 3 / C_status.md). The curve and STRIPS
stages are real and wired against B's actual code.

A's separate ablation grid (A3) is NOT part of this single-command pipeline
-- final-parallel-team-plan.md describes it as its own "unattended script",
and it doesn't appear in the Sec 7 pipeline diagram either. It still has to
be run once (producing outputs/ablation_results.csv) before packaging; the
smoke test below checks for it but doesn't run it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DATA_CLEAN = Path("data/clean")
OUTPUTS = Path("outputs")

OUTPUTS_MANIFEST = [
    "curve_grid.csv",
    "curve_plot.png",
    "repricing_checks.json",
    "comparison_overlay.png",
    "comparison_diff_bps.png",
    "comparison_table.csv",
    "comparison_summary.json",
    "ablation_results.csv",
    "strips_table_scenario1.csv",
    "strips_table_scenario2.csv",
]

COMPARISON_TENORS = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 14, 20, 30, 40]


def stage_curve(settle):
    """B's stage: fit the ZCYC and write its three outputs."""
    from src.curve import CurveConfig, build_zcyc

    bonds_df = pd.read_csv(DATA_CLEAN / "bonds.csv")
    tbills_df = pd.read_csv(DATA_CLEAN / "tbills.csv")

    curve = build_zcyc(bonds_df, tbills_df, settle, CurveConfig())
    curve.assert_checks()

    OUTPUTS.mkdir(exist_ok=True)
    grid = curve.grid()
    grid.to_csv(OUTPUTS / "curve_grid.csv", index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(grid["tenor_years"], grid["zcy_semi"], label="Zero (semi)")
    ax.plot(grid["tenor_years"], grid["par_semi"], label="Par (semi)")
    ax.set_xlabel("Tenor (years)")
    ax.set_ylabel("Yield (%)")
    ax.legend()
    fig.savefig(OUTPUTS / "curve_plot.png")
    plt.close(fig)

    checks = curve.checks().set_index("check")["passed"]
    solver_mode = "sequential_fallback" if curve.diagnostics.get("solver") == "sequential" else "global"
    with open(OUTPUTS / "repricing_checks.json", "w") as f:
        json.dump(
            {
                "max_repricing_error": curve.diagnostics.get("max_abs_reprice_error"),
                "c2_continuity_ok": bool(checks["max 2nd-derivative jump at knots (relative)"]),
                "df_monotone_ok": bool(checks["largest DF step (must be negative)"]),
                "solver_mode": solver_mode,
            },
            f,
            indent=2,
        )

    return grid


def stage_compare(grid: pd.DataFrame):
    """A's stage: compare our grid against FBIL's published curve."""
    from src.compare import compare

    fbil_grid = pd.read_csv(DATA_CLEAN / "fbil_zcyc.csv")
    # B's Curve.grid() already outputs FBIL-matching zcy_* column names
    # directly, rather than the zero_* names technical-schema.md Sec 5.2
    # documents as compare()'s expected input -- renamed here to match the
    # frozen contract's text. See C_status.md divergence D3.
    our_grid = grid.rename(columns={"zcy_semi": "zero_semi", "zcy_annual": "zero_annual"})

    metrics_df, summary = compare(our_grid, fbil_grid)

    metrics_df[metrics_df["tenor_years"].isin(COMPARISON_TENORS)].to_csv(
        OUTPUTS / "comparison_table.csv", index=False
    )
    with open(OUTPUTS / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # comparison_overlay.png / comparison_diff_bps.png are A's own plots --
    # no frozen function produces them, so they aren't generated here. See
    # C_status.md divergence D9.
    return metrics_df, summary


def stage_strips(settle):
    """C's stage: bonus SDL STRIP pricing, both book-value scenarios."""
    from src.strips import price_sdl

    sdl_bond = pd.read_csv(DATA_CLEAN / "sdl_bond.csv")
    sdl_zcyc = pd.read_csv(DATA_CLEAN / "fbil_sdl_zcyc.csv")
    row = sdl_bond.iloc[0]

    for name, col in [("scenario1", "book_value_scenario_1"), ("scenario2", "book_value_scenario_2")]:
        table = price_sdl(row, sdl_zcyc, settle, book_value_col=col)
        table.to_csv(OUTPUTS / f"strips_table_{name}.csv", index=False)


def smoke_test_outputs():
    """Integration check (SPEC.md Sec 7): every manifest file must exist."""
    missing = [f for f in OUTPUTS_MANIFEST if not (OUTPUTS / f).exists()]
    if missing:
        raise FileNotFoundError(
            "run_all.py finished, but these outputs/ files are still missing: "
            + ", ".join(missing)
            + ". (ablation_results.csv comes from A's separate ablation script, "
            "not this pipeline -- run that too before packaging.)"
        )


def main():
    parser = argparse.ArgumentParser(description="Run the full ZCYC + STRIPS pipeline.")
    parser.add_argument("--date", required=True, help="Valuation date, YYYY-MM-DD")
    args = parser.parse_args()

    from src.bond import settlement_date

    settle = settlement_date(args.date)

    grid = stage_curve(settle)
    stage_compare(grid)
    stage_strips(settle)
    smoke_test_outputs()
    print(f"Pipeline complete for valuation date {args.date} (settle {settle}). See outputs/.")


if __name__ == "__main__":
    main()
