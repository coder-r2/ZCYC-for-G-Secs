"""Single command chaining the whole pipeline: data -> curve -> comparison -> STRIPS.

    python run_all.py --date YYYY-MM-DD

Pipeline (SPEC.md Sec 7 / technical-schema.md Sec 0):

    data/clean/*.csv (A, produced ahead of time by data_loader.py)
        -> selection.py + curve.py (B)  -> Curve object
        -> compare.py (A)               -> our curve vs FBIL's published curve
        -> strips.py (C)                -> STRIP prices for the bonus SDL

data_loader.py has no frozen call signature (technical-schema.md only
freezes the *schemas* it must produce, Sec 2) -- confirmed against A's real
code: it's a separate CLI (`python -m src.data_loader --date ...`) that
populates data/clean/*.csv ahead of time, not something this script calls.
This script starts by reading those already-clean CSVs, same as every
other stage reads its predecessor's output.

A's real compare.py exposes write_comparison_outputs() (compare() + both
plots + table + summary, one call, exact frozen filenames) and ablation.py
exposes run_ablation()/write_ablation_results() the same composable way --
both wired in below. compare()'s own _normalise_ours() already accepts
either zero_semi or zcy_semi, so no column rename is needed on this side
either (see C_status.md D3/D9/D11 -- all resolved once A's real code
existed to check against, no changes needed to A's or B's code).
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


def stage_compare(grid: pd.DataFrame, provenance: str):
    """A's stage: compare our grid against FBIL's published curve, both
    plots, table and summary -- all four frozen outputs in one call."""
    from src.compare import write_comparison_outputs

    fbil_grid = pd.read_csv(DATA_CLEAN / "fbil_zcyc.csv")
    return write_comparison_outputs(grid, fbil_grid, outputs_dir=OUTPUTS, provenance=provenance)


def stage_ablation(settle):
    """A's stage: the 16-configuration sensitivity grid."""
    from src.ablation import run_ablation, write_ablation_results

    bonds_df = pd.read_csv(DATA_CLEAN / "bonds.csv")
    tbills_df = pd.read_csv(DATA_CLEAN / "tbills.csv")
    fbil_grid = pd.read_csv(DATA_CLEAN / "fbil_zcyc.csv")

    results, failures = run_ablation(bonds_df, tbills_df, fbil_grid, settle, verbose=False)
    write_ablation_results(results, outputs_dir=OUTPUTS)
    return results, failures


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
            "run_all.py finished, but these outputs/ files are still missing: " + ", ".join(missing)
        )


def main():
    parser = argparse.ArgumentParser(description="Run the full ZCYC + STRIPS pipeline.")
    parser.add_argument("--date", required=True, help="Valuation date, YYYY-MM-DD")
    args = parser.parse_args()

    from src.bond import settlement_date
    from src.data_loader import clean_data_provenance

    settle = settlement_date(args.date)
    provenance = clean_data_provenance(DATA_CLEAN)

    grid = stage_curve(settle)
    stage_compare(grid, provenance)
    stage_ablation(settle)
    stage_strips(settle)
    smoke_test_outputs()

    print(f"Pipeline complete for valuation date {args.date} (settle {settle}). See outputs/.")
    if provenance != "FBIL":
        print(f"WARNING: data/clean/ provenance is '{provenance}', not FBIL -- do not submit these outputs.")


if __name__ == "__main__":
    main()
