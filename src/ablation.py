"""A3 -- the 2x2x2x2 ablation grid.

Refits the curve under all sixteen combinations of the four modelling switches
and scores each one by its overall RMSE against FBIL's published curve. Runs
unattended:

    python -m src.ablation --date YYYY-MM-DD

This is reported as a **sensitivity analysis, not a model search**. The headline
curve uses the defaults locked in docs/assumptions.md, chosen before any
comparison against FBIL was run. If some other configuration scores better it
gets said out loud on the slide rather than quietly promoted to headline --
picking the config that happens to match FBIL best, after seeing the answer,
would make the comparison meaningless.

A failed configuration is a result. ``build_zcyc`` raises by design when its
auto-checks fail, so the exception is caught, the row is kept with
``rmse_bps = NaN``, and the reason is printed for the slide notes. That keeps
the table at the sixteen rows the contract asks for; pandas sorts NaN last, so
the failures land at the bottom where they belong.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd

from src.compare import compare
from src.curve import BOUNDARIES, SPLINE_VARIABLES, CurveConfig, build_zcyc

# Frozen output column names. These are the contract's spellings, which are not
# quite CurveConfig's field names (`thinning_90day` vs `thin_90d`); the mapping
# lives in `_row` and nowhere else.
RESULT_COLUMNS = ["spline_variable", "boundary", "thinning_90day", "time_convention", "rmse_bps"]
RESULTS_FILE = "ablation_results.csv"

THINNING = (True, False)
TIME_CONVENTIONS = ("act365", "halfyear")


def configurations() -> list[CurveConfig]:
    """All sixteen configurations, built by keyword.

    Keyword construction rather than positional: the curve engine is free to
    reorder or insert CurveConfig fields, and a positional call would silently
    start ablating the wrong axis.
    """
    return [
        CurveConfig(
            spline_variable=spline_variable,
            boundary=boundary,
            thin_90d=thin_90d,
            time_convention=time_convention,
        )
        for spline_variable, boundary, thin_90d, time_convention in itertools.product(
            SPLINE_VARIABLES, BOUNDARIES, THINNING, TIME_CONVENTIONS
        )
    ]


def _row(cfg: CurveConfig, rmse_bps: float) -> dict:
    return {
        "spline_variable": cfg.spline_variable,
        "boundary": cfg.boundary,
        "thinning_90day": cfg.thin_90d,
        "time_convention": cfg.time_convention,
        "rmse_bps": rmse_bps,
    }


def run_ablation(
    bonds_df: pd.DataFrame,
    tbills_df: pd.DataFrame,
    fbil_grid_df: pd.DataFrame,
    settle,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Fit every configuration and score it against FBIL's curve.

    Returns ``(results_df, failures)`` -- sixteen rows sorted by ``rmse_bps``
    ascending, and the reason each failed configuration raised, keyed by the
    curve engine's own config label.
    """
    rows, failures = [], {}

    for i, cfg in enumerate(configurations(), start=1):
        label = cfg.label()
        try:
            curve = build_zcyc(bonds_df, tbills_df, settle, cfg)
            curve.assert_checks()
            _, summary = compare(curve.grid(), fbil_grid_df)
            rmse = float(summary["overall"]["rmse_bps"])
            if verbose:
                print(f"[{i:2d}/16] {label:<44} rmse {rmse:8.2f} bps")
        except Exception as exc:
            # Includes AssertionError from assert_checks(): a curve that fails
            # its own repricing or monotonicity checks has no business being
            # scored, so it is recorded as a failure rather than as a number.
            rmse = float("nan")
            failures[label] = f"{type(exc).__name__}: {exc}".replace("\n", " ")[:300]
            if verbose:
                print(f"[{i:2d}/16] {label:<44} FAILED   {failures[label]}")
        rows.append(_row(cfg, rmse))

    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    # NaN sorts last by default, which is exactly where failed runs belong.
    return results.sort_values("rmse_bps").reset_index(drop=True), failures


def write_ablation_results(results: pd.DataFrame, outputs_dir: str | Path = "outputs") -> Path:
    outputs = Path(outputs_dir)
    outputs.mkdir(parents=True, exist_ok=True)
    path = outputs / RESULTS_FILE
    results.to_csv(path, index=False)
    return path


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run the 16-configuration ablation grid.")
    parser.add_argument("--date", required=True, help="Valuation date, YYYY-MM-DD")
    parser.add_argument("--clean-dir", default="data/clean")
    parser.add_argument("--outputs-dir", default="outputs")
    args = parser.parse_args(argv)

    from src.bond import settlement_date
    from src.data_loader import clean_data_provenance

    clean = Path(args.clean_dir)
    bonds = pd.read_csv(clean / "bonds.csv")
    tbills = pd.read_csv(clean / "tbills.csv")
    fbil = pd.read_csv(clean / "fbil_zcyc.csv")
    settle = settlement_date(args.date)
    provenance = clean_data_provenance(clean)

    print(f"valuation date {args.date} (settle {settle}), data provenance: {provenance}\n")
    results, failures = run_ablation(bonds, tbills, fbil, settle)
    path = write_ablation_results(results, args.outputs_dir)

    print(f"\n{results.to_string(index=False)}")
    print(f"\nwrote {path}")

    headline = CurveConfig()
    best = results.iloc[0]
    if results["rmse_bps"].notna().any() and (
        best["spline_variable"] != headline.spline_variable
        or best["boundary"] != headline.boundary
        or bool(best["thinning_90day"]) != headline.thin_90d
        or best["time_convention"] != headline.time_convention
    ):
        print(
            "\nNote: the best-scoring configuration is not the locked headline default. "
            "Report this as sensitivity on the slide -- the headline curve does not change."
        )
    if failures:
        print("\nFailed configurations (copy into slide-notes-a.md):")
        for label, reason in failures.items():
            print(f"  {label}: {reason}")
    if provenance != "FBIL":
        print(f"\nWARNING: built from {provenance} data -- no ranking here is a result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
