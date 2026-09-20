# Mini Project 1 — G-Sec ZCYC Bootstrap + SDL STRIPS

Bootstraps a Zero Coupon Yield Curve (ZCYC) for Government Securities using a cubic-spline
model, compares it against FBIL's published curve, and (bonus) prices STRIPS on a State
Development Loan using the RBI stripping guidelines.

## Setup

```bash
pip install -r requirements.txt
```

## How to run

```bash
python run_all.py --date YYYY-MM-DD
```

Runs the full pipeline: `data/clean/*.csv` → curve → comparison → STRIPS, and writes every table/plot in `outputs/`. `--date` is the valuation date; settlement is computed as T+1 (rolled over weekends).

`data/clean/*.csv` must already exist before running — they're produced by the data-cleaning step, not by `run_all.py` itself. The ablation grid (`outputs/ablation_results.csv`) is also a separate, unattended script, run once ahead of packaging rather than as part of this single command.

## Tests

```bash
pytest
```

## Folder map

```
run_all.py           single command: data -> curve -> comparison -> STRIPS
data/raw/            unmodified downloads
data/clean/          cleaned CSVs in the schemas documented alongside the project
src/
  bond.py            coupon schedules, accrued interest, price <-> YTM
  selection.py        curve-node selection (T-bills, 90-day thinning, long-end buckets)
  curve.py            cubic-spline ZCYC bootstrap + solver + auto-checks
  compare.py          our curve vs FBIL's published curve
  strips.py           STRIP engine + bonus SDL pricing
tests/                pytest suite, one file per module
outputs/              result tables, plots, and JSON diagnostics (see below)
docs/assumptions.md   locked modeling decisions -- read this for what's assumed vs. modeled
presentation/         final slide deck
```

## Outputs produced by `run_all.py`

| File | From |
|---|---|
| `curve_grid.csv`, `curve_plot.png`, `repricing_checks.json` | curve engine |
| `comparison_table.csv`, `comparison_summary.json`, `comparison_overlay.png`, `comparison_diff_bps.png` | comparison vs FBIL |
| `ablation_results.csv` | separate ablation script (run once, not by `run_all.py`) |
| `strips_table_scenario1.csv`, `strips_table_scenario2.csv` | bonus SDL STRIP pricing, two book-value scenarios |
