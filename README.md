# G-Sec ZCYC Bootstrap + SDL STRIPS

Bootstraps a Zero Coupon Yield Curve (ZCYC) for Government Securities using a cubic-spline
model, compares it against FBIL's published curve, and (bonus) prices STRIPS on a State
Development Loan using the RBI stripping guidelines.

## Acknowledgement

This mini-project was done as part of the course MG251: Finance & Accounts (Fall '26 Semester) at IISc Bangalore.

Team Members:
- Abinav Thangaraju Sethupathy
- A.S. Kretik
- Khaja Aflal H
- Rishe Raghavendira Gnanasekaran

## Slides

`presentation/Project-Presentation.pdf` contains the slide deck for this project.

## Setup

```bash
pip install -r requirements.txt
```

## How to run

```bash
python -m src.data_loader --date YYYY-MM-DD   # builds data/clean/*.csv from data/raw/
python run_all.py --date YYYY-MM-DD           # curve -> comparison -> ablation -> STRIPS
```

`data/clean/*.csv` are already committed in this repo (built from the raw FBIL files under `data/raw/`), so the first command is only needed if you want to rebuild them from scratch or re-run against a different date. `data_loader.py` does not hit the network — it reads the raw FBIL exports already sitting in `data/raw/`.

`run_all.py` runs the full pipeline: `data/clean/*.csv` → curve → comparison → ablation → STRIPS, and writes every table/plot in `outputs/`. `--date` is the valuation date; settlement is computed as T+1 (rolled over weekends).

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
| `ablation_results.csv` | 16-configuration sensitivity grid |
| `strips_table_scenario1.csv`, `strips_table_scenario2.csv` | bonus SDL STRIP pricing, two book-value scenarios |
