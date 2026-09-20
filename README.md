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

Runs the full pipeline: data → curve → comparison → STRIPS. Outputs (tables, plots) land in `outputs/`.

## Tests

```bash
pytest
```

## Folder map

```
data/{raw,clean}/   input data (raw downloads, cleaned CSVs)
src/                 bond.py, selection.py, curve.py, compare.py, strips.py
tests/               pytest suite
outputs/             result tables and charts
docs/assumptions.md  locked modeling decisions
presentation/        final slide deck
```
