"""Download the real FBIL publications into data/raw/.

FBIL's public site (https://www.fbil.org.in) is an Angular front end over a
JSON/file API at https://www.fbil.org.in/wasdm. Everything the front end shows
to a logged-out visitor is reachable with `authenticated=false`, and that is
exactly the free tier: the public archive lags the live benchmark by about a
week, which is fine for this project -- we need one settled business day, not
today's rate.

Two endpoints matter:

    GET /wasdm/<benchmark>/fetchfiltered?fromDate&toDate&authenticated=false
        -> the archive index, one entry per published business day.
    GET /wasdm/<benchmark>/download?date=YYYY-MM-DD
        -> the published workbook for that day, exactly as FBIL ships it.

`/tbill/download` returns HTTP 500 on the public tier, so the T-bill rates come
from `/tbill/fetchfiltered`, which serves the same published numbers as JSON.

What lands where
----------------
``data/raw/fbil_source/`` keeps every byte FBIL served, under FBIL's own
filenames, untouched. That is the archival copy and the thing to re-check a
number against.

``data/raw/`` gets the five role-named files ``data_loader.discover_raw_files``
looks for. Three of them are FBIL's workbooks copied verbatim; two are derived,
and only in shape, never in value:

  * ``fbil_gsec_zcyc_<date>.csv`` joins FBIL's G-Sec zero curve (the ZCYC sheet
    of the STRIPS workbook) to FBIL's G-Sec par curve (the Par Yield sheet of
    the G-Sec workbook) on tenor. FBIL publishes the two halves of one curve in
    two different workbooks; the clean schema wants them in one frame. Column
    headers are carried across unchanged so the alias table still does the
    matching.
  * ``fbil_tbills_<date>.csv`` is the T-bill JSON written out as a two-column
    table.

No number in either derived file is computed, interpolated or filled.
"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from src.bond import to_date

API = "https://www.fbil.org.in/wasdm"
# The API answers anonymous requests but only with a browser-ish Referer.
HEADERS = {
    "Referer": "https://www.fbil.org.in/",
    "Accept": "application/json, application/octet-stream, */*",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
}
TIMEOUT = 60

# The benchmarks this project needs, and what each one is for.
BENCHMARKS = {
    "gsec": "G-Sec prices/YTMs (sheet 'G-Sec') and the par curve (sheet 'Par Yield')",
    "strips": "GOI STRIPS prices and the G-Sec zero curve (sheet 'ZCYC')",
    "sdl": "SDL/SGS prices and YTMs (sheet 'SDL')",
    "sdlzcyc": "SDL/SGS zero curve (sheet 'SDL_ZCYC')",
}
# Benchmarks that must all have published on a day for it to be usable.
REQUIRED = ("gsec", "strips", "sdl", "sdlzcyc", "tbill")

SOURCE_SUBDIR = "fbil_source"
ARCHIVE_LOOKBACK_DAYS = 120


# ------------------------------------------------------------------- transport

def _get(path: str, **params) -> bytes:
    params.setdefault("authenticated", "false")
    url = f"{API}/{path}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"FBIL returned HTTP {exc.code} for {url}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"could not reach FBIL at {url}: {exc.reason}") from exc


def _get_json(path: str, **params):
    return json.loads(_get(path, **params))


def published_dates(benchmark: str, lookback_days: int = ARCHIVE_LOOKBACK_DAYS) -> list[str]:
    """ISO dates `benchmark` has published on, newest first."""
    today = date.today()
    rows = _get_json(
        f"{benchmark}/fetchfiltered",
        fromDate=(today - timedelta(days=lookback_days)).isoformat(),
        toDate=today.isoformat(),
    )
    dates = {str(row["processRunDate"])[:10] for row in rows}
    return sorted(dates, reverse=True)


def latest_common_date(lookback_days: int = ARCHIVE_LOOKBACK_DAYS) -> str:
    """Newest business day on which every benchmark we need has published.

    Taking the newest date per benchmark separately would mix publication days
    inside one curve if FBIL is mid-release on one of them; the whole point of
    working off the lagged archive is that the day is settled.
    """
    per_benchmark = {b: set(published_dates(b, lookback_days)) for b in REQUIRED}
    common = set.intersection(*per_benchmark.values())
    if not common:
        raise RuntimeError(
            "no business day in the last "
            f"{lookback_days} days has all of {list(REQUIRED)} published. "
            f"Latest per benchmark: { {b: max(d, default=None) for b, d in per_benchmark.items()} }"
        )
    return max(common)


# ------------------------------------------------------------------ downloading

def download_source(valuation_date: str, source_dir: Path) -> dict[str, Path]:
    """Fetch FBIL's files for `valuation_date` verbatim into `source_dir`."""
    source_dir.mkdir(parents=True, exist_ok=True)
    stamp = to_date(valuation_date).strftime("%d%m%Y")
    saved: dict[str, Path] = {}

    for benchmark in BENCHMARKS:
        payload = _get(f"{benchmark}/download", date=valuation_date)
        # An xlsx is a zip; anything else means FBIL served an error page or an
        # empty body for a day it has not actually published.
        if not payload.startswith(b"PK"):
            raise RuntimeError(
                f"{benchmark} for {valuation_date} is not an xlsx "
                f"(first bytes {payload[:40]!r}). FBIL may not have published that day."
            )
        path = source_dir / f"{benchmark}_{stamp}.xlsx"
        path.write_bytes(payload)
        saved[benchmark] = path

    rows = _get_json("tbill/fetchfiltered", fromDate=valuation_date, toDate=valuation_date)
    rows = [r for r in rows if str(r["processRunDate"])[:10] == valuation_date]
    if not rows:
        raise RuntimeError(f"FBIL published no T-bill rates for {valuation_date}")
    path = source_dir / f"tbill_{stamp}.json"
    path.write_text(json.dumps(rows, indent=2))
    saved["tbill"] = path
    return saved


# --------------------------------------------------------------- role files

def _sheet(path: Path, sheet: str) -> pd.DataFrame:
    """A published sheet with its letterhead stripped and FBIL's header kept.

    FBIL puts two to four rows of title, entity name and date above the real
    header. We find the header the same way the loader does -- by looking for
    the row that names a tenor -- rather than hard-coding a skiprows that would
    break the week FBIL adds a line to the letterhead.
    """
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    for i in range(min(20, len(raw))):
        cells = [str(v).strip().lower() for v in raw.iloc[i].tolist() if pd.notna(v)]
        if any(c.startswith("tenor") for c in cells):
            table = raw.iloc[i + 1:].copy()
            table.columns = [str(c).strip() for c in raw.iloc[i].tolist()]
            table = table.loc[:, [c for c in table.columns if c and c.lower() != "nan"]]
            first = table.columns[0]
            return table[pd.to_numeric(table[first], errors="coerce").notna()].reset_index(drop=True)
    raise ValueError(f"no header row naming a tenor in {path.name} sheet {sheet!r}")


def build_role_files(valuation_date: str, source: dict[str, Path], raw_dir: Path) -> dict[str, Path]:
    """Turn the archived downloads into the files the loader discovers."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    d = to_date(valuation_date).isoformat()
    written: dict[str, Path] = {}

    # Copied verbatim -- in each of these the sheet the loader needs is sheet 0.
    for role, benchmark in (("gsec_bonds", "gsec"), ("fbil_sdl_zcyc", "sdlzcyc"), ("fbil_sdl_valuation", "sdl")):
        target = raw_dir / f"{role}_{d}.xlsx"
        shutil.copyfile(source[benchmark], target)
        written[role] = target

    # G-Sec zero curve and par curve, published in two workbooks, joined on tenor.
    zero = _sheet(source["strips"], "ZCYC")
    par = _sheet(source["gsec"], "Par Yield")
    tenor_zero, tenor_par = zero.columns[0], par.columns[0]
    zero[tenor_zero] = pd.to_numeric(zero[tenor_zero]).round(2)
    par[tenor_par] = pd.to_numeric(par[tenor_par]).round(2)
    merged = zero.merge(par, left_on=tenor_zero, right_on=tenor_par, how="inner")
    if not len(merged) == len(zero) == len(par):
        raise RuntimeError(
            f"FBIL's G-Sec zero curve ({len(zero)} tenors) and par curve ({len(par)} tenors) "
            f"do not sit on the same grid; the join kept {len(merged)}."
        )
    if tenor_par != tenor_zero:
        merged = merged.drop(columns=[tenor_par])
    target = raw_dir / f"fbil_gsec_zcyc_{d}.csv"
    merged.to_csv(target, index=False)
    written["fbil_gsec_zcyc"] = target

    # T-bills: the published JSON as a table, every tenor FBIL quoted that day.
    rows = json.loads(source["tbill"].read_text())
    tbills = pd.DataFrame({"Tenor": [r["tenorName"] for r in rows], "Rate": [float(r["rate"]) for r in rows]})
    target = raw_dir / f"fbil_tbills_{d}.csv"
    tbills.to_csv(target, index=False)
    written["fbil_tbills"] = target
    return written


def fetch(valuation_date: str | None = None, raw_dir: str | Path = "data/raw") -> tuple[str, dict[str, Path]]:
    """Download FBIL's publications and lay out data/raw/. Returns (date, files)."""
    raw_dir = Path(raw_dir)
    valuation_date = to_date(valuation_date).isoformat() if valuation_date else latest_common_date()
    source = download_source(valuation_date, raw_dir / SOURCE_SUBDIR)
    return valuation_date, build_role_files(valuation_date, source, raw_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--date", help="FBIL business day, YYYY-MM-DD (default: newest published)")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--list", action="store_true", help="list published dates and exit")
    args = parser.parse_args(argv)

    if args.list:
        for benchmark in REQUIRED:
            print(f"{benchmark:9s} {', '.join(published_dates(benchmark)[:10])}")
        return 0

    valuation_date, written = fetch(args.date, args.raw_dir)
    print(f"FBIL valuation date: {valuation_date}")
    for role, path in written.items():
        print(f"  {role:20s} -> {path}")
    print(f"\nNow run:  python -c \"from src.data_loader import build_clean_data; "
          f"build_clean_data(valuation_date='{valuation_date}')\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
