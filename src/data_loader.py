"""Raw FBIL downloads -> the five clean CSVs the rest of the pipeline reads.

    data/raw/<whatever FBIL gave us>  ->  data/clean/{bonds, tbills, fbil_zcyc,
                                          fbil_sdl_zcyc, sdl_bond}.csv

FBIL publishes the same numbers in a different shape most weeks: sometimes an
xlsx with three rows of letterhead above the header, sometimes a csv, sometimes
an html table; the header text drifts ("Coupon", "Coupon Rate", "Coupon (%)").
So nothing here is positional. Files are matched to a role by filename, the
header row is found by looking for the row that actually contains recognisable
column names, and columns are matched through an alias table. A column that
cannot be matched is an error, never a silent drop -- getting `ytm` and
`price` the wrong way round would produce a plausible-looking curve that is
entirely wrong.

Units, fixed by the interface contract and not negotiable here:
  * Rates stay in PERCENT in every output CSV. 6.05 means 6.05%. The
    percent -> decimal conversion belongs at the point of use in curve.py and
    strips.py, so this module never divides a rate by 100.
  * Dates are ISO strings, YYYY-MM-DD.
  * Money is absolute rupees (Rs 5 crore -> 50000000). `price` and
    `market_price` are the exception: those are per 100 face, as quoted.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.bond import accrued_30_360, settlement_date, to_date

CLEAN_FILES = ("bonds", "tbills", "fbil_zcyc", "fbil_sdl_zcyc", "sdl_bond")

SCHEMAS: dict[str, list[str]] = {
    "bonds": ["isin", "desc", "coupon", "maturity", "price", "ytm", "volume"],
    "tbills": ["tenor", "tenor_years", "rate"],
    "fbil_zcyc": ["tenor_years", "zcy_semi", "zcy_annual", "par_semi", "par_annual"],
    "fbil_sdl_zcyc": ["tenor_years", "zcy_semi", "zcy_annual", "par_semi", "par_annual"],
    "sdl_bond": [
        "isin",
        "desc",
        "coupon",
        "maturity",
        "market_price",
        "face_stripped",
        "book_value_scenario_1",
        "book_value_scenario_2",
    ],
}

# Published ZCYC grids: (first tenor, last tenor, step). Both were read off
# FBIL's actual files: the G-Sec curve runs 0.25 to 50.00 (200 points), not to
# 40.00 as the interface contract guessed, and the SDL curve stops at 14.00
# (56 points) as expected.
ZCYC_GRID = (0.25, 50.0, 0.25)
SDL_ZCYC_GRID = (0.25, 14.0, 0.25)

# Columns of a published ZCYC file that FBIL does not always publish. It gives
# a par curve alongside the G-Sec zero curve but none alongside the SDL one,
# so for SDL these columns exist in the schema and are empty. Nothing is
# invented to fill them.
OPTIONAL_ZCYC_COLUMNS = ("par_semi", "par_annual")

# T-bill points the short end is built from, in the order the schema lists them.
TBILL_TENORS = (("7D", 7 / 365), ("6M", 0.5), ("12M", 1.0))

PLAUSIBLE_RATE_RANGE = (0.0, 20.0)   # percent, for coupon / ytm / zero rates
SAMPLE_SUFFIX = "_SAMPLE"
SOURCE_FILE = "SOURCE.md"

# The published SDL ZCYC stops at 14y, so a longer SDL has no curve to be
# discounted off and cannot be the bonus parent.
SDL_MAX_RESIDUAL_YEARS = 14.0
CRORE = 10_000_000
DEFAULT_FACE_STRIPPED = 5 * CRORE


# --------------------------------------------------------------------- aliases

def _norm(name) -> str:
    """Column header -> comparable key: lowercase, alphanumerics only."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


# Right-hand side is a list of normalised header spellings seen in FBIL's
# files, most specific first. Add to these rather than renaming by position.
BOND_ALIASES: dict[str, tuple[str, ...]] = {
    "isin": ("isin", "isinno", "isincode", "securityisin"),
    "desc": ("desc", "description", "securitydescription", "security", "securityname", "nomenclature"),
    "coupon": ("coupon", "couponrate", "coupon", "couponinpercent", "couponrateinpercent"),
    "maturity": ("maturity", "maturitydate", "dateofmaturity", "redemptiondate", "maturitydt",
                 "maturityddmmmyyyy"),
    "price": ("price", "cleanprice", "valuationprice", "priceper100", "fimmdaprice", "pricers"),
    "ytm": ("ytm", "yieldtomaturity", "ytminpercent", "annualisedytm", "yield", "valuationytm",
            "ytmpasemiannual"),
    "volume": ("volume", "tradedvolume", "turnover", "tradedvalue", "totaltradedvalue", "nosoftrades", "trades"),
    # Not part of the `bonds` schema and dropped before it is returned. Read
    # only so parse_bonds can see FBIL's "FRB" flag -- see FLOATER_REMARKS.
    "remark": ("remark1", "remark", "remarks", "securitytype"),
}

# Values of FBIL's Remark column that mark a security whose cash flows are not
# a fixed nominal coupon stream: floating-rate bonds reset off a benchmark and
# inflation-indexed bonds pay on an indexed principal. Their quoted YTM is not
# comparable with a fixed-coupon YTM, so fitting a nominal zero curve through
# them is wrong however good the fit looks. FBIL excludes them from its own
# curve too -- none of them carries the "Input Point" flag.
FLOATER_REMARKS = ("frb", "iib", "floating", "inflation")

TBILL_ALIASES: dict[str, tuple[str, ...]] = {
    "tenor": ("tenor", "tenure", "maturity", "period", "tenorindays", "residualmaturity"),
    "rate": ("rate", "tbillrate", "yield", "ytm", "impliedyield", "annualisedyield", "rateinpercent"),
}

ZCYC_ALIASES: dict[str, tuple[str, ...]] = {
    "tenor_years": ("tenoryears", "tenor", "tenorinyears", "tenoryrs", "maturityyears", "years", "term",
                    "tenoryear"),
    "zcy_semi": ("zcysemi", "zcycsemiannual", "zerocouponyieldsemiannual", "zcycsemi", "zerosemi",
                 "zcycsemiannualised", "semiannualzcyc", "zerocouponsemiannual"),
    "zcy_annual": ("zcyannual", "zcycannualised", "zerocouponyieldannualised", "zcycannual",
                   "zeroannual", "annualisedzcyc", "zcycannualized", "zerocouponannualized"),
    # FBIL heads its par curve "YTM% p.a.(Semi-Annual)" -- the same words it
    # uses for a bond's own YTM, which is why the par spellings only ever get
    # looked up against a ZCYC file.
    "par_semi": ("parsemi", "parcurvesemiannual", "paryieldsemiannual", "parsemiannual", "semiannualpar",
                 "ytmpasemiannual"),
    "par_annual": ("parannual", "parcurveannualised", "paryieldannualised", "parannualised",
                   "annualisedpar", "parannualized", "ytmpaannualized"),
}

SDL_BOND_ALIASES: dict[str, tuple[str, ...]] = {
    **{k: v for k, v in BOND_ALIASES.items() if k in ("isin", "desc", "coupon", "maturity")},
    "market_price": ("marketprice", "price", "cleanprice", "valuationprice", "priceper100", "pricers"),
}

# Filename fragments that identify each raw file's role. Checked in order, and
# the first role whose `must` fragments all appear (and whose `must_not` ones do
# not) wins -- "sdl" has to be ruled out before a file counts as the G-Sec one.
ROLE_PATTERNS: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("fbil_sdl_zcyc", ("sdl", "zcyc"), ()),
    ("fbil_sdl_zcyc", ("sgs", "zcyc"), ()),
    ("sdl_bond", ("sdl",), ("zcyc",)),
    ("sdl_bond", ("sgs",), ("zcyc",)),
    ("fbil_zcyc", ("zcyc",), ("sdl", "sgs")),
    ("tbills", ("tbill",), ()),
    ("tbills", ("treasurybill",), ()),
    ("bonds", ("bond",), ("sdl", "sgs")),
    ("bonds", ("gsec",), ("sdl", "sgs", "zcyc")),
    ("bonds", ("govtsec",), ("sdl", "sgs", "zcyc")),
    ("bonds", ("valuation",), ("sdl", "sgs", "zcyc")),
)


# ----------------------------------------------------------------- raw reading

def _read_any(path: Path) -> pd.DataFrame:
    """Read csv / xlsx / html into a frame with no header applied yet."""
    suffix = path.suffix.lower()
    if suffix in (".csv", ".txt"):
        return pd.read_csv(path, header=None, dtype=object, skip_blank_lines=False)
    if suffix in (".xlsx", ".xls", ".xlsm"):
        return pd.read_excel(path, header=None, dtype=object)
    if suffix in (".html", ".htm"):
        tables = pd.read_html(path, header=None)
        if not tables:
            raise ValueError(f"no tables found in {path}")
        # The valuation table is the biggest one on the page; FBIL wraps it in
        # navigation tables that would otherwise win on document order.
        table = max(tables, key=len)
        # read_html always promotes a <th> row into `columns`, so push it back
        # down into the data and let _header_row find it the same way it does
        # for csv and xlsx. Where there was no <th> the promoted row is a plain
        # RangeIndex, which simply scores zero and is ignored.
        return pd.concat(
            [pd.DataFrame([table.columns.tolist()]), pd.DataFrame(table.to_numpy())],
            ignore_index=True,
        )
    raise ValueError(f"unsupported raw file type {path.suffix!r} for {path.name}")


def _header_row(raw: pd.DataFrame, aliases: dict[str, tuple[str, ...]], min_hits: int = 2) -> int:
    """Index of the row that is actually the header.

    FBIL's xlsx files carry a title and a date line above the real header, and
    the number of preamble rows changes between files. Rather than hard-coding
    a `skiprows`, score each of the first few rows by how many schema columns it
    names and take the best -- a file whose header moved down a row still loads.
    """
    best_row, best_hits = None, 0
    for i in range(min(20, len(raw))):
        cells = {_norm(v) for v in raw.iloc[i].tolist() if pd.notna(v)}
        hits = sum(1 for spellings in aliases.values() if cells & set(spellings))
        if hits > best_hits:
            best_row, best_hits = i, hits
    if best_row is None or best_hits < min_hits:
        raise ValueError(
            "could not find a header row naming at least "
            f"{min_hits} known columns; looked for {sorted(aliases)}"
        )
    return best_row


def _apply_header(raw: pd.DataFrame, row: int) -> pd.DataFrame:
    out = raw.iloc[row + 1:].copy()
    out.columns = [str(c).strip() for c in raw.iloc[row].tolist()]
    return out.dropna(how="all").reset_index(drop=True)


def _map_columns(
    df: pd.DataFrame,
    aliases: dict[str, tuple[str, ...]],
    required: tuple[str, ...],
    source: str,
) -> pd.DataFrame:
    """Rename `df` into schema names via the alias table.

    A required column that matches nothing raises, naming the file and the
    headers actually present -- the caller can then add one spelling to the
    alias table. Silently continuing would hand the curve engine a frame with
    the right shape and the wrong contents.
    """
    present = {_norm(c): c for c in df.columns}
    out = pd.DataFrame(index=df.index)
    for target, spellings in aliases.items():
        match = next((present[s] for s in spellings if s in present), None)
        if match is None:
            if target in required:
                raise ValueError(
                    f"{source}: no column matches {target!r}. Headers present: "
                    f"{list(df.columns)}. Add the spelling to the alias table in data_loader.py."
                )
            continue
        out[target] = df[match]
    return out


def _to_number(series: pd.Series) -> pd.Series:
    """Coerce a quoted column to float: strips commas, currency, %, footnotes."""
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
        .replace({"": None, "-": None, "--": None, "NA": None, "N.A.": None, "nan": None, "None": None})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _to_iso_date(series: pd.Series) -> pd.Series:
    """Coerce a date column to ISO strings.

    `dayfirst=True` because FBIL quotes Indian-format dates (14/05/2031). The
    result is re-read from the ISO string by every downstream consumer, so an
    ambiguous date parsed the wrong way here would be invisible later -- any
    value that fails to parse raises rather than becoming NaT.
    """
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
    if parsed.isna().any():
        bad = series[parsed.isna()].unique()[:5]
        raise ValueError(f"unparseable dates: {list(bad)}")
    return parsed.dt.strftime("%Y-%m-%d")


# ------------------------------------------------------------ role discovery

def discover_raw_files(raw_dir: str | Path = "data/raw") -> dict[str, Path]:
    """Map each of the five roles to a file in `raw_dir`, by filename.

    Sample files (``*_SAMPLE.*``) are used only when no real download claims
    the same role, so dropping the real FBIL file into data/raw/ is enough to
    take over -- there is no flag to remember to flip.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"raw data directory {raw_dir} does not exist")

    real: dict[str, Path] = {}
    sample: dict[str, Path] = {}
    for path in sorted(raw_dir.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() not in (".csv", ".txt", ".xlsx", ".xls", ".xlsm", ".html", ".htm"):
            continue
        key = _norm(path.stem)
        role = next(
            (
                r
                for r, must, must_not in ROLE_PATTERNS
                if all(m in key for m in must) and not any(n in key for n in must_not)
            ),
            None,
        )
        if role is None:
            continue
        target = sample if path.stem.upper().endswith(SAMPLE_SUFFIX) else real
        target.setdefault(role, path)

    found = {**sample, **real}
    missing = [r for r in CLEAN_FILES if r not in found]
    if missing:
        raise FileNotFoundError(
            f"no raw file in {raw_dir} matches these roles: {missing}. "
            f"Files seen: {[p.name for p in sorted(raw_dir.iterdir()) if p.is_file()]}. "
            "Name the download so the role is in the filename, e.g. 'gsec_bonds_2026-09-18.xlsx', "
            "'tbills_...', 'fbil_zcyc_...', 'sdl_zcyc_...', 'sdl_bond_...'."
        )
    return found


def is_sample(path: Path) -> bool:
    return Path(path).stem.upper().endswith(SAMPLE_SUFFIX)


# --------------------------------------------------------------------- parsers

def parse_bonds(path: Path, valuation_date) -> pd.DataFrame:
    """G-Sec valuation file -> the frozen `bonds` schema."""
    raw = _read_any(Path(path))
    table = _apply_header(raw, _header_row(raw, BOND_ALIASES))
    df = _map_columns(table, BOND_ALIASES, ("isin", "coupon", "maturity"), Path(path).name)

    df = df[df["isin"].notna()].copy()
    df["isin"] = df["isin"].astype(str).str.strip().str.upper()
    df = df[df["isin"].str.match(r"^IN[A-Z0-9]{8,}$", na=False)].reset_index(drop=True)

    if "desc" not in df.columns:
        df["desc"] = ""
    df["desc"] = df["desc"].astype(str).str.strip()
    df["maturity"] = _to_iso_date(df["maturity"])
    for col in ("coupon", "price", "ytm"):
        df[col] = _to_number(df[col]) if col in df.columns else np.nan

    # `volume` is nullable but the column must exist: selection.py reads it to
    # break ties between bonds in the same long-end bucket and has a documented
    # fallback for NaN values, but a missing column would raise a KeyError.
    df["volume"] = _to_number(df["volume"]) if "volume" in df.columns else np.nan

    # Drop floating-rate and inflation-indexed bonds. Their published YTM is
    # not a fixed-coupon YTM, so they cannot price off a nominal zero curve;
    # left in, five FRBs would drag the 2-9 year segment with yields that mean
    # something else entirely. The remark column is FBIL's own flag.
    if "remark" in df.columns:
        flag = df["remark"].astype(str).str.lower()
        df = df[~flag.str.contains("|".join(FLOATER_REMARKS), na=False)]

    df = df.drop_duplicates(subset="isin", keep="first")
    d = to_date(valuation_date)
    df = df[pd.to_datetime(df["maturity"]).dt.date > d].reset_index(drop=True)
    return df[SCHEMAS["bonds"]]


def parse_tbills(path: Path) -> pd.DataFrame:
    """FBIL T-bill rates -> the frozen `tbills` schema (7D / 6M / 12M).

    The published file quotes whatever bills exist on the day (91D/182D/364D on
    some dates, 7D/6M/12M on others), so each schema tenor takes the rate of
    the nearest published bill.

    `tenor_years` is the schema's nominal figure (7/365, 0.5, 1.0), not the
    stand-in bill's own tenor. selection.py re-derives the tenor from the
    `tenor` label rather than reading this column, so quoting 364/365 here
    while the curve engine used 1.0 would make the CSV disagree with the curve
    built from it. Carrying a 364-day rate onto the 1.0y point is a timing
    simplification of at most a day, recorded in docs/assumptions.md.
    """
    raw = _read_any(Path(path))
    table = _apply_header(raw, _header_row(raw, TBILL_ALIASES))
    df = _map_columns(table, TBILL_ALIASES, ("tenor", "rate"), Path(path).name)
    df["rate"] = _to_number(df["rate"])
    df = df[df["rate"].notna()].reset_index(drop=True)

    from src.selection import parse_tenor

    df["published_years"] = [parse_tenor(v) for v in df["tenor"]]

    rows = []
    for label, years in TBILL_TENORS:
        idx = (df["published_years"] - years).abs().idxmin()
        rows.append({"tenor": label, "tenor_years": years, "rate": float(df.loc[idx, "rate"])})
    return pd.DataFrame(rows)[SCHEMAS["tbills"]]


def parse_zcyc(path: Path, kind: str) -> pd.DataFrame:
    """FBIL published ZCYC (+ par curve where there is one) -> the `fbil_zcyc` schema.

    The par columns are optional: FBIL publishes a G-Sec par curve but no SDL
    one. A missing par column becomes an empty column rather than a computed
    one -- the par yields implied by the published zeros are a derivation, not
    a publication, and this file is only ever read as "what FBIL said".
    """
    raw = _read_any(Path(path))
    table = _apply_header(raw, _header_row(raw, ZCYC_ALIASES))
    required = tuple(c for c in SCHEMAS[kind] if c not in OPTIONAL_ZCYC_COLUMNS)
    df = _map_columns(table, ZCYC_ALIASES, required, Path(path).name)
    for col in SCHEMAS[kind]:
        df[col] = _to_number(df[col]) if col in df.columns else np.nan
    df = df.dropna(subset=["tenor_years"]).sort_values("tenor_years").reset_index(drop=True)
    df["tenor_years"] = df["tenor_years"].round(2)
    return df[SCHEMAS[kind]]


def parse_sdl_bond(path: Path, valuation_date, face_stripped: float | None = None) -> pd.DataFrame:
    """SDL valuation file -> the one-row `sdl_bond` schema for the bonus.

    Picks the SDL that best demonstrates stripping: residual maturity inside
    the 14-year SDL ZCYC limit (the guidelines only allow the published curve
    to discount within its own published range), and among those the longest,
    since a longer parent has more remaining coupons and therefore more
    Coupon STRIPS on the slide.

    The two book-value scenarios are set to straddle market value so the
    min(book, market) normalisation can be shown binding each way. That is a
    presentational choice, not market data, and it is recorded as such in
    docs/assumptions.md.
    """
    raw = _read_any(Path(path))
    table = _apply_header(raw, _header_row(raw, SDL_BOND_ALIASES))
    df = _map_columns(table, SDL_BOND_ALIASES, ("isin", "coupon", "maturity", "market_price"), Path(path).name)

    df = df[df["isin"].notna()].copy()
    df["isin"] = df["isin"].astype(str).str.strip().str.upper()
    if "desc" not in df.columns:
        df["desc"] = ""
    df["desc"] = df["desc"].astype(str).str.strip()
    df["maturity"] = _to_iso_date(df["maturity"])
    df["coupon"] = _to_number(df["coupon"])
    df["market_price"] = _to_number(df["market_price"])
    df = df.dropna(subset=["coupon", "market_price"]).reset_index(drop=True)

    settle = settlement_date(valuation_date)
    residual = (pd.to_datetime(df["maturity"]).dt.date - settle).map(lambda td: td.days / 365.0)
    eligible = df[(residual > 1.0) & (residual <= SDL_MAX_RESIDUAL_YEARS)].copy()
    if eligible.empty:
        raise ValueError(
            "no SDL in the valuation file has a residual maturity in (1, "
            f"{SDL_MAX_RESIDUAL_YEARS}] years from settlement {settle}; the published SDL ZCYC "
            "does not extend past 14y, so a longer parent cannot be discounted"
        )
    row = eligible.loc[pd.to_datetime(eligible["maturity"]).idxmax()]

    face = float(face_stripped if face_stripped is not None else DEFAULT_FACE_STRIPPED)
    if face % CRORE != 0:
        raise ValueError(f"face_stripped {face:,.0f} must be a whole multiple of Rs 1 crore")

    coupon, maturity = float(row["coupon"]), row["maturity"]
    dirty_per_100 = float(row["market_price"]) + accrued_30_360(coupon, maturity, settle)
    market_value = dirty_per_100 * face / 100.0

    return pd.DataFrame(
        [
            {
                "isin": row["isin"],
                "desc": row["desc"],
                "coupon": coupon,
                "maturity": maturity,
                "market_price": float(row["market_price"]),
                "face_stripped": face,
                # Scenario 1 sits above market value (so market binds), scenario 2
                # below it (so book binds). +/-3% is far enough clear of the
                # normalisation to be unambiguous on a slide.
                "book_value_scenario_1": round(market_value * 1.03, 2),
                "book_value_scenario_2": round(market_value * 0.97, 2),
            }
        ]
    )[SCHEMAS["sdl_bond"]]


# ------------------------------------------------------------------ validation

def _check_grid(df: pd.DataFrame, name: str, grid: tuple[float, float, float]) -> None:
    """Assert the published grid's step and endpoints -- never its row count.

    The interface contract quotes a row count for the G-Sec grid that does not
    match its own step and endpoints (0.25 to 40.00 in 0.25 steps is 160 points,
    not 159). Checking the step and the two endpoints pins the same property
    without depending on which of the two numbers was the typo.
    """
    start, stop, step = grid
    t = df["tenor_years"].to_numpy(dtype=float)
    if len(t) < 2:
        raise ValueError(f"{name}: need at least two grid points, got {len(t)}")
    if np.any(np.diff(t) <= 0):
        raise ValueError(f"{name}: tenor_years must be strictly ascending")
    if not np.allclose(np.diff(t), step, atol=1e-9):
        offenders = np.unique(np.round(np.diff(t)[~np.isclose(np.diff(t), step, atol=1e-9)], 4))
        raise ValueError(f"{name}: tenor_years step must be {step}; found steps {list(offenders)}")
    if not (np.isclose(t[0], start) and np.isclose(t[-1], stop)):
        raise ValueError(f"{name}: grid must run {start} to {stop}, got {t[0]} to {t[-1]}")

    for col in ("zcy_semi", "zcy_annual", "par_semi", "par_annual"):
        if col in OPTIONAL_ZCYC_COLUMNS and df[col].isna().all():
            continue    # FBIL published no par curve for this benchmark.
        _check_percent(df[col], f"{name}.{col}")
    if (df["zcy_annual"] < df["zcy_semi"]).any():
        raise ValueError(f"{name}: annualised rate below the semiannual one -- columns look swapped")


def _check_percent(series: pd.Series, name: str) -> None:
    """Catch a rate column that somebody already divided by 100.

    A G-Sec zero rate of 0.0605 is not a 0.06% yield, it is 6.05% that has been
    converted to a decimal upstream. Feeding that to the curve engine produces a
    curve that is wrong by a factor of 100 and still fits, so this has to raise.
    """
    values = series.dropna().astype(float)
    if values.empty:
        raise ValueError(f"{name}: no values")
    lo, hi = PLAUSIBLE_RATE_RANGE
    if values.max() < 1.0:
        raise ValueError(
            f"{name}: every value is below 1.0 (max {values.max():.6f}). These look like decimals; "
            "this pipeline keeps rates in percent (6.05 means 6.05%)."
        )
    outside = values[(values < lo) | (values > hi)]
    if not outside.empty:
        raise ValueError(f"{name}: values outside a plausible [{lo}, {hi}] percent range: {list(outside[:5])}")


def validate_clean(frames: dict[str, pd.DataFrame], valuation_date) -> None:
    """Every check the loader enforces. Raises on the first failure, never warns."""
    for name in CLEAN_FILES:
        if name not in frames:
            raise ValueError(f"missing clean frame {name!r}")
        missing = [c for c in SCHEMAS[name] if c not in frames[name].columns]
        extra = [c for c in frames[name].columns if c not in SCHEMAS[name]]
        if missing or extra:
            raise ValueError(f"{name}.csv columns wrong: missing {missing}, unexpected {extra}")

    d = to_date(valuation_date)

    bonds = frames["bonds"]
    if bonds["isin"].duplicated().any():
        dupes = bonds.loc[bonds["isin"].duplicated(), "isin"].tolist()
        raise ValueError(f"bonds.csv: duplicate ISINs {dupes[:5]}")
    if bonds.empty:
        raise ValueError("bonds.csv: no bonds survived cleaning")
    matured = bonds[pd.to_datetime(bonds["maturity"]).dt.date <= d]
    if not matured.empty:
        raise ValueError(f"bonds.csv: {len(matured)} bonds mature on or before the valuation date {d}")
    _check_percent(bonds["coupon"], "bonds.coupon")
    # curve.py prices a bond off its `price` if there is one and falls back to
    # `ytm`, so either column may be empty -- but not both, and whichever is
    # present still has to be in the right units.
    if bonds["price"].isna().all() and bonds["ytm"].isna().all():
        raise ValueError("bonds.csv: every bond needs a price or a ytm; both columns are empty")
    if bonds["ytm"].notna().any():
        _check_percent(bonds["ytm"], "bonds.ytm")
    if bonds["price"].notna().any() and not bonds["price"].dropna().between(20, 200).all():
        raise ValueError("bonds.csv: prices are meant to be per 100 face; found values outside [20, 200]")

    tbills = frames["tbills"]
    if len(tbills) != len(TBILL_TENORS):
        raise ValueError(f"tbills.csv: expected {len(TBILL_TENORS)} rows, got {len(tbills)}")
    if list(tbills["tenor"]) != [label for label, _ in TBILL_TENORS]:
        raise ValueError(f"tbills.csv: tenor labels must be {[l for l, _ in TBILL_TENORS]}")
    _check_percent(tbills["rate"], "tbills.rate")

    _check_grid(frames["fbil_zcyc"], "fbil_zcyc.csv", ZCYC_GRID)
    _check_grid(frames["fbil_sdl_zcyc"], "fbil_sdl_zcyc.csv", SDL_ZCYC_GRID)

    sdl = frames["sdl_bond"]
    if len(sdl) != 1:
        raise ValueError(f"sdl_bond.csv: expected exactly one row, got {len(sdl)}")
    row = sdl.iloc[0]
    settle = settlement_date(d)
    residual = (to_date(row["maturity"]) - settle).days / 365.0
    if not 0 < residual <= SDL_MAX_RESIDUAL_YEARS:
        raise ValueError(
            f"sdl_bond.csv: residual maturity {residual:.2f}y is outside (0, {SDL_MAX_RESIDUAL_YEARS}]; "
            "the published SDL ZCYC stops at 14y"
        )
    if float(row["face_stripped"]) % CRORE != 0:
        raise ValueError("sdl_bond.csv: face_stripped must be a whole multiple of Rs 1 crore")
    _check_percent(sdl["coupon"], "sdl_bond.coupon")

    dirty = float(row["market_price"]) + accrued_30_360(float(row["coupon"]), row["maturity"], settle)
    market_value = dirty * float(row["face_stripped"]) / 100.0
    bv1, bv2 = float(row["book_value_scenario_1"]), float(row["book_value_scenario_2"])
    if not bv1 > market_value:
        raise ValueError(f"sdl_bond.csv: book_value_scenario_1 ({bv1:,.2f}) must exceed market value ({market_value:,.2f})")
    if not bv2 < market_value:
        raise ValueError(f"sdl_bond.csv: book_value_scenario_2 ({bv2:,.2f}) must be below market value ({market_value:,.2f})")


# ----------------------------------------------------------------- entry point

def build_clean_data(
    raw_dir: str = "data/raw",
    clean_dir: str = "data/clean",
    valuation_date: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Parse raw FBIL downloads, write the five data/clean/*.csv files, and
    return them keyed by name ('bonds', 'tbills', 'fbil_zcyc', 'fbil_sdl_zcyc',
    'sdl_bond').

    `valuation_date` is the FBIL business day the files were published for. It
    is not read out of the files themselves: FBIL stamps the date in a title
    row whose position and wording change between formats, and guessing it
    wrong would silently shift every cash flow time. Pass it explicitly.
    """
    if valuation_date is None:
        raise ValueError(
            "valuation_date is required -- pass the FBIL business day the raw files "
            "were published for, as 'YYYY-MM-DD'"
        )
    valuation_date = to_date(valuation_date).isoformat()

    sources = discover_raw_files(raw_dir)
    frames = {
        "bonds": parse_bonds(sources["bonds"], valuation_date),
        "tbills": parse_tbills(sources["tbills"]),
        "fbil_zcyc": parse_zcyc(sources["fbil_zcyc"], "fbil_zcyc"),
        "fbil_sdl_zcyc": parse_zcyc(sources["fbil_sdl_zcyc"], "fbil_sdl_zcyc"),
        "sdl_bond": parse_sdl_bond(sources["sdl_bond"], valuation_date),
    }
    validate_clean(frames, valuation_date)

    clean_path = Path(clean_dir)
    clean_path.mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        df.to_csv(clean_path / f"{name}.csv", index=False)
    _write_source_note(clean_path, sources, valuation_date, len(frames["bonds"]))
    return frames


def _write_source_note(clean_dir: Path, sources: dict[str, Path], valuation_date: str, n_bonds: int) -> None:
    """Record which raw file produced each clean CSV.

    Without this there is no way to tell a clean CSV built from a real FBIL
    download from one built from the sample, and they look identical. Every
    consumer that cares -- the comparison charts, the slide notes -- reads the
    provenance from here.
    """
    sampled = sorted(role for role, path in sources.items() if is_sample(path))
    lines = [
        "# data/clean provenance",
        "",
        "Written by `src.data_loader.build_clean_data`. Do not edit by hand.",
        "",
        f"- Valuation date: **{valuation_date}**",
        f"- Settlement date (T+1): **{settlement_date(valuation_date).isoformat()}**",
        f"- Bonds after cleaning: {n_bonds}",
        f"- Provenance: **{'SAMPLE' if sampled else 'FBIL'}**",
        "",
        "| Clean file | Built from | Real FBIL download? |",
        "|---|---|---|",
    ]
    for role in CLEAN_FILES:
        path = sources[role]
        lines.append(f"| `{role}.csv` | `{path.as_posix()}` | {'no -- SAMPLE' if is_sample(path) else 'yes'} |")
    if sampled:
        lines += [
            "",
            "> **These CSVs are not market data.** The roles marked SAMPLE above were built by",
            "> `src.data_loader.write_sample_raw()` so the pipeline could be developed and tested",
            "> before the FBIL download arrived. Replace the `*_SAMPLE.*` files in `data/raw/` with",
            "> the real downloads and re-run the loader before submission; no number produced from",
            "> these files may be quoted as a result.",
        ]
    (clean_dir / SOURCE_FILE).write_text("\n".join(lines) + "\n")


def clean_data_provenance(clean_dir: str | Path = "data/clean") -> str:
    """'FBIL', 'SAMPLE', or 'UNKNOWN' -- what the current clean CSVs came from."""
    note = Path(clean_dir) / SOURCE_FILE
    if not note.exists():
        return "UNKNOWN"
    text = note.read_text()
    match = re.search(r"Provenance: \*\*(\w+)\*\*", text)
    return match.group(1) if match else "UNKNOWN"


# ----------------------------------------------------------- sample generation

def write_sample_raw(raw_dir: str | Path = "data/raw", valuation_date: str = "2026-09-18") -> dict[str, Path]:
    """Write a `*_SAMPLE` stand-in for each raw FBIL download.

    This exists so the loader, the comparison and the ablation grid can be
    written and tested before the FBIL files are in hand, and so the shapes the
    parsers must cope with (a title row above the header, Indian-format dates,
    comma-separated amounts, percent signs) are exercised by the test suite.

    The numbers come from a smooth parametric curve, not from any market. Every
    file it writes ends in `_SAMPLE` and `data/clean/SOURCE.md` records the
    substitution, because the one genuinely unacceptable outcome for this
    project is submitting invented data as though it were FBIL's.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    d = to_date(valuation_date)
    settle = settlement_date(d)
    rng = np.random.default_rng(20260918)

    def zero_semi(t):
        """Smooth upward-sloping curve, ~5.4% at the short end to ~7.4% at 40y."""
        t = np.asarray(t, dtype=float)
        return 5.40 + 1.75 * (1.0 - np.exp(-t / 5.0)) + 0.25 * (1.0 - np.exp(-t / 22.0))

    def df_of(t):
        return (1.0 + zero_semi(t) / 200.0) ** (-2.0 * np.asarray(t, dtype=float))

    def annualise_pct(z):
        return ((1.0 + np.asarray(z, dtype=float) / 200.0) ** 2 - 1.0) * 100.0

    def par_semi(T):
        times = np.arange(T, 1e-9, -0.5)[::-1]
        dfs = df_of(times)
        accrual = np.full(len(times), 0.5)
        accrual[0] = times[0]
        return (1.0 - df_of(T)) / float(np.sum(accrual * dfs)) * 100.0

    paths: dict[str, Path] = {}

    # --- G-Sec valuation file: xlsx with two preamble rows, as FBIL ships it.
    rows = []
    for i, years in enumerate(np.concatenate([np.arange(0.3, 1.0, 0.15), np.arange(1.05, 39.0, 0.42)])):
        maturity = pd.Timestamp(settle) + pd.Timedelta(days=int(round(365.25 * years)))
        t = (maturity.date() - settle).days / 365.0
        coupon = round(float(zero_semi(max(t, 0.5))) + float(rng.normal(0, 0.35)), 2)
        # Price off the true curve, then perturb the yield a touch: real quotes
        # never sit exactly on any single smooth curve, and a perfectly
        # consistent input set would make the ablation grid meaningless.
        from src.bond import cashflows, dirty_price, ytm_from_price

        cf = cashflows(coupon, maturity.date(), settle)
        dirty = float((cf["cf"] * df_of(cf["t"].to_numpy())).sum())
        ytm = ytm_from_price(coupon, maturity.date(), settle, dirty, price_type="dirty")
        ytm += float(rng.normal(0, 0.012))
        clean = dirty_price(coupon, maturity.date(), settle, ytm) - accrued_30_360(coupon, maturity.date(), settle)
        rows.append(
            {
                "ISIN": f"IN00200{i:05d}",
                "Security Description": f"{coupon:.2f}% GS {maturity:%Y}",
                "Coupon Rate": coupon,
                "Maturity Date": maturity.strftime("%d/%m/%Y"),
                "Price": round(clean, 4),
                "YTM": round(ytm, 4),
                "Traded Volume": float(rng.choice([0, 0, 5e6, 2.5e7, 1e8, 5e8])),
            }
        )
    paths["bonds"] = _write_with_preamble(
        pd.DataFrame(rows), raw_dir / "gsec_bonds_SAMPLE.xlsx", "G-Sec Valuation (SAMPLE, not FBIL data)", d
    )

    # --- T-bills: csv, quoted as FBIL does with 91D/182D/364D labels.
    tbills = pd.DataFrame(
        {
            "Tenor": ["7D", "91D", "182D", "364D"],
            "Rate (%)": [f"{zero_semi(t):.4f}" for t in (7 / 365, 0.25, 0.5, 1.0)],
        }
    )
    paths["tbills"] = raw_dir / "fbil_tbills_SAMPLE.csv"
    tbills.to_csv(paths["tbills"], index=False)

    # --- Published ZCYC files. FBIL's own curve is the sample's true curve
    # shifted by a small smooth term structure, so comparing our bootstrap
    # against it produces a non-zero but realistic difference profile.
    def zcyc_frame(stop: float, spread):
        t = np.round(np.arange(0.25, stop + 1e-9, 0.25), 2)
        z = zero_semi(t) + spread(t)
        return pd.DataFrame(
            {
                "Tenor (Yrs)": t,
                "ZCYC (Semi-Annual)": np.round(z, 4),
                "ZCYC (Annualised)": np.round(annualise_pct(z), 4),
                "Par (Semi-Annual)": np.round([par_semi(float(x)) + float(spread(x)) for x in t], 4),
                "Par (Annualised)": np.round(annualise_pct([par_semi(float(x)) + float(spread(x)) for x in t]), 4),
            }
        )

    paths["fbil_zcyc"] = raw_dir / "fbil_gsec_zcyc_SAMPLE.csv"
    zcyc_frame(ZCYC_GRID[1], lambda t: 0.02 * np.sin(np.asarray(t, dtype=float) / 6.0)).to_csv(
        paths["fbil_zcyc"], index=False
    )

    paths["fbil_sdl_zcyc"] = raw_dir / "fbil_sdl_zcyc_SAMPLE.csv"
    # SDLs trade at a spread over the sovereign curve; ~40bp widening with tenor.
    zcyc_frame(SDL_ZCYC_GRID[1], lambda t: 0.30 + 0.012 * np.asarray(t, dtype=float)).to_csv(
        paths["fbil_sdl_zcyc"], index=False
    )

    # --- SDL valuation file, in html this time so that reader is exercised too.
    sdl_rows = []
    for i, years in enumerate((3.4, 6.2, 8.7, 11.3, 13.4, 17.0)):
        maturity = pd.Timestamp(settle) + pd.Timedelta(days=int(round(365.25 * years)))
        t = (maturity.date() - settle).days / 365.0
        coupon = round(float(zero_semi(t)) + 0.45, 2)
        from src.bond import cashflows as _cashflows

        cf = _cashflows(coupon, maturity.date(), settle)
        sdl_df = df_of(cf["t"].to_numpy()) * np.exp(-0.004 * cf["t"].to_numpy())
        dirty = float((cf["cf"] * sdl_df).sum())
        sdl_rows.append(
            {
                "ISIN": f"IN19200{i:05d}",
                "Security Description": f"{coupon:.2f}% Maharashtra SDL {maturity:%Y}",
                "Coupon Rate": coupon,
                "Maturity Date": maturity.strftime("%d/%m/%Y"),
                "Price": round(dirty - accrued_30_360(coupon, maturity.date(), settle), 4),
            }
        )
    paths["sdl_bond"] = raw_dir / "fbil_sdl_valuation_SAMPLE.html"
    paths["sdl_bond"].write_text(
        f"<html><body><h2>SDL Valuation (SAMPLE, not FBIL data) as on {d:%d/%m/%Y}</h2>"
        + pd.DataFrame(sdl_rows).to_html(index=False)
        + "</body></html>"
    )
    return paths


def _write_with_preamble(df: pd.DataFrame, path: Path, title: str, d) -> Path:
    """Write an xlsx with two title rows above the header, like FBIL's files."""
    preamble = pd.DataFrame([[title] + [None] * (len(df.columns) - 1),
                             [f"As on {d:%d/%m/%Y}"] + [None] * (len(df.columns) - 1)])
    body = pd.concat([preamble, pd.DataFrame([df.columns.tolist()]), pd.DataFrame(df.values)], ignore_index=True)
    body.to_excel(path, index=False, header=False)
    return path


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build data/clean/*.csv from the raw FBIL downloads.")
    parser.add_argument("--date", required=True, help="Valuation date the raw files were published for, YYYY-MM-DD")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--clean-dir", default="data/clean")
    parser.add_argument(
        "--write-sample",
        action="store_true",
        help="First write *_SAMPLE stand-ins into --raw-dir. Development only; never for submission.",
    )
    args = parser.parse_args(argv)

    if args.write_sample:
        write_sample_raw(args.raw_dir, args.date)

    frames = build_clean_data(args.raw_dir, args.clean_dir, args.date)
    provenance = clean_data_provenance(args.clean_dir)
    for name in CLEAN_FILES:
        print(f"{name+'.csv':<20} {len(frames[name]):>5} rows")
    print(f"\nprovenance: {provenance} (see {args.clean_dir}/{SOURCE_FILE})")
    if provenance != "FBIL":
        print("WARNING: at least one clean CSV is built from SAMPLE data and must not be submitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
