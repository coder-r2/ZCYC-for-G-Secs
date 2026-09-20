"""Tests for A1 -- the raw -> data/clean loader.

Two halves:

* Round-trip tests build a full set of raw files in a tmp directory, run
  ``build_clean_data`` over them, read the CSVs back off disk and assert the
  frozen schema. Reading back off disk matters: every downstream consumer gets
  the CSV, not the in-memory frame, so a dtype that only survives in memory is
  not a pass.
* Validation tests corrupt one thing at a time and assert the loader raises.
  The loader's job is to make a bad download impossible to use silently; a
  check that only warns is the same as no check.

The committed ``data/clean/*.csv`` are also checked, so whatever is actually in
the repo has to satisfy the same schema as the freshly-built files.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.bond import settlement_date, to_date  # noqa: E402
from src.data_loader import (  # noqa: E402
    CLEAN_FILES,
    CRORE,
    SCHEMAS,
    SDL_MAX_RESIDUAL_YEARS,
    TBILL_TENORS,
    build_clean_data,
    discover_raw_files,
    parse_bonds,
    validate_clean,
    write_sample_raw,
)

VALUATION_DATE = "2026-09-18"
CLEAN_DIR = REPO_ROOT / "data" / "clean"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """A complete raw -> clean run in a tmp directory."""
    root = tmp_path_factory.mktemp("loader")
    raw, clean = root / "raw", root / "clean"
    write_sample_raw(raw, VALUATION_DATE)
    frames = build_clean_data(str(raw), str(clean), VALUATION_DATE)
    return {"raw": raw, "clean": clean, "frames": frames}


@pytest.fixture(scope="module")
def on_disk(built):
    """The five CSVs as a downstream consumer actually reads them."""
    return {name: pd.read_csv(built["clean"] / f"{name}.csv") for name in CLEAN_FILES}


# ------------------------------------------------------------------- discovery

def test_discovery_finds_all_five_roles(built):
    found = discover_raw_files(built["raw"])
    assert set(found) == set(CLEAN_FILES)


def test_discovery_prefers_a_real_download_over_a_sample(built, tmp_path):
    """Dropping the real file in is enough -- no flag to remember to flip."""
    raw = tmp_path / "raw"
    raw.mkdir()
    for path in Path(built["raw"]).iterdir():
        (raw / path.name).write_bytes(path.read_bytes())
    real = raw / "fbil_tbills_2026-09-18.csv"
    real.write_bytes((raw / "fbil_tbills_SAMPLE.csv").read_bytes())

    assert discover_raw_files(raw)["tbills"] == real


def test_discovery_raises_when_a_role_is_missing(tmp_path):
    (tmp_path / "gsec_bonds_SAMPLE.csv").write_text("isin,coupon\n")
    with pytest.raises(FileNotFoundError, match="tbills"):
        discover_raw_files(tmp_path)


# ---------------------------------------------------------------------- schema

@pytest.mark.parametrize("name", CLEAN_FILES)
def test_columns_are_exactly_the_frozen_schema(on_disk, name):
    assert list(on_disk[name].columns) == SCHEMAS[name]


@pytest.mark.parametrize("name", CLEAN_FILES)
def test_numeric_columns_are_numeric_after_a_csv_round_trip(on_disk, name):
    text_columns = {"isin", "desc", "tenor", "maturity"}
    for col in SCHEMAS[name]:
        if col in text_columns:
            continue
        assert pd.api.types.is_numeric_dtype(on_disk[name][col]), f"{name}.{col} is not numeric"


def test_dates_are_iso_strings(on_disk):
    for name, col in (("bonds", "maturity"), ("sdl_bond", "maturity")):
        values = on_disk[name][col].astype(str)
        assert values.str.match(r"^\d{4}-\d{2}-\d{2}$").all(), f"{name}.{col} is not ISO YYYY-MM-DD"


def test_volume_column_exists_even_when_the_raw_file_has_none(tmp_path):
    """selection.py tolerates NaN volumes but not a missing column."""
    path = tmp_path / "gsec_bonds_SAMPLE.csv"
    path.write_text(
        "ISIN,Security Description,Coupon Rate,Maturity Date,Price,YTM\n"
        "IN0020240001,7.10% GS 2034,7.10,15/04/2034,101.2,6.95\n"
    )
    bonds = parse_bonds(path, VALUATION_DATE)
    assert "volume" in bonds.columns
    assert bonds["volume"].isna().all()


# ----------------------------------------------------------------------- bonds

def test_no_duplicate_isins(on_disk):
    assert not on_disk["bonds"]["isin"].duplicated().any()


def test_no_bond_matures_on_or_before_the_valuation_date(on_disk):
    maturities = pd.to_datetime(on_disk["bonds"]["maturity"]).dt.date
    assert (maturities > to_date(VALUATION_DATE)).all()


def test_coupon_and_ytm_are_plausible_percentages(on_disk):
    bonds = on_disk["bonds"]
    for col in ("coupon", "ytm"):
        values = bonds[col].dropna()
        assert values.between(0, 20).all(), f"bonds.{col} outside [0, 20] percent"
        # A column of decimals would sit entirely below 1.0 -- the single
        # failure mode that produces a curve wrong by a factor of 100.
        assert values.max() > 1.0, f"bonds.{col} looks like decimals, not percent"


def test_prices_are_per_100_face(on_disk):
    assert on_disk["bonds"]["price"].dropna().between(20, 200).all()


def test_cleaning_does_not_drop_any_live_isin(built, on_disk):
    """The only rows the loader may remove are duplicates and matured bonds."""
    raw = pd.read_excel(built["raw"] / "gsec_bonds_SAMPLE.xlsx", header=2)
    live = raw[pd.to_datetime(raw["Maturity Date"], dayfirst=True).dt.date > to_date(VALUATION_DATE)]
    assert len(on_disk["bonds"]) == live["ISIN"].nunique()
    assert set(on_disk["bonds"]["isin"]) == set(live["ISIN"].astype(str).str.upper())


# ---------------------------------------------------------------------- tbills

def test_tbills_carry_the_three_schema_tenors(on_disk):
    tbills = on_disk["tbills"]
    assert list(tbills["tenor"]) == [label for label, _ in TBILL_TENORS]
    assert np.allclose(tbills["tenor_years"], [years for _, years in TBILL_TENORS])


def test_tbill_rates_are_percentages(on_disk):
    assert on_disk["tbills"]["rate"].between(0, 20).all()


def test_tbills_take_the_nearest_published_bill(built):
    """The sample file quotes 91D/182D/364D; 6M must come from the 182D bill."""
    published = pd.read_csv(built["raw"] / "fbil_tbills_SAMPLE.csv")
    rate_182 = float(published.loc[published["Tenor"] == "182D", "Rate (%)"].iloc[0])
    tbills = pd.read_csv(built["clean"] / "tbills.csv")
    assert float(tbills.loc[tbills["tenor"] == "6M", "rate"].iloc[0]) == pytest.approx(rate_182)


# ------------------------------------------------------------------ zcyc grids

@pytest.mark.parametrize("name,stop", [("fbil_zcyc", 40.0), ("fbil_sdl_zcyc", 14.0)])
def test_published_grid_is_a_clean_quarter_year_ladder(on_disk, name, stop):
    """Step and endpoints are asserted; the row count is not hard-coded.

    The interface contract quotes 159 rows for a 0.25-to-40.00 grid in 0.25
    steps, which is 160 points. Pinning the step and the two endpoints tests the
    same property without depending on which number was the typo.
    """
    t = on_disk[name]["tenor_years"].to_numpy(dtype=float)
    assert t[0] == pytest.approx(0.25)
    assert t[-1] == pytest.approx(stop)
    assert np.all(np.diff(t) > 0)
    assert np.allclose(np.diff(t), 0.25)
    assert len(t) == round(stop / 0.25)


@pytest.mark.parametrize("name", ["fbil_zcyc", "fbil_sdl_zcyc"])
def test_published_rates_are_percentages_and_annual_exceeds_semi(on_disk, name):
    df = on_disk[name]
    for col in ("zcy_semi", "zcy_annual", "par_semi", "par_annual"):
        assert df[col].between(0, 20).all(), f"{name}.{col} outside [0, 20] percent"
        assert df[col].max() > 1.0, f"{name}.{col} looks like decimals, not percent"
    assert (df["zcy_annual"] >= df["zcy_semi"]).all()


def test_sdl_curve_sits_above_the_sovereign_curve(on_disk):
    """Sanity check that the two ZCYC files did not get swapped."""
    gsec = on_disk["fbil_zcyc"].set_index("tenor_years")["zcy_semi"]
    sdl = on_disk["fbil_sdl_zcyc"].set_index("tenor_years")["zcy_semi"]
    assert (sdl - gsec.loc[sdl.index] > 0).all()


# -------------------------------------------------------------------- sdl bond

def test_sdl_bond_is_a_single_eligible_row(on_disk):
    sdl = on_disk["sdl_bond"]
    assert len(sdl) == 1
    settle = settlement_date(VALUATION_DATE)
    residual = (to_date(sdl.iloc[0]["maturity"]) - settle).days / 365.0
    assert 0 < residual <= SDL_MAX_RESIDUAL_YEARS


def test_face_stripped_is_a_whole_number_of_crore(on_disk):
    assert float(on_disk["sdl_bond"].iloc[0]["face_stripped"]) % CRORE == 0


def test_book_value_scenarios_straddle_market_value(on_disk):
    """C's min(BV, MV) demonstration needs one scenario to bind each way."""
    from src.bond import accrued_30_360

    row = on_disk["sdl_bond"].iloc[0]
    settle = settlement_date(VALUATION_DATE)
    dirty = float(row["market_price"]) + accrued_30_360(float(row["coupon"]), row["maturity"], settle)
    market_value = dirty * float(row["face_stripped"]) / 100.0
    assert float(row["book_value_scenario_1"]) > market_value
    assert float(row["book_value_scenario_2"]) < market_value


# ------------------------------------------------------------------ validation

def _frames(built):
    return {name: df.copy() for name, df in built["frames"].items()}


def test_validation_rejects_rates_already_converted_to_decimals(built):
    frames = _frames(built)
    frames["bonds"]["ytm"] = frames["bonds"]["ytm"] / 100.0
    with pytest.raises(ValueError, match="decimals"):
        validate_clean(frames, VALUATION_DATE)


def test_validation_rejects_duplicate_isins(built):
    frames = _frames(built)
    frames["bonds"] = pd.concat([frames["bonds"], frames["bonds"].head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate ISIN"):
        validate_clean(frames, VALUATION_DATE)


def test_validation_rejects_a_gap_in_the_published_grid(built):
    frames = _frames(built)
    frames["fbil_zcyc"] = frames["fbil_zcyc"].drop(index=10).reset_index(drop=True)
    with pytest.raises(ValueError, match="step"):
        validate_clean(frames, VALUATION_DATE)


def test_validation_rejects_a_renamed_column(built):
    frames = _frames(built)
    frames["bonds"] = frames["bonds"].rename(columns={"ytm": "yield"})
    with pytest.raises(ValueError, match="columns wrong"):
        validate_clean(frames, VALUATION_DATE)


def test_validation_rejects_an_sdl_beyond_the_published_curve(built):
    frames = _frames(built)
    frames["sdl_bond"].loc[0, "maturity"] = "2045-06-30"
    with pytest.raises(ValueError, match="residual maturity"):
        validate_clean(frames, VALUATION_DATE)


def test_validation_rejects_book_values_that_do_not_straddle_market_value(built):
    frames = _frames(built)
    frames["sdl_bond"].loc[0, "book_value_scenario_1"] = 1.0
    with pytest.raises(ValueError, match="book_value_scenario_1"):
        validate_clean(frames, VALUATION_DATE)


# ------------------------------------------- the CSVs actually committed to git

@pytest.mark.skipif(not (CLEAN_DIR / "bonds.csv").exists(), reason="data/clean not built yet")
@pytest.mark.parametrize("name", CLEAN_FILES)
def test_committed_clean_csvs_match_the_frozen_schema(name):
    df = pd.read_csv(CLEAN_DIR / f"{name}.csv")
    assert list(df.columns) == SCHEMAS[name]
    assert len(df) > 0


@pytest.mark.skipif(not (CLEAN_DIR / "bonds.csv").exists(), reason="data/clean not built yet")
def test_committed_clean_csvs_pass_every_validation_rule():
    frames = {name: pd.read_csv(CLEAN_DIR / f"{name}.csv") for name in CLEAN_FILES}
    validate_clean(frames, VALUATION_DATE)
