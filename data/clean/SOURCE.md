# data/clean provenance

Written by `src.data_loader.build_clean_data`. Do not edit by hand.

- Valuation date: **2026-09-18**
- Settlement date (T+1): **2026-09-21**
- Bonds after cleaning: 96
- Provenance: **SAMPLE**

| Clean file | Built from | Real FBIL download? |
|---|---|---|
| `bonds.csv` | `data/raw/gsec_bonds_SAMPLE.xlsx` | no -- SAMPLE |
| `tbills.csv` | `data/raw/fbil_tbills_SAMPLE.csv` | no -- SAMPLE |
| `fbil_zcyc.csv` | `data/raw/fbil_gsec_zcyc_SAMPLE.csv` | no -- SAMPLE |
| `fbil_sdl_zcyc.csv` | `data/raw/fbil_sdl_zcyc_SAMPLE.csv` | no -- SAMPLE |
| `sdl_bond.csv` | `data/raw/fbil_sdl_valuation_SAMPLE.html` | no -- SAMPLE |

> **These CSVs are not market data.** The roles marked SAMPLE above were built by
> `src.data_loader.write_sample_raw()` so the pipeline could be developed and tested
> before the FBIL download arrived. Replace the `*_SAMPLE.*` files in `data/raw/` with
> the real downloads and re-run the loader before submission; no number produced from
> these files may be quoted as a result.
