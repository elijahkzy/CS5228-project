# CS5228 HDB rental feature pipeline

## Purpose

The feature pipeline converts the raw rental records and collected auxiliary sources into two aligned model-training tables:

- `<data root>/processed/train_features.parquet`: 150,000 rows, predictors, identifiers, and `MONTHLY_RENT`.
- `<data root>/processed/test_features.parquet`: 50,000 rows with the same predictors and identifiers, without the target.

The pipeline preserves the original row order through `ROW_ID`. It does not combine training and test records into one modeling file because the target is available only for training and the Kaggle submission must retain test order.

## Build command

From the project root:

```powershell
.\.venv\Scripts\python.exe .\src\download_macro_data.py
.\.venv\Scripts\python.exe .\src\build_features.py
```

These commands automatically detect the adjacent `CS5228 Project Data/data`
folder. For a Google Drive copy in another location, set
`CS5228_DATA_ROOT` or pass `--data-root`:

```powershell
.\.venv\Scripts\python.exe .\src\build_features.py `
  --data-root "G:\My Drive\CS5228 Project Data\data"
```

The first command refreshes the selected official Singapore macroeconomic
series and constructs their point-in-time monthly panel. See
`docs/MACRO_FEATURES.md` for source provenance, availability assumptions, and
fixed-origin forecasting guidance.

The script can also be run from another directory:

```powershell
python path\to\CS5228 Project\src\build_features.py `
  --project-root "path\to\CS5228 Project" `
  --data-root "path\to\CS5228 Project Data\data"
```

The build creates:

```text
<data root>/processed/
├── train_features.parquet
├── test_features.parquet
├── feature_dictionary.csv
└── feature_build_report.json
```

`feature_dictionary.csv` is the authoritative column list. `feature_build_report.json` records row counts, date ranges, join coverage, historical-feature coverage, and column-level missingness.

## Processing sequence

1. Read `<data root>/train.csv` and `<data root>/test.csv` without changing row order.
2. Normalize text case, whitespace, punctuation, street abbreviations, and flat-type labels.
3. Parse monthly approval dates and numeric property fields.
4. Remove `FURNISHED` and `FEE`, which are constant in the supplied release.
5. Create time, seasonality, lease-age, and address-key features.
6. Join HDB block attributes by normalized town, block, and street.
7. Derive nearest-station, MRT-network, station-ridership, mall, and school features at block level.
8. Join planning-area and subzone demographics.
9. Aggregate COE and stock data to monthly one-period-lagged indicators.
10. Join six official Singapore macro and housing-market indicators using only
    observations assumed to have been published before each rental month.
11. Calculate address, town-flat-type, and flat-type rent histories using strictly earlier months.
12. Restore original train and test order and validate row counts and schema parity.
13. Save typed Parquet tables, a feature dictionary, and a build report.

## Final feature groups

### Identifiers and source fields

| Feature | Use |
|---|---|
| `ROW_ID` | Preserves original train/test and Kaggle submission order. Do not use as a predictor. |
| `ADDRESS_KEY` | Normalized town, block, and street key used for joins and history. It may be used as a CatBoost category, but can encourage memorization. |
| `BLOCK`, `STREET`, `POSTAL_CODE` | High-cardinality identifiers. Prefer CatBoost categorical handling or leakage-safe historical aggregates. |
| `TOWN`, `FLAT_TYPE`, `FLAT_MODEL` | Core categorical property features. |

### Core property and time features

| Features | Description |
|---|---|
| `FLOOR_AREA_SQM` | Flat floor area. |
| `LEASE_COMMENCE_DATE`, `LEASE_AGE`, `REMAINING_LEASE_PROXY` | Lease timing and approximate remaining lease. |
| `RENT_YEAR`, `RENT_MONTH`, `RENT_QUARTER`, `MONTH_INDEX` | Calendar and long-term trend variables. |
| `MONTH_SIN`, `MONTH_COS` | Cyclical month-of-year representation for seasonality. |

`MONTH_INDEX` is important because rent changes strongly over time. Tree models do not extrapolate trends reliably beyond the training range. Compare them with a linear trend, GAM, or a two-stage market-index model.

### HDB block and geography

| Features | Description |
|---|---|
| `LATITUDE`, `LONGITUDE` | Block coordinates. |
| `MAX_FLOOR`, `YEAR_COMPLETED`, `BUILDING_AGE` | Physical block attributes. |
| `SUBZONE`, `PLANNING_AREA`, `REGION` | Nested geographic categories. |
| `HDB_BLOCK_MATCHED` | Indicates whether the address matched the block reference table. |

The original HDB file uses `-1` for some missing maximum-floor and completion-year values. The pipeline converts these values to missing rather than treating them as real measurements.

### MRT accessibility

The pipeline selects the nearest station whose current status is `open` and creates:

- nearest-station distance and station name;
- station and line counts within 0.5, 0.8, 1, and 2 kilometres;
- line count and interchange status for the nearest station;
- rail-network distance from the nearest station to Raffles Place, Orchard, Jurong East, and Woodlands;
- mean May–July 2026 passenger-volume statistics for the nearest station;
- passenger-volume coverage and missingness indicators.

Rail-network distances use the supplied weighted graph. Some edges use haversine fallback distances where OpenStreetMap rail geometry was unavailable.

### Mall accessibility

Features include:

- distance and name of the nearest mall;
- mall counts within 0.5, 1, and 2 kilometres;
- the nearest mall's Wikipedia-derived floor area, number of stores, floors, anchors, and opening year;
- summed known mall floor area and store counts within 2 kilometres;
- Wikipedia status and missing-detail indicator.

Wikipedia coverage is incomplete. A missing mall-size value means that the attribute was unavailable, not that the mall has zero floor area or stores. The sum features represent known values only and should be interpreted with the missingness indicator.

### School accessibility

Features include:

- distances to the nearest primary, secondary, and junior-college records;
- school counts within relevant distance bands;
- most selective available secondary `pg3_max` within 2 kilometres;
- most selective available JC `L1R5` maximum within 4 kilometres;
- count of high-fee or independent secondary schools within 2 kilometres.

Lower entry-score cutoffs generally indicate greater selectivity, but PSLE and L1R5 fields are different measures and must not be combined directly. Primary schools have no academic entry cutoff. Some specialized schools legitimately have no cutoff.

### Demographics

Planning-area features use the `PA_` prefix and include population, age structure, unemployment, household counts, housing mix, income bands, household size, ethnicity, owner occupation, and PMET share.

Subzone features use the `SZ_` prefix and include population, age structure, and sex composition. `PA_DEMOGRAPHICS_MATCHED` and `SZ_DEMOGRAPHICS_MATCHED` expose join coverage.

These are 2020 census snapshots. They describe neighbourhood structure but do not vary over rental months. Planning-area economic fields may be missing for small or non-residential areas.

### Lagged market indicators

The feature table contains one-month-lagged aggregates:

- mean and maximum COE premium;
- COE bid-to-quota ratio;
- equal-weighted stock return;
- stock-return volatility.

These variables may act mainly as proxies for date. Keep them only when they improve several forward validation folds. The first available month is missing by design because no earlier month exists in the supplied data.

### Official macroeconomic indicators

The compact official macro panel contains:

- CPI excluding accommodation, expressed as year-over-year inflation;
- the 3-month compounded SORA level;
- real GDP year-over-year growth;
- the seasonally adjusted total unemployment rate;
- HDB resale price-index year-over-year growth; and
- private residential rental-index year-over-year growth.

These variables were selected for distinct economic channels: general cost
pressure, financing conditions, economic activity, labour demand, the HDB
rent-versus-buy alternative, and broader rental-market pressure. They are joined
by month using conservative availability dates, not by matching the observation
period directly. `<data root>/macro/sg_macro_observations.csv` records the source period
and availability assumption for every observation.

Use these features as one ablation group. With only about 51 distinct training
months, adding many related macro series would create more apparent predictors
without adding comparable independent time-series evidence.

### Historical rental features

The pipeline produces prior median, last observed rent, number of prior months, and months since the latest observation for:

- exact address;
- town and flat type;
- flat type across Singapore.

For a record in month `t`, only monthly rental medians from months strictly before `t` are eligible. Records from the same month never supply one another's labels.

These are the most leakage-sensitive fields. They are safe for chronological validation as materialized, but not for random cross-validation. A random fold can allow a later row, whose history contains an earlier validation label, into the training fold. Use rolling or forward time splits only.

## Recommended model inputs

### CatBoost

Pass cleaned categories directly, including town, flat type, flat model, subzone, planning area, region, and optionally address or nearest-station name. Exclude `ROW_ID`. Let CatBoost handle numeric missing values, but retain explicit missingness indicators where absence has meaning.

### LightGBM or XGBoost

Encode low-cardinality categories consistently. Frequency encoding or leakage-safe encoding is preferable to large one-hot matrices for block, street, and address. Fit encoders on each training fold only.

### Linear model or GAM

One-hot encode moderate-cardinality categories and standardize continuous variables. Exclude high-cardinality identifiers unless regularization is strong. GAMs are useful for nonlinear time, area, age, and distance effects.

## Example loading code

```python
from pathlib import Path
import sys
import pandas as pd

project_root = Path.cwd()
sys.path.insert(0, str(project_root / "src"))
from data_paths import resolve_data_root

processed = resolve_data_root(project_root) / "processed"
train = pd.read_parquet(processed / "train_features.parquet")
test = pd.read_parquet(processed / "test_features.parquet")

target = "MONTHLY_RENT"
non_predictors = ["ROW_ID", "RENT_APPROVAL_DATE"]
features = [c for c in test.columns if c not in non_predictors]

X = train[features]
y = train[target]
X_test = test[features]
```

Use `ROW_ID` when preparing the submission:

```python
submission = pd.DataFrame({
    "Id": test["ROW_ID"],
    "Predicted": predictions,
}).sort_values("Id")
submission.to_csv("submission.csv", index=False)
```

## Validation requirements

Use forward or rolling-origin validation. A recommended initial holdout is the final 12 training months. Report performance and prediction bias by month because the Kaggle test period extends further into the future.

For stronger evidence, use several folds such as:

1. train through March 2023 and validate the next six months;
2. train through September 2023 and validate the next six months;
3. train through March 2024 and validate the next twelve months.

Do not select features based only on a random split or public leaderboard changes.

## Missingness and modeling caveats

### Do not convert every missing value to zero

Zero has a real meaning for facility counts. It does not mean the same thing as unknown mall size, unavailable demographics, or missing ridership. Use median imputation, native missing-value handling, or explicit unknown categories as appropriate.

### Snapshot data and temporal mismatch

Several sources are current snapshots applied to earlier rentals:

- May–July 2026 station ridership;
- current station status and network;
- current mall and school attributes;
- 2020 demographics.

They should be described as stable location proxies, not contemporaneous historical measurements. Station opening dates would be needed to reconstruct historically accurate accessibility.

### Target leakage

Historical rents, rental indices, and target encodings must only use earlier months. If a new validation design is introduced, rebuild any label-derived transformations consistently. Never calculate a full-data target mean and then split the data.

The official macro panel also requires point-in-time treatment. A value must have
an assumed availability date strictly before the rental month. For a fixed-origin
forecast, pass the same information cutoff to `download_macro_data.py` for every
future row. Never join realized future macro values merely because they are
available at the time of retrospective analysis. The downloader uses current
revised official history rather than archived first-release vintages, so a small
revision-risk caveat remains.

### Market trend and extrapolation

The test period extends beyond most training dates. Gradient-boosted trees generally cannot extrapolate a rising time trend. Compare direct raw-rent prediction with a two-stage method that forecasts a monthly market index and predicts the property-specific residual or relative premium.

### Correlated features

Town, region, planning area, subzone, coordinates, MRT distance, demographics, and amenity density overlap. Permutation importance may divide importance among correlated variables. Use group ablation and time-based validation rather than selecting features solely from individual importance values.

### High-cardinality categories

Address, block, street, station, and mall names can encourage memorization. Evaluate performance separately for addresses with and without prior rental history. Ensure unknown categories are supported at inference time.

### Static school selectivity

Current entry cutoffs do not represent every historical rental year. They are location-quality proxies and should not be interpreted causally. School proximity may matter mainly for family-sized flats, so test interactions with flat type.

### External-data licensing and reproducibility

Document OpenStreetMap, Wikipedia, LTA, OneMap, MOE, and data.gov.sg sources in the report. Retain raw caches only where licensing permits. Never commit `keys.txt`, API tokens, account credentials, or downloaded secrets.

## Quality checks performed by the build

The script fails if:

- training and test row counts change;
- row order is not preserved;
- train and test predictor schemas differ; or
- the training target becomes missing.

After every build, review `feature_build_report.json` for join coverage and missingness. Before modeling, also check numerical ranges, category drift, duplicated identifiers, and whether a feature is available in the intended prediction setting.
