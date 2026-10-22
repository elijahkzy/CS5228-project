# Singapore macroeconomic features

## Purpose

The macro panel adds a small set of national time-series features to the HDB
rental tables. These variables describe market-wide conditions. They complement
property and location features, but they cannot explain differences between two
flats approved in the same month.

The selection is intentionally limited to six features. The training data cover
about 51 distinct months, so a large collection of related indicators would add
multicollinearity and make time-series overfitting likely.

## Selected features

| Feature | Official source | Construction | Why it was selected |
|---|---|---|---|
| `MACRO_CPI_EX_ACCOM_YOY` | SingStat `M213752`, All Items Less Accommodation | Year-over-year percentage change | Captures broad cost pressure without directly embedding accommodation inflation, which is too close to the rental outcome. |
| `MACRO_SORA_3M_PCT` | SingStat `M700071`, 3-month compounded SORA | Published percentage level | Measures financing conditions that may affect landlords, housing demand, and the rent-versus-buy decision. |
| `MACRO_REAL_GDP_YOY` | SingStat `M015631`, real GDP in chained 2015 dollars | Published year-over-year growth rate | Represents broad economic and income conditions without adding many overlapping activity indicators. |
| `MACRO_UNEMPLOYMENT_SA_PCT` | SingStat `M182342`, total unemployment rate | Seasonally adjusted percentage level | Adds a direct labour-demand and household-income-risk signal that GDP does not fully capture. |
| `MACRO_HDB_RESALE_RPI_YOY` | SingStat `M212161`, HDB resale price index | Year-over-year percentage change | Represents the closest observable purchase alternative to HDB renting and the rent-versus-buy substitution channel. |
| `MACRO_PRIVATE_RENT_RPI_YOY` | SingStat `M212311`, private residential rental index | Year-over-year percentage change | Provides a separate indicator of common Singapore-wide rental pressure. It does not contain HDB rental labels. |

Official table pages:

- <https://tablebuilder.singstat.gov.sg/table/TS/M213752>
- <https://tablebuilder.singstat.gov.sg/table/TS/M700071>
- <https://tablebuilder.singstat.gov.sg/table/TS/M015631>
- <https://tablebuilder.singstat.gov.sg/table/TS/M182342>
- <https://tablebuilder.singstat.gov.sg/table/TS/M212161>
- <https://tablebuilder.singstat.gov.sg/table/TS/M212311>

The downloader stores the source table ID, series number, original period,
transformation, retrieval date, source update date, rationale, and assumed
availability date in `<data root>/macro/sg_macro_observations.csv`.

## Leakage control

Rental approval dates are available only at monthly precision. The feature value
for a rental month is therefore selected using an as-of join against the first
day of that month:

```text
macro availability date < first day of rental month
```

The inequality is strict. A value released on the first day of a rental month is
not considered known for that month's record.

The panel uses conservative release-delay assumptions:

| Series | Assumed availability |
|---|---|
| Monthly CPI | 25 days after month-end; this generally produces a two-month information lag at the start of a rental month |
| Monthly 3-month SORA | At month-end; the next rental month may use the completed prior month |
| Quarterly real GDP growth | 50 days after quarter-end |
| Quarterly unemployment | 45 days after quarter-end |
| Quarterly HDB resale price index | 30 days after quarter-end |
| Quarterly private rental index | 30 days after quarter-end |

These dates are modeling assumptions rather than a historical release-calendar
archive. They are deliberately later than typical initial releases. Review them
if a formal vintage-data source becomes available.

The official API exposes the latest historical revision, not the value exactly as
first published. This creates a residual revision risk, especially for GDP and
seasonally adjusted unemployment. The project should disclose this limitation
and avoid causal claims.

The raw HDB rental dataset published on data.gov.sg is deliberately excluded. It
contains the same outcome domain and now covers the hidden test period, so using
its later rental records or a rental index calculated from them would risk direct
target leakage.

## Build and refresh

From the project root, refresh the official data and then rebuild all features:

```powershell
.\.venv\Scripts\python.exe .\src\download_macro_data.py
.\.venv\Scripts\python.exe .\src\build_features.py
```

Both commands automatically detect an adjacent `CS5228 Project Data/data`
folder. When the Google Drive copy is elsewhere, set `CS5228_DATA_ROOT` or pass
`--data-root "G:\My Drive\CS5228 Project Data\data"` to both commands.

The downloader creates:

```text
<data root>/macro/
├── sg_macro_observations.csv
└── sg_macro_features_monthly.csv
```

`sg_macro_observations.csv` is the audit and provenance table.
`sg_macro_features_monthly.csv` is the compact table joined to the main feature
pipeline by `RENT_APPROVAL_DATE`.

### Fixed-origin forecasting

The default panel implements a rolling-origin information set: each month can use
values released before that month. This is suitable for chronological validation
that simulates monthly model updates and for retrospective prediction when those
releases were genuinely available.

To simulate producing every future test prediction from one fixed date, use an
information cutoff. For example:

```powershell
.\.venv\Scripts\python.exe .\src\download_macro_data.py `
  --information-cutoff 2025-03-01
```

Observations available on or after the cutoff are excluded. Later test months
then carry the last value known before the cutoff. A more advanced fixed-origin
experiment should forecast the macro series using only pre-cutoff observations;
it must not substitute realized future macro values.

## Appropriate uses

Use the six features:

1. As a feature group in CatBoost, LightGBM, XGBoost, a regularized linear model,
   or a GAM.
2. With forward or rolling-origin validation and the same availability rules in
   every fold.
3. In group ablation: compare the identical model with and without all six macro
   features.
4. In a two-stage model where the macro variables estimate the national monthly
   rental level and property/location features estimate each flat's premium or
   discount relative to that level.

Do not use them:

- with random cross-validation;
- without a publication delay;
- as evidence of causal effects;
- by inserting actual macro observations that were unavailable at the declared
  forecast origin;
- as replacements for property and location variables; or
- simply because individual feature-importance values are high. Most macro
  variables are correlated with calendar time.

## Evaluation

Compare at least these experiments with identical forward validation folds:

1. Existing property, location, time, and historical-rent features.
2. The same model plus all six official macro features.
3. A two-stage market-level and property-residual model.

Report overall error, monthly error, monthly prediction bias, and the change in
performance on the latest fold. Retain the macro group only if it improves more
than one forward fold and does not merely compensate for a poorly specified time
trend.
