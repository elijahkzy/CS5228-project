# Final dataset quality and join audit

Audit date: 2026-09-11

## Outcome

The final feature tables require review before modeling. Their row integrity, exact-key joins, distance calculations, monthly lags, and target preservation pass. The MRT availability assumption does not pass because the station source labels planned stations as `open`.

The detailed workbook is available at `reports/final_dataset_quality_and_join_audit.xlsx`. The structured audit results are available at `reports/final_dataset_audit.json`.

## Main findings

### Passed checks

- `train_features.parquet` contains 150,000 rows and `test_features.parquet` contains 50,000 rows.
- `ROW_ID` preserves the source order and the predictor schemas match.
- `MONTHLY_RENT` matches `train.csv` exactly.
- The normalized HDB block key is unique across all 9,464 source rows and matches 100% of train and test records.
- Planning-area and subzone demographic keys are unique and match 100% of records.
- The subzone-to-planning-area hierarchy has no observed conflicts.
- Independent Haversine calculations reproduce the MRT, mall, and school nearest distances and radius counts.
- Lagged COE and stock values reproduce the one-month-shifted monthly source aggregates.
- A deterministic sample of 1,996 train and test rows confirms that historical-rent features use only strictly earlier calendar months.
- No infinite numeric values, invalid approval dates, implausible floor areas, or Singapore coordinate-bound violations were found.

### MRT availability failure

The local MRT node files label planned stations as `open`. The official Land Transport Authority Jurong Region Line page currently lists JRL openings in 2028 and 2029, contradicting those source statuses for JRL stations.

As a diagnostic proxy, the audit excludes stations that have no May–July 2026 passenger-volume record:

- the nearest MRT changes for 23,077 training rows and 7,948 test rows;
- at least one MRT radius count changes for 74,849 training rows and 24,861 test rows;
- the affected nearest-station list contains 31 station names;
- the weighted rail graph has two components, leaving all four hub-distance fields missing for 1,381 training and 500 test rows.

The passenger-volume filter is an audit proxy, not a replacement for an authoritative station-opening table. Correct the source status and add opening dates before rebuilding each rental month's accessibility features.

Official validation source: [Land Transport Authority — Jurong Region Line](https://www.lta.gov.sg/content/ltagov/en/upcoming_projects/rail_expansion/jurong_region_line.html/1000)

### Missing values

The 42 training columns with missing values fall into these groups:

| Cause | Columns | Interpretation |
|---|---:|---|
| Historical cold start | 9 | No earlier rental month exists for the relevant group. |
| Planning-area suppression | 8 | Some 2020 economic and household fields are unavailable. |
| MRT station-status issue | 6 | Nearest stations absent from the passenger feed include stations incorrectly marked open. |
| Incomplete mall metadata | 5 | Wikipedia does not provide every mall attribute. |
| First-month market lag | 5 | January 2021 has no December 2020 source value. |
| Disconnected MRT graph | 4 | The nearest station cannot reach the four named hubs. |
| HDB source sentinel | 3 | `-1` was correctly converted to missing. |
| Conditional school cutoff | 2 | No eligible school with a published cutoff exists inside the radius. |

Do not replace all missing values with zero. Missing mall size, demographic suppression, historical cold start, and a real facility count of zero have different meanings.

### Historical-rent validation assumption

The implemented feature values do not use the current or a future month. They are valid for a monthly-updated prediction workflow in which prior-month rents are observable before the next prediction.

They are not directly valid for a fixed-origin multi-month holdout. In the final 12-month training holdout:

- 71.1% of rows have address-history features that use a label from an earlier holdout month;
- 91.0% have town-and-flat-type history with that dependency;
- 91.1% have flat-type history with that dependency.

For a fixed six- or twelve-month forecast, rebuild all label-derived features using only labels available at the fold cutoff.

### Other issues requiring a decision

- One training row (`ROW_ID=31739`) has `LEASE_COMMENCE_DATE=2022` for a November 2021 rental, producing `LEASE_AGE=-1` and `REMAINING_LEASE_PROXY=100`.
- 836 training rows and 10,099 test rows belong to exact duplicate groups. They may represent separate rental contracts; do not delete them without a transaction-level rule.
- `MALL_FLOOR_AREA_SUM_WITHIN_2KM` is zero for 28,405 rows where at least one mall is nearby but no nearby mall has known floor area.
- `MALL_STORES_SUM_WITHIN_2KM` has the same unknown-versus-zero ambiguity for 34,726 rows.
- No unseen values were found in test for the audited major categorical fields.

## Recommended correction order

1. Create an authoritative MRT station availability table with opening and closure dates.
2. Rebuild MRT nearest-station, radius-count, ridership, and rail-distance features as of each rental month.
3. Represent mall aggregate coverage explicitly, for example with counts of nearby malls having known floor area and store counts.
4. Verify or null the invalid lease-derived fields for `ROW_ID=31739`.
5. Rerun `src/audit_final_dataset.py`.
6. Evaluate fixed-origin and monthly-updated validation as separate prediction scenarios.

## Reproducing the audit

Install the project requirements, build the feature tables, and run:

```bash
python src/audit_final_dataset.py
```

The script writes `reports/final_dataset_audit.json`. It does not modify the processed feature tables.
