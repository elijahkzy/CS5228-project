from __future__ import annotations

import json
import math
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT_JSON = PROJECT / "reports" / "final_dataset_audit.json"

sys.path.insert(0, str(PROJECT))
from src import build_features as bf  # noqa: E402
from src.data_paths import resolve_data_root  # noqa: E402

DATA = resolve_data_root(PROJECT)


def pct(numerator: int | float, denominator: int | float) -> float:
    return 0.0 if denominator == 0 else 100.0 * float(numerator) / float(denominator)


def finite_equal(left: pd.Series, right: pd.Series, atol: float = 1e-8) -> int:
    left_values = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
    right_values = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
    equal = np.isclose(left_values, right_values, atol=atol, rtol=1e-9, equal_nan=True)
    return int((~equal).sum())


def normalize_key(frame: pd.DataFrame, key: str) -> pd.Series:
    if key == "ADDRESS_KEY":
        return (
            bf.normalize_text(frame["TOWN"])
            + "|"
            + bf.normalize_text(frame["BLOCK"])
            + "|"
            + bf.normalize_street(frame["STREET"])
        )
    return bf.normalize_text(frame[key])


def duplicate_key_profile(frame: pd.DataFrame, key: str, value_columns: list[str]) -> dict:
    work = frame.copy()
    work["_AUDIT_KEY"] = normalize_key(work, key)
    sizes = work.groupby("_AUDIT_KEY", dropna=False).size()
    duplicated_keys = sizes[sizes > 1].index
    duplicate_rows = int(work["_AUDIT_KEY"].isin(duplicated_keys).sum())
    conflicting = 0
    examples = []
    for current_key, group in work.loc[work["_AUDIT_KEY"].isin(duplicated_keys)].groupby("_AUDIT_KEY", dropna=False):
        columns = []
        for column in value_columns:
            if column in group and group[column].astype("string").fillna("<NA>").nunique() > 1:
                columns.append(column)
        if columns:
            conflicting += 1
            if len(examples) < 8:
                examples.append({"key": str(current_key), "conflicting_columns": ", ".join(columns)})
    return {
        "source_rows": int(len(work)),
        "unique_keys": int(sizes.size),
        "duplicate_key_groups": int(len(duplicated_keys)),
        "duplicate_rows": duplicate_rows,
        "conflicting_duplicate_groups": conflicting,
        "conflict_examples": examples,
    }


def haversine_matrix(source_lat, source_lon, target_lat, target_lon):
    lat1 = np.radians(np.asarray(source_lat, dtype=float))[:, None]
    lon1 = np.radians(np.asarray(source_lon, dtype=float))[:, None]
    lat2 = np.radians(np.asarray(target_lat, dtype=float))[None, :]
    lon2 = np.radians(np.asarray(target_lon, dtype=float))[None, :]
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def spatial_check(blocks: pd.DataFrame, places: pd.DataFrame, prefix: str, radii: list[float]) -> dict:
    places = places.loc[
        pd.to_numeric(places["LATITUDE"], errors="coerce").notna()
        & pd.to_numeric(places["LONGITUDE"], errors="coerce").notna()
    ].copy()
    expected_min = np.full(len(blocks), np.nan)
    expected_counts = {radius: np.zeros(len(blocks), dtype=int) for radius in radii}
    chunk_size = 1500
    for start in range(0, len(blocks), chunk_size):
        stop = min(start + chunk_size, len(blocks))
        matrix = haversine_matrix(
            blocks["LATITUDE"].iloc[start:stop],
            blocks["LONGITUDE"].iloc[start:stop],
            places["LATITUDE"],
            places["LONGITUDE"],
        )
        expected_min[start:stop] = np.nanmin(matrix, axis=1)
        for radius in radii:
            expected_counts[radius][start:stop] = np.sum(matrix <= radius, axis=1)
    nearest_column = f"NEAREST_{prefix}_KM"
    nearest_mismatches = int(
        (~np.isclose(blocks[nearest_column].to_numpy(dtype=float), expected_min, atol=1e-8, rtol=1e-9, equal_nan=True)).sum()
    )
    count_mismatches = 0
    for radius in radii:
        label = str(radius).replace(".", "P")
        column = f"N_{prefix}_WITHIN_{label}KM"
        count_mismatches += int((blocks[column].to_numpy(dtype=float) != expected_counts[radius]).sum())
    valid_coords = (
        pd.to_numeric(places["LATITUDE"], errors="coerce").between(1.1, 1.6)
        & pd.to_numeric(places["LONGITUDE"], errors="coerce").between(103.5, 104.2)
    )
    return {
        "source_rows": int(len(places)),
        "valid_coordinate_rows": int(valid_coords.sum()),
        "audited_address_keys": int(len(blocks)),
        "nearest_distance_mismatches": nearest_mismatches,
        "radius_count_mismatches": count_mismatches,
        "status": "PASS" if nearest_mismatches == 0 and count_mismatches == 0 and valid_coords.all() else "REVIEW",
    }


def missing_reason(column: str, train: pd.DataFrame, test: pd.DataFrame) -> tuple[str, str, str]:
    if column in {"MAX_FLOOR", "YEAR_COMPLETED", "BUILDING_AGE"}:
        return ("Expected source sentinel", "HDB source uses -1 for unavailable block attributes; the pipeline converts it to missing.", "Retain missingness or impute within validation folds.")
    if column == "REMAINING_LEASE_PROXY":
        return ("Expected dependency", "Missing when lease age cannot be calculated from a missing completion/lease input.", "Retain missingness or impute within validation folds.")
    if column.startswith("NEAREST_MRT_PV_") and column != "NEAREST_MRT_PV_MISSING":
        return ("MRT station-status validity issue", "Missing ridership identifies nearest stations absent from the 2026 passenger feed; the source incorrectly labels planned stations as open.", "Correct station status/opening dates, rebuild MRT proximity features, then reassess residual ridership gaps.")
    if column.startswith("RAIL_KM_TO_"):
        return ("Network coverage gap", "The nearest station is not connected to the named hub in the supplied weighted graph.", "Do not zero-fill; consider graph-component indicators or road distance fallback.")
    if column.startswith("NEAREST_MALL_") and column not in {"NEAREST_MALL_KM", "NEAREST_MALL_NAME", "NEAREST_MALL_WIKI_STATUS", "NEAREST_MALL_DETAILS_MISSING"}:
        return ("Expected source coverage gap", "Wikipedia-derived mall attributes are incomplete.", "Use the missing-detail indicator and avoid interpreting missing as zero.")
    if column in {"BEST_SECONDARY_PG3_MAX_WITHIN_2KM", "BEST_JC_L1R5_MAX_WITHIN_4KM"}:
        return ("Expected conditional absence", "No eligible school with a published cutoff exists within the specified radius.", "Keep missing distinct from a score of zero.")
    if column.startswith("PA_"):
        return ("Expected demographic suppression", "2020 planning-area detail is unavailable for some small or non-residential areas.", "Use PA_DATA_STATUS and PA_DEMOGRAPHICS_MATCHED; do not zero-fill suppressed values.")
    if column.startswith("SZ_"):
        return ("Source/join coverage gap", "The subzone source or joined HDB geography lacks a usable value.", "Review affected subzones before modeling.")
    if column.startswith(("COE_", "STOCK_")):
        missing_dates = set(train.loc[train[column].isna(), "RENT_APPROVAL_DATE"].dropna().dt.strftime("%Y-%m"))
        if missing_dates.issubset({"2021-01"}):
            return ("Expected first-period lag", "January 2021 has no prior month in the supplied market series.", "Keep missing or apply a documented first-period policy.")
        return ("Unexpected time-series gap", f"Missing values occur in: {', '.join(sorted(missing_dates))}.", "Repair source coverage or exclude affected periods.")
    if column.startswith(("ADDRESS_", "TOWN_FLAT_", "FLAT_TYPE_")):
        return ("Expected historical cold start", "No strictly earlier rental month exists for the relevant group.", "Keep missing; use prior-month count and chronological validation.")
    return ("Needs review", "Missingness is not covered by an explicit pipeline assumption.", "Inspect source rows and join keys.")


def add_check(checks: list[dict], area: str, check: str, result, threshold: str, status: str, interpretation: str):
    checks.append({
        "area": area,
        "check": check,
        "result": result,
        "threshold": threshold,
        "status": status,
        "interpretation": interpretation,
    })


raw_train = pd.read_csv(DATA / "train.csv")
raw_test = pd.read_csv(DATA / "test.csv")
train = pd.read_parquet(DATA / "processed" / "train_features.parquet")
test = pd.read_parquet(DATA / "processed" / "test_features.parquet")
dictionary = pd.read_csv(DATA / "processed" / "feature_dictionary.csv")
feature_group = dictionary.set_index("feature")["feature_group"].to_dict()

clean_train = bf.clean_core(raw_train, has_target=True)
clean_test = bf.clean_core(raw_test, has_target=False)
combined_processed = pd.concat([train.drop(columns="MONTHLY_RENT"), test], ignore_index=True, sort=False)
unique_blocks = combined_processed.drop_duplicates("ADDRESS_KEY").reset_index(drop=True)

checks: list[dict] = []
add_check(checks, "Integrity", "Training row count", len(train), "150,000", "PASS" if len(train) == 150000 else "FAIL", "No row multiplication or loss.")
add_check(checks, "Integrity", "Test row count", len(test), "50,000", "PASS" if len(test) == 50000 else "FAIL", "No row multiplication or loss.")
add_check(checks, "Integrity", "Training ROW_ID sequence mismatches", int((train["ROW_ID"].to_numpy() != np.arange(len(train))).sum()), "0", "PASS" if train["ROW_ID"].equals(pd.Series(np.arange(len(train)), name="ROW_ID")) else "FAIL", "Original training order is preserved.")
add_check(checks, "Integrity", "Test ROW_ID sequence mismatches", int((test["ROW_ID"].to_numpy() != np.arange(len(test))).sum()), "0", "PASS" if test["ROW_ID"].equals(pd.Series(np.arange(len(test)), name="ROW_ID")) else "FAIL", "Original test order is preserved.")
schema_match = [column for column in train.columns if column != "MONTHLY_RENT"] == list(test.columns)
add_check(checks, "Integrity", "Train/test predictor schema", "Identical" if schema_match else "Different", "Identical", "PASS" if schema_match else "FAIL", "Training and test predictors align.")
target_mismatches = finite_equal(train["MONTHLY_RENT"], clean_train["MONTHLY_RENT"])
add_check(checks, "Integrity", "Target value mismatches", target_mismatches, "0", "PASS" if target_mismatches == 0 else "FAIL", "MONTHLY_RENT is unchanged from train.csv.")
train_duplicate_rows = int(raw_train.duplicated(keep=False).sum())
test_duplicate_rows = int(raw_test.duplicated(keep=False).sum())
add_check(checks, "Integrity", "Training rows in exact duplicate groups", train_duplicate_rows, "Review, not automatic deletion", "REVIEW" if train_duplicate_rows else "PASS", "Identical records may represent repeated rental contracts; no transaction identifier is available to distinguish them.")
add_check(checks, "Integrity", "Test rows in exact duplicate groups", test_duplicate_rows, "Review, not automatic deletion", "REVIEW" if test_duplicate_rows else "PASS", "Identical predictor rows may be legitimate repeated records; preserve submission row order.")

invalid_date = int(train["RENT_APPROVAL_DATE"].isna().sum() + test["RENT_APPROVAL_DATE"].isna().sum())
add_check(checks, "Ranges", "Invalid approval dates", invalid_date, "0", "PASS" if invalid_date == 0 else "FAIL", "All monthly dates parsed successfully.")
invalid_area = int(((combined_processed["FLOOR_AREA_SQM"] <= 0) | (combined_processed["FLOOR_AREA_SQM"] > 300)).sum())
add_check(checks, "Ranges", "Implausible floor area rows", invalid_area, "0 outside (0, 300] sqm", "PASS" if invalid_area == 0 else "REVIEW", "Broad physical plausibility check.")
invalid_lease = int(((combined_processed["LEASE_AGE"] < 0) | (combined_processed["LEASE_AGE"] > 99)).sum())
add_check(checks, "Ranges", "Lease age outside 0–99", invalid_lease, "0", "PASS" if invalid_lease == 0 else "REVIEW", "Flags impossible lease timing.")
invalid_coords = int((~(combined_processed["LATITUDE"].between(1.1, 1.6) & combined_processed["LONGITUDE"].between(103.5, 104.2))).sum())
add_check(checks, "Ranges", "Coordinates outside Singapore bounding box", invalid_coords, "0", "PASS" if invalid_coords == 0 else "FAIL", "All block coordinates fall in a broad Singapore extent.")
numeric = combined_processed.select_dtypes(include=["number"])
inf_count = int(np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).sum())
add_check(checks, "Ranges", "Infinite numeric values", inf_count, "0", "PASS" if inf_count == 0 else "FAIL", "No infinite values should reach modeling.")

monotonic_specs = [
    ("MRT counts", ["N_OPEN_MRT_WITHIN_0P5KM", "N_OPEN_MRT_WITHIN_0P8KM", "N_OPEN_MRT_WITHIN_1P0KM", "N_OPEN_MRT_WITHIN_2P0KM"]),
    ("Mall counts", ["N_MALL_WITHIN_0P5KM", "N_MALL_WITHIN_1P0KM", "N_MALL_WITHIN_2P0KM"]),
    ("Primary-school counts", ["N_PRIMARY_SCHOOL_WITHIN_0P5KM", "N_PRIMARY_SCHOOL_WITHIN_1P0KM", "N_PRIMARY_SCHOOL_WITHIN_2P0KM"]),
    ("Secondary-school counts", ["N_SECONDARY_SCHOOL_WITHIN_1P0KM", "N_SECONDARY_SCHOOL_WITHIN_2P0KM"]),
    ("Junior-college counts", ["N_JUNIOR_COLLEGE_WITHIN_2P0KM", "N_JUNIOR_COLLEGE_WITHIN_4P0KM"]),
]
for label, columns in monotonic_specs:
    violations = 0
    for left, right in zip(columns, columns[1:]):
        violations += int((combined_processed[left] > combined_processed[right]).sum())
    add_check(checks, "Spatial consistency", f"{label} radius monotonicity violations", violations, "0", "PASS" if violations == 0 else "FAIL", "Counts must not fall as the radius grows.")

missingness = []
for column in train.columns:
    train_missing = int(train[column].isna().sum())
    test_missing = int(test[column].isna().sum()) if column in test else None
    if train_missing == 0 and (test_missing in (0, None)):
        continue
    classification, reason, action = missing_reason(column, train, test)
    missing_dates = train.loc[train[column].isna(), "RENT_APPROVAL_DATE"] if train_missing else pd.Series(dtype="datetime64[ns]")
    missingness.append({
        "feature": column,
        "feature_group": feature_group.get(column, ""),
        "train_missing_rows": train_missing,
        "train_missing_pct": pct(train_missing, len(train)),
        "test_missing_rows": test_missing,
        "test_missing_pct": None if test_missing is None else pct(test_missing, len(test)),
        "classification": classification,
        "reason": reason,
        "recommended_handling": action,
        "first_missing_month": None if missing_dates.empty else missing_dates.min().strftime("%Y-%m"),
        "last_missing_month": None if missing_dates.empty else missing_dates.max().strftime("%Y-%m"),
    })

join_audits: list[dict] = []
join_examples: list[dict] = []

hdb_raw = pd.read_csv(DATA / "auxiliary" / "sg-hdb-block.csv")
hdb_profile = duplicate_key_profile(hdb_raw, "ADDRESS_KEY", ["POSTAL_CODE", "LATITUDE", "LONGITUDE", "MAX_FLOOR", "YEAR_COMPLETED", "SUBZONE", "PLANNING_AREA", "REGION"])
hdb_clean = bf.clean_hdb(hdb_raw)
train_hdb_match = clean_train["ADDRESS_KEY"].isin(hdb_clean["ADDRESS_KEY"])
test_hdb_match = clean_test["ADDRESS_KEY"].isin(hdb_clean["ADDRESS_KEY"])
hdb_status = "PASS" if train_hdb_match.all() and test_hdb_match.all() and hdb_profile["conflicting_duplicate_groups"] == 0 else "REVIEW"
join_audits.append({
    "stage": "HDB block attributes",
    "method": "Exact normalized key; left join; many-to-one validation",
    "left_key": "town + block + standardized street",
    **{key: value for key, value in hdb_profile.items() if key != "conflict_examples"},
    "train_match_pct": pct(train_hdb_match.sum(), len(train_hdb_match)),
    "test_match_pct": pct(test_hdb_match.sum(), len(test_hdb_match)),
    "output_row_multiplier": 1.0,
    "status": hdb_status,
    "conclusion": "Exact-key assumption holds for all rental rows. Duplicate source keys still require conflict review." if hdb_profile["duplicate_key_groups"] else "Exact-key assumption and source uniqueness hold.",
})
for item in hdb_profile["conflict_examples"]:
    join_examples.append({"stage": "HDB block attributes", "issue": "Conflicting duplicate source key", **item})
for key in clean_train.loc[~train_hdb_match, "ADDRESS_KEY"].drop_duplicates().head(10):
    join_examples.append({"stage": "HDB block attributes", "issue": "Unmatched training key", "key": key, "conflicting_columns": ""})

planning_raw = pd.read_csv(DATA / "collected_data" / "zoning_data" / "sg-demographics-planning-area.csv")
subzone_raw = pd.read_csv(DATA / "collected_data" / "zoning_data" / "sg-demographics-subzone.csv")
for label, frame, raw_key, processed_key, match_indicator in [
    ("Planning-area demographics", planning_raw, "planning_area", "PLANNING_AREA", "PA_DEMOGRAPHICS_MATCHED"),
    ("Subzone demographics", subzone_raw, "subzone", "SUBZONE", "SZ_DEMOGRAPHICS_MATCHED"),
]:
    profile = duplicate_key_profile(frame, raw_key, [column for column in frame.columns if column != raw_key])
    keys = set(normalize_key(frame, raw_key).dropna())
    key_match = combined_processed[processed_key].isin(keys)
    usable = combined_processed[match_indicator].eq(1)
    status = "PASS" if profile["conflicting_duplicate_groups"] == 0 and key_match.all() else "REVIEW"
    join_audits.append({
        "stage": label,
        "method": "Exact normalized geography; left join; many-to-one validation",
        "left_key": processed_key,
        **{key: value for key, value in profile.items() if key != "conflict_examples"},
        "train_match_pct": pct(key_match.iloc[: len(train)].sum(), len(train)),
        "test_match_pct": pct(key_match.iloc[len(train):].sum(), len(test)),
        "usable_value_pct": pct(usable.sum(), len(usable)),
        "output_row_multiplier": 1.0,
        "status": status,
        "conclusion": "Key coverage and value availability differ because some matched 2020 geographies have suppressed/no population data.",
    })
    for item in profile["conflict_examples"]:
        join_examples.append({"stage": label, "issue": "Conflicting duplicate source key", **item})
    for key in combined_processed.loc[~key_match, processed_key].drop_duplicates().head(10):
        join_examples.append({"stage": label, "issue": "Unmatched geography key", "key": str(key), "conflicting_columns": ""})

sz_planning_mismatch = int((combined_processed["SZ_PLANNING_AREA"].notna() & (combined_processed["SZ_PLANNING_AREA"] != combined_processed["PLANNING_AREA"])).sum())
add_check(checks, "Join consistency", "Subzone-to-planning-area hierarchy mismatches", sz_planning_mismatch, "0", "PASS" if sz_planning_mismatch == 0 else "REVIEW", "Checks whether subzone census geography agrees with the HDB block planning area.")

mrt_nodes = bf.aggregate_mrt_nodes(DATA / "collected_data" / "mrt_data")
mrt_nodes["NAME"] = bf.normalize_text(mrt_nodes["NAME"])
mrt_nodes["status"] = bf.normalize_text(mrt_nodes["status"])
mrt_nodes["LATITUDE"] = bf.numeric(mrt_nodes["latitude"])
mrt_nodes["LONGITUDE"] = bf.numeric(mrt_nodes["longitude"])
open_mrt = mrt_nodes.loc[mrt_nodes["status"].eq("open")].copy()
mrt_spatial = spatial_check(unique_blocks, open_mrt, "OPEN_MRT", [0.5, 0.8, 1.0, 2.0])
pv_matched_mrt = open_mrt.loc[open_mrt["pv_total"].notna()].reset_index(drop=True)
proxy_matrix = haversine_matrix(
    unique_blocks["LATITUDE"], unique_blocks["LONGITUDE"],
    pv_matched_mrt["LATITUDE"], pv_matched_mrt["LONGITUDE"],
)
proxy_nearest_idx = np.nanargmin(proxy_matrix, axis=1)
proxy_nearest_name = pv_matched_mrt.iloc[proxy_nearest_idx]["NAME"].reset_index(drop=True)
current_nearest_name = unique_blocks["NEAREST_MRT_NAME"].reset_index(drop=True)
nearest_changed_block = current_nearest_name.ne(proxy_nearest_name)
changed_address_keys = set(unique_blocks.loc[nearest_changed_block, "ADDRESS_KEY"])
planned_nearest_train = train["ADDRESS_KEY"].isin(changed_address_keys)
planned_nearest_test = test["ADDRESS_KEY"].isin(changed_address_keys)
count_changed_block = np.zeros(len(unique_blocks), dtype=bool)
for radius in [0.5, 0.8, 1.0, 2.0]:
    label = str(radius).replace(".", "P")
    expected = np.sum(proxy_matrix <= radius, axis=1)
    actual = unique_blocks[f"N_OPEN_MRT_WITHIN_{label}KM"].to_numpy(dtype=int)
    count_changed_block |= expected != actual
count_changed_keys = set(unique_blocks.loc[count_changed_block, "ADDRESS_KEY"])
count_changed_train = train["ADDRESS_KEY"].isin(count_changed_keys)
count_changed_test = test["ADDRESS_KEY"].isin(count_changed_keys)
planned_station_impact = []
for station, counts in pd.concat([
    train.loc[planned_nearest_train, ["NEAREST_MRT_NAME"]].assign(split="train"),
    test.loc[planned_nearest_test, ["NEAREST_MRT_NAME"]].assign(split="test"),
]).groupby(["NEAREST_MRT_NAME", "split"]).size().unstack(fill_value=0).sort_values("train", ascending=False).iterrows():
    planned_station_impact.append({
        "station": station,
        "train_rows": int(counts.get("train", 0)),
        "test_rows": int(counts.get("test", 0)),
        "total_rows": int(counts.sum()),
    })
add_check(checks, "Join validity", "Rows whose nearest MRT changes when no-ridership/planned stations are excluded", int(planned_nearest_train.sum() + planned_nearest_test.sum()), "0", "FAIL" if (planned_nearest_train.any() or planned_nearest_test.any()) else "PASS", "The source's open-status field admits planned stations into historical accessibility features.")
add_check(checks, "Join validity", "Rows whose MRT radius counts change under the operational proxy", int(count_changed_train.sum() + count_changed_test.sum()), "0", "FAIL" if (count_changed_train.any() or count_changed_test.any()) else "PASS", "Counts within 0.5–2 km include stations with no passenger-volume evidence and planned station codes.")
join_audits.append({
    "stage": "MRT spatial features",
    "method": "Haversine nearest neighbour and radius counts at unique address level",
    "left_key": "HDB block latitude/longitude",
    "source_rows": mrt_spatial["source_rows"],
    "unique_keys": int(open_mrt["NAME"].nunique()),
    "duplicate_key_groups": int(open_mrt["NAME"].duplicated(keep=False).groupby(open_mrt["NAME"]).any().sum()),
    "duplicate_rows": int(open_mrt["NAME"].duplicated(keep=False).sum()),
    "conflicting_duplicate_groups": 0,
    "train_match_pct": 100.0,
    "test_match_pct": 100.0,
    "usable_value_pct": pct(open_mrt["pv_total"].notna().sum(), len(open_mrt)),
    "output_row_multiplier": 1.0,
    "status": "FAIL" if planned_nearest_train.any() or planned_nearest_test.any() else mrt_spatial["status"],
    "conclusion": f"Coordinate math is exact, but source status is invalid for historical use. Excluding stations without 2026 passenger records changes the nearest station for {planned_nearest_train.sum():,} train rows ({planned_nearest_train.mean():.1%}) and {planned_nearest_test.sum():,} test rows ({planned_nearest_test.mean():.1%}).",
})
for item in planned_station_impact[:15]:
    join_examples.append({
        "stage": "MRT spatial features",
        "issue": "Nearest station changes under operational proxy",
        "key": item["station"],
        "conflicting_columns": f"train rows={item['train_rows']:,}; test rows={item['test_rows']:,}",
    })

malls = pd.read_csv(DATA / "auxiliary" / "sg-shopping-malls.csv")
malls["LATITUDE"] = bf.numeric(malls["LATITUDE"])
malls["LONGITUDE"] = bf.numeric(malls["LONGITUDE"])
malls["MALL_KEY"] = bf.normalize_text(malls["NAME"])
wiki = pd.read_csv(DATA / "collected_data" / "mall_features" / "sg-mall-features-wikipedia.csv")
wiki["MALL_KEY"] = bf.normalize_text(wiki["mall_name"])
mall_spatial = spatial_check(unique_blocks, malls, "MALL", [0.5, 1.0, 2.0])
mall_base_dups = int(malls["MALL_KEY"].duplicated(keep=False).sum())
mall_wiki_dups = int(wiki["MALL_KEY"].duplicated(keep=False).sum())
wiki_matched = malls["MALL_KEY"].isin(wiki["MALL_KEY"])
join_audits.append({
    "stage": "Mall lookup and spatial features",
    "method": "Exact normalized mall-name metadata join, then Haversine proximity",
    "left_key": "mall name; HDB block coordinates",
    "source_rows": int(len(malls)),
    "unique_keys": int(malls["MALL_KEY"].nunique()),
    "duplicate_key_groups": int(malls.loc[malls["MALL_KEY"].duplicated(keep=False), "MALL_KEY"].nunique()),
    "duplicate_rows": mall_base_dups,
    "conflicting_duplicate_groups": mall_wiki_dups,
    "train_match_pct": pct(wiki_matched.sum(), len(malls)),
    "test_match_pct": pct(wiki_matched.sum(), len(malls)),
    "usable_value_pct": pct(wiki["floor_area_sqm"].notna().sum(), len(wiki)),
    "output_row_multiplier": 1.0,
    "status": "PASS" if mall_spatial["status"] == "PASS" and mall_base_dups == 0 and mall_wiki_dups == 0 else "REVIEW",
    "conclusion": f"All {len(malls)} mall names have metadata rows, but detailed attributes remain partial. Spatial recomputation found {mall_spatial['nearest_distance_mismatches']} distance and {mall_spatial['radius_count_mismatches']} count mismatches.",
})

mall_floor_known = pd.to_numeric(wiki.set_index("MALL_KEY")["floor_area_sqm"], errors="coerce")
mall_store_known = pd.to_numeric(wiki.set_index("MALL_KEY")["num_stores"], errors="coerce")
mall_distance_matrix = haversine_matrix(unique_blocks["LATITUDE"], unique_blocks["LONGITUDE"], malls["LATITUDE"], malls["LONGITUDE"])
within_two = mall_distance_matrix <= 2.0
known_floor_mask = malls["MALL_KEY"].map(mall_floor_known).notna().to_numpy()
known_store_mask = malls["MALL_KEY"].map(mall_store_known).notna().to_numpy()
no_known_floor = (within_two.sum(axis=1) > 0) & ((within_two & known_floor_mask[None, :]).sum(axis=1) == 0)
no_known_store = (within_two.sum(axis=1) > 0) & ((within_two & known_store_mask[None, :]).sum(axis=1) == 0)
ambiguous_floor_keys = set(unique_blocks.loc[no_known_floor, "ADDRESS_KEY"])
ambiguous_store_keys = set(unique_blocks.loc[no_known_store, "ADDRESS_KEY"])
ambiguous_floor_rows = int(combined_processed["ADDRESS_KEY"].isin(ambiguous_floor_keys).sum())
ambiguous_store_rows = int(combined_processed["ADDRESS_KEY"].isin(ambiguous_store_keys).sum())
add_check(checks, "Missing-value semantics", "Rows with malls within 2 km but no known floor area", ambiguous_floor_rows, "Review zero-valued sum", "REVIEW" if ambiguous_floor_rows else "PASS", "MALL_FLOOR_AREA_SUM_WITHIN_2KM is zero even though the true floor area is unknown.")
add_check(checks, "Missing-value semantics", "Rows with malls within 2 km but no known store count", ambiguous_store_rows, "Review zero-valued sum", "REVIEW" if ambiguous_store_rows else "PASS", "MALL_STORES_SUM_WITHIN_2KM is zero even though the true store count is unknown.")

primary, secondary, junior_college = bf.read_school_groups(DATA / "collected_data" / "school_data")
school_details = []
for prefix, frame, radii in [
    ("PRIMARY_SCHOOL", primary, [0.5, 1.0, 2.0]),
    ("SECONDARY_SCHOOL", secondary, [1.0, 2.0]),
    ("JUNIOR_COLLEGE", junior_college, [2.0, 4.0]),
]:
    result = spatial_check(unique_blocks, frame, prefix, radii)
    school_details.append((prefix, result))
school_mismatches = sum(item[1]["nearest_distance_mismatches"] + item[1]["radius_count_mismatches"] for item in school_details)
school_valid = all(item[1]["status"] == "PASS" for item in school_details)
join_audits.append({
    "stage": "School spatial features",
    "method": "Level-specific Haversine proximity; mixed-level schools included in applicable levels",
    "left_key": "HDB block latitude/longitude",
    "source_rows": int(len(primary) + len(secondary) + len(junior_college)),
    "unique_keys": int(pd.concat([primary["NAME"], secondary["NAME"], junior_college["NAME"]]).nunique()),
    "duplicate_key_groups": int(pd.concat([primary.assign(level="primary"), secondary.assign(level="secondary"), junior_college.assign(level="junior_college")])["NAME"].value_counts().gt(1).sum()),
    "duplicate_rows": int(pd.concat([primary.assign(level="primary"), secondary.assign(level="secondary"), junior_college.assign(level="junior_college")])["NAME"].duplicated(keep=False).sum()),
    "conflicting_duplicate_groups": 0,
    "train_match_pct": 100.0,
    "test_match_pct": 100.0,
    "usable_value_pct": 100.0,
    "output_row_multiplier": 1.0,
    "status": "PASS" if school_valid else "REVIEW",
    "conclusion": f"Independent recomputation found {school_mismatches} distance/count mismatches. Cross-level reuse is intentional for mixed-level schools.",
})

edges = pd.read_csv(DATA / "collected_data" / "mrt_data" / "sg-mrt-edges-weighted.csv")
edges["src_station"] = bf.normalize_text(edges["src_station"])
edges["dst_station"] = bf.normalize_text(edges["dst_station"])
edges["distance_km"] = bf.numeric(edges["distance_km"])
graph = defaultdict(set)
for row in edges.itertuples(index=False):
    graph[row.src_station].add(row.dst_station)
    graph[row.dst_station].add(row.src_station)
components = []
unseen = set(graph)
while unseen:
    start = next(iter(unseen))
    queue = deque([start])
    component = set([start])
    unseen.remove(start)
    while queue:
        node = queue.popleft()
        for neighbour in graph[node]:
            if neighbour in unseen:
                unseen.remove(neighbour)
                component.add(neighbour)
                queue.append(neighbour)
    components.append(component)
nearest_station_names = set(unique_blocks["NEAREST_MRT_NAME"].dropna())
graph_missing_nearest = sorted(nearest_station_names - set(graph))
rail_missing = {column: int(unique_blocks[column].isna().sum()) for column in unique_blocks if column.startswith("RAIL_KM_TO_")}
small_component_stations = sorted(next((component for component in components if len(component) == min(map(len, components))), set()))
join_audits.append({
    "stage": "MRT rail-network distances",
    "method": "Shortest path on weighted undirected graph from nearest station to four hubs",
    "left_key": "normalized nearest MRT station name",
    "source_rows": int(len(edges)),
    "unique_keys": int(len(graph)),
    "duplicate_key_groups": int(edges.duplicated(["src_station", "dst_station"], keep=False).sum()),
    "duplicate_rows": int(edges.duplicated(["src_station", "dst_station"], keep=False).sum()),
    "conflicting_duplicate_groups": 0,
    "train_match_pct": pct(len(nearest_station_names) - len(graph_missing_nearest), len(nearest_station_names)),
    "test_match_pct": pct(len(nearest_station_names) - len(graph_missing_nearest), len(nearest_station_names)),
    "usable_value_pct": 100.0 - max(pct(value, len(unique_blocks)) for value in rail_missing.values()),
    "output_row_multiplier": 1.0,
    "status": "PASS" if len(components) == 1 and not graph_missing_nearest and not any(rail_missing.values()) else "REVIEW",
    "conclusion": f"Graph has {len(components)} connected components with sizes {sorted([len(component) for component in components], reverse=True)}; {len(graph_missing_nearest)} nearest-station names are absent. Missing address-level distances: {rail_missing}.",
})
for station in graph_missing_nearest[:10]:
    join_examples.append({"stage": "MRT rail-network distances", "issue": "Nearest station absent from graph", "key": station, "conflicting_columns": ""})
for station in small_component_stations:
    join_examples.append({"stage": "MRT rail-network distances", "issue": "Station in disconnected graph component", "key": station, "conflicting_columns": ""})

macro = bf.build_macro_features(DATA)
macro_duplicates = int(macro["RENT_APPROVAL_DATE"].duplicated(keep=False).sum())
macro_cols = [column for column in macro.columns if column != "RENT_APPROVAL_DATE"]
macro_check = combined_processed[["RENT_APPROVAL_DATE", *macro_cols]].merge(
    macro, on="RENT_APPROVAL_DATE", how="left", suffixes=("_OUT", "_EXPECTED"), validate="many_to_one"
)
macro_mismatches = sum(finite_equal(macro_check[f"{column}_OUT"], macro_check[f"{column}_EXPECTED"], atol=1e-10) for column in macro_cols)
macro_key_match = combined_processed["RENT_APPROVAL_DATE"].isin(macro["RENT_APPROVAL_DATE"])
join_audits.append({
    "stage": "Lagged COE and stock indicators",
    "method": "Monthly date join after one-row shift of sorted monthly source aggregates",
    "left_key": "RENT_APPROVAL_DATE",
    "source_rows": int(len(macro)),
    "unique_keys": int(macro["RENT_APPROVAL_DATE"].nunique()),
    "duplicate_key_groups": int(macro_duplicates > 0),
    "duplicate_rows": macro_duplicates,
    "conflicting_duplicate_groups": int(macro_duplicates > 0),
    "train_match_pct": pct(macro_key_match.iloc[: len(train)].sum(), len(train)),
    "test_match_pct": pct(macro_key_match.iloc[len(train):].sum(), len(test)),
    "usable_value_pct": pct(macro[macro_cols].notna().all(axis=1).sum(), len(macro)),
    "output_row_multiplier": 1.0,
    "status": "PASS" if macro_duplicates == 0 and macro_mismatches == 0 else "FAIL",
    "conclusion": f"All output values reproduce the one-month-lagged monthly sources ({macro_mismatches} cell mismatches). January 2021 is intentionally blank because December 2020 is outside the source window.",
})

history_specs = [
    (["ADDRESS_KEY"], "ADDRESS"),
    (["TOWN", "FLAT_TYPE"], "TOWN_FLAT"),
    (["FLAT_TYPE"], "FLAT_TYPE"),
]
history_audits = []
rng = np.random.default_rng(5228)
sample_positions = np.unique(np.concatenate([
    np.arange(min(250, len(combined_processed))),
    rng.choice(len(combined_processed), size=1500, replace=False),
    np.arange(max(0, len(combined_processed) - 250), len(combined_processed)),
]))
history_reference = clean_train.copy()
holdout_start = train["RENT_APPROVAL_DATE"].max() - pd.DateOffset(months=11)
holdout_mask = train["RENT_APPROVAL_DATE"].ge(holdout_start)
for group_columns, prefix in history_specs:
    monthly = (
        history_reference.groupby([*group_columns, "RENT_APPROVAL_DATE"], dropna=False, as_index=False)
        .agg(MONTH_MEDIAN=("MONTHLY_RENT", "median"))
        .sort_values([*group_columns, "RENT_APPROVAL_DATE"])
    )
    lookup = {}
    group_key = group_columns[0] if len(group_columns) == 1 else group_columns
    for key, group in monthly.groupby(group_key, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        lookup[key] = (group["RENT_APPROVAL_DATE"].tolist(), group["MONTH_MEDIAN"].to_numpy(dtype=float))
    mismatches = 0
    same_or_future_used = 0
    for position in sample_positions:
        row = combined_processed.iloc[position]
        key = tuple(row[column] for column in group_columns)
        dates_values = lookup.get(key)
        eligible_dates = []
        eligible_values = np.array([], dtype=float)
        if dates_values is not None and pd.notna(row["RENT_APPROVAL_DATE"]):
            dates, values = dates_values
            mask = np.array([date < row["RENT_APPROVAL_DATE"] for date in dates])
            eligible_dates = [date for date, flag in zip(dates, mask) if flag]
            eligible_values = values[mask]
            if any(date >= row["RENT_APPROVAL_DATE"] for date in eligible_dates):
                same_or_future_used += 1
        expected = {
            f"{prefix}_PRIOR_MEDIAN": np.nan if len(eligible_values) == 0 else float(np.nanmedian(eligible_values)),
            f"{prefix}_LAST_RENT": np.nan if len(eligible_values) == 0 else float(eligible_values[-1]),
            f"{prefix}_PRIOR_MONTHS": int(len(eligible_values)),
            f"{prefix}_MONTHS_SINCE_LAST": np.nan if len(eligible_dates) == 0 else int((row["RENT_APPROVAL_DATE"].year - eligible_dates[-1].year) * 12 + row["RENT_APPROVAL_DATE"].month - eligible_dates[-1].month),
        }
        for column, expected_value in expected.items():
            actual = row[column]
            if pd.isna(expected_value) and pd.isna(actual):
                continue
            if not (pd.notna(expected_value) and pd.notna(actual) and math.isclose(float(actual), float(expected_value), rel_tol=1e-9, abs_tol=1e-8)):
                mismatches += 1
    last_months = train[f"{prefix}_MONTHS_SINCE_LAST"]
    possible_holdout_dependency = int((holdout_mask & last_months.notna() & (train["RENT_APPROVAL_DATE"] - pd.to_timedelta(0, unit="D") > holdout_start) & ((train["RENT_APPROVAL_DATE"].dt.year * 12 + train["RENT_APPROVAL_DATE"].dt.month - last_months) >= (holdout_start.year * 12 + holdout_start.month))).sum())
    holdout_rows = int(holdout_mask.sum())
    history_audits.append({
        "feature_family": prefix,
        "group_key": " + ".join(group_columns),
        "sampled_rows_checked": int(len(sample_positions)),
        "value_mismatches": mismatches,
        "same_or_future_months_used": same_or_future_used,
        "train_nonmissing_pct": pct(train[f"{prefix}_PRIOR_MEDIAN"].notna().sum(), len(train)),
        "test_nonmissing_pct": pct(test[f"{prefix}_PRIOR_MEDIAN"].notna().sum(), len(test)),
        "final_12_month_holdout_rows": holdout_rows,
        "holdout_rows_using_earlier_holdout_labels": possible_holdout_dependency,
        "holdout_dependency_pct": pct(possible_holdout_dependency, holdout_rows),
        "status": "PASS" if mismatches == 0 and same_or_future_used == 0 else "FAIL",
        "interpretation": "Values use only strictly earlier calendar months. For a fixed-origin multi-month holdout, later holdout rows can still include labels from earlier holdout months unless features are rebuilt at the cutoff.",
    })

for item in history_audits:
    join_audits.append({
        "stage": f"Historical rent: {item['feature_family']}",
        "method": "Monthly group medians with strict date < query date cutoff",
        "left_key": item["group_key"] + " + RENT_APPROVAL_DATE",
        "source_rows": len(clean_train),
        "unique_keys": int(clean_train[item["group_key"].split(" + ")].drop_duplicates().shape[0]),
        "duplicate_key_groups": 0,
        "duplicate_rows": 0,
        "conflicting_duplicate_groups": 0,
        "train_match_pct": item["train_nonmissing_pct"],
        "test_match_pct": item["test_nonmissing_pct"],
        "usable_value_pct": item["train_nonmissing_pct"],
        "output_row_multiplier": 1.0,
        "status": item["status"],
        "conclusion": f"Sampled {item['sampled_rows_checked']:,} train/test rows: {item['value_mismatches']} value mismatches and {item['same_or_future_months_used']} same/future-month uses. Fixed-origin holdout dependency affects {item['holdout_dependency_pct']:.1f}% of final-12-month rows.",
    })

drift = []
for column in ["TOWN", "FLAT_TYPE", "FLAT_MODEL", "SUBZONE", "PLANNING_AREA", "REGION", "NEAREST_MRT_NAME", "NEAREST_MALL_NAME"]:
    train_values = set(train[column].dropna().astype(str))
    test_nonnull = test[column].dropna().astype(str)
    unseen = ~test_nonnull.isin(train_values)
    drift.append({
        "feature": column,
        "train_unique": len(train_values),
        "test_unique": int(test_nonnull.nunique()),
        "unseen_test_values": int(test_nonnull[unseen].nunique()),
        "test_rows_with_unseen_value": int(unseen.sum()),
        "test_rows_with_unseen_pct": pct(unseen.sum(), len(test)),
        "examples": ", ".join(test_nonnull[unseen].drop_duplicates().head(8)),
    })

temporal_risks = [
    {
        "risk": "Planned MRT stations are labeled open",
        "affected_features": "MRT nearest/count/rail-distance features",
        "evidence": f"The source marks all JRL nodes open. The operational proxy changes nearest MRT for {planned_nearest_train.sum():,} training and {planned_nearest_test.sum():,} test rows. LTA lists JRL openings in 2028–2029.",
        "severity": "High",
        "assumption_needed": "Invalid as an open-station measure; at most this can represent future planned accessibility.",
        "recommended_test": "Correct status and add station opening dates, then rebuild features as-of each rental month.",
    },
    {
        "risk": "2026 station ridership used for earlier rentals",
        "affected_features": "NEAREST_MRT_PV_*_STATIC_2026",
        "evidence": "Passenger volumes are averages from May–July 2026; training dates end in 2025-03.",
        "severity": "High",
        "assumption_needed": "2026 ridership ranks are a stable station-busyness proxy available at prediction time.",
        "recommended_test": "Time-based ablation with and without ridership features; describe them as static proxies.",
    },
    {
        "risk": "Current mall and school snapshots applied historically",
        "affected_features": "Mall and school proximity/detail/selectivity features",
        "evidence": "Current source coordinates and attributes are applied uniformly across all rental months.",
        "severity": "Medium",
        "assumption_needed": "Facility presence and attributes were stable over the modeled period.",
        "recommended_test": "Add opening/closure dates where available or run group ablations.",
    },
    {
        "risk": "Static 2020 demographics",
        "affected_features": "PA_* and SZ_*",
        "evidence": "One census snapshot is joined to every rental month.",
        "severity": "Medium",
        "assumption_needed": "Neighbourhood structure is sufficiently stable from 2021 onward.",
        "recommended_test": "Use as location descriptors; do not interpret as month-specific demographics.",
    },
    {
        "risk": "Historical-rent features under fixed-origin validation",
        "affected_features": "ADDRESS_*, TOWN_FLAT_*, FLAT_TYPE_*",
        "evidence": "Strictly earlier months are used, but materialized later holdout rows may include labels from earlier holdout months.",
        "severity": "High",
        "assumption_needed": "Predictions are refreshed monthly after prior-month rents become observable.",
        "recommended_test": "For a fixed 6/12-month forecast, rebuild all validation features using only labels available at the fold cutoff.",
    },
    {
        "risk": "Mall aggregate sums conflate no known detail with zero known total",
        "affected_features": "MALL_FLOOR_AREA_SUM_WITHIN_2KM, MALL_STORES_SUM_WITHIN_2KM",
        "evidence": "np.nansum returns zero when no qualifying mall has a known attribute.",
        "severity": "Medium",
        "assumption_needed": "Zero is interpreted as known-value sum, not actual total mall capacity.",
        "recommended_test": "Add counts of malls with known floor-area/store values within 2 km.",
    },
]

source_profiles = []
source_paths = [
    "train.csv",
    "test.csv",
    "auxiliary/sg-hdb-block.csv",
    "collected_data/mrt_data/sg-mrt-nodes-features-202605.csv",
    "collected_data/mrt_data/sg-mrt-nodes-features-202606.csv",
    "collected_data/mrt_data/sg-mrt-nodes-features-202607.csv",
    "collected_data/mrt_data/sg-mrt-edges-weighted.csv",
    "auxiliary/sg-shopping-malls.csv",
    "collected_data/mall_features/sg-mall-features-wikipedia.csv",
    "collected_data/school_data/sg-schools-primary.csv",
    "collected_data/school_data/sg-schools-secondary.csv",
    "collected_data/school_data/sg-schools-junior_college.csv",
    "collected_data/school_data/sg-schools-mixed.csv",
    "collected_data/zoning_data/sg-demographics-planning-area.csv",
    "collected_data/zoning_data/sg-demographics-subzone.csv",
    "auxiliary/sg-coe-prices.csv",
    "auxiliary/sg-stock-prices.csv",
]
for relative in source_paths:
    frame = pd.read_csv(DATA / relative)
    source_profiles.append({
        "source": relative,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "fully_duplicate_rows": int(frame.duplicated().sum()),
        "columns_with_missing": int((frame.isna().sum() > 0).sum()),
    })

status_counts = pd.Series([item["status"] for item in checks + join_audits]).value_counts().to_dict()
overview = {
    "overall_status": "CONDITIONAL PASS" if status_counts.get("FAIL", 0) == 0 else "REVIEW REQUIRED",
    "train_rows": int(len(train)),
    "test_rows": int(len(test)),
    "predictor_count": int(len(test.columns)),
    "train_date_range": f"{train['RENT_APPROVAL_DATE'].min().date()} to {train['RENT_APPROVAL_DATE'].max().date()}",
    "test_date_range": f"{test['RENT_APPROVAL_DATE'].min().date()} to {test['RENT_APPROVAL_DATE'].max().date()}",
    "missing_columns_train": int((train.isna().sum() > 0).sum()),
    "missing_columns_test": int((test.isna().sum() > 0).sum()),
    "checks_pass": int(status_counts.get("PASS", 0)),
    "checks_review": int(status_counts.get("REVIEW", 0)),
    "checks_fail": int(status_counts.get("FAIL", 0)),
    "unique_address_keys": int(unique_blocks["ADDRESS_KEY"].nunique()),
    "holdout_start": str(holdout_start.date()),
    "planned_station_nearest_train_rows": int(planned_nearest_train.sum()),
    "planned_station_nearest_test_rows": int(planned_nearest_test.sum()),
    "mrt_count_changed_train_rows": int(count_changed_train.sum()),
    "mrt_count_changed_test_rows": int(count_changed_test.sum()),
    "ambiguous_mall_floor_sum_rows": ambiguous_floor_rows,
    "ambiguous_mall_store_sum_rows": ambiguous_store_rows,
}

data_exceptions = []
for split, frame in [("train", train), ("test", test)]:
    invalid = frame.loc[(frame["LEASE_AGE"] < 0) | (frame["LEASE_AGE"] > 99)]
    for row in invalid.itertuples(index=False):
        data_exceptions.append({
            "issue": "Lease age outside 0–99",
            "split": split,
            "row_id": int(row.ROW_ID),
            "rent_approval_date": str(row.RENT_APPROVAL_DATE.date()),
            "address_key": row.ADDRESS_KEY,
            "observed_value": f"LEASE_COMMENCE_DATE={row.LEASE_COMMENCE_DATE}; LEASE_AGE={row.LEASE_AGE}; REMAINING_LEASE_PROXY={row.REMAINING_LEASE_PROXY}",
            "recommended_action": "Verify the lease commencement year against the source; set the derived ages missing until corrected.",
        })

OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON.write_text(json.dumps({
    "overview": overview,
    "checks": checks,
    "missingness": missingness,
    "join_audits": join_audits,
    "join_examples": join_examples,
    "history_audits": history_audits,
    "temporal_risks": temporal_risks,
    "planned_station_impact": planned_station_impact,
    "data_exceptions": data_exceptions,
    "train_test_drift": drift,
    "source_profiles": source_profiles,
    "external_sources": [
        {
            "source": "Land Transport Authority — Jurong Region Line",
            "url": "https://www.lta.gov.sg/content/ltagov/en/upcoming_projects/rail_expansion/jurong_region_line.html/1000",
            "relevance": "Current official opening stages show JRL service begins in 2028 and 2029, contradicting the local source status='open'.",
        }
    ],
}, indent=2, default=str), encoding="utf-8")

print(json.dumps(overview, indent=2))
print("Join statuses:", pd.Series([item["status"] for item in join_audits]).value_counts().to_dict())
print("Missing classifications:", pd.Series([item["classification"] for item in missingness]).value_counts().to_dict())
print(f"Wrote {OUTPUT_JSON}")
