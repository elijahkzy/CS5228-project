"""Build aligned model-ready feature tables for the CS5228 HDB rental project.

Run from any directory:
    python src/build_features.py

Inputs and outputs use the resolved external data root. See docs/DATA_STORAGE.md.
"""

from __future__ import annotations

import argparse
import bisect
import heapq
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from data_paths import resolve_data_root


EARTH_RADIUS_KM = 6371.0088
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def normalize_text(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .str.replace(r"[-_/]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
    )


def normalize_street(series: pd.Series) -> pd.Series:
    replacements = {
        r"\bavenue\b": "ave",
        r"\broad\b": "rd",
        r"\bstreet\b": "st",
        r"\bdrive\b": "dr",
        r"\bcentral\b": "ctrl",
        r"\bplace\b": "pl",
        r"\bclose\b": "cl",
        r"\bcrescent\b": "cres",
    }
    out = normalize_text(series)
    for pattern, replacement in replacements.items():
        out = out.str.replace(pattern, replacement, regex=True)
    return out.str.replace(r"\s+", " ", regex=True).str.strip()


def normalize_flat_type(series: pd.Series) -> pd.Series:
    out = normalize_text(series)
    return out.str.replace(r"^(\d) room$", r"\1-room", regex=True)


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def clean_core(df: pd.DataFrame, has_target: bool) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "ROW_ID", np.arange(len(out), dtype=np.int64))
    out["RENT_APPROVAL_DATE"] = pd.to_datetime(
        out["RENT_APPROVAL_DATE"], format="%Y-%m", errors="coerce"
    )
    for column in ["TOWN", "BLOCK", "FLAT_MODEL", "FURNISHED"]:
        out[column] = normalize_text(out[column])
    out["STREET"] = normalize_street(out["STREET"])
    out["FLAT_TYPE"] = normalize_flat_type(out["FLAT_TYPE"])
    numeric_columns = ["FLOOR_AREA_SQM", "LEASE_COMMENCE_DATE", "FEE"]
    if has_target:
        numeric_columns.append("MONTHLY_RENT")
    for column in numeric_columns:
        out[column] = numeric(out[column])

    out["RENT_YEAR"] = out["RENT_APPROVAL_DATE"].dt.year.astype("Int64")
    out["RENT_MONTH"] = out["RENT_APPROVAL_DATE"].dt.month.astype("Int64")
    out["RENT_QUARTER"] = out["RENT_APPROVAL_DATE"].dt.quarter.astype("Int64")
    out["MONTH_INDEX"] = (
        (out["RENT_YEAR"] - 2021) * 12 + out["RENT_MONTH"] - 1
    ).astype("Int64")
    out["MONTH_SIN"] = np.sin(2 * np.pi * out["RENT_MONTH"].astype(float) / 12)
    out["MONTH_COS"] = np.cos(2 * np.pi * out["RENT_MONTH"].astype(float) / 12)
    out["LEASE_AGE"] = out["RENT_YEAR"] - out["LEASE_COMMENCE_DATE"]
    out["REMAINING_LEASE_PROXY"] = 99 - out["LEASE_AGE"]
    out["ADDRESS_KEY"] = out["TOWN"] + "|" + out["BLOCK"] + "|" + out["STREET"]

    # Known zero-variance fields in the supplied release.
    out = out.drop(columns=["FURNISHED", "FEE"], errors="ignore")
    return out


def clean_hdb(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in ["TOWN", "BLOCK", "SUBZONE", "PLANNING_AREA", "REGION"]:
        out[column] = normalize_text(out[column])
    out["STREET"] = normalize_street(out["STREET"])
    for column in ["LATITUDE", "LONGITUDE", "MAX_FLOOR", "YEAR_COMPLETED"]:
        out[column] = numeric(out[column])
    out[["MAX_FLOOR", "YEAR_COMPLETED"]] = out[
        ["MAX_FLOOR", "YEAR_COMPLETED"]
    ].replace(-1, np.nan)
    out["POSTAL_CODE"] = out["POSTAL_CODE"].astype("string").str.replace(r"\.0$", "", regex=True)
    out["ADDRESS_KEY"] = out["TOWN"] + "|" + out["BLOCK"] + "|" + out["STREET"]
    return out.drop_duplicates("ADDRESS_KEY", keep="first")


def pairwise_haversine_km(
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
) -> np.ndarray:
    lat1 = np.radians(np.asarray(source_lat, dtype=float))[:, None]
    lon1 = np.radians(np.asarray(source_lon, dtype=float))[:, None]
    lat2 = np.radians(np.asarray(target_lat, dtype=float))[None, :]
    lon2 = np.radians(np.asarray(target_lon, dtype=float))[None, :]
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def spatial_summary(
    blocks: pd.DataFrame,
    places: pd.DataFrame,
    prefix: str,
    radii_km: tuple[float, ...] = (),
) -> tuple[pd.DataFrame, np.ndarray]:
    result = pd.DataFrame(index=blocks.index)
    matrix = pairwise_haversine_km(
        blocks["LATITUDE"].to_numpy(),
        blocks["LONGITUDE"].to_numpy(),
        places["LATITUDE"].to_numpy(),
        places["LONGITUDE"].to_numpy(),
    )
    valid = np.isfinite(matrix)
    safe = np.where(valid, matrix, np.inf)
    nearest_idx = safe.argmin(axis=1)
    nearest_distance = safe[np.arange(len(blocks)), nearest_idx]
    nearest_distance[~np.isfinite(nearest_distance)] = np.nan
    result[f"NEAREST_{prefix}_KM"] = nearest_distance
    for radius in radii_km:
        label = str(radius).replace(".", "P")
        result[f"N_{prefix}_WITHIN_{label}KM"] = np.sum(safe <= radius, axis=1)
    return result, nearest_idx


def row_min_or_nan(values: np.ndarray) -> np.ndarray:
    """Return row minima without warnings when a row contains only missing values."""
    safe = np.where(np.isfinite(values), values, np.inf)
    result = safe.min(axis=1)
    result[~np.isfinite(result)] = np.nan
    return result


def aggregate_mrt_nodes(mrt_dir: Path) -> pd.DataFrame:
    files = sorted(mrt_dir.glob("sg-mrt-nodes-features-*.csv"))
    if not files:
        nodes = pd.read_csv(mrt_dir / "sg-mrt-nodes.csv")
        nodes["PV_MONTHS_AVAILABLE"] = 0
        return nodes
    frames = [pd.read_csv(path) for path in files]
    combined = pd.concat(frames, ignore_index=True)
    identity = [
        "NAME", "member_codes", "lines", "n_lines", "status", "latitude",
        "longitude", "subzone", "planning_area", "region", "is_interchange",
    ]
    pv_columns = [
        "pv_tap_in", "pv_tap_out", "pv_total", "pv_weekday_total",
        "pv_weekend_total", "pv_am_peak_total", "pv_pm_peak_total", "pv_busiest_hour",
    ]
    for column in pv_columns + ["n_lines", "latitude", "longitude"]:
        combined[column] = numeric(combined[column])
    aggregated = combined.groupby("NAME", as_index=False).agg(
        {**{column: "first" for column in identity if column != "NAME"},
         **{column: "mean" for column in pv_columns},
         "year_month": "nunique"}
    )
    return aggregated.rename(columns={"year_month": "PV_MONTHS_AVAILABLE"})


def dijkstra_distances(edges: pd.DataFrame, source: str) -> dict[str, float]:
    graph: dict[str, list[tuple[str, float]]] = {}
    for row in edges.itertuples(index=False):
        graph.setdefault(row.src_station, []).append((row.dst_station, row.distance_km))
        graph.setdefault(row.dst_station, []).append((row.src_station, row.distance_km))
    distances = {source: 0.0}
    queue = [(0.0, source)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances.get(node):
            continue
        for neighbour, weight in graph.get(node, []):
            candidate = distance + weight
            if candidate < distances.get(neighbour, np.inf):
                distances[neighbour] = candidate
                heapq.heappush(queue, (candidate, neighbour))
    return distances


def build_mrt_features(blocks: pd.DataFrame, collected_dir: Path) -> pd.DataFrame:
    mrt_dir = collected_dir / "mrt_data"
    nodes = aggregate_mrt_nodes(mrt_dir)
    nodes["NAME"] = normalize_text(nodes["NAME"])
    nodes["status"] = normalize_text(nodes["status"])
    nodes["LATITUDE"] = numeric(nodes["latitude"])
    nodes["LONGITUDE"] = numeric(nodes["longitude"])
    nodes["n_lines"] = numeric(nodes["n_lines"])
    nodes["is_interchange"] = (
        normalize_text(nodes["is_interchange"]).map({"true": 1, "false": 0}).astype("Int64")
    )
    open_nodes = nodes.loc[nodes["status"].eq("open")].reset_index(drop=True)
    summary, nearest_idx = spatial_summary(blocks, open_nodes, "OPEN_MRT", (0.5, 0.8, 1.0, 2.0))
    nearest = open_nodes.iloc[nearest_idx].reset_index(drop=True)
    summary["NEAREST_MRT_NAME"] = nearest["NAME"].to_numpy()
    summary["NEAREST_MRT_N_LINES"] = nearest["n_lines"].to_numpy()
    summary["NEAREST_MRT_IS_INTERCHANGE"] = nearest["is_interchange"].to_numpy()
    summary["NEAREST_MRT_PV_MONTHS_AVAILABLE"] = nearest["PV_MONTHS_AVAILABLE"].to_numpy()
    for column in [
        "pv_total", "pv_weekday_total", "pv_weekend_total", "pv_am_peak_total",
        "pv_pm_peak_total", "pv_busiest_hour",
    ]:
        summary[f"NEAREST_MRT_{column.upper()}_STATIC_2026"] = numeric(nearest[column]).to_numpy()
    summary["NEAREST_MRT_PV_MISSING"] = summary["NEAREST_MRT_PV_TOTAL_STATIC_2026"].isna().astype("int8")

    edges = pd.read_csv(mrt_dir / "sg-mrt-edges-weighted.csv")
    edges["src_station"] = normalize_text(edges["src_station"])
    edges["dst_station"] = normalize_text(edges["dst_station"])
    edges["distance_km"] = numeric(edges["distance_km"])
    hubs = {
        "RAFFLES_PLACE": "raffles place mrt station",
        "ORCHARD": "orchard mrt station",
        "JURONG_EAST": "jurong east mrt station",
        "WOODLANDS": "woodlands mrt station",
    }
    for label, station in hubs.items():
        distances = dijkstra_distances(edges, station)
        summary[f"RAIL_KM_TO_{label}"] = summary["NEAREST_MRT_NAME"].map(distances)
    return summary


def build_mall_features(
    blocks: pd.DataFrame, base_malls_path: Path, collected_dir: Path
) -> pd.DataFrame:
    malls = pd.read_csv(base_malls_path)
    malls["MALL_KEY"] = normalize_text(malls["NAME"])
    malls["LATITUDE"] = numeric(malls["LATITUDE"])
    malls["LONGITUDE"] = numeric(malls["LONGITUDE"])
    wiki = pd.read_csv(collected_dir / "mall_features" / "sg-mall-features-wikipedia.csv")
    wiki["MALL_KEY"] = normalize_text(wiki["mall_name"])
    for column in ["floors", "num_stores", "num_anchors", "opening_year", "floor_area_sqm"]:
        wiki[column] = numeric(wiki[column])
    malls = malls.merge(wiki, on="MALL_KEY", how="left", validate="one_to_one")
    summary, nearest_idx = spatial_summary(blocks, malls, "MALL", (0.5, 1.0, 2.0))
    nearest = malls.iloc[nearest_idx].reset_index(drop=True)
    summary["NEAREST_MALL_NAME"] = nearest["NAME"].to_numpy()
    summary["NEAREST_MALL_WIKI_STATUS"] = nearest["wiki_status"].astype("string").to_numpy()
    for column in ["floors", "num_stores", "num_anchors", "opening_year", "floor_area_sqm"]:
        summary[f"NEAREST_MALL_{column.upper()}"] = nearest[column].to_numpy()
    summary["NEAREST_MALL_DETAILS_MISSING"] = summary["NEAREST_MALL_FLOOR_AREA_SQM"].isna().astype("int8")

    distances = pairwise_haversine_km(
        blocks["LATITUDE"].to_numpy(), blocks["LONGITUDE"].to_numpy(),
        malls["LATITUDE"].to_numpy(), malls["LONGITUDE"].to_numpy(),
    )
    within_2km = distances <= 2.0
    floor_area = malls["floor_area_sqm"].to_numpy(dtype=float)
    stores = malls["num_stores"].to_numpy(dtype=float)
    summary["MALL_FLOOR_AREA_SUM_WITHIN_2KM"] = np.nansum(
        np.where(within_2km, floor_area[None, :], np.nan), axis=1
    )
    summary["MALL_STORES_SUM_WITHIN_2KM"] = np.nansum(
        np.where(within_2km, stores[None, :], np.nan), axis=1
    )
    return summary


def read_school_groups(school_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary = pd.read_csv(school_dir / "sg-schools-primary.csv")
    secondary = pd.read_csv(school_dir / "sg-schools-secondary.csv")
    junior_college = pd.read_csv(school_dir / "sg-schools-junior_college.csv")
    mixed = pd.read_csv(school_dir / "sg-schools-mixed.csv")
    for frame in [primary, secondary, junior_college, mixed]:
        frame["LATITUDE"] = numeric(frame["LATITUDE"])
        frame["LONGITUDE"] = numeric(frame["LONGITUDE"])
        frame["fee_tier"] = normalize_text(frame["fee_tier"])
    secondary = pd.concat([secondary, mixed], ignore_index=True, sort=False)
    junior_college = pd.concat([junior_college, mixed], ignore_index=True, sort=False)
    return primary, secondary, junior_college


def build_school_features(blocks: pd.DataFrame, collected_dir: Path) -> pd.DataFrame:
    primary, secondary, junior_college = read_school_groups(collected_dir / "school_data")
    result = pd.DataFrame(index=blocks.index)
    for prefix, places, radii in [
        ("PRIMARY_SCHOOL", primary, (0.5, 1.0, 2.0)),
        ("SECONDARY_SCHOOL", secondary, (1.0, 2.0)),
        ("JUNIOR_COLLEGE", junior_college, (2.0, 4.0)),
    ]:
        summary, _ = spatial_summary(blocks, places, prefix, radii)
        result = pd.concat([result, summary], axis=1)

    secondary_dist = pairwise_haversine_km(
        blocks["LATITUDE"].to_numpy(), blocks["LONGITUDE"].to_numpy(),
        secondary["LATITUDE"].to_numpy(), secondary["LONGITUDE"].to_numpy(),
    )
    pg3_max = numeric(secondary.get("pg3_max", pd.Series(index=secondary.index, dtype=float))).to_numpy()
    result["BEST_SECONDARY_PG3_MAX_WITHIN_2KM"] = row_min_or_nan(
        np.where((secondary_dist <= 2.0) & np.isfinite(pg3_max)[None, :], pg3_max[None, :], np.nan),
    )
    high_fee = secondary["fee_tier"].str.contains("independent|high", na=False).to_numpy()
    result["N_HIGH_FEE_SECONDARY_WITHIN_2KM"] = np.sum(
        (secondary_dist <= 2.0) & high_fee[None, :], axis=1
    )

    jc_dist = pairwise_haversine_km(
        blocks["LATITUDE"].to_numpy(), blocks["LONGITUDE"].to_numpy(),
        junior_college["LATITUDE"].to_numpy(), junior_college["LONGITUDE"].to_numpy(),
    )
    jc_score_columns = [
        column for column in ["l1r5_arts_max", "l1r5_science_max", "l1r5_commerce_max",
                              "l1r5_international_baccalaureate_max"]
        if column in junior_college.columns
    ]
    jc_scores = junior_college[jc_score_columns].apply(numeric).min(axis=1).to_numpy()
    result["BEST_JC_L1R5_MAX_WITHIN_4KM"] = row_min_or_nan(
        np.where((jc_dist <= 4.0) & np.isfinite(jc_scores)[None, :], jc_scores[None, :], np.nan),
    )
    return result


def clean_demographics(path: Path, prefix: str, key: str) -> pd.DataFrame:
    out = pd.read_csv(path)
    out[key] = normalize_text(out[key])
    status_columns = [column for column in ["data_status", "onemap_query", "year"] if column in out]
    numeric_columns = [column for column in out.columns if column not in [key, *status_columns, "planning_area"]]
    for column in numeric_columns:
        out[column] = numeric(out[column])
    rename = {column: f"{prefix}_{column.upper()}" for column in out.columns if column != key}
    return out.rename(columns=rename).drop_duplicates(key)


def build_macro_features(data_dir: Path) -> pd.DataFrame:
    auxiliary_dir = data_dir / "auxiliary"
    coe_path = auxiliary_dir / "sg-coe-prices.csv" if auxiliary_dir.exists() else data_dir / "sg-coe-prices.csv"
    stock_path = auxiliary_dir / "sg-stock-prices.csv" if auxiliary_dir.exists() else data_dir / "sg-stock-prices.csv"
    coe = pd.read_csv(coe_path)
    month_lookup = {month.lower(): index for index, month in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1
    )}
    coe["MONTH_NUM"] = normalize_text(coe["MONTH"]).map(month_lookup)
    coe["RENT_APPROVAL_DATE"] = pd.to_datetime(
        dict(year=numeric(coe["YEAR"]), month=coe["MONTH_NUM"], day=1)
    )
    for column in ["QUOTA_PREMIUM", "PREVAILING_QUOTA_PREMIUM", "QUOTA", "BIDS_RECEIVED"]:
        coe[column] = numeric(coe[column].astype(str).str.replace(r"[$,]", "", regex=True))
    coe["BID_QUOTA_RATIO"] = coe["BIDS_RECEIVED"] / coe["QUOTA"].replace(0, np.nan)
    coe_monthly = coe.groupby("RENT_APPROVAL_DATE").agg(
        COE_PREMIUM_MEAN=("QUOTA_PREMIUM", "mean"),
        COE_PREMIUM_MAX=("QUOTA_PREMIUM", "max"),
        COE_BID_QUOTA_RATIO=("BID_QUOTA_RATIO", "mean"),
    ).sort_index().shift(1).add_suffix("_LAG1")

    stocks = pd.read_csv(stock_path)
    stocks["DATE"] = pd.to_datetime(stocks["DATE"], errors="coerce")
    stocks["ADJUSTED_CLOSE"] = numeric(stocks["ADJUSTED_CLOSE"])
    stocks = stocks.sort_values(["SYMBOL", "DATE"])
    stocks["DAILY_RETURN"] = stocks.groupby("SYMBOL")["ADJUSTED_CLOSE"].pct_change(fill_method=None)
    stocks["RENT_APPROVAL_DATE"] = stocks["DATE"].dt.to_period("M").dt.to_timestamp()
    stock_monthly = stocks.groupby("RENT_APPROVAL_DATE").agg(
        STOCK_MARKET_RETURN=("DAILY_RETURN", "mean"),
        STOCK_MARKET_VOLATILITY=("DAILY_RETURN", "std"),
    ).sort_index().shift(1).add_suffix("_LAG1")
    return coe_monthly.join(stock_monthly, how="outer").reset_index()


def load_official_macro_features(data_dir: Path) -> pd.DataFrame:
    """Load the compact point-in-time macro panel produced by the downloader."""
    path = data_dir / "macro" / "sg_macro_features_monthly.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run 'python src/download_macro_data.py' first."
        )
    macro = pd.read_csv(path)
    macro["RENT_APPROVAL_DATE"] = pd.to_datetime(
        macro["RENT_APPROVAL_DATE"], errors="raise"
    )
    feature_columns = [column for column in macro if column.startswith("MACRO_")]
    if not feature_columns:
        raise ValueError("Official macro panel contains no MACRO_ feature columns")
    if macro["RENT_APPROVAL_DATE"].duplicated().any():
        raise ValueError("Official macro panel has duplicate rental months")
    for column in feature_columns:
        macro[column] = numeric(macro[column])
    return macro[["RENT_APPROVAL_DATE", *feature_columns]]


def historical_group_features(
    train: pd.DataFrame,
    queries: pd.DataFrame,
    group_columns: list[str],
    prefix: str,
) -> pd.DataFrame:
    monthly = (
        train.groupby([*group_columns, "RENT_APPROVAL_DATE"], dropna=False, as_index=False)
        .agg(MONTH_MEDIAN=("MONTHLY_RENT", "median"))
        .sort_values([*group_columns, "RENT_APPROVAL_DATE"])
    )
    histories: dict[tuple, tuple[list[pd.Timestamp], np.ndarray]] = {}
    group_key = group_columns[0] if len(group_columns) == 1 else group_columns
    for key, group in monthly.groupby(group_key, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        histories[key] = (
            group["RENT_APPROVAL_DATE"].tolist(),
            group["MONTH_MEDIAN"].to_numpy(dtype=float),
        )

    prior_median = np.full(len(queries), np.nan)
    prior_last = np.full(len(queries), np.nan)
    prior_months = np.zeros(len(queries), dtype=np.int32)
    months_since_last = np.full(len(queries), np.nan)
    keys = queries[group_columns].itertuples(index=False, name=None)
    dates = queries["RENT_APPROVAL_DATE"].tolist()
    for row_index, (key, query_date) in enumerate(zip(keys, dates)):
        history = histories.get(tuple(key))
        if history is None or pd.isna(query_date):
            continue
        history_dates, values = history
        cutoff = bisect.bisect_left(history_dates, query_date)
        if cutoff == 0:
            continue
        eligible = values[:cutoff]
        prior_median[row_index] = np.nanmedian(eligible)
        prior_last[row_index] = eligible[-1]
        prior_months[row_index] = cutoff
        last_date = history_dates[cutoff - 1]
        months_since_last[row_index] = (query_date.year - last_date.year) * 12 + query_date.month - last_date.month

    return pd.DataFrame({
        f"{prefix}_PRIOR_MEDIAN": prior_median,
        f"{prefix}_LAST_RENT": prior_last,
        f"{prefix}_PRIOR_MONTHS": prior_months,
        f"{prefix}_MONTHS_SINCE_LAST": months_since_last,
    }, index=queries.index)


def feature_description(column: str) -> tuple[str, str, str]:
    if column == "MONTHLY_RENT":
        return "target", "Observed monthly rent in SGD", "Training table only"
    if column in {"ROW_ID", "ADDRESS_KEY", "BLOCK", "STREET", "POSTAL_CODE"}:
        return "identifier", "Identifier retained for ordering, joins, and diagnostics", "Do not treat numeric-looking identifiers as continuous"
    if column.startswith(("ADDRESS_", "TOWN_FLAT_", "FLAT_TYPE_")):
        return "historical rent", "Rental statistic calculated from strictly earlier months", "Use time-based validation only; never recompute with validation or test labels"
    if column.startswith("PA_"):
        return "planning-area demographics", "2020 planning-area demographic feature", "Static census snapshot; some small areas have missing values"
    if column.startswith("SZ_"):
        return "subzone demographics", "2020 subzone demographic feature", "Static census snapshot"
    if "MRT" in column or column.startswith("RAIL_"):
        return "MRT accessibility", "Nearest-station, network, or station-busyness feature", "Ridership is a static May-Jul 2026 proxy, not historical"
    if "MALL" in column:
        return "mall accessibility", "Mall proximity or Wikipedia-derived size feature", "Wikipedia attributes have incomplete coverage"
    if any(token in column for token in ["SCHOOL", "SECONDARY", "JUNIOR_COLLEGE", "JC_L1R5"]):
        return "school accessibility", "School proximity, count, fee tier, or entry-score feature", "Current school snapshot; entry-score meaning varies by level"
    if column.startswith("MACRO_"):
        return "official macroeconomic", "Point-in-time Singapore macro or housing-market indicator", "Uses a conservative publication delay; validate as a group with forward time splits"
    if column.startswith(("COE_", "STOCK_")):
        return "macroeconomic", "One-month-lagged monthly market indicator", "May proxy for time; validate incrementally"
    if column in {"LATITUDE", "LONGITUDE", "MAX_FLOOR", "YEAR_COMPLETED", "BUILDING_AGE", "SUBZONE", "PLANNING_AREA", "REGION"}:
        return "HDB block", "HDB block or geographic attribute", "HDB -1 sentinels converted to missing"
    if column.startswith("N_"):
        return "proximity count", "Number of facilities within the stated radius", "Correlated with urban density and other location variables"
    if column in {"RENT_APPROVAL_DATE", "RENT_YEAR", "RENT_MONTH", "RENT_QUARTER", "MONTH_INDEX", "MONTH_SIN", "MONTH_COS"}:
        return "time", "Rental approval date or derived trend/seasonality feature", "Tree models cannot reliably extrapolate beyond observed time"
    return "core property", "Cleaned core rental or derived property feature", "Review categorical encoding and missing values in the model pipeline"


def build_feature_dictionary(train_output: pd.DataFrame) -> pd.DataFrame:
    records = []
    for column in train_output.columns:
        group, description, notes = feature_description(column)
        records.append({
            "feature": column,
            "feature_group": group,
            "dtype": str(train_output[column].dtype),
            "description": description,
            "training_notes": notes,
        })
    return pd.DataFrame(records)


def build_tables(
    project_root: Path, data_root: Path | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    data_dir = resolve_data_root(project_root, data_root)
    auxiliary_dir = data_dir / "auxiliary"
    collected_dir = data_dir / "collected_data"
    hdb_path = auxiliary_dir / "sg-hdb-block.csv" if auxiliary_dir.exists() else data_dir / "sg-hdb-block.csv"
    mrt_path = auxiliary_dir / "sg-mrt-stations.csv" if auxiliary_dir.exists() else data_dir / "sg-mrt-stations.csv"
    malls_path = auxiliary_dir / "sg-shopping-malls.csv" if auxiliary_dir.exists() else data_dir / "sg-shopping-malls.csv"

    train = clean_core(pd.read_csv(data_dir / "train.csv"), has_target=True)
    test = clean_core(pd.read_csv(data_dir / "test.csv"), has_target=False)
    train["SOURCE_SPLIT"] = "train"
    test["SOURCE_SPLIT"] = "test"
    combined = pd.concat([train.drop(columns="MONTHLY_RENT"), test], ignore_index=True, sort=False)

    hdb = clean_hdb(pd.read_csv(hdb_path))
    hdb_columns = [
        "ADDRESS_KEY", "POSTAL_CODE", "LATITUDE", "LONGITUDE", "MAX_FLOOR",
        "YEAR_COMPLETED", "SUBZONE", "PLANNING_AREA", "REGION",
    ]
    combined = combined.merge(hdb[hdb_columns], on="ADDRESS_KEY", how="left", validate="many_to_one")
    combined["HDB_BLOCK_MATCHED"] = combined["LATITUDE"].notna().astype("int8")
    combined["BUILDING_AGE"] = combined["RENT_YEAR"] - combined["YEAR_COMPLETED"]

    block_frame = combined[hdb_columns].drop_duplicates("ADDRESS_KEY").reset_index(drop=True)
    mrt_features = build_mrt_features(block_frame, collected_dir)
    mall_features = build_mall_features(block_frame, malls_path, collected_dir)
    school_features = build_school_features(block_frame, collected_dir)
    block_enriched = pd.concat([block_frame, mrt_features, mall_features, school_features], axis=1)
    extra_columns = [column for column in block_enriched.columns if column not in hdb_columns]
    combined = combined.merge(
        pd.concat([block_enriched[["ADDRESS_KEY"]], block_enriched[extra_columns]], axis=1),
        on="ADDRESS_KEY", how="left", validate="many_to_one",
    )

    zoning_dir = collected_dir / "zoning_data"
    planning = clean_demographics(
        zoning_dir / "sg-demographics-planning-area.csv", "PA", "planning_area"
    ).rename(columns={"planning_area": "PLANNING_AREA"})
    subzone = clean_demographics(
        zoning_dir / "sg-demographics-subzone.csv", "SZ", "subzone"
    ).rename(columns={"subzone": "SUBZONE"})
    combined = combined.merge(planning, on="PLANNING_AREA", how="left", validate="many_to_one")
    combined = combined.merge(subzone, on="SUBZONE", how="left", validate="many_to_one")
    combined["PA_DEMOGRAPHICS_MATCHED"] = combined.filter(regex=r"^PA_TOTAL_POPULATION$").notna().any(axis=1).astype("int8")
    combined["SZ_DEMOGRAPHICS_MATCHED"] = combined.filter(regex=r"^SZ_TOTAL_POPULATION$").notna().any(axis=1).astype("int8")

    combined = combined.merge(build_macro_features(data_dir), on="RENT_APPROVAL_DATE", how="left", validate="many_to_one")
    combined = combined.merge(
        load_official_macro_features(data_dir),
        on="RENT_APPROVAL_DATE",
        how="left",
        validate="many_to_one",
    )

    history_specs = [
        (["ADDRESS_KEY"], "ADDRESS"),
        (["TOWN", "FLAT_TYPE"], "TOWN_FLAT"),
        (["FLAT_TYPE"], "FLAT_TYPE"),
    ]
    for group_columns, prefix in history_specs:
        history = historical_group_features(train, combined, group_columns, prefix)
        combined = pd.concat([combined, history], axis=1)

    train_output = combined.loc[combined["SOURCE_SPLIT"].eq("train")].copy()
    test_output = combined.loc[combined["SOURCE_SPLIT"].eq("test")].copy()
    train_output = train_output.merge(train[["ROW_ID", "MONTHLY_RENT"]], on="ROW_ID", how="left", validate="one_to_one")
    train_output = train_output.sort_values("ROW_ID").reset_index(drop=True)
    test_output = test_output.sort_values("ROW_ID").reset_index(drop=True)
    train_output = train_output.drop(columns="SOURCE_SPLIT")
    test_output = test_output.drop(columns="SOURCE_SPLIT")

    predictor_columns = [column for column in train_output.columns if column != "MONTHLY_RENT"]
    test_output = test_output[predictor_columns]
    if len(train_output) != 150_000 or len(test_output) != 50_000:
        raise ValueError(f"Unexpected row counts: train={len(train_output)}, test={len(test_output)}")
    if not train_output["ROW_ID"].equals(pd.Series(np.arange(len(train_output)), name="ROW_ID")):
        raise ValueError("Training row order was not preserved")
    if not test_output["ROW_ID"].equals(pd.Series(np.arange(len(test_output)), name="ROW_ID")):
        raise ValueError("Test row order was not preserved")
    if predictor_columns != list(test_output.columns):
        raise ValueError("Train and test predictor schemas differ")
    if train_output["MONTHLY_RENT"].isna().any():
        raise ValueError("Training target contains missing values after feature construction")

    report = {
        "train_rows": len(train_output),
        "test_rows": len(test_output),
        "predictor_count": len(predictor_columns),
        "train_date_min": str(train_output["RENT_APPROVAL_DATE"].min().date()),
        "train_date_max": str(train_output["RENT_APPROVAL_DATE"].max().date()),
        "test_date_min": str(test_output["RENT_APPROVAL_DATE"].min().date()),
        "test_date_max": str(test_output["RENT_APPROVAL_DATE"].max().date()),
        "hdb_match_train_pct": float(100 * train_output["HDB_BLOCK_MATCHED"].mean()),
        "hdb_match_test_pct": float(100 * test_output["HDB_BLOCK_MATCHED"].mean()),
        "address_history_train_pct": float(100 * train_output["ADDRESS_PRIOR_MEDIAN"].notna().mean()),
        "address_history_test_pct": float(100 * test_output["ADDRESS_PRIOR_MEDIAN"].notna().mean()),
        "official_macro_feature_count": len(
            [column for column in predictor_columns if column.startswith("MACRO_")]
        ),
        "official_macro_complete_train_pct": float(
            100
            * train_output.filter(regex=r"^MACRO_").notna().all(axis=1).mean()
        ),
        "official_macro_complete_test_pct": float(
            100
            * test_output.filter(regex=r"^MACRO_").notna().all(axis=1).mean()
        ),
        "columns_with_missing_train": int((train_output.isna().sum() > 0).sum()),
        "columns_with_missing_test": int((test_output.isna().sum() > 0).sum()),
        "missingness_train_pct": {
            column: round(float(value), 4)
            for column, value in (100 * train_output.isna().mean()).items() if value > 0
        },
        "missingness_test_pct": {
            column: round(float(value), 4)
            for column, value in (100 * test_output.isna().mean()).items() if value > 0
        },
    }
    return train_output, test_output, report


def save_outputs(data_root: Path, train: pd.DataFrame, test: pd.DataFrame, report: dict) -> None:
    output_dir = data_root / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(output_dir / "train_features.parquet", index=False)
    test.to_parquet(output_dir / "test_features.parquet", index=False)
    build_feature_dictionary(train).to_csv(output_dir / "feature_dictionary.csv", index=False)
    with (output_dir / "feature_build_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Folder containing train.csv; overrides CS5228_DATA_ROOT and auto-detection",
    )
    arguments = parser.parse_args()
    project_root = arguments.project_root.resolve()
    data_root = resolve_data_root(project_root, arguments.data_root)
    train, test, report = build_tables(project_root, data_root)
    save_outputs(data_root, train, test, report)
    print(json.dumps({key: value for key, value in report.items() if not key.startswith("missingness_")}, indent=2))
    print(f"Data root: {data_root}")
    print(f"Outputs written to {data_root / 'processed'}")


if __name__ == "__main__":
    main()
