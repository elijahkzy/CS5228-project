"""Download and prepare a compact, leakage-controlled Singapore macro panel.

The source tables are official SingStat time series. Each observation receives a
conservative assumed availability date. A rental month can use an observation
only when that availability date is strictly earlier than the first day of the
rental month.

Run from the project root:
    python src/download_macro_data.py
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from data_paths import resolve_data_root


API_BASE = "https://tablebuilder.singstat.gov.sg/api/table/tabledata"
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUEST_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (compatible; CS5228-academic-project/1.0)",
    "Referer": "https://tablebuilder.singstat.gov.sg/",
}


@dataclass(frozen=True)
class SeriesSpec:
    feature: str
    table_id: str
    series_no: str
    expected_name: str
    frequency: str
    transformation: str
    availability_days: int
    rationale: str


SERIES = (
    SeriesSpec(
        feature="MACRO_CPI_EX_ACCOM_YOY",
        table_id="M213752",
        series_no="2",
        expected_name="All Items Less Accommodation",
        frequency="monthly",
        transformation="year-over-year percent change",
        availability_days=25,
        rationale=(
            "Measures broad cost pressure without embedding accommodation costs, "
            "which are too close to the rental outcome."
        ),
    ),
    SeriesSpec(
        feature="MACRO_SORA_3M_PCT",
        table_id="M700071",
        series_no="23",
        expected_name="Compounded Singapore Overnight Rate Average (SORA) - 3 Month",
        frequency="monthly",
        transformation="level in percent per annum",
        availability_days=0,
        rationale=(
            "Represents financing conditions that can affect landlords, housing "
            "demand, and the rent-versus-buy decision."
        ),
    ),
    SeriesSpec(
        feature="MACRO_REAL_GDP_YOY",
        table_id="M015631",
        series_no="2",
        expected_name="GDP In Chained (2015) Dollars",
        frequency="quarterly",
        transformation="published year-over-year growth rate",
        availability_days=50,
        rationale=(
            "Captures broad real economic demand and income conditions without "
            "adding many overlapping activity indicators."
        ),
    ),
    SeriesSpec(
        feature="MACRO_UNEMPLOYMENT_SA_PCT",
        table_id="M182342",
        series_no="1",
        expected_name="Total Unemployment Rate, (SA)",
        frequency="quarterly",
        transformation="seasonally adjusted level in percent",
        availability_days=45,
        rationale=(
            "Provides a direct labour-demand and household-income-risk signal that "
            "is not fully represented by GDP."
        ),
    ),
    SeriesSpec(
        feature="MACRO_HDB_RESALE_RPI_YOY",
        table_id="M212161",
        series_no="1",
        expected_name="HDB Resale Price Index",
        frequency="quarterly",
        transformation="year-over-year percent change",
        availability_days=30,
        rationale=(
            "Tracks the closest observable alternative to HDB renting and therefore "
            "the rent-versus-buy substitution channel."
        ),
    ),
    SeriesSpec(
        feature="MACRO_PRIVATE_RENT_RPI_YOY",
        table_id="M212311",
        series_no="1",
        expected_name="Rental Index Of Residential Properties",
        frequency="quarterly",
        transformation="year-over-year percent change",
        availability_days=30,
        rationale=(
            "Tracks common Singapore-wide rental-market pressure while remaining "
            "separate from the HDB rental labels."
        ),
    ),
)


def fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers=REQUEST_HEADERS)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def parse_period(period: str, frequency: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    if frequency == "monthly":
        start = pd.to_datetime(period, format="%Y %b", errors="raise")
        return start, start + pd.offsets.MonthEnd(0)
    match = re.fullmatch(r"(\d{4})\s+([1-4])Q", period)
    if not match:
        raise ValueError(f"Unexpected quarterly period: {period!r}")
    quarter = pd.Period(f"{match.group(1)}Q{match.group(2)}", freq="Q")
    return quarter.start_time.normalize(), quarter.end_time.normalize()


def transform_values(values: pd.Series, transformation: str) -> pd.Series:
    if transformation == "year-over-year percent change":
        periods = 12 if values.index.freqstr and values.index.freqstr.startswith("M") else 4
        return values.pct_change(periods=periods, fill_method=None) * 100
    return values


def download_series(spec: SeriesSpec) -> pd.DataFrame:
    query = urllib.parse.urlencode({"seriesNoORrowNo": spec.series_no})
    url = f"{API_BASE}/{spec.table_id}?{query}"
    payload = fetch_json(url)
    data = payload.get("Data", {})
    rows = data.get("row", [])
    row = next((item for item in rows if str(item.get("seriesNo")) == spec.series_no), None)
    if row is None:
        raise ValueError(f"Series {spec.series_no} not returned for {spec.table_id}")
    if row.get("rowText") != spec.expected_name:
        raise ValueError(
            f"Series label changed for {spec.table_id}/{spec.series_no}: "
            f"{row.get('rowText')!r}"
        )

    records = []
    for item in row.get("columns", []):
        period_start, period_end = parse_period(item["key"], spec.frequency)
        records.append(
            {
                "OBSERVATION_PERIOD": item["key"],
                "PERIOD_DATE": period_start,
                "PERIOD_END": period_end,
                "RAW_VALUE": pd.to_numeric(item.get("value"), errors="coerce"),
            }
        )
    frame = pd.DataFrame(records).sort_values("PERIOD_DATE").reset_index(drop=True)

    indexed = frame.set_index(
        pd.PeriodIndex(
            frame["PERIOD_DATE"], freq="M" if spec.frequency == "monthly" else "Q"
        )
    )["RAW_VALUE"]
    frame["VALUE"] = transform_values(indexed, spec.transformation).to_numpy()
    frame["AVAILABLE_DATE"] = frame["PERIOD_END"] + pd.to_timedelta(
        spec.availability_days, unit="D"
    )
    frame["FEATURE"] = spec.feature
    frame["SOURCE_TABLE_ID"] = spec.table_id
    frame["SERIES_NO"] = spec.series_no
    frame["SERIES_NAME"] = row["rowText"]
    frame["UNIT"] = row.get("uoM", "")
    frame["FREQUENCY"] = spec.frequency
    frame["TRANSFORMATION"] = spec.transformation
    frame["AVAILABILITY_ASSUMPTION_DAYS_AFTER_PERIOD_END"] = spec.availability_days
    frame["SOURCE_LAST_UPDATED"] = data.get("dataLastUpdated")
    frame["RETRIEVED_ON"] = date.today().isoformat()
    frame["SOURCE_URL"] = f"https://tablebuilder.singstat.gov.sg/table/TS/{spec.table_id}"
    frame["RATIONALE"] = spec.rationale
    return frame


def build_monthly_panel(
    observations: pd.DataFrame,
    start_month: str,
    end_month: str,
    information_cutoff: str | None = None,
) -> pd.DataFrame:
    panel = pd.DataFrame(
        {
            "RENT_APPROVAL_DATE": pd.date_range(
                pd.Timestamp(start_month), pd.Timestamp(end_month), freq="MS"
            )
        }
    )
    for spec in SERIES:
        source = observations.loc[
            observations["FEATURE"].eq(spec.feature) & observations["VALUE"].notna(),
            ["AVAILABLE_DATE", "VALUE"],
        ].sort_values("AVAILABLE_DATE")
        if information_cutoff is not None:
            cutoff = pd.Timestamp(information_cutoff)
            source = source.loc[source["AVAILABLE_DATE"] < cutoff]
        matched = pd.merge_asof(
            panel[["RENT_APPROVAL_DATE"]].sort_values("RENT_APPROVAL_DATE"),
            source,
            left_on="RENT_APPROVAL_DATE",
            right_on="AVAILABLE_DATE",
            direction="backward",
            allow_exact_matches=False,
        )
        if (matched["AVAILABLE_DATE"] >= matched["RENT_APPROVAL_DATE"]).fillna(False).any():
            raise ValueError(f"Availability leakage detected for {spec.feature}")
        panel[spec.feature] = matched["VALUE"].to_numpy()

    feature_columns = [spec.feature for spec in SERIES]
    if panel[feature_columns].isna().any().any():
        missing = panel.loc[panel[feature_columns].isna().any(axis=1), "RENT_APPROVAL_DATE"]
        raise ValueError(f"Macro panel has missing values for: {missing.dt.strftime('%Y-%m').tolist()}")
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Folder containing train.csv; overrides CS5228_DATA_ROOT and auto-detection",
    )
    parser.add_argument("--start-month", default="2021-01-01")
    parser.add_argument("--end-month", default="2026-07-01")
    parser.add_argument(
        "--information-cutoff",
        default=None,
        help=(
            "Optional fixed forecast-origin date. Observations available on or "
            "after this date are excluded and later months carry the last earlier value."
        ),
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    data_root = resolve_data_root(project_root, args.data_root)
    output_dir = data_root / "macro"
    output_dir.mkdir(parents=True, exist_ok=True)

    observations = pd.concat(
        [download_series(spec) for spec in SERIES], ignore_index=True
    ).sort_values(["FEATURE", "PERIOD_DATE"])
    panel = build_monthly_panel(
        observations,
        args.start_month,
        args.end_month,
        information_cutoff=args.information_cutoff,
    )

    observations.to_csv(output_dir / "sg_macro_observations.csv", index=False)
    panel.to_csv(output_dir / "sg_macro_features_monthly.csv", index=False)
    print(
        json.dumps(
            {
                "features": [spec.feature for spec in SERIES],
                "months": len(panel),
                "start_month": str(panel["RENT_APPROVAL_DATE"].min().date()),
                "end_month": str(panel["RENT_APPROVAL_DATE"].max().date()),
                "information_cutoff": args.information_cutoff,
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
