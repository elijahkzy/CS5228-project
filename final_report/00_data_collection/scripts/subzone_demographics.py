"""Task 3 (subzone): genuine per-subzone demographics from the 2020 Census via
data.gov.sg (CKAN API, free, no key). This is REAL subzone-resolution data, not
imputed from the parent planning area.

Dataset d_d95ae740... = "Resident Population by Planning Area/Subzone, Age Group
and Sex (Census 2020)". Rows are a mix of the national "Total", planning-area
"<PA> - Total" delimiters, and bare subzone rows. We keep the subzone rows,
track the current planning area from the delimiters, and left-join onto our 224
subzone keys.

Only basic demographics are published at subzone level (population, age, sex);
income/occupation/etc. are not (counts too small) — those stay planning-area only.

Output: sg-demographics-subzone.csv (our 224 subzones + features + match status).
"""

import re
import requests
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))      # final_report: data/config.py holds every path of the project
from data import config as cfg      # noqa: E402
SUBZONE_CSV = str(cfg.SUBZONE_CSV)
OUT = str(cfg.DEMOGRAPHICS_SUBZONE_CSV)
CKAN = "https://data.gov.sg/api/action/datastore_search"
RID = "d_d95ae740c0f8961a0b10435836660ce0"   # Census 2020, PA/subzone x age x sex
YEAR = 2020


def num(x):
    try:
        return float(str(x).replace(",", ""))
    except (TypeError, ValueError):
        return 0.0


def age_lo(col):
    m = re.search(r"_(\d+)", col)
    return int(m.group(1)) if m else None


def fetch_records():
    r = requests.get(CKAN, params={"resource_id": RID, "limit": 1000}, timeout=60)
    recs = r.json()["result"]["records"]
    return sorted(recs, key=lambda d: int(d["_id"]))


def subzone_features(recs):
    """Return {subzone_lower: feature dict}, tracking parent planning area."""
    age_cols = [c for c in recs[0] if c.startswith("Total_") and c != "Total_Total"]
    out = {}
    current_pa = None
    for row in recs:
        label = row["Number"].strip()
        if label == "Total":
            continue
        if label.endswith("- Total"):
            current_pa = label[:-len("- Total")].strip()
            continue
        total = num(row["Total_Total"])
        y = sum(num(row[c]) for c in age_cols if age_lo(c) <= 10)
        w = sum(num(row[c]) for c in age_cols if 15 <= age_lo(c) <= 60)
        e = sum(num(row[c]) for c in age_cols if age_lo(c) >= 65)
        males = num(row.get("Males_Total"))
        out[label.lower()] = {
            "planning_area": (current_pa or "").lower(),
            "total_population": int(total),
            "pct_age_0_14": round(100 * y / total, 2) if total else None,
            "pct_age_15_64": round(100 * w / total, 2) if total else None,
            "pct_age_65_plus": round(100 * e / total, 2) if total else None,
            "pct_male": round(100 * males / total, 2) if total else None,
        }
    return out


def main():
    ours = pd.read_csv(SUBZONE_CSV)
    feats = subzone_features(fetch_records())
    print(f"census subzones available: {len(feats)} | our keys: {len(ours)}")

    rows = []
    for sz in ours.SUBZONE:
        f = feats.get(str(sz).lower())
        if f:
            rows.append({"subzone": sz, "year": YEAR, "data_status": "ok", **f})
        else:
            rows.append({"subzone": sz, "year": YEAR, "data_status": "no_match"})
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)

    ok = (out.data_status == "ok").sum()
    print(f"\n{len(out)} subzones -> {OUT}")
    print(f"matched to census: {ok} | unmatched: {len(out)-ok}")
    if len(out) - ok:
        print("unmatched keys:", out.loc[out.data_status != "ok", "subzone"].tolist())
    print("\nsample (genuine within-PA variation, e.g. Bedok subzones):")
    ex = out[(out.data_status == "ok") & (out.get("planning_area") == "bedok")]
    print(ex[["subzone", "total_population", "pct_age_65_plus"]].head(6).to_string(index=False))


if __name__ == "__main__":
    main()
