"""Task 3: planning-area demographics from OneMap popapi (2020 census).

OneMap serves demographics ONLY at planning-area granularity (no subzone; region
must be rolled up separately). Keyed by our planning-area names so it joins onto
schools / HDB / MRT nodes (all carry PLANNING_AREA).

Every (endpoint, planning-area) response is cached to disk, so the short-lived
OneMap token (3-day expiry) is spent at most once per cell.

Output: sg-demographics-planning-area.csv — derived features per planning area.
Raw JSON is kept in _onemap_cache/ for anything not flattened here.
"""

import os
import re
import json
import time
import requests
import pandas as pd

ROOT = "toy_data/graph_data"
PA_CSV = "toy_data/zoning_data/planning_area.csv"
CACHE = f"{ROOT}/_onemap_cache"
OUT = f"{ROOT}/sg-demographics-planning-area.csv"
KEYS = "data_collection_scripts/keys.txt"
BASE = "https://www.onemap.gov.sg/api/public/popapi/"
YEAR = "2020"

ENDPOINTS = ["getPopulationAgeGroup", "getEconomicStatus",
             "getTypeOfDwellingHousehold", "getHouseholdMonthlyIncomeWork",
             "getHouseholdSize", "getEthnicGroup", "getTenancy", "getOccupation"]

# our name -> OneMap planning-area name where a plain .title() doesn't match
NAME_FIX = {"seng kang": "Sengkang"}


def token():
    lines = [l.strip() for l in open(KEYS).read().splitlines()]
    return next(lines[i + 1] for i, l in enumerate(lines)
               if "onemap" in l.lower())


def fetch(ep, pa_query, tok):
    """Cached OneMap call -> list of records (may be empty)."""
    safe = re.sub(r"[^a-z0-9]+", "_", pa_query.lower())
    cpath = os.path.join(CACHE, f"{ep}__{safe}.json")
    if os.path.exists(cpath):
        return json.load(open(cpath))
    h = {"Authorization": tok}
    for attempt in range(4):
        try:
            r = requests.get(BASE + ep, headers=h,
                             params={"planningArea": pa_query, "year": YEAR},
                             timeout=30)
            if r.status_code == 200:
                j = r.json()
                data = j if isinstance(j, list) else []
                os.makedirs(CACHE, exist_ok=True)
                json.dump(data, open(cpath, "w"))
                time.sleep(0.15)
                return data
            if r.status_code in (429, 500, 502, 503):
                time.sleep(2 * (attempt + 1))
                continue
            return []
        except Exception:
            time.sleep(2 * (attempt + 1))
    return []


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def pct(a, b):
    return round(100 * a / b, 2) if b else None


def features_for(recs):
    """Compute derived demographic features from the 8 endpoints' records."""
    f = {}
    age = recs["getPopulationAgeGroup"]
    if age:
        total = sum(num(r.get("total")) for r in age)
        buckets = {}
        for r in age:
            for k, v in r.items():
                if k.startswith("age_"):
                    lo = int(re.search(r"age_(\d+)", k).group(1))
                    buckets[lo] = buckets.get(lo, 0) + num(v)
        y = sum(v for lo, v in buckets.items() if lo <= 10)      # 0-14
        w = sum(v for lo, v in buckets.items() if 15 <= lo <= 60)  # 15-64
        e = sum(v for lo, v in buckets.items() if lo >= 65)      # 65+
        f.update(total_population=int(total), pct_age_0_14=pct(y, total),
                 pct_age_15_64=pct(w, total), pct_age_65_plus=pct(e, total))

    econ = recs["getEconomicStatus"]
    if econ:
        emp = sum(num(r.get("employed")) for r in econ)
        un = sum(num(r.get("unemployed")) for r in econ)
        f["unemployment_rate"] = pct(un, emp + un)

    dw = recs["getTypeOfDwellingHousehold"]
    if dw:
        d = dw[0]
        hdb = num(d.get("total_hdb"))
        priv = num(d.get("condominiums_and_other_apartments")) + num(d.get("landed_properties"))
        tot = hdb + priv + num(d.get("others"))
        f.update(total_households=int(tot), pct_hdb=pct(hdb, tot),
                 pct_private_housing=pct(priv, tot))

    inc = recs["getHouseholdMonthlyIncomeWork"]
    if inc:
        d = inc[0]
        t = num(d.get("total"))
        f["pct_hh_income_ge_10k"] = pct(num(d.get("sgd_10000_over")), t)
        f["pct_hh_income_below_2k"] = pct(
            num(d.get("below_sgd_1000")) + num(d.get("sgd_1000_to_1999")), t)

    hs = recs["getHouseholdSize"]
    if hs:
        d = hs[0]
        keys = {"person1": 1, "person2": 2, "person3": 3, "person4": 4,
                "person5": 5, "person6": 6, "person7": 7, "person_more_8": 9}
        tot = sum(num(d.get(k)) for k in keys)
        ppl = sum(num(d.get(k)) * w for k, w in keys.items())
        f["avg_household_size"] = round(ppl / tot, 2) if tot else None

    eth = recs["getEthnicGroup"]
    if eth:
        d = eth[0]
        t = num(d.get("chinese")) + num(d.get("malays")) + num(d.get("indian")) + num(d.get("others"))
        f.update(pct_chinese=pct(num(d.get("chinese")), t),
                 pct_malay=pct(num(d.get("malays")), t),
                 pct_indian=pct(num(d.get("indian")), t))

    ten = recs["getTenancy"]
    if ten:
        d = ten[0]
        t = num(d.get("owner")) + num(d.get("tenant")) + num(d.get("others"))
        f["pct_owner_occupied"] = pct(num(d.get("owner")), t)

    occ = recs["getOccupation"]
    if occ:
        d = occ[0]
        pmet = (num(d.get("professionals")) + num(d.get("associate_professionals_tech"))
                + num(d.get("senior_officials_managers")))
        t = sum(num(v) for k, v in d.items() if k not in ("year", "planning_area"))
        f["pct_pmet"] = pct(pmet, t)

    return f


def main():
    pas = pd.read_csv(PA_CSV).PLANNING_AREA.tolist()
    tok = token()
    rows = []
    for name in pas:
        query = NAME_FIX.get(name, name.title())
        recs = {ep: fetch(ep, query, tok) for ep in ENDPOINTS}
        feat = features_for(recs)
        status = "ok" if feat.get("total_population") else "no_data"
        rows.append({"planning_area": name, "onemap_query": query,
                     "year": int(YEAR), "data_status": status, **feat})
        print(f"[{status:7}] {name:24} pop={feat.get('total_population')}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    ok = (out.data_status == "ok").sum()
    print(f"\n{len(out)} planning areas -> {OUT}")
    print(f"with data: {ok} | no_data: {len(out)-ok}")
    print("\nsample (richest areas):")
    cols = ["planning_area", "total_population", "pct_age_65_plus", "pct_hdb",
            "pct_owner_occupied", "pct_pmet", "avg_household_size"]
    print(out[out.data_status == "ok"].nlargest(6, "total_population")[cols]
          .to_string(index=False))


if __name__ == "__main__":
    main()
