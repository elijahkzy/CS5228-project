"""Task 2: train-station node features from LTA DataMall PV/Train (source #4).

DataMall reports ridership at the physical-station level: interchange rows use a
combined slash code (e.g. 'EW24/NS1') with tap volumes already summed across the
interchange. So we aggregate the hourly CSV to one row per PT_CODE, explode the
slash codes to individual codes, and match those against each NAME node's
member_codes. Planned-only stations (no operational code) get null ridership and
are flagged matched=False.

The API only exposes a rolling ~3-month window (no history back to 2021), so we
pin and pull each available month explicitly and save one self-contained
node-feature snapshot per month. Ridership = monthly totals for that month; it
encodes station popularity/busyness, not a historical time series.

Output: sg-mrt-nodes-features-<YYYYMM>.csv, one per month.
OSM station tags (source #6) are added in a later step.
"""

import io
import os
import zipfile
import requests
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))      # final_report: data/config.py holds every path of the project
from data import config as cfg      # noqa: E402
NODES_CSV = str(cfg.MRT_NODES_CSV)
CACHE_DIR = str(cfg.PV_CACHE_DIR)          # raw monthly CSVs, fetched at most once
KEYS_TXT = str(cfg.KEYS_FILE)
PV_TRAIN = "https://datamall2.mytransport.sg/ltaodataservice/PV/Train"

# The PV/Train endpoint has its own strict per-endpoint quota (separate from the
# general 10M/day account cap). So: cache every month to disk, fetch each month
# at most once, NEVER retry against a quota error, and stop-and-resume cleanly.


class QuotaExceeded(Exception):
    """Raised when PV/Train returns its rate/quota violation (do not retry)."""

# The only months the API currently serves (probed 2026-09-02). Pinned for
# reproducibility rather than relying on "latest".
MONTHS = ["202605", "202606", "202607"]

AM_PEAK = {7, 8, 9}
PM_PEAK = {17, 18, 19}


def load_account_key():
    # keys.txt may be in cwd or alongside this script (post-reorg).
    here = os.path.dirname(os.path.abspath(__file__))
    path = next((p for p in (KEYS_TXT, os.path.join(here, "keys.txt"))
                 if os.path.exists(p)), KEYS_TXT)
    lines = [l.strip() for l in open(path).read().splitlines()]
    return next(lines[i + 1] for i, l in enumerate(lines)
               if "account key" in l.lower())


def _cache_path(date):
    return os.path.join(CACHE_DIR, f"transport_node_train_{date}.csv")


def fetch_pv_train(key, date, net_tries=3):
    """Return (df, source) of PV/Train hourly data for a YYYYMM month.

    Cached-first: a month already on disk costs NO API call. A quota response
    raises QuotaExceeded immediately (no retry — retrying only drains the
    per-endpoint bucket). Only genuine transient network/5xx errors are retried.
    """
    import time
    cpath = _cache_path(date)
    if os.path.exists(cpath):
        return pd.read_csv(cpath), "cache"

    h = {"AccountKey": key, "accept": "application/json"}
    last = None
    for attempt in range(net_tries):
        try:
            r = requests.get(PV_TRAIN, headers=h, params={"Date": date},
                             timeout=40)
        except Exception as ex:                      # connection/timeout: retry
            last = ex
            print(f"  ! {date} network error: {ex} -> retry in {3*(attempt+1)}s")
            time.sleep(3 * (attempt + 1))
            continue

        body = r.text or ""
        if "Quota limit" in body or "Rate limit" in body:
            raise QuotaExceeded(date)                # stop; do not burn more calls
        try:
            r.raise_for_status()
            val = r.json().get("value", [])
            if not val:
                raise ValueError("empty 'value' (month not available yet?)")
            blob = requests.get(val[0]["Link"], timeout=180).content
            zf = zipfile.ZipFile(io.BytesIO(blob))
            df = pd.read_csv(zf.open(zf.namelist()[0]))
            os.makedirs(CACHE_DIR, exist_ok=True)
            df.to_csv(cpath, index=False)            # cache so we never re-fetch
            return df, "api"
        except Exception as ex:                      # transient (bad zip/5xx): retry
            last = ex
            print(f"  ! {date} transient error: {ex} -> retry in {3*(attempt+1)}s")
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"PV/Train fetch failed for {date}: {last}")


def aggregate_by_ptcode(df):
    """Collapse hourly rows to one feature row per combined PT_CODE."""
    df = df.copy()
    df["vol"] = df.TOTAL_TAP_IN_VOLUME + df.TOTAL_TAP_OUT_VOLUME
    is_wd = df.DAY_TYPE.str.upper().eq("WEEKDAY")
    is_am = df.TIME_PER_HOUR.isin(AM_PEAK)
    is_pm = df.TIME_PER_HOUR.isin(PM_PEAK)

    g = df.groupby("PT_CODE")
    feat = pd.DataFrame({
        "pv_tap_in": g.TOTAL_TAP_IN_VOLUME.sum(),
        "pv_tap_out": g.TOTAL_TAP_OUT_VOLUME.sum(),
        "pv_total": g.vol.sum(),
        "pv_weekday_total": df[is_wd].groupby("PT_CODE").vol.sum(),
        "pv_weekend_total": df[~is_wd].groupby("PT_CODE").vol.sum(),
        "pv_am_peak_total": df[is_wd & is_am].groupby("PT_CODE").vol.sum(),
        "pv_pm_peak_total": df[is_wd & is_pm].groupby("PT_CODE").vol.sum(),
    }).fillna(0)
    # busiest hour (by total throughput, across all day types)
    hourly = df.groupby(["PT_CODE", "TIME_PER_HOUR"]).vol.sum()
    busiest = hourly.groupby(level=0).idxmax().map(lambda t: t[1])
    feat["pv_busiest_hour"] = busiest
    feat["year_month"] = df.YEAR_MONTH.iloc[0]
    return feat.reset_index()


def code_to_ptcode(feat):
    """Map each individual code (upper) -> its combined PT_CODE row index."""
    m = {}
    for idx, pc in feat.PT_CODE.items():
        for c in str(pc).upper().split("/"):
            m[c] = idx
    return m


PV_COLS = ["pv_tap_in", "pv_tap_out", "pv_total", "pv_weekday_total",
           "pv_weekend_total", "pv_am_peak_total", "pv_pm_peak_total",
           "pv_busiest_hour", "year_month"]


def build_month(nodes, key, date):
    """Fetch one month, join to nodes, write a snapshot CSV, return summary."""
    raw, source = fetch_pv_train(key, date)
    feat = aggregate_by_ptcode(raw)
    code2idx = code_to_ptcode(feat)

    # For each node, find the DataMall row via any operational member_code.
    matched_idx = []
    for codes in nodes.member_codes:
        hit = None
        for c in str(codes).split(";"):
            j = code2idx.get(c.upper())
            if j is not None:
                hit = j
                break
        matched_idx.append(hit)

    attached = pd.DataFrame(
        [feat.loc[j, PV_COLS].to_dict() if j is not None
         else {c: np.nan for c in PV_COLS} for j in matched_idx]
    )
    out = pd.concat([nodes.reset_index(drop=True), attached], axis=1)
    out["pv_matched"] = [j is not None for j in matched_idx]

    out_path = cfg.MRT_NODE_FEATURES_CSV_TEMPLATE.format(date=date)
    out.to_csv(out_path, index=False)

    used = {j for j in matched_idx if j is not None}
    return {
        "date": raw.YEAR_MONTH.iloc[0], "rows": len(raw), "source": source,
        "pt_codes": raw.PT_CODE.nunique(), "matched": int(out.pv_matched.sum()),
        "unmatched": int((~out.pv_matched).sum()),
        "unused_ptcodes": len(set(feat.index) - used),
        "top": out.loc[out.pv_matched].nlargest(3, "pv_total")
                  [["NAME", "pv_total"]].values.tolist(),
        "path": out_path,
    }


def main():
    import time
    nodes = pd.read_csv(NODES_CSV)
    key = load_account_key()
    done, pending, api_calls = [], [], 0
    for date in MONTHS:
        # pace only genuine API calls; cached months are free and instant.
        if api_calls and not os.path.exists(_cache_path(date)):
            time.sleep(20)
        try:
            s = build_month(nodes, key, date)
        except QuotaExceeded:
            # Stop cleanly — whatever was fetched/cached is kept; resume later.
            print(f"[{date}] PV quota hit -> stopping. Re-run after the window "
                  f"resets; cached months won't be re-fetched.")
            pending = [d for d in MONTHS if not os.path.exists(_cache_path(d))]
            break
        if s["source"] == "api":
            api_calls += 1
        done.append(date)
        print(f"[{s['date']}] ({s['source']}) rows={s['rows']} "
              f"pt_codes={s['pt_codes']} | matched={s['matched']} "
              f"unmatched={s['unmatched']} unused_dm={s['unused_ptcodes']} "
              f"-> {s['path']}")
        print("   busiest:", ", ".join(f"{n} ({int(v):,})" for n, v in s["top"]))

    print(f"\ncollected: {done or '(none)'}  | api_calls_used: {api_calls}")
    if pending:
        print(f"pending (quota): {pending} — safe to re-run to resume.")


if __name__ == "__main__":
    main()
