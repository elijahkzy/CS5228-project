"""Task 5: collect latest school entry-cutoff + fee-tier features from MOE
SchoolFinder, segmented into one CSV per level (schemas differ by level).

Source: SchoolFinder per-school detail pages (SSR HTML) at
  https://www.moe.gov.sg/schoolfinder/schooldetail/<slug>
The cut-off data is embedded as JSON in the RSC payload:
  Secondary : {"course":"POSTING GROUP N","lower","upper"}   (PSLE AL)
  JC / MI   : {"stream":"Arts|Science","code","lower","upper","group":"L1R5"}
Primary schools have no academic cut-off (proximity/balloting) and are not
fetched. Fees are not per-school on the page (shared client-filtered schedule),
so fee_tier is derived from the school's governance TYPE_CODE — the actual
determinant of SG school fees.

Politeness: descriptive UA, on-disk cache (fetch each school once), throttle
between live requests.

Outputs (one per level bucket):
  sg-schools-primary.csv, sg-schools-secondary.csv,
  sg-schools-junior_college.csv, sg-schools-mixed.csv
"""

import os
import re
import time
import requests
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))      # final_report: data/config.py holds every path of the project
from data import config as cfg      # noqa: E402
SCHOOLS_CSV = str(cfg.SCHOOLS_CSV)
CACHE_DIR = str(cfg.SCHOOL_CACHE_DIR)
BASE = "https://www.moe.gov.sg/schoolfinder/schooldetail/"
SITEMAP = "https://www.moe.gov.sg/sitemap.xml"
HEADERS = {"User-Agent": "sg-mrt-graph/1.0 (non-commercial research; school features)"}
THROTTLE_S = 1.0

IDENT_COLS = ["NAME", "MAINLEVEL_CODE", "TYPE_CODE", "NATRUE_CODE",
              "PLANNING_AREA", "REGION", "MRT_STATIONS", "LATITUDE",
              "LONGITUDE", "URL"]


def level_bucket(mainlevel):
    m = str(mainlevel).lower()
    if m == "primary":
        return "primary"
    if m.startswith("secondary"):
        return "secondary"
    if "junior college" in m or "centralised institute" in m:
        return "junior_college"
    if m.startswith("mixed"):
        return "mixed"
    return "other"


def slugify(name):
    s = str(name).lower().strip().replace("&", " and ")
    s = re.sub(r"[.'’,()]", "", s)        # drop punctuation
    s = re.sub(r"[^a-z0-9]+", "-", s)     # non-alnum -> hyphen
    return s.strip("-")


def fee_tier(type_code):
    t = str(type_code).lower()
    if "independent" in t and "special" in t:
        return "specialised_independent (high, sets own)"
    if "independent" in t:
        return "independent (high, sets own)"
    if "special" in t:
        return "specialised (varies)"
    if "autonomous" in t:
        return "autonomous (low + autonomous fee)"
    if "aided" in t:
        return "government_aided (low, nominal)"
    if "government" in t:
        return "government (low, nominal)"
    return t or "unknown"


def load_sitemap_slugs():
    """Authoritative schooldetail slugs from the sitemap (cached)."""
    cpath = os.path.join(CACHE_DIR, "sitemap_slugs.txt")
    if os.path.exists(cpath):
        return set(open(cpath).read().split())
    r = requests.get(SITEMAP, headers=HEADERS, timeout=60)
    slugs = set(re.findall(r"schooldetail/([a-z0-9-]+)", r.text))
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cpath, "w") as f:
        f.write("\n".join(sorted(slugs)))
    return slugs


def resolve_slugs(name, sitemap):
    """Map a school to its real slug(s). Through-train schools are split into
    per-level pages (`<base>-secondary`, `<base>-junior-college`, ...)."""
    base = slugify(name)
    if base in sitemap:
        return [base]
    return sorted(s for s in sitemap if s.startswith(base + "-"))


def fetch(slug, live_counter):
    """Return (html, status). Cached-first; 404 => slug miss; throttle live."""
    cpath = os.path.join(CACHE_DIR, f"{slug}.html")
    if os.path.exists(cpath):
        return open(cpath, encoding="utf-8").read(), "cache"
    for attempt in range(3):
        try:
            r = requests.get(BASE + slug, headers=HEADERS, timeout=40,
                             allow_redirects=True)
        except Exception as ex:
            print(f"  ! {slug} net error: {ex}")
            time.sleep(3 * (attempt + 1))
            continue
        finally:
            pass
        live_counter[0] += 1
        time.sleep(THROTTLE_S)                 # be polite between live hits
        if r.status_code == 404:
            return None, "404_slug_miss"
        if r.status_code == 200 and len(r.text) > 5000:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(cpath, "w", encoding="utf-8") as f:
                f.write(r.text)
            return r.text, "ok"
        time.sleep(2 * (attempt + 1))
    return None, "fetch_failed"


def parse_cutoffs(html):
    """Return list of dicts from the embedded cut-off JSON objects."""
    h = html.replace('\\"', '"')
    objs = set(re.findall(r'\{[^{}]*"(?:course|stream)":"[^"}]*"[^{}]*\}', h))
    recs = []
    for o in objs:
        f = dict(re.findall(r'"(\w+)":"?([^",}]*)"?', o))
        recs.append(f)
    return recs


def cutoff_columns(recs):
    """Flatten cut-off records into per-school columns (level-agnostic)."""
    out = {}
    for f in recs:
        lo, up = f.get("lower"), f.get("upper")
        if "course" in f and f["course"]:                 # Secondary posting groups
            m = re.search(r"(\d+)", f["course"])
            if m:
                out[f"pg{m.group(1)}_min"] = lo
                out[f"pg{m.group(1)}_max"] = up
        elif "stream" in f and f["stream"]:               # JC L1R5 streams
            s = re.sub(r"[^a-z0-9]+", "_", f["stream"].lower()).strip("_")
            out[f"l1r5_{s}_min"] = lo
            out[f"l1r5_{s}_max"] = up
            if f.get("code"):
                out[f"{s}_course_code"] = f["code"]
    return out


def main():
    df = pd.read_csv(SCHOOLS_CSV)
    sitemap = load_sitemap_slugs()
    buckets = {"primary": [], "secondary": [], "junior_college": [],
               "mixed": [], "other": []}
    live = [0]
    stats = {}

    for _, row in df.iterrows():
        bucket = level_bucket(row.MAINLEVEL_CODE)
        rec = {c: row.get(c) for c in IDENT_COLS}
        rec["fee_tier"] = fee_tier(row.TYPE_CODE)

        if bucket == "primary":
            rec["resolved_slugs"] = ""
            rec["scrape_status"] = "skipped_no_cutoff"
            buckets[bucket].append(rec)
            continue

        # Resolve real slug(s) — through-train schools have one page per level.
        slugs = resolve_slugs(row.NAME, sitemap)
        rec["resolved_slugs"] = ";".join(slugs)
        if not slugs:
            rec["scrape_status"] = "404_slug_miss"
            buckets[bucket].append(rec)
            continue

        cols, any_ok = {}, False
        for slug in slugs:
            html, status = fetch(slug, live)
            if html is not None:
                any_ok = True
                cols.update(cutoff_columns(parse_cutoffs(html)))
        rec.update(cols)
        rec["scrape_status"] = ("ok" if cols else
                                "ok_no_cutoff_found" if any_ok else "fetch_failed")
        buckets[bucket].append(rec)

    for name, rows in buckets.items():
        if not rows:
            continue
        out = pd.DataFrame(rows)
        path = cfg.SCHOOL_CSV_TEMPLATE.format(name=name)
        out.to_csv(path, index=False)
        sc = out.scrape_status.value_counts().to_dict()
        stats[name] = (len(out), sc, path)

    print(f"live fetches: {live[0]}\n")
    for name, (n, sc, path) in stats.items():
        print(f"[{name}] {n} schools -> {path}")
        print(f"   status: {sc}")
        # show a couple with cutoffs
        cut = [c for c in pd.DataFrame(buckets[name]).columns
               if c.startswith(("pg", "l1r5"))]
        if cut:
            ex = pd.DataFrame(buckets[name])
            ex = ex[ex.scrape_status == "ok"].head(2)
            for _, r in ex.iterrows():
                vals = {c: r[c] for c in cut if pd.notna(r.get(c))}
                print(f"   e.g. {r['NAME']}: {vals}")


if __name__ == "__main__":
    main()
