"""Task 4: mall size/importance features from Wikipedia (MediaWiki API, free).

For each mall we resolve its Wikipedia article (direct title, else search),
verify it's actually a shopping mall (Infobox shopping mall / "shopping
centre"), and parse the infobox for:
  floor_area (sqm, GFA/NLA), floors, number_of_stores, number_of_anchors,
  opening year, developer, owner.

Coverage is inherently partial — major malls have articles, smaller ones don't;
`wiki_status` records match / no_article / not_a_mall so gaps are explicit.

Output: sg-mall-features-wikipedia.csv (one row per mall).
"""

import os
import re
import json
import time
import requests
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))      # final_report: data/config.py holds every path of the project
from data import config as cfg      # noqa: E402
MALLS_CSV = str(cfg.SHOPPING_MALLS_CSV)
CACHE = str(cfg.WIKI_CACHE_DIR)
OUT = str(cfg.MALL_FEATURES_WIKIPEDIA_CSV)
API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "sg-mrt-graph/1.0 (non-commercial research; mall features)"}


def api_get(params, tries=6):
    params = {**params, "format": "json", "formatversion": "2"}
    for a in range(tries):
        r = requests.get(API, params=params, headers=HEADERS, timeout=40)
        if r.status_code == 429:                       # rate limited: back off
            wait = int(r.headers.get("Retry-After", "0")) or 5 * (a + 1)
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()


def resolve_wikitext(name):
    """Return (title, wikitext) for a mall, or (None, None)."""
    safe = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    cpath = os.path.join(CACHE, f"{safe}.json")
    if os.path.exists(cpath):
        d = json.load(open(cpath))
        return d.get("title"), d.get("wikitext")

    title, wikitext = None, None
    # 1) direct title (follow redirects)
    j = api_get({"action": "query", "prop": "revisions", "rvprop": "content",
                 "rvslots": "main", "redirects": "1", "titles": name})
    page = j.get("query", {}).get("pages", [{}])[0]
    if "revisions" in page:
        title = page.get("title")
        wikitext = page["revisions"][0]["slots"]["main"]["content"]
    else:
        # 2) search fallback
        time.sleep(0.5)
        s = api_get({"action": "query", "list": "search",
                     "srsearch": f"{name} shopping mall Singapore", "srlimit": 1})
        hits = s.get("query", {}).get("search", [])
        if hits:
            t = hits[0]["title"]
            time.sleep(0.5)
            j2 = api_get({"action": "query", "prop": "revisions",
                          "rvprop": "content", "rvslots": "main",
                          "redirects": "1", "titles": t})
            p2 = j2.get("query", {}).get("pages", [{}])[0]
            if "revisions" in p2:
                title = p2.get("title")
                wikitext = p2["revisions"][0]["slots"]["main"]["content"]
    os.makedirs(CACHE, exist_ok=True)
    json.dump({"title": title, "wikitext": wikitext}, open(cpath, "w"))
    time.sleep(0.5)
    return title, wikitext


def clean(v):
    v = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", v)   # [[a|b]] -> b
    v = re.sub(r"\{\{[^{}]*\}\}", "", v)                        # drop templates
    v = re.sub(r"<[^>]+>", " ", v)                             # drop html
    v = re.sub(r"'{2,}", "", v)
    return re.sub(r"\s+", " ", v).strip(" .,")


def field(infobox, key):
    m = re.search(r"\|\s*" + key + r"\s*=\s*(.*)", infobox)
    return m.group(1).strip() if m else None


def parse(wikitext):
    """Extract mall features from the Infobox shopping mall block."""
    low = wikitext.lower()
    is_mall = ("infobox shopping mall" in low or "shopping centre" in low
               or "shopping mall" in low)
    if not is_mall:
        return None
    f = {}
    fa = field(wikitext, "floor_area")
    if fa:
        # {{convert|N|unit}} / {{cvt|N|unit}}, else "N unit"; unit sqft or m2.
        m = re.search(r"\{\{\s*(?:convert|cvt)\s*\|\s*([\d,]+)\s*\|\s*"
                      r"(sq\s?ft|sqft|ft2|m2|sqm|m²)", fa, re.I)
        if not m:
            m = re.search(r"([\d,]+)\s*(sq\s?ft|sqft|ft2|m2|sqm|m²|"
                          r"square\s+(?:feet|met\w+))", fa, re.I)
        if m:
            val = int(m.group(1).replace(",", ""))
            unit = m.group(2).lower()
            if "ft" in unit or "feet" in unit:
                val = round(val * 0.092903)          # sqft -> sqm
            f["floor_area_sqm"] = val
            f["floor_area_type"] = ("net_lettable" if "lettable" in fa.lower()
                                    else "gross" if "gross" in fa.lower() else "unknown")
    for col, key in [("floors", "floors"), ("num_stores", "number_of_stores"),
                     ("num_anchors", "number_of_anchors")]:
        v = field(wikitext, key)
        if v:
            m = re.search(r"\d[\d,]*", v)              # must start with a digit
            if m:
                f[col] = int(m.group(0).replace(",", ""))
    op = field(wikitext, "opening_date")
    if op:
        yr = re.search(r"(19|20)\d{2}", op)
        if yr:
            f["opening_year"] = int(yr.group(0))
    for col, key in [("developer", "developer"), ("owner", "owner")]:
        v = field(wikitext, key)
        if v:
            v = re.split(r"\s\|\s", v)[0]             # stop at next infobox field
            f[col] = clean(v)[:80] or None
    return f


def main():
    malls = pd.read_csv(MALLS_CSV)
    rows = []
    for name in malls.NAME:
        title, wt = resolve_wikitext(name)
        if not wt:
            rows.append({"mall_name": name, "wiki_status": "no_article"})
            continue
        feat = parse(wt)
        if feat is None:
            rows.append({"mall_name": name, "wiki_title": title,
                         "wiki_status": "not_a_mall"})
        else:
            rows.append({"mall_name": name, "wiki_title": title,
                         "wiki_status": "ok", **feat})

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    st = out.wiki_status.value_counts().to_dict()
    print(f"{len(out)} malls -> {OUT}")
    print("status:", st)
    ok = out[out.wiki_status == "ok"]
    for c in ["floor_area_sqm", "num_stores", "floors", "opening_year"]:
        if c in ok:
            print(f"  {c}: populated for {ok[c].notna().sum()}/{len(ok)} matched malls")
    print("\nsample:")
    cols = [c for c in ["mall_name", "floor_area_sqm", "num_stores", "floors",
                        "opening_year", "developer"] if c in ok]
    print(ok.nlargest(8, "floor_area_sqm")[cols].to_string(index=False))


if __name__ == "__main__":
    main()
