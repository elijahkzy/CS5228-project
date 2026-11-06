"""Task 1 (option B): attach rail-distance weights to the collapsed edge table.

Primary source : OpenStreetMap track geometry (Overpass), per line.
Fallback       : haversine straight-line * detour factor, when OSM geometry is
                 missing or the projection is unreliable.

Every output edge is tagged with `method`:
  * osm                -> distance from OSM track polyline (3rd-party success)
  * haversine_fallback -> straight-line * DETOUR (OSM unavailable/unreliable)

Design notes for reliability (per the null-value concern):
  * Overpass is queried ONE LINE AT A TIME (small queries), with retries.
  * A per-line fetch that fails just routes that line's edges to fallback;
    it can never blank out a weight.
  * A projection residual guard + a sane-distance range guard send any
    dubious OSM measurement to fallback rather than emitting a wrong number.

Output: sg-mrt-edges-weighted.csv  (src_station, dst_station, distance_km, method)
Units  : kilometres, everywhere.
"""

import re
import time
import math
import requests
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))      # final_report: data/config.py holds every path of the project
from data import config as cfg      # noqa: E402
STN_CSV = str(cfg.MRT_STATIONS_CSV)
COLLAPSED_CSV = str(cfg.MRT_EDGES_COLLAPSED_CSV)
OUT_CSV = str(cfg.MRT_EDGES_WEIGHTED_CSV)

# Public Overpass mirrors, tried in order per attempt (failover). A missing /
# generic User-Agent gets a 406 from these endpoints, so we send a descriptive
# one identifying the tool and purpose.
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
HEADERS = {
    "User-Agent": "sg-mrt-graph/1.0 (non-commercial research; MRT rail-distance edge weights)",
}
DETOUR = 1.2                 # straight-line -> approx track length
RESIDUAL_MAX_M = 300.0       # station must lie within this of the OSM track
DIST_MIN_M, DIST_MAX_M = 120.0, 6000.0  # plausible adjacent-station spacing

# our line prefix -> (OSM route type, OSM ref). Prefixes not listed here always
# use the haversine fallback (CRL cr/cp, LRT bp/pe/pw/ptc, rts, ...).
PREFIX_TO_OSM = {
    "ns": ("subway", "NSL"),
    "ew": ("subway", "EWL"), "cg": ("subway", "EWL"),
    "ne": ("subway", "NEL"),
    "cc": ("subway", "CCL"), "ce": ("subway", "CCL"),
    "dt": ("subway", "DTL"), "de": ("subway", "DTL"),
    "te": ("subway", "TEL"),
    "js": ("subway", "JRL"), "jw": ("subway", "JRL"), "je": ("subway", "JRL"),
    "sk": ("light_rail", "SKLRT"), "se": ("light_rail", "SKLRT"),
    "sw": ("light_rail", "SKLRT"), "stc": ("light_rail", "SKLRT"),
}

_CODE_RE = re.compile(r"^([a-z]+?)(\d+)([a-z]*)$")
LAT0 = 1.35  # reference latitude for the local equirectangular projection
R_EARTH = 6371000.0


def parse_code(code):
    m = _CODE_RE.match(code)
    if m:
        return m.group(1), int(m.group(2)), m.group(3)
    return code, -1, ""


def to_xy(lat, lon):
    """Local equirectangular projection to metres (fine over Singapore)."""
    x = np.radians(lon) * math.cos(math.radians(LAT0)) * R_EARTH
    y = np.radians(lat) * R_EARTH
    return np.column_stack([np.ravel(x), np.ravel(y)])


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R_EARTH * math.asin(math.sqrt(a)) / 1000.0


def fetch_line_polyline(route_type, ref, tries=4):
    """Return an (N,2) array of xy vertices for one OSM line, or None."""
    q = (f'[out:json][timeout:60];'
         f'relation["route"="{route_type}"]["ref"="{ref}"];'
         f'out geom;')
    for attempt in range(tries):
        endpoint = OVERPASS_ENDPOINTS[attempt % len(OVERPASS_ENDPOINTS)]
        try:
            r = requests.post(endpoint, data={"data": q},
                              headers=HEADERS, timeout=90)
            r.raise_for_status()
            els = [e for e in r.json().get("elements", [])
                   if e.get("type") == "relation"]
            if not els:
                return None
            # First relation = one direction; its member ways span the line.
            coords = []
            for m in els[0].get("members", []):
                if m.get("type") == "way" and "geometry" in m:
                    coords += [(g["lat"], g["lon"]) for g in m["geometry"]]
            if len(coords) < 2:
                return None
            lat = np.array([c[0] for c in coords])
            lon = np.array([c[1] for c in coords])
            return to_xy(lat, lon)
        except Exception as ex:
            host = endpoint.split("/")[2]
            print(f"  ! {ref} attempt {attempt+1} ({host}) failed: {ex}")
            time.sleep(2 * (attempt + 1))  # backoff
    return None


def build_chainage(verts):
    """Cumulative distance (m) along the polyline at each vertex."""
    seg = np.linalg.norm(np.diff(verts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)]), seg


def project_point(pt, verts, cum):
    """Return (chainage_m, residual_m) of pt projected onto the polyline."""
    a = verts[:-1]
    b = verts[1:]
    ab = b - a
    ap = pt - a
    denom = np.einsum("ij,ij->i", ab, ab)
    denom[denom == 0] = 1e-9
    t = np.clip(np.einsum("ij,ij->i", ap, ab) / denom, 0.0, 1.0)
    proj = a + t[:, None] * ab
    d = np.linalg.norm(proj - pt, axis=1)
    i = int(np.argmin(d))
    seg_len = math.sqrt(denom[i])
    chainage = cum[i] + t[i] * seg_len
    return chainage, float(d[i])


def main():
    stn = pd.read_csv(STN_CSV).drop_duplicates(subset="CODE")
    code2name = dict(zip(stn.CODE, stn.NAME))
    code2lat = dict(zip(stn.CODE, stn.LATITUDE))
    code2lon = dict(zip(stn.CODE, stn.LONGITUDE))

    parsed = {c: parse_code(c) for c in stn.CODE}
    stn = stn.assign(line=[parsed[c][0] for c in stn.CODE],
                     num=[parsed[c][1] for c in stn.CODE],
                     suffix=[parsed[c][2] for c in stn.CODE])

    # --- along-line adjacent code pairs (same structure used to build edges).
    pairs = []  # (code_a, code_b, prefix)
    for prefix, grp in stn.groupby("line"):
        seq = grp.sort_values(["num", "suffix"]).CODE.tolist()
        for a, b in zip(seq, seq[1:]):
            pairs.append((a, b, prefix))

    # --- fetch OSM geometry once per needed line and project every station.
    needed_refs = {PREFIX_TO_OSM[p] for _, _, p in pairs if p in PREFIX_TO_OSM}
    lines = {}     # ref -> (verts, cum)
    for route_type, ref in sorted(needed_refs):
        print(f"fetching OSM line {ref} ({route_type}) ...")
        verts = fetch_line_polyline(route_type, ref)
        if verts is not None:
            cum, _ = build_chainage(verts)
            lines[ref] = (verts, cum)
            print(f"  ok: {len(verts)} vertices, {cum[-1]/1000:.1f} km track")
        else:
            print(f"  MISS: {ref} -> edges on this line will use fallback")
        time.sleep(1)  # be polite to the public endpoint

    def chainage_of(code):
        pfx = parsed[code][0]
        key = PREFIX_TO_OSM.get(pfx)
        if not key or key[1] not in lines:
            return None
        verts, cum = lines[key[1]]
        pt = to_xy(code2lat[code], code2lon[code])[0]
        ch, resid = project_point(pt, verts, cum)
        return (ch, resid) if resid <= RESIDUAL_MAX_M else None

    # --- weight each along-line pair, OSM first then fallback.
    rows = []
    for a, b, pfx in pairs:
        ca, cb = chainage_of(a), chainage_of(b)
        method, dist_km = None, None
        if ca is not None and cb is not None:
            d_m = abs(ca[0] - cb[0])
            if DIST_MIN_M <= d_m <= DIST_MAX_M:
                method, dist_km = "osm", d_m / 1000.0
        if method is None:
            dist_km = (haversine_km(code2lat[a], code2lon[a],
                                    code2lat[b], code2lon[b]) * DETOUR)
            method = "haversine_fallback"
        rows.append({"src_station": code2name[a], "dst_station": code2name[b],
                     "distance_km": round(dist_km, 3), "method": method})

    edges = pd.DataFrame(rows)
    # canonicalise to NAME order and collapse parallel edges (keep shortest).
    lo = edges[["src_station", "dst_station"]].min(axis=1)
    hi = edges[["src_station", "dst_station"]].max(axis=1)
    edges = edges.assign(src_station=lo, dst_station=hi)
    edges = (edges.sort_values("distance_km")
                  .drop_duplicates(subset=["src_station", "dst_station"], keep="first")
                  .sort_values(["src_station", "dst_station"])
                  .reset_index(drop=True))

    # sanity: must align 1:1 with the existing collapsed edge table.
    collapsed = pd.read_csv(COLLAPSED_CSV)
    key = lambda df: set(map(tuple, df[["src_station", "dst_station"]].values))
    missing = key(collapsed) - key(edges)
    extra = key(edges) - key(collapsed)

    edges.to_csv(OUT_CSV, index=False)
    print(f"\nweighted edges: {len(edges)} -> {OUT_CSV}")
    print(f"unmatched vs collapsed table  : missing={len(missing)} extra={len(extra)}")
    print("method breakdown:\n" + edges.method.value_counts().to_string())
    print(f"distance_km  min={edges.distance_km.min():.3f} "
          f"median={edges.distance_km.median():.3f} "
          f"max={edges.distance_km.max():.3f}")


if __name__ == "__main__":
    main()
