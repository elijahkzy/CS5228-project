"""Tag each MRT station with the shopping malls it is connected to via
sheltered walkways / underground tunnels (best-effort, from OSM).

There is no authoritative mall<->station link dataset, so we reconstruct it from
OSM pedestrian infrastructure: ways tagged covered=yes/arcade (sheltered),
tunnel=yes (underground), or highway=corridor (indoor). We build a graph of that
network and connect a station to a nearby mall (<= RADIUS_M straight-line) by:

  method C (primary)  : station and mall both attach (within *_TOL_M) to nodes in
                        the SAME connected component -> a real sheltered/tunnel
                        path exists between them. We also extract the path and
                        classify it underground (any tunnel edge) vs sheltered.
  method B (fallback) : C fails, but a SINGLE covered/tunnel way passes within
                        HEUR_TOL_M of BOTH the station and the mall (covers OSM
                        gaps where the network isn't node-connected).

Each linkage is tagged with the method used, per request.

Output: cfg.STATION_MALL_LINKS_CSV (data/features/raw/collected_data/mrt_data/sg-station-mall-links.csv)
        (station_name, mall_name, straight_m, method, connection_type, path_m)
"""

import os
import json
import math
import requests
import numpy as np
import pandas as pd
import networkx as nx

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))      # final_report: data/config.py holds every path of the project
from data import config as cfg      # noqa: E402
NODES_CSV = str(cfg.MRT_NODES_CSV)
MALLS_CSV = str(cfg.SHOPPING_MALLS_CSV)
OUT_CSV = str(cfg.STATION_MALL_LINKS_CSV)
OSM_CACHE = str(cfg.OSM_SHELTERED_NETWORK_JSON)

BBOX = (1.20, 103.58, 1.48, 104.05)
RADIUS_M = 400.0      # max straight-line station->mall to even consider
STATION_TOL_M = 110.0  # snap station centroid to nearest network node
MALL_TOL_M = 110.0    # snap mall point (building centroid) to nearest node
HEUR_TOL_M = 50.0     # method B: a way must pass this close to both
BRIDGE_EPS_M = 5.0    # de-fragment: join near-coincident endpoints across ways

LAT0 = 1.35
R_EARTH = 6371000.0

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
HEADERS = {"User-Agent": "sg-mrt-graph/1.0 (non-commercial research)"}


def to_xy(lat, lon):
    x = np.radians(np.asarray(lon)) * math.cos(math.radians(LAT0)) * R_EARTH
    y = np.radians(np.asarray(lat)) * R_EARTH
    return np.column_stack([np.ravel(x), np.ravel(y)])


def fetch_network():
    """Bulk-fetch covered/tunnel/corridor ways + nodes (cached to disk)."""
    if os.path.exists(OSM_CACHE):
        with open(OSM_CACHE) as f:
            return json.load(f)
    s, w, n, e = BBOX
    q = (f'[out:json][timeout:180];'
         f'(way["highway"~"footway|pedestrian|path|steps"]["covered"~"yes|arcade"]({s},{w},{n},{e});'
         f' way["highway"~"footway|pedestrian|path|steps"]["tunnel"="yes"]({s},{w},{n},{e});'
         f' way["highway"="corridor"]({s},{w},{n},{e}););'
         f'out body;>;out skel qt;')
    for i, ep in enumerate(OVERPASS_ENDPOINTS):
        try:
            r = requests.post(ep, data={"data": q}, headers=HEADERS, timeout=200)
            if r.status_code == 200 and r.text.strip().startswith("{"):
                data = r.json()
                os.makedirs(os.path.dirname(OSM_CACHE), exist_ok=True)
                with open(OSM_CACHE, "w") as f:
                    json.dump(data, f)
                print(f"fetched network via {ep.split('/')[2]} "
                      f"({len(data.get('elements', []))} elements)")
                return data
            print(f"  bad {ep.split('/')[2]}: {r.status_code} {r.text[:80]!r}")
        except Exception as ex:
            print(f"  err {ep.split('/')[2]}: {ex}")
    raise RuntimeError("could not fetch OSM sheltered network")


def edge_kind(tags):
    if tags.get("tunnel") == "yes":
        return "tunnel"
    if tags.get("highway") == "corridor":
        return "corridor"
    return "covered"


def build_graph(data):
    """Return (G, node_ids, node_xy, node2ways, way_kind)."""
    elements = data["elements"]
    node_elems = [e for e in elements if e["type"] == "node" and "lat" in e]
    ids = np.array([e["id"] for e in node_elems])
    xy = to_xy(np.array([e["lat"] for e in node_elems]),
               np.array([e["lon"] for e in node_elems]))
    xy_of = {nid: xy[k] for k, nid in enumerate(ids)}  # id -> (x, y) metres

    G = nx.Graph()
    node2ways = {}
    way_kind = {}
    for e in elements:
        if e["type"] != "way":
            continue
        kind = edge_kind(e.get("tags", {}))
        way_kind[e["id"]] = kind
        nodes = [nid for nid in e.get("nodes", []) if nid in xy_of]
        for nid in nodes:
            node2ways.setdefault(nid, []).append(e["id"])
        for a, b in zip(nodes, nodes[1:]):
            if G.has_edge(a, b):
                if kind == "tunnel":       # tunnel priority when ways overlap
                    G[a][b]["kind"] = "tunnel"
            else:
                length = float(np.hypot(*(xy_of[a] - xy_of[b])))
                G.add_edge(a, b, kind=kind, length=length)
    # De-fragment: OSM covered/tunnel ways are often drawn coincident but not
    # joined at a shared node, splitting the network into thousands of stray
    # components. Bridge nodes from *different* ways that sit within BRIDGE_EPS_M
    # with a zero-length 'bridge' edge (neutral kind, ignored when classifying).
    from collections import defaultdict
    ids_arr = np.array(list(G.nodes))
    xy_arr = np.array([xy_of[n] for n in ids_arr])
    grid = defaultdict(list)
    for k, (x, y) in enumerate(xy_arr):
        grid[(int(x // BRIDGE_EPS_M), int(y // BRIDGE_EPS_M))].append(k)
    bridges = 0
    for (gx, gy), idxs in grid.items():
        cand = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cand += grid.get((gx + dx, gy + dy), [])
        for a in idxs:
            na = ids_arr[a]
            ways_a = set(node2ways.get(na, []))
            for b in cand:
                if b <= a:
                    continue
                nb = ids_arr[b]
                if ways_a & set(node2ways.get(nb, [])):
                    continue  # same way already connected
                if np.hypot(*(xy_arr[a] - xy_arr[b])) <= BRIDGE_EPS_M \
                        and not G.has_edge(na, nb):
                    G.add_edge(na, nb, kind="bridge", length=0.0)
                    bridges += 1
    print(f"de-fragment: added {bridges} bridge edges")

    node_ids = np.array(list(G.nodes))
    node_xy = np.array([xy_of[n] for n in node_ids])
    return G, node_ids, node_xy, node2ways, way_kind


def nearest_and_within(pt_xy, node_xy, tol):
    """Return (nearest_idx, nearest_dist, mask_within_tol)."""
    d = np.hypot(node_xy[:, 0] - pt_xy[0], node_xy[:, 1] - pt_xy[1])
    i = int(np.argmin(d))
    return i, float(d[i]), d <= tol


def classify_path(G, path):
    kinds = {G[u][v]["kind"] for u, v in zip(path, path[1:])}
    return "underground" if "tunnel" in kinds else "sheltered"


def main():
    stations = pd.read_csv(NODES_CSV)[["NAME", "latitude", "longitude"]]
    malls = pd.read_csv(MALLS_CSV)
    data = fetch_network()
    G, node_ids, node_xy, node2ways, way_kind = build_graph(data)
    id2pos = {nid: k for k, nid in enumerate(node_ids)}
    comp = {n: c for c, nodes in enumerate(nx.connected_components(G)) for n in nodes}
    print(f"network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"{len(set(comp.values()))} components")

    st_xy = to_xy(stations.latitude.values, stations.longitude.values)
    ml_xy = to_xy(malls.LATITUDE.values, malls.LONGITUDE.values)

    # snap malls once
    mall_attach, mall_near_ways = [], []
    for k in range(len(malls)):
        i, dist, mask = nearest_and_within(ml_xy[k], node_xy, MALL_TOL_M)
        mall_attach.append(node_ids[i] if dist <= MALL_TOL_M else None)
        near_ids = node_ids[nearest_and_within(ml_xy[k], node_xy, HEUR_TOL_M)[2]]
        mall_near_ways.append({w for nid in near_ids for w in node2ways.get(nid, [])})

    rows = []
    for s in range(len(stations)):
        sx = st_xy[s]
        # candidate malls within RADIUS_M straight-line
        dmall = np.hypot(ml_xy[:, 0] - sx[0], ml_xy[:, 1] - sx[1])
        cands = np.where(dmall <= RADIUS_M)[0]
        if len(cands) == 0:
            continue
        si, sdist, _ = nearest_and_within(sx, node_xy, STATION_TOL_M)
        st_node = node_ids[si] if sdist <= STATION_TOL_M else None
        st_near_ways = {w for nid in node_ids[nearest_and_within(sx, node_xy, HEUR_TOL_M)[2]]
                        for w in node2ways.get(nid, [])}

        for m in cands:
            straight = float(dmall[m])
            method = ctype = None
            path_m = np.nan
            # --- method C: same connected component -> real path
            mn = mall_attach[m]
            if st_node is not None and mn is not None and \
               comp.get(st_node) == comp.get(mn):
                try:
                    path = nx.shortest_path(G, st_node, mn, weight="length")
                    path_m = float(nx.shortest_path_length(G, st_node, mn, weight="length"))
                    ctype = classify_path(G, path)
                    method = "C"
                except nx.NetworkXNoPath:
                    pass
            # --- method B fallback: one way passes near both
            if method is None:
                shared = st_near_ways & mall_near_ways[m]
                if shared:
                    kinds = {way_kind[w] for w in shared}
                    ctype = "underground" if "tunnel" in kinds else "sheltered"
                    method = "B"
            if method:
                rows.append({
                    "station_name": stations.NAME[s], "mall_name": malls.NAME[m],
                    "straight_m": round(straight, 1), "method": method,
                    "connection_type": ctype,
                    "path_m": round(path_m, 1) if not np.isnan(path_m) else np.nan,
                })

    links = pd.DataFrame(rows).sort_values(["station_name", "straight_m"])
    links.to_csv(OUT_CSV, index=False)

    print(f"\nlinks: {len(links)} -> {OUT_CSV}")
    print("by method:\n" + links.method.value_counts().to_string())
    print("by connection_type:\n" + links.connection_type.value_counts().to_string())
    print(f"stations with >=1 connected mall: {links.station_name.nunique()}/{len(stations)}")
    print("\ntop connected stations:")
    print(links.groupby("station_name").size().sort_values(ascending=False)
          .head(8).to_string())
    print("\nsample links:")
    print(links.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
