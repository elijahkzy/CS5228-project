"""Relaxed mall feature: link each MRT station to every shopping mall within
WALK_M straight-line metres (default 500m ~ walking distance). No connectivity
inference — just proximity, which gives usable coverage for model training.

Straight-line ("as the crow flies") distance via a local equirectangular
projection (accurate over Singapore). Actual walking distance is somewhat
longer, so 500m straight-line ~ a short walk.

Outputs:
  * sg-station-mall-proximity.csv : long form (station, mall, distance_m, rank)
  * sg-station-mall-counts.csv     : per-station node feature
        (n_malls_<WALK>m, nearest_mall, nearest_mall_m, malls_<WALK>m list)
"""

import math
import numpy as np
import pandas as pd

ROOT = "toy_data/graph_data"
AUX = "toy_data/auxiliary"
NODES_CSV = f"{ROOT}/sg-mrt-nodes.csv"
MALLS_CSV = f"{AUX}/sg-shopping-malls.csv"
OUT_LINKS = f"{ROOT}/sg-station-mall-proximity.csv"
OUT_COUNTS = f"{ROOT}/sg-station-mall-counts.csv"

WALK_M = 500.0
LAT0 = 1.35
R_EARTH = 6371000.0


def to_xy(lat, lon):
    x = np.radians(np.asarray(lon, float)) * math.cos(math.radians(LAT0)) * R_EARTH
    y = np.radians(np.asarray(lat, float)) * R_EARTH
    return np.column_stack([np.ravel(x), np.ravel(y)])


def main():
    stations = pd.read_csv(NODES_CSV)[["NAME", "latitude", "longitude"]]
    malls = pd.read_csv(MALLS_CSV)

    st_xy = to_xy(stations.latitude.values, stations.longitude.values)
    ml_xy = to_xy(malls.LATITUDE.values, malls.LONGITUDE.values)

    # full pairwise distance matrix (192 x 113 — tiny)
    dist = np.hypot(st_xy[:, 0][:, None] - ml_xy[:, 0][None, :],
                    st_xy[:, 1][:, None] - ml_xy[:, 1][None, :])

    link_rows, count_rows = [], []
    for s in range(len(stations)):
        d = dist[s]
        within = np.where(d <= WALK_M)[0]
        order = within[np.argsort(d[within])]  # nearest first
        for rank, m in enumerate(order, start=1):
            link_rows.append({
                "station_name": stations.NAME[s], "mall_name": malls.NAME[m],
                "distance_m": round(float(d[m]), 1), "rank": rank,
            })
        nearest = int(np.argmin(d))
        count_rows.append({
            "station_name": stations.NAME[s],
            f"n_malls_{int(WALK_M)}m": len(order),
            "nearest_mall": malls.NAME[nearest],
            "nearest_mall_m": round(float(d[nearest]), 1),
            f"malls_{int(WALK_M)}m": ";".join(malls.NAME[order].tolist()),
        })

    links = pd.DataFrame(link_rows).sort_values(["station_name", "rank"])
    counts = pd.DataFrame(count_rows)
    links.to_csv(OUT_LINKS, index=False)
    counts.to_csv(OUT_COUNTS, index=False)

    n_col = f"n_malls_{int(WALK_M)}m"
    print(f"links: {len(links)} -> {OUT_LINKS}")
    print(f"counts: {len(counts)} stations -> {OUT_COUNTS}")
    print(f"\nstations with >=1 mall within {int(WALK_M)}m: "
          f"{(counts[n_col] > 0).sum()}/{len(counts)}")
    print(f"malls-per-station: mean={counts[n_col].mean():.2f} "
          f"median={counts[n_col].median():.0f} max={counts[n_col].max()}")
    print(f"stations with 0 malls: {(counts[n_col] == 0).sum()}")
    print("\nmost mall-dense stations:")
    print(counts.nlargest(8, n_col)[["station_name", n_col, "nearest_mall",
          "nearest_mall_m"]].to_string(index=False))
    print("\nsample links:")
    print(links.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
