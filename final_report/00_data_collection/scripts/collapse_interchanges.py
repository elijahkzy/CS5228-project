"""Collapse interchange stations into a single node, keyed by NAME.

Codes that share a NAME (e.g. cc9 == ew8 == Paya Lebar) are the same physical
station, so we merge them into one node identified by NAME. Consequences:
  * old interchange edges (cc9-ew8) become self-loops -> dropped
  * two lines running between the same pair of stations can now collapse to
    the same NAME-NAME edge -> deduplicated

The CODE -> NAME mapping already lives in the source stations CSV, so we don't
re-emit it here.

Output: sg-mrt-edges-collapsed.csv with src_station, dst_station (both NAMEs).
"""

import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))      # final_report: data/config.py holds every path of the project
from data import config as cfg      # noqa: E402
EDGES_CSV = str(cfg.MRT_EDGES_CSV)
STN_CSV = str(cfg.MRT_STATIONS_CSV)
OUT_EDGES = str(cfg.MRT_EDGES_COLLAPSED_CSV)


def main():
    edges = pd.read_csv(EDGES_CSV)
    stn = pd.read_csv(STN_CSV)

    code2name = dict(zip(stn.CODE, stn.NAME))

    # Remap both endpoints from CODE to NAME.
    src = edges.src_station.map(code2name)
    dst = edges.dst_station.map(code2name)
    collapsed = pd.DataFrame({"src_station": src, "dst_station": dst})

    # Drop interchange self-loops (both codes were the same station).
    n_selfloops = (collapsed.src_station == collapsed.dst_station).sum()
    collapsed = collapsed[collapsed.src_station != collapsed.dst_station]

    # Re-canonicalise (smaller NAME as src) and dedup any parallel edges.
    lo = collapsed[["src_station", "dst_station"]].min(axis=1)
    hi = collapsed[["src_station", "dst_station"]].max(axis=1)
    collapsed = (
        pd.DataFrame({"src_station": lo, "dst_station": hi})
        .drop_duplicates()
        .sort_values(["src_station", "dst_station"])
        .reset_index(drop=True)
    )

    collapsed.to_csv(OUT_EDGES, index=False)

    print(f"code-level edges in           : {len(edges)}")
    print(f"interchange self-loops dropped: {n_selfloops}")
    print(f"parallel edges deduped        : "
          f"{len(edges) - n_selfloops - len(collapsed)}")
    print(f"collapsed NAME-level edges    : {len(collapsed)} -> {OUT_EDGES}")
    print(f"unique NAME nodes             : "
          f"{pd.concat([collapsed.src_station, collapsed.dst_station]).nunique()}")


if __name__ == "__main__":
    main()
