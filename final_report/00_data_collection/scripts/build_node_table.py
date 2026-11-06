"""Collapse the stations dataset to one node per NAME (matches the collapsed
edge table, which is keyed by NAME).

Each NAME node aggregates its member codes:
  * member_codes  : all codes that share this NAME (';'-joined)
  * lines         : distinct line prefixes serving the station
  * n_lines       : number of distinct lines
  * is_interchange: n_lines > 1
  * status        : 'open' if any member code is open, else the remaining
                    status (planned / under construction) — a station is
                    usable as soon as one line opens
  * latitude/longitude : centroid (mean) of member codes
  * subzone / planning_area / region : shared location attributes (first;
                    a warning prints if members disagree)

Output: sg-mrt-nodes.csv, one row per NAME.
"""

import re
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))      # final_report: data/config.py holds every path of the project
from data import config as cfg      # noqa: E402
STN_CSV = str(cfg.MRT_STATIONS_CSV)
OUT_NODES = str(cfg.MRT_NODES_CSV)

_CODE_RE = re.compile(r"^([a-z]+?)(\d+)([a-z]*)$")


def line_of(code):
    m = _CODE_RE.match(code)
    return m.group(1) if m else code  # no-digit codes: whole code is the line


def combine_status(s):
    order = ["open", "under construction", "planned"]
    present = set(s)
    for st in order:
        if st in present:
            return st
    return ";".join(sorted(present))


def first_with_check(col_name):
    def agg(s):
        vals = s.dropna().unique()
        if len(vals) > 1:
            print(f"  ! {col_name} varies within a NAME: {list(vals)}")
        return vals[0] if len(vals) else None
    return agg


def main():
    stn = pd.read_csv(STN_CSV)
    stn["line"] = stn.CODE.map(line_of)

    grouped = stn.groupby("NAME")
    nodes = pd.DataFrame({
        "member_codes": grouped.CODE.apply(lambda s: ";".join(sorted(s))),
        "lines": grouped.line.apply(lambda s: ";".join(sorted(s.unique()))),
        "n_lines": grouped.line.nunique(),
        "status": grouped.STATUS.apply(combine_status),
        "latitude": grouped.LATITUDE.mean(),
        "longitude": grouped.LONGITUDE.mean(),
        "subzone": grouped.SUBZONE.agg(first_with_check("subzone")),
        "planning_area": grouped.PLANNING_AREA.agg(first_with_check("planning_area")),
        "region": grouped.REGION.agg(first_with_check("region")),
    })
    nodes["is_interchange"] = nodes.n_lines > 1
    nodes = nodes.reset_index()  # NAME becomes the first column / primary key

    nodes.to_csv(OUT_NODES, index=False)
    print(f"\n{len(nodes)} NAME nodes -> {OUT_NODES}")
    print(f"interchanges (n_lines>1): {int(nodes.is_interchange.sum())}")
    print(f"status breakdown:\n{nodes.status.value_counts().to_string()}")


if __name__ == "__main__":
    main()
