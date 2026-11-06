"""Build an undirected edge table for the Singapore MRT/LRT network.

The topology is latent in the station CODE column (e.g. ns10, ew9, cg2):
  * along-line edges  -> stations consecutive in sequence on the same line
  * interchange edges -> different codes that share the same station NAME

The graph is undirected, so each edge is canonicalised with the smaller CODE
(string comparison) as src_station and the larger as dst_station, then
deduplicated. Nodes are keyed by CODE; the node table is simply the input
dataset itself, so unopened (planned / under-construction) stations are kept.

Output: two columns, src_station, dst_station.
"""

import re
from itertools import combinations

import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))      # final_report: data/config.py holds every path of the project
from data import config as cfg      # noqa: E402
IN_CSV = str(cfg.MRT_STATIONS_CSV)
OUT_CSV = str(cfg.MRT_EDGES_CSV)

# CODE = line-prefix + optional number + optional single-letter suffix.
#   ns10  -> ('ns', 10, '')     ew9   -> ('ew', 9, '')
#   ns3a  -> ('ns', 3, 'a')     te22a -> ('te', 22, 'a')   (infill stations)
#   cg    -> ('cg', -1, '')     stc/ptc/rts (no digits -> sentinel -1)
_CODE_RE = re.compile(r"^([a-z]+?)(\d+)([a-z]*)$")


def parse_code(code):
    """Return (line_prefix, number, suffix) used to order stations on a line."""
    m = _CODE_RE.match(code)
    if m:
        return m.group(1), int(m.group(2)), m.group(3)
    # No digits (e.g. 'cg', 'stc'): whole code is the prefix. -1 sorts it
    # ahead of numbered peers, so 'cg' -> 'cg1' -> 'cg2' links correctly.
    return code, -1, ""


def build_edges(df):
    """Yield (code_a, code_b) pairs for along-line and interchange links."""
    # One row per CODE (dataset has no dup codes, but stay defensive).
    nodes = df.drop_duplicates(subset="CODE").copy()
    parsed = nodes["CODE"].map(parse_code)
    nodes["line"] = parsed.map(lambda t: t[0])
    nodes["num"] = parsed.map(lambda t: t[1])
    nodes["suffix"] = parsed.map(lambda t: t[2])

    # --- along-line edges: connect stations consecutive in line order.
    # Sorting by (num, suffix) puts infill stops in place (te22 < te22a < te23)
    # and connecting only consecutive *existing* codes bridges reserved-number
    # gaps automatically.
    for _, grp in nodes.groupby("line"):
        seq = grp.sort_values(["num", "suffix"])["CODE"].tolist()
        for a, b in zip(seq, seq[1:]):
            yield a, b

    # --- interchange edges: same physical station, different lines/codes.
    for _, grp in nodes.groupby("NAME"):
        codes = grp["CODE"].tolist()
        if len(codes) > 1:
            yield from combinations(codes, 2)


def main():
    df = pd.read_csv(IN_CSV)

    edges = set()
    for a, b in build_edges(df):
        if a == b:
            continue  # guard against self-loops
        # Canonical undirected form: smaller CODE (string order) is the source.
        edges.add((a, b) if a < b else (b, a))

    out = (
        pd.DataFrame(sorted(edges), columns=["src_station", "dst_station"])
    )
    out.to_csv(OUT_CSV, index=False)
    print(f"{len(out)} undirected edges -> {OUT_CSV}")


if __name__ == "__main__":
    main()
