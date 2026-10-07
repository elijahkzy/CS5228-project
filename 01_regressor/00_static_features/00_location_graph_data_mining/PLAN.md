# Graph-Based Regression Model — Project Plan

## Objective

Perform a regression task on `train_df` (housing rental prices over time) using
**graph methods** to extract features and, potentially, to train a graph
convolutional network (via **DGL**) that performs the regression directly.

Tree-based models are a valid alternative approach but are **out of scope** for
this effort — the focus here is deliberately on graph-based feature extraction
and graph neural network modelling.

---

## Node Types

The graph is built from the following node types. House, MRT, and Mall each have
known latitude/longitude coordinates:

| Node type | Role |
|-----------|------|
| **House** | The target node type — regression (rental price) is performed over these nodes. |
| **Train station (MRT)** | Represents the train lines connecting different regions of Singapore. |
| **Mall** | Represents local commercial activity within a region. |
| **Region** | A more generalised encoding of houses — houses are aggregated to the region they fall in. Provides a coarser, more deterministic unit that generalises across unseen houses (see Part 3). |

**Default distance metric:** Haversine distance, denominated in **kilometers**.

### Linkage Semantics

Two distinct linkage rules apply depending on the node types involved:

- **Distance-based (House ↔ MRT ↔ Mall):** edges are created between these nodes
  when they fall within the maximum Haversine linkage distance (≈ 0.75 km; see
  Part 2). Edges are distance-weighted.
- **Containment-based (Region ↔ MRT / Mall):** a Region node is linked to a train
  station or mall node when that station/mall **lies geographically within the
  region's boundary** — not by a distance cutoff. Likewise each House belongs to
  the Region that contains it.

> **Note:** containment requires region boundary polygons (planning-area
> shapefiles), not just point coordinates. A point-in-polygon step is required to
> assign houses/stations/malls to regions.

#### Region Granularity — Planning Area (chosen)

Regions are defined at the **planning-area** level (~55 areas in Singapore),
**not** subzone (~330).

- **Rationale — sample size:** the training set has only **~15,000** points.
  At subzone granularity this is only ~40–50 samples per subzone, too sparse for
  stable per-region estimates. Planning areas keep enough houses per region for
  meaningful aggregation.
- **Rationale — connectivity:** finer subzones may contain **no train station**,
  which would leave those region nodes with no MRT edge and risk a
  **disconnected graph**. Planning areas are large enough to almost always
  contain at least one station.

**Connectivity requirement:** every node should have **at least one edge**.
Before finalising the graph, verify this — e.g.:
- Count isolated nodes per node type (regions with no MRT/mall, houses with no
  neighbour inside the distance cutoff).
- Visualise the graph / plot region polygons against station and mall locations
  to spot regions that would be left unconnected.
- If isolated nodes remain, consider a fallback linkage (e.g. connect an isolated
  region to its **nearest** station/mall regardless of containment, or relax the
  distance cutoff for isolated houses).

> **Concern — non-residential / MRT-less planning areas:** even at planning-area
> granularity, a few areas may genuinely contain **no MRT station** (e.g.
> outlying or non-residential areas such as Western Water Catchment, Tuas, or
> Marina South). These would produce isolated region nodes. Furthermore, purely
> non-residential planning areas may have **zero houses** in the dataset — such
> regions could be **dropped entirely** rather than force-connected, since the
> regression is over houses. Consider a houses-present filter during region
> construction.

**Task:** verify that **every planning area has at least one MRT connection**
(i.e. contains, or under the fallback rule links to, at least one train station).
Flag any planning area that does not, and decide per case whether to apply the
nearest-station fallback or drop the region (if it has no houses).

---

## Part 1 — Graph Feature Generation on House Nodes

Feature-engineering approaches applied to the House nodes. Each should be
evaluated for **information value against the rental-price label**.

1. **DBSCAN clustering**
   - Run basic clustering (DBSCAN) on House nodes across both train and test data.
   - Predict rental price from the resulting `cluster_id` to gauge the
     information value of the clustering.

2. **KNN / Epsilon-ball graph + Laplacian Eigenmap embedding**
   - Build a K-NN or epsilon-ball graph linkage over House nodes using housing
     features.
   - Compute a low-dimensional **Laplacian Eigenmap** (3–5 dimensions) to
     produce a housing embedding vector.
   - Evaluate the embedding's information value against the label.

3. **Graphlet Degree Vector (GDV)**
   - Compute a GDV on the House nodes using the KNN or epsilon graph.
   - Evaluate its usefulness as a feature.

### Part 1.2 Notebook Series — Graph Construction, Embedding & Extensions

The graph is built over the **HDB block** node set (`sg-hdb-block.csv`, ~9.5k
distinct locations); per-block features are joined back to the rental records via
the composite key `[TOWN, BLOCK, STREET]`.

Notebook map:
- `02_house_dbscan_clustering` — Part 1.1 DBSCAN clustering (baseline clusters).
- `03a_epsilon_ball_graph` — early epsilon-ball prototype (too dense; superseded).
- `03c_knn_epsilon_graph` — **radius-capped k-NN graph** (top-`K` within a walking
  radius, union-symmetrised) → **Laplacian Eigenmap** embedding → **Nyström**
  out-of-sample extension → basic linear-regression evaluation vs raw lat/long.

**Observed behaviour to exploit.** The radius-capped k-NN graph is **fragmented**
into many connected components (66 at `K_NEIGHBORS=5`, `RADIUS_WALK_KM=2`) — i.e.
multiple disconnected manifolds rather than one continuous manifold. This limits a
plain Laplacian Eigenmap (no continuous feature diffusion across the whole dataset),
but the fragmentation and the edge-weighting freedom are themselves useful. Three
follow-up works, each in its own notebook:

1. **`03d` — Connected components as a clustering feature (DBSCAN validation). [BUILT]**
   Use the k-NN graph's connected components as a cluster assignment and compare it
   against the DBSCAN output as an independent validation.
   - DBSCAN currently yields **18 clusters**; the k-NN graph identifies **66**
     distinct components.
   - The dataset has **2,735 distinct blocks** across **26 towns**, so a 66-way
     component labelling sits between town-level (26) and block-level (2,735)
     granularity — it encodes structure **not already present** in the raw
     `TOWN`/`BLOCK` fields, so it is a genuine new categorical feature.
   - Deliver the component id as a feature and evaluate its information value
     against the label.

2. **`03e` — Fix fragmentation + inverse-distance edge weighting. [PENDING REVIEW]**
   Tune the graph-construction logic in `03c` to reduce/eliminate fragmentation
   (e.g. raise `K`/radius, or bridge components), and replace binary edges with an
   **inverse-distance weight** (penalising longer edges). This yields a better-behaved
   manifold that models the **flow of housing value across geographic distance**.

3. **`03f` — Feature-similarity edge weighting (feature diffusion). [PENDING REVIEW]**
   Building on `03e`, replace the inverse-Haversine distance weight with **node
   feature similarity** as the edge-weighting criterion, so the Laplacian Eigenmap
   models **feature diffusion** (feature-aware embedding) rather than pure geography.
   - Consistency: the Nyström extension must use the same feature-similarity kernel,
     so test-house features feed the out-of-sample weights.
   - Leakage guard: build edge weights from covariates only — never from
     `MONTHLY_RENT`.

All three should be delivered and benchmarked for information value against the
rental-price label.

---

## Part 2 — Heterogeneous Graph (House + MRT + Mall)

A **multi-node-type, single undirected edge type** heterogeneous graph across
MRT, House, and Mall nodes, with **distance-weighted edges**.

Two categories of output can be delivered from this setup:
- **Feature encodings** (see below), and
- The **regression prediction** itself (see Part 3).

### Maximum Linkage Distance — 1 km walking ≈ 0.74 km Haversine

The threshold is defined behaviourally as a **1 km walking distance** — most
people are unlikely to walk more than ~1 km from home to a train station or mall,
so links beyond this are behaviourally implausible.

Because edges are measured with **Haversine (straight-line)** distance, the
walking cutoff must be converted. Real walking paths detour around roads and
blocks, so walking distance exceeds straight-line distance by a **detour factor**
of roughly **1.3–1.4** (a common urban default is ~1.35). Dividing out this
factor:

$$d_{\text{haversine}} \approx \frac{1\ \text{km}}{1.35} \approx 0.74\ \text{km}$$

So edges are created only between nodes within a **maximum Haversine linkage
distance of ≈ 0.74 km** (use **0.75 km** as a round working value). Beyond this
radius no edge is formed, so the graph is **sparse** rather than fully connected.

- This bounds edge count (avoiding the quadratic blow-up of a fully connected
  graph) and keeps each House node's neighbourhood locally meaningful.
- **Note:** the detour factor is itself an assumption; if actual street-network
  walking distances are available (e.g. from a routing API), prefer those over
  the Haversine-plus-detour approximation.

The 1 km figure is a sensible prior but should be **tuned empirically**. Candidate
methods for estimating/validating the threshold:

1. **KNN distance / elbow (k-distance) plot** — for each House node, plot the
   sorted distance to its k-th nearest MRT/mall. Look for the "knee" where
   distance rises sharply; it marks where neighbourhoods stop being local. (This
   is also the standard heuristic for choosing DBSCAN's `eps`, so it reuses
   Part 1's tooling.)
2. **Predictive cross-validation over the radius** — sweep the Haversine
   threshold around the ~0.75 km working value (e.g. 0.5, 0.65, 0.75, 0.9, 1.1 km),
   rebuild the graph/features at each, and pick the radius that maximises held-out
   regression performance. Most direct signal, but the most compute-intensive.
3. **Connectivity / degree-distribution analysis** — inspect how the giant
   connected component, isolated-node count, and mean degree evolve as the radius
   grows. Choose the smallest radius that avoids excessive isolated House nodes
   while keeping degree from exploding.
4. **Empirical walking-behaviour / catchment priors** — calibrate against
   published transit "walk access" catchment studies or local planning norms
   (e.g. typical MRT walk-catchment radii), optionally adjusting for the road-
   network detour factor since Haversine underestimates true walking distance.
5. **Variogram / spatial-autocorrelation range** — estimate the distance at which
   spatial autocorrelation of rental price decays (the variogram range) and use
   it as a data-driven characteristic interaction scale.

### 2a — Heterogeneous Graph Feature Encoding

1. **Basic aggregations**
   - e.g. count of linked train stations and malls per House node.

2. **Homogeneous conversion + diffusion / GDV**
   - Convert the heterogeneous graph to a **homogeneous, undirected, weighted**
     graph.
   - Run a diffusion method using the **graph Laplacian** to encode graph
     connectivity, **or**
   - Compute a **GDV** after converting House nodes into **region** nodes.

---

## Part 3 — Graph Convolution Network (Regression Model)

Because some of the test set may contain nodes not present in the training set,
model options are:

1. **Heterogeneous attention model** (primary candidate)
   - Handles unseen nodes via attention over the heterogeneous graph.

2. **Region-aggregated GCN with spectral encoding** (more deterministic)
   - Convert House nodes to their **region**, then apply a graph model with
     **Laplacian spectral encoding** (e.g. GCN) for prediction.

3. **Graphlet kernel + SVM**
   - Use a graphlet kernel with an SVM for the regression task.
   - **Open question:** feasibility on a heterogeneous graph is uncertain.

---

## Open Questions / Notes

- Confirm whether the graphlet-kernel + SVM approach is viable on a
  heterogeneous graph, or whether it requires the homogeneous/region conversion.
- Each feature-generation method should be benchmarked for information value
  against the label before committing to the full GCN pipeline.
