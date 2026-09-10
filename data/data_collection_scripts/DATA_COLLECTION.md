# MRT Graph — Data Collection Plan & Status

Building an undirected, weighted graph of Singapore MRT/LRT stations (for a model
whose training data spans **2021–2026**). Nodes are stations keyed by `NAME`
(interchanges collapsed into one node); edges are the rail network. This file
tracks every data-collection task considered, the sources for each, and status.

_Last updated: 2026-09-02._

---

## 0. Graph construction (foundation — no external data)

Derived purely from the station `CODE` column in `toy_data/auxiliary/sg-mrt-stations.csv`.

| Output | What | Script |
|---|---|---|
| `sg-mrt-edges.csv` | code-level edges (along-line + interchange) | `build_station_edges.py` |
| `sg-mrt-edges-collapsed.csv` | 224 undirected edges, interchanges collapsed to `NAME` | `collapse_interchanges.py` |
| `sg-mrt-nodes.csv` | 192 `NAME` nodes + `member_codes`, `lines`, `is_interchange`, region cols | `build_node_table.py` |

**Status: ✅ complete.**

---

## Task 1 — Edge weights (rail distance)

Weight each edge by inter-station rail distance.

**Sources considered:**
| # | Source | Free? | Decision |
|---|---|---|---|
| 1 | **OpenStreetMap** (Overpass API) — track geometry from `route=subway` relations | Free (ODbL) | **Used** (primary) |
| 3 | LTA Master Plan 2019 rail line (data.gov.sg) | Free | **Rejected** — 2019 vintage, no station attribution, corridor centrelines only |
| — | Haversine × detour factor (1.2) | n/a | **Used** (fallback, option B) |

**Method:** project each station onto its OSM line polyline, measure chainage
difference; fall back to haversine×1.2 where OSM lacks the line. Every edge tagged
with a `method` column.

**Result:** `sg-mrt-edges-weighted.csv` — 224 edges, **176 `osm` / 48 `haversine_fallback`**,
units in km. Fallbacks are non-operational lines (CRL, LRT, Changi branch) whose
public data is unreliable anyway.

**Status: ✅ complete.** Script: `weight_edges.py`.

---

## Task 2 — Node features (train-station characteristics only)

Scope deliberately narrowed to intrinsic station traits (demographics moved to Task 3).

### 2a. Ridership / busyness — LTA DataMall (source #4)
- **Source:** DataMall `PV/Train` (Passenger Volume by Train Stations). Free, needs AccountKey (in `keys.txt`).
- **Signal:** per-station tap-in/out volume → `pv_total`, weekday/weekend, AM/PM peak, busiest hour.
- **Join:** DataMall reports combined interchange codes (`EW24/NS1`); matched to node `member_codes`. 144/192 nodes match (rest are planned stations).
- **History constraint:** API only serves a **rolling ~3-month window** (currently 2026-05/06/07). **No history back to 2021.** Decision: keep the 3 months as a static busyness/popularity metric, one CSV per month.
- **⚠ Quota:** the PV endpoints have their **own strict per-endpoint quota**, separate from the 10M/day account cap. Exhausted by exploratory calls; script now caches each month, never retries on quota, and resumes.
- **Status: ✅ collected** (2026-09-03) — quota lifted; all 3 months pulled in exactly 3 API calls → `sg-mrt-nodes-features-202605/202606/202607.csv` (144/192 matched each month; busiest Jurong East / Orchard / Woodlands). Script: `node_features.py`.

### 2b. OSM station tags (source #6)
- **Source:** OSM/Overpass station node tags. Free, no key.
- **Finding:** too thin — only `operator` (61%) and `wheelchair` (59%) well-populated; `platforms` 3%, `start_date` 9%.
- **Status: 🚫 shelved** — not worth it as a main feature.

### 2c. Mall proximity feature (added mid-task as a richer signal)
- **Source:** `toy_data/auxiliary/sg-shopping-malls.csv` (113 malls, NAME + lat/long) + OSM pedestrian network.
- **First attempt (strict connectivity):** OSM sheltered-walkway (`covered=yes`) / underground-tunnel (`tunnel=yes`) routing via networkx — method **C** (same connected component) + fallback **B** (one way near both), tagged by method. Result only 28 links / 25 stations because OSM covered/tunnel tagging is fragmented (both ends attach but land in different components). Output: `sg-station-mall-links.csv`. Script: `mall_links.py`.
- **Relaxed (current):** straight-line **proximity ≤ 500m** — no connectivity needed. Result **112 links / 66 of 192 stations**. Outputs `sg-station-mall-proximity.csv` (long form) and `sg-station-mall-counts.csv` (`n_malls_500m`, `nearest_mall`, `nearest_mall_m`). **`nearest_mall_m` is the dense, training-ready feature** (`n_malls_500m` is sparse — median 0). Script: `mall_proximity.py`.
- **Status: ✅ complete** (relaxed version).

---

## Task 3 — Demographic aggregation by geography

Demographics over unique **subzone / planning area / region** codes (in
`toy_data/zoning_data/`), decoupled from MRT so it joins onto any dataset carrying
those columns (HDB, schools, malls, MRT nodes).

**Source:** OneMap `popapi` (2020 census), keyed by **planning area** only.

**Key-availability finding (empirical):**
- **Planning area (47): ✅ OneMap** — 43 return demographics. 4 no_data: `marina east`/`marina south`/`straits view` (≈0 residential pop) + `central` (invalid PA name). `seng kang` is a spelling dup of `sengkang` (both carry data via NAME_FIX).
- **Subzone (224): ✅ data.gov.sg census** — OneMap has none, but the **2020 Census on data.gov.sg** publishes real per-subzone population/age/sex (finer than PA, genuine within-PA variation — NOT reuse). All 224 matched.
- **Region (9): ❌ closed** — no independent source; region is definitionally a rollup of planning areas → collinear/redundant. Not collected (user decision). File also mixes two naming schemes (would need reconciling if ever aggregated).

**Status: ✅ planning-area collected** (2026-09-03, `demographics.py`) →
`sg-demographics-planning-area.csv` (47 rows, 43 with data). Features: total_pop,
age structure, unemployment, HDB/private mix, income bands, household size, ethnic
mix, owner-occupied %, PMET %. Household/economic features null for 12 small/
non-residential areas (SingStat doesn't publish breakdowns for tiny pops — expected).
Raw JSON cached in `_onemap_cache/`. **OneMap token expires ~3 days** — for reruns
past that, store account email+password in `keys.txt` to auto-mint tokens.

**Subzone collected** (2026-09-03, `subzone_demographics.py`) from data.gov.sg
Census 2020 (CKAN API, no key) → `sg-demographics-subzone.csv` (224/224 matched;
population, age structure, %male per subzone — real within-PA variation, e.g.
Bedok subzones range 12.6–21.8% elderly vs PA avg 19.3%). Income/occupation not
available at subzone level (planning-area only).

**Task 3 status: ✅ complete** — planning area (OneMap) + subzone (census). Region
intentionally closed (redundant rollup).

---

## Task 4 — Mall features (importance via foot traffic)

Attach features to the malls linked in Task 2c. Primary signal for mall
**importance = foot traffic**. True footfall is proprietary/paid, so the free
plan is to use **proxies** — notably Google Popular Times, which mirrors the
station-ridership busyness feature (hourly relative busyness).

**Sources considered:**
| Source | Signal | Free | Key | Notes |
|---|---|---|---|---|
| **Google Popular Times** (via BestTime.app free tier, or `populartimes` lib) | hourly relative busyness per mall | Free tier | BestTime API key | Best free proxy. **Relative 0–100, not absolute counts**; unofficial (not a Google API). Match mall → Google place by name+coords. |
| Google Places API (`user_ratings_total`, rating) | popularity proxy | Free tier → paid | API key | Review count correlates with popularity, not footfall directly |
| OSM building footprint / GFA | mall size (capacity proxy) | ✅ ODbL | No | Size, not traffic |
| Derived: nearby station ridership (Task 2a) | inherited footfall estimate | ✅ internal | — | Malls near busy stations likely busier; reuses our own data |
| Wikipedia / mall directories | GFA, number of stores | ✅ | No | Manual/scrape; major malls only |
| Commercial (Unacast ~$5k/mo, GapMaps, Placer.ai, Singtel DataSpark) | true footfall | 💲 Paid | — | Out of scope (not free) |

**Recommended:** Google Popular Times via **BestTime.app free tier** as the
primary busyness metric — **on hold** (user deferred Google sources).

**Collected so far (Wikipedia, 2026-09-03, `mall_wikipedia.py`):** size/importance
features via MediaWiki API (free, no key) → `sg-mall-features-wikipedia.csv`
(113 malls, 106 matched). Features: `floor_area_sqm` (sqft auto-converted),
`floors`, `num_stores`, `num_anchors`, `opening_year`, `developer`, `owner`.
Coverage is partial (Wikipedia-dependent): opening_year 62/113, floors 57, stores
49, floor_area 45; gaps explicit via `wiki_status`. Values verified (Jewel 135,700
sqm, VivoCity 99,987 NLA/340 stores, Ngee Ann City 110,452). Wikitext cached in
`_wiki_cache/`.

**Station-ridership-derived footfall: dropped** (user). **OSM footprint: dropped**
(user — Wikipedia only).

**Status: 🅿 ON HOLD** — user wants the actual **foot-traffic** signal, which needs
Google Popular Times via **BestTime.app free tier** (blocked on a BestTime/Places
API key). Wikipedia size features (`sg-mall-features-wikipedia.csv`) are collected
and kept as a complementary layer, but do NOT provide traffic. Resume once the key
is available.

---

## Task 5 — School features

Enrich schools (`toy_data/auxiliary/sg-schools.csv`, 337 rows — this is the
data.gov.sg MOE "General information of schools" directory, and already has an
`MRT_STATIONS` column linking schools to stations).

**Scope (updated 2026-09-02):** **latest data only** (no history needed), to feed
a later **school-ranking** step. Required features — everything else is out of scope:
1. **Entry score cutoff** requirements (all levels except Primary)
2. **School fees**, if any

_National exam / graduation performance was considered but **removed** — no clear
government source exists (school-level results are not published)._

**Sources considered:**
| Source | Feature | Free | Key | Status / Notes |
|---|---|---|---|---|
| data.gov.sg MOE "General information of schools" | attributes/location | ✅ | No | **Already have it** (= our file) |
| data.gov.sg MOE — CCA, subjects offered, distinctive programmes, MTL | attribute enrichment | ✅ | No | Optional add-ons |
| **MOE SchoolFinder** (schools.moe.gov.sg) | **Secondary PSLE AL cut-off ranges** (per school, per Posting Group) | ✅ | No (scrape) | **Available** — not clean CSV; scrape SchoolFinder per school |
| MOE "Indicative PSLE Score Ranges for Individual Secondary Schools" (PDF) | Secondary cutoffs | ✅ | No | Annual PDF, parse |
| MOE **JAE cut-off points** (annual publication) | **JC (L1R5)** and **Poly (ELR2B2 COP)** cutoffs | ✅ | No | Available — PDF/scrape; poly COP is per-course |
| School-level exam **results** (PSLE/O/A averages, pass rates) | graduation performance | — | — | 🚫 **Out of scope** — not published (rankings abolished ~2012); no official source. Feature removed. |
| MOE SchoolFinder / school websites / MOE fees page | **School fees** (misc fees; higher for independent/autonomous & non-citizens) | ✅ | No (scrape) | Available — likely on SchoolFinder detail page; confirm during scrape |

**Key findings / caveats:**
- **Graduation performance removed from scope** — school-level A-level/O-level results are not published; no clear source. (The entry cutoff already encodes the *admitted cohort's* national-exam standard, so it also serves as a de-facto quality signal for ranking.)
- **Entry cutoff (feature 1)** is published on SchoolFinder per level: **JC = 2026 JAE L1R5 net aggregate by stream** (Arts/Science, `lower`–`upper` range; verified on VJC: Arts 4–8 code 33A, Science 2–6 code 33S — embedded as JSON `{stream,code,lower,upper,group:"L1R5"}`), **Secondary = PSLE AL cutoff**, **Poly = ELR2B2 COP**.
- **Entry cutoff (feature 2) is available but not as tidy CSV** — lives in SchoolFinder (scrape per school, per level: Secondary AL ranges, JC L1R5, Poly COP).
- **School fees (feature 3)** — standardised & minor for government/aided; higher for independent/autonomous schools and non-citizens. Expected on the SchoolFinder detail page / school sites.
- **Primary schools have no academic cutoff** (proximity/balloting) — excluded.
- Latest-year only now needed, so the T-score→AL (2021) history break and the PDF/Wayback back-fill are **no longer relevant**.

**How to access SchoolFinder (investigated 2026-09-02):** No documented API.
SchoolFinder is a Next.js app over a **Directus CMS** (`cms.moe.gov.sg`); CMS
schema is locked (403) and JSON proxy routes are POST/server-actions. **But each
school's detail page is server-side rendered with the cut-off data in the HTML** —
e.g. `https://www.moe.gov.sg/schoolfinder/schooldetail/victoria-junior-college`
(HTTP 200) contains the JC `Arts`/`Science` `L1R5` cut-offs. **Method = SSR-page
scrape:** enumerate slugs (kebab-cased names from our 337-school directory) →
GET `…/schooldetail/<slug>` (no headless browser) → parse the cut-off block.
Throttle/cache/respect robots.txt.

**History:** not needed (latest year only) — SchoolFinder's current-year data is
sufficient, so no PDF/Wayback back-fill required.

**Status: ✅ collected** (`scrape_schools.py`). Slugs resolved via the MOE
**sitemap** (through-train schools split into per-level pages, e.g.
`raffles-institution-secondary` + `-junior-college`). Output **segmented by level**:
- `sg-schools-primary.csv` (179) — fees/attributes only, no cutoff (as expected)
- `sg-schools-secondary.csv` (133; 128 with PSLE-AL posting-group cutoffs pg1–pg3)
- `sg-schools-junior_college.csv` (11; L1R5 by stream — arts/science/commerce)
- `sg-schools-mixed.csv` (14; through-train schools carry both posting + L1R5/IB)

Notes: `fee_tier` derived from `TYPE_CODE` (government/aided = low, independent/
specialised = high) since per-school dollar fees aren't published on the page.
8 schools legitimately have no cutoff (NUS High/SOTA/Sports School — non-academic
admission; NorthLight/Assumption/Crest/Spectra — N(T)/no-PSLE-cutoff). 4 SAP
schools carry a `(D)/(M)` Higher-Chinese-Language grade requirement alongside the
aggregate (e.g. `6(D)`), splittable into numeric + CL-grade if needed.

---

## Data sources — master list

| Source | Used for | Free | Key | Status |
|---|---|---|---|---|
| Station `CODE` column | graph topology, Task 1/2 joins | ✅ | — | Used |
| OpenStreetMap / Overpass | Task 1 rail geometry, 2b tags, 2c walkways | ✅ ODbL | No | Used |
| LTA Master Plan 2019 (data.gov.sg) | Task 1 (considered) | ✅ | No | Rejected |
| Haversine fallback | Task 1 | n/a | — | Used |
| LTA DataMall `PV/Train` | Task 2a ridership | ✅ | AccountKey | Pending (quota) |
| `sg-shopping-malls.csv` | Task 2c malls | ✅ local | — | Used |
| OneMap `popapi` | Task 3 demographics (planning area) | ✅ | Token | Used ✅ |
| SingStat / data.gov.sg | Task 3 demographics | ✅ | No | Not needed (OneMap covered it) |
| Wikipedia (MediaWiki API) | Task 4 mall size/GFA/stores | ✅ | No | Used ✅ |
| Google Popular Times (BestTime.app / `populartimes`) | Task 4 mall foot traffic | ✅ free tier | BestTime key | On hold |
| Google Places API | Task 4 mall popularity | ✅ free tier | API key | On hold |
| OSM footprint / station-ridership | Task 4 (considered) | ✅ | — | Dropped (user) |
| Commercial footfall (Unacast, GapMaps, Placer.ai, DataSpark) | Task 4 (considered) | 💲 | — | Rejected (paid) |
| data.gov.sg MOE school datasets | Task 5 attributes | ✅ | No | Have directory; add-ons planned |
| MOE SchoolFinder / JAE cut-offs | Task 5 entry cutoffs | ✅ | No (scrape) | Planned |
| School-level exam results | Task 5 graduation perf | — | — | Not published (rejected) |

## Outputs in `toy_data/graph_data/`

- `sg-mrt-nodes.csv`, `sg-mrt-edges.csv`, `sg-mrt-edges-collapsed.csv` — graph
- `sg-mrt-edges-weighted.csv` — Task 1 (rail distances + method)
- `sg-station-mall-links.csv` — Task 2c strict connectivity (superseded)
- `sg-station-mall-proximity.csv`, `sg-station-mall-counts.csv` — Task 2c relaxed
- `sg-mrt-nodes-features-YYYYMM.csv` — Task 2a ridership (pending quota)
