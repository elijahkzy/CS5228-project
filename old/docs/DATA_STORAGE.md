# External data storage

## Canonical location

The repository contains code and documentation only. The canonical data is kept
in the adjacent folder `CS5228 Project Data`, which is ready to upload to Google
Drive:

```text
CS5228 Project Data/
├── README.md
├── manifest.csv
└── data/
    ├── train.csv
    ├── test.csv
    ├── example-submission.csv
    ├── auxiliary/
    ├── collected_data/
    ├── macro/
    ├── location_graph/
    └── processed/
```

The six auxiliary CSV files exist only in `data/auxiliary/`; their former
root-level duplicates were deliberately omitted. Data-collection scripts are
version-controlled under `src/data_collection/`, not stored in the Drive bundle.

## Connecting the code to the data

If `CS5228 Project` and `CS5228 Project Data` remain sibling folders, the main
scripts and notebooks find the data automatically. After syncing the data to a
different Google Drive location, set the environment variable for the current
PowerShell session:

```powershell
$env:CS5228_DATA_ROOT = "G:\My Drive\CS5228 Project Data\data"
```

Alternatively, pass the location explicitly:

```powershell
.\.venv\Scripts\python.exe .\src\build_features.py `
  --data-root "G:\My Drive\CS5228 Project Data\data"
```

The expected value is the inner `data` folder that directly contains
`train.csv`, not the parent bundle folder.

## Folder roles

| Folder | Contents | Required? |
|---|---|---|
| data root | Competition train, test, and submission-template files | Yes |
| `auxiliary/` | Provided HDB, MRT, mall, school, COE, and stock tables | Yes for the current pipeline |
| `collected_data/` | Enriched MRT, mall, school, and zoning inputs | Yes for the current feature build |
| `collected_data/**/_*_cache/` | Source-response caches retained for reproducibility | Optional for normal model training |
| `macro/` | Leakage-controlled official macro panel and provenance audit | Yes when macro features are enabled |
| `location_graph/` | Graph-model feature stores and raw MRT/HDB graph inputs | Required only for the location-graph notebooks |
| `processed/` | Model-ready Parquet outputs, dictionary, and build report | Regenerable |

The location-graph notebooks retain their original relative paths. Follow
`00_location_graph_data_mining/DATA_STORAGE.md` to create local directory
junctions to the external graph folders before running those notebooks.

## Integrity and security

`manifest.csv` lists every file under the external `data` folder, including its
size, role, and SHA-256 checksum. Recompute a file's checksum with
`Get-FileHash -Algorithm SHA256` if a sync appears incomplete.

Do not place API keys, account credentials, `.env` files, or other secrets in
the data bundle. Google Drive sharing permissions should be limited to project
members who are allowed to access the competition data.

## Git history caveat

Removing data from the current working tree prevents future data commits, but it
does not remove blobs from earlier Git commits. Repository-history cleanup is a
separate, disruptive operation requiring coordination with every collaborator
and a force-push; it is intentionally not part of this storage change.
