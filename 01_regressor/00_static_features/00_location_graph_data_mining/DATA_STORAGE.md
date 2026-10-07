# Location-graph data storage

The CSV inputs and generated feature stores used by these notebooks are kept
outside Git in:

```text
<data root>/location_graph/
├── graph_feature_store/
└── raw_graph_data/
```

The existing notebooks refer to `graph_feature_store/` and `raw_graph_data/`
relative to this directory. On Windows, create local directory junctions after
cloning the repository so those paths continue to work without copying data
back into Git:

```powershell
$env:CS5228_DATA_ROOT = "G:\My Drive\CS5228 Project Data\data"

New-Item -ItemType Junction `
  -Path ".\00_location_graph_data_mining\graph_feature_store" `
  -Target "$env:CS5228_DATA_ROOT\location_graph\graph_feature_store"

New-Item -ItemType Junction `
  -Path ".\00_location_graph_data_mining\raw_graph_data" `
  -Target "$env:CS5228_DATA_ROOT\location_graph\raw_graph_data"
```

If the two local paths already exist, verify their contents have been preserved
in the external bundle before removing them and creating the junctions. Both
paths are ignored by Git. Do not commit their contents or any credentials.
