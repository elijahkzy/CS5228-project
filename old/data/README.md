# Data is stored externally

Project datasets are not stored in Git. See `docs/DATA_STORAGE.md` for the Drive
bundle structure and setup instructions.

The scripts automatically find an adjacent folder named `CS5228 Project Data`,
or you can point them to another synced copy:

```powershell
$env:CS5228_DATA_ROOT = "G:\My Drive\CS5228 Project Data\data"
```

Do not commit data files, source-response caches, credentials, or API keys here.
