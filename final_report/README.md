# CS5228 Group 13: predicting Singapore HDB monthly rents

Code, data and results behind the report. The task is a regression problem: predict the monthly rent (`MONTHLY_RENT`) of HDB flats rented out between 2021-01 and 2026-07.

The folders are numbered in the order the work was done: collect the data, turn it into features, fit models, evaluate them.

```text
final_report/
├── 00_data_collection/    EDA and the scripts that gathered the extra data
├── 01_data_processing/    raw data -> model-ready feature tables, plus feature studies
├── 02_modelling/          every model that was fitted
├── 03_evaluation/         predictions of all models, their comparison and the final submission
└── data/                  all data (inputs, intermediate tables, final features) and data/config.py
```

The processed data and trained models are already in the folder, so you can open any notebook and run it without redoing the steps before it.

## Quick start

1. Install the packages used by the notebooks: `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`, `pyarrow`, `torch`, `catboost`, `lightgbm`, `xgboost`, `statsmodels`, `plotly` and `contextily`. The scraping scripts also use `requests`, `geopy` and `pandarallel`.
2. Open a notebook and run it. Notebooks can be started from any folder inside the project.
3. Every path lives in [data/config.py](data/config.py), so the folder can be moved or renamed freely. Run `python data/config.py` to list every path and whether it exists on your machine.

## What is in each folder

### `00_data_collection/`: understanding the data and gathering extra data

| File | What it does |
| --- | --- |
| `00_eda.ipynb` | Exploratory analysis of the train and test rentals; writes the distinct-house tables `data/df_keys/*-house.csv`. |
| `01_data_viz.ipynb` | Charts and maps of rents by location, flat type, lease, schools, malls and transport. |
| `scripts/` | Scripts that downloaded or built the external data (MRT graph, mall links, school cut-offs, census demographics, ridership, bus stops). Their outputs are already in `data/features/raw/`; you only need these to collect the data again. `node_features.py` and `demographics.py` need an API key in `scripts/keys.txt` (not included). |

### `01_data_processing/`: from raw data to features

Run the subfolders in order.

| Folder | What it does |
| --- | --- |
| `00_temporal_features/` | Month-level features that do not depend on the flat: stock prices and COE (car ownership) prices (`00_feature_engineering`), then the merge and selection of the temporal features (`01_temporal_feature_coverage`). |
| `01_clustering_features/` | DBSCAN clusters of the flats' coordinates (`00_dbscan_location_clusters`). The fitted clusterers are saved in `models/`. |
| `02_static_features/` | Features of each flat's surroundings: `00_railway_aggregation` (nearest stations, traffic, malls), `01_school_aggregation` (nearest schools and their cut-offs), `02_feature_join` (joins everything with the clusters and demographics and one-hot encodes). |
| `03_feature_analysis/` | `00_visualise_all_features` joins static and temporal features, splits train and validation and scales them, which writes the final tables in `data/features/final/`. `01_feature_ablation` measures how much each of the 329 features matters (results in `result/`). `static_feature_correlations` and `temporal_feature_correlations` look at how features relate to rent. |
| `04_pca_study/` | Checks whether PCA is worth using on the 177 temporal features. |

### `02_modelling/`: the models

| Folder / file | What it does |
| --- | --- |
| `baseline_features.ipynb` | Shows the final input features the models use. |
| `nn_utils.py` | Shared helpers for loading the data, the networks, and saving models and predictions. |
| `single_stage_models/` | One model predicts rent directly: `catboost_team_main_model.ipynb` (CatBoost) and `neural_network/static_temporal_ffn.ipynb` (a network on static and temporal features together). |
| `two_stage_models/` | Stage 1 predicts the market level of each month, stage 2 predicts the flat relative to it. `two_stage_lr_forest.ipynb`: Lasso, then random forest. `neural_network/temporal_ffn.ipynb` and `neural_network/two_stage_nn_forest.ipynb`: network, then random forest. `time_series_plus_residual_model.ipynb` and `gaussian_process_baseline.ipynb`: time-series or Gaussian-process stage 1. |
| `neural_network_retraining/` | Refits the chosen models on train plus validation months for the final Kaggle predictions. |

Each model folder keeps its fitted models in `results/models/` and training statistics in `results/tables/`.

### `03_evaluation/`: comparing the models

| File / folder | What it does |
| --- | --- |
| `overall_model_evaluation.ipynb` | Compares all models on the same validation window, chooses the winner and writes the Kaggle submission (`overall_model_outputs/final_submission.csv`). |
| `evaluation_utils.py` | Helpers for reading predictions, computing metrics and drawing charts. |
| `predictions/` | Each model's predictions: `train/`, `val/`, `test/`, `levels/` (monthly rent level) and `submission/` (the files sent to Kaggle). |
| `charts/`, `tables/` | Figures and metric tables for the report. |

### `data/`

| Path | Contents |
| --- | --- |
| `config.py` | Every path of the project. |
| `feature_config_static.py`, `feature_config_temporal.py` | The lists of feature names used by the models. |
| `original_dataset/` | The competition `train.csv` and `test.csv`, and `auxiliary/` with the supplied tables (HDB blocks, schools, malls, MRT stations, COE and stock prices, map boundaries). |
| `df_keys/` | One row per distinct house in the train and test sets. |
| `features/raw/` | Collected data: MRT graph, school, mall and demographic tables, and the per-flat nearest station and school. |
| `features/cleaned/` | Features by family: temporal, clusters, joined static, one-hot encoded. |
| `features/final/` | The model-ready tables and the scalers that produced them. |
| `train_features.parquet`, `test_features.parquet` | Row-level tables used by the time-series and Gaussian-process notebooks. |

## Time periods

Months are counted from 2021-01. Models are fitted on 2021-01 to 2024-03, scored on the validation window 2024-04 to 2025-03, and predict the test months 2025-03 to 2026-07. The retrained models also use the validation months.