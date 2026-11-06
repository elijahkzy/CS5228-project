"""Every path of the project, in one place.

Nothing here is tied to a machine: ROOT is worked out from the location of this file (``final_report/data/config.py``), and every other path is built from it.
Copy or move the folder anywhere (another user, another OS, another folder name) and every notebook and script follows.

How to use it
-------------
Notebooks (the same 4 lines everywhere; they walk up from the working directory until they find ``data/config.py``, so they run from any folder inside the project;
set the environment variable ``RENT_PROJECT_ROOT`` if the notebook is launched from outside the project)::

    import os, sys
    from pathlib import Path
    _here = Path(os.environ.get('RENT_PROJECT_ROOT') or Path.cwd()).resolve()
    sys.path.insert(0, str(next(p for p in [_here, *_here.parents] if (p / 'data' / 'config.py').exists())))
    from data import config as cfg

Modules and scripts: ``sys.path.insert(0, str(Path(__file__).resolve().parents[N]))`` (N = how many folders up final_report is), then ``from data import config as cfg``.

Then ``pd.read_csv(cfg.TRAIN_CSV)``, ``cfg.FINAL_DIR / 'temporal_scaled.csv'``, ``f'{cfg.LOCATION_MINING_DIR}/station-keys.csv'``, and so on. Every value is a ``pathlib.Path``.

Run ``python data/config.py`` to list every path and whether it exists on this machine.

Optional environment variables
------------------------------
RENT_PROJECT_ROOT          where the notebooks start looking for this file (only needed when they are launched from outside the project)
CS5228_PREDICTION_BUNDLES  extra folders (os.pathsep-separated) holding train/ val/ test/ prediction files, for 03_evaluation/overall_model_evaluation.ipynb
CS5228_LABEL_DATA_ROOT     a folder with train.csv and test.csv, if the original dataset is kept outside the project
CS5228_KEYS_FILE           the API keys file read by the data collection scripts (default: 00_data_collection/scripts/keys.txt, not shipped: it holds secrets)
"""
import os
from pathlib import Path

# ============================================================================================================================================ project root
ROOT = Path(__file__).resolve().parents[1]                                   # final_report

# ============================================================================================================================================ pipeline folders
DATA_COLLECTION_DIR = ROOT / '00_data_collection'
SCRIPTS_DIR = DATA_COLLECTION_DIR / 'scripts'                                # the scraping / API scripts
DATA_PROCESSING_DIR = ROOT / '01_data_processing'
MODELLING_DIR = ROOT / '02_modelling'                                        # holds nn_utils.py
EVALUATION_DIR = ROOT / '03_evaluation'                                      # holds evaluation_utils.py

# ============================================================================================================================================ data/
DATA_DIR = ROOT / 'data'
FEATURE_CONFIG_STATIC_PY = DATA_DIR / 'feature_config_static.py'
FEATURE_CONFIG_TEMPORAL_PY = DATA_DIR / 'feature_config_temporal.py'

# ---- data/original_dataset: the competition files
ORIGINAL_DIR = DATA_DIR / 'original_dataset'
TRAIN_CSV = ORIGINAL_DIR / 'train.csv'
TEST_CSV = ORIGINAL_DIR / 'test.csv'

# ---- data/original_dataset/auxiliary: the supplied external tables
AUXILIARY_DIR = ORIGINAL_DIR / 'auxiliary'
HDB_BLOCK_CSV = AUXILIARY_DIR / 'sg-hdb-block.csv'
HDB_PROPERTY_INFO_CSV = AUXILIARY_DIR / 'HDBPropertyInformation.csv'
RENTING_OUT_OF_FLATS_CSV = AUXILIARY_DIR / 'RentingOutofFlatsfromJan2021.csv'
SCHOOLS_CSV = AUXILIARY_DIR / 'sg-schools.csv'
SHOPPING_MALLS_CSV = AUXILIARY_DIR / 'sg-shopping-malls.csv'
MRT_STATIONS_CSV = AUXILIARY_DIR / 'sg-mrt-stations.csv'
COE_PRICES_CSV = AUXILIARY_DIR / 'sg-coe-prices.csv'
STOCK_PRICES_CSV = AUXILIARY_DIR / 'sg-stock-prices.csv'
SUBZONE_GEOJSON = AUXILIARY_DIR / 'MasterPlan2019SubzoneBoundaryNoSeaGEOJSON.geojson'
PLANNING_AREA_GEOJSON = AUXILIARY_DIR / 'MasterPlan2019PlanningAreaBoundaryNoSea.geojson'
# Read by the notebooks of 00_data_collection and 02_modelling, but NOT in the folder as shipped (produced by scripts/get_bus_stops.py and the data_viz /
# time-series notebooks; kept at the location those notebooks expect):
BUS_STOPS_CSV = AUXILIARY_DIR / 'sg-bus-stops.csv'
HDB_ACCESSIBILITY_CSV = AUXILIARY_DIR / 'hdb_features_with_accessibility.csv'

# ---- data/processed: intermediate tables written by 00_data_collection/01_data_viz.ipynb (created on first use)
PROCESSED_DIR = DATA_DIR / 'processed'
HDB_FEATURES_BEFORE_ACCESSIBILITY_CSV = PROCESSED_DIR / 'hdb_features_before_accessibility.csv'
HDB_FEATURES_WITH_ACCESSIBILITY_CSV = PROCESSED_DIR / 'hdb_features_with_accessibility.csv'
ACCESSIBILITY_ROUTE_CACHE_CSV = PROCESSED_DIR / 'accessibility_route_cache.csv'
HDB_TRAVEL_TIME_INPUT_CSV = PROCESSED_DIR / 'hdb_df_for_travel_time.csv'     # input of scripts/get_distance_to_place.py

# ---- data/df_keys: the distinct houses (address keys) of the train and test sets
DF_KEYS_DIR = DATA_DIR / 'df_keys'
TRAIN_HOUSE_CSV = DF_KEYS_DIR / 'train-house.csv'
TEST_HOUSE_CSV = DF_KEYS_DIR / 'test-house.csv'

# ---- data/*.parquet: row-level tables used by the two-stage time-series / Gaussian-process notebooks
TRAIN_FEATURES_PARQUET = DATA_DIR / 'train_features.parquet'
TEST_FEATURES_PARQUET = DATA_DIR / 'test_features.parquet'

# ---- data/model_outputs: predictions and metrics of the CatBoost, time-series and Gaussian-process notebooks (created on first use)
MODEL_OUTPUTS_DIR = DATA_DIR / 'model_outputs'
CATBOOST_OUTPUT_DIR = MODEL_OUTPUTS_DIR / 'catboost'
GAUSSIAN_PROCESS_OUTPUT_DIR = DATA_DIR                                       # gaussian_process_*.csv are written straight into data/
TIME_SERIES_OUTPUT_DIR = DATA_DIR                                            # *time_series_plus_residual*.csv are written straight into data/

# ============================================================================================================================================ data/features
FEATURES_DIR = DATA_DIR / 'features'

# ---- raw: collected by 00_data_collection/scripts and aggregated by 01_data_processing/02_static_features
RAW_DIR = FEATURES_DIR / 'raw'

RAW_GRAPH_DIR = RAW_DIR / 'raw_graph_data'                                   # the MRT graph (nodes, edges) and the station / school keys
MRT_NODES_CSV = RAW_GRAPH_DIR / 'sg-mrt-nodes.csv'
MRT_EDGES_CSV = RAW_GRAPH_DIR / 'sg-mrt-edges.csv'
MRT_EDGES_COLLAPSED_CSV = RAW_GRAPH_DIR / 'sg-mrt-edges-collapsed.csv'
MRT_EDGES_WEIGHTED_CSV = RAW_GRAPH_DIR / 'sg-mrt-edges-weighted.csv'
RAW_GRAPH_STATION_KEYS_CSV = RAW_GRAPH_DIR / 'station-keys.csv'
RAW_GRAPH_HDB_BLOCK_CSV = RAW_GRAPH_DIR / 'sg-hdb-block.csv'

LOCATION_MINING_DIR = RAW_DIR / 'location_data_mining'                       # per-house nearest station / school, station keys and traffic
STATION_KEYS_CSV = LOCATION_MINING_DIR / 'station-keys.csv'
STATION_TRAFFIC_CSV = LOCATION_MINING_DIR / 'station-traffic.csv'
STATION_TRAFFIC_STD_CSV = LOCATION_MINING_DIR / 'station-traffic-std.csv'
HDB_NEAREST_STATION_CSV = LOCATION_MINING_DIR / 'hdb_nearest_station.csv'
HDB_NEAREST_SCHOOL_CSV = LOCATION_MINING_DIR / 'hdb_nearest_school.csv'
LOCATION_MINING_SCALERS_DIR = LOCATION_MINING_DIR / 'scalers'
TRAFFIC_SCALER_PKL = LOCATION_MINING_SCALERS_DIR / 'traffic_scale.pkl'

COLLECTED_DIR = RAW_DIR / 'collected_data'                                   # outputs of the scraping scripts
MRT_DATA_DIR = COLLECTED_DIR / 'mrt_data'
STATION_MALL_LINKS_CSV = MRT_DATA_DIR / 'sg-station-mall-links.csv'
STATION_MALL_PROXIMITY_CSV = MRT_DATA_DIR / 'sg-station-mall-proximity.csv'
STATION_MALL_COUNTS_CSV = MRT_DATA_DIR / 'sg-station-mall-counts.csv'
MRT_NODE_FEATURES_CSV_TEMPLATE = str(MRT_DATA_DIR / 'sg-mrt-nodes-features-{date}.csv')    # .format(date='202605')
SCHOOL_DATA_DIR = COLLECTED_DIR / 'school_data'                              # sg-schools-<primary|secondary|junior_college|mixed>.csv
SCHOOL_CSV_TEMPLATE = str(SCHOOL_DATA_DIR / 'sg-schools-{name}.csv')         # .format(name='primary')
MALL_FEATURES_DIR = COLLECTED_DIR / 'mall_features'
MALL_FEATURES_WIKIPEDIA_CSV = MALL_FEATURES_DIR / 'sg-mall-features-wikipedia.csv'
ZONING_DATA_DIR = COLLECTED_DIR / 'zoning_data'
PLANNING_AREA_CSV = ZONING_DATA_DIR / 'planning_area.csv'
SUBZONE_CSV = ZONING_DATA_DIR / 'subzone.csv'
REGION_CSV = ZONING_DATA_DIR / 'region.csv'
DEMOGRAPHICS_PLANNING_AREA_CSV = ZONING_DATA_DIR / 'sg-demographics-planning-area.csv'
DEMOGRAPHICS_SUBZONE_CSV = ZONING_DATA_DIR / 'sg-demographics-subzone.csv'

# Download caches of the scraping scripts (so a rerun does not hit the web again)
ONEMAP_CACHE_DIR = ZONING_DATA_DIR / '_onemap_cache'
PV_CACHE_DIR = MRT_DATA_DIR / '_pv_cache'
OSM_CACHE_DIR = MRT_DATA_DIR / '_osm_cache'
OSM_SHELTERED_NETWORK_JSON = OSM_CACHE_DIR / 'sheltered_network.json'
WIKI_CACHE_DIR = MALL_FEATURES_DIR / '_wiki_cache'
SCHOOL_CACHE_DIR = SCHOOL_DATA_DIR / '_school_cache'

# API keys for the scraping scripts (a text file kept out of the shared folder; point CS5228_KEYS_FILE at it)
KEYS_FILE = Path(os.environ.get('CS5228_KEYS_FILE') or SCRIPTS_DIR / 'keys.txt').expanduser()

# ---- cleaned: one table per feature family
CLEANED_DIR = FEATURES_DIR / 'cleaned'

TEMPORAL_FEATURES_DIR = CLEANED_DIR / 'temporal_features'                    # stock, COE and macro features by month
STOCK_FEATURES_CSV = TEMPORAL_FEATURES_DIR / 'stock_features.csv'
COE_FEATURES_MONTHLY_CSV = TEMPORAL_FEATURES_DIR / 'coe_features_monthly.csv'
MACRO_FEATURES_CSV = TEMPORAL_FEATURES_DIR / 'sg_macro_features_monthly.csv'
FEATURE_COVERAGE_CSV = TEMPORAL_FEATURES_DIR / 'feature_coverage.csv'
SELECTED_TEMPORAL_FEATURES_CSV = TEMPORAL_FEATURES_DIR / 'selected_temporal_features.csv'
SELECTED_TEMPORAL_FEATURES_JSON = TEMPORAL_FEATURES_DIR / 'selected_temporal_features.json'

STATIC_FEATURES_DIR = CLEANED_DIR / 'static_features'
LOCATION_CLUSTER_DIR = STATIC_FEATURES_DIR / 'location_cluster'              # DBSCAN cluster of every house
TRAIN_DBSCAN_CLUSTER_CSV = LOCATION_CLUSTER_DIR / 'train_dbscan_cluster.csv'
TEST_DBSCAN_CLUSTER_CSV = LOCATION_CLUSTER_DIR / 'test_dbscan_cluster.csv'
UNPROCESSED_DIR = STATIC_FEATURES_DIR / 'unprocessed'                        # all static features joined, before encoding
TRAIN_FEATURE_RAW_CSV = UNPROCESSED_DIR / 'train_feature_raw.csv'
TEST_FEATURE_RAW_CSV = UNPROCESSED_DIR / 'test_feature_raw.csv'
ONE_HOT_DIR = STATIC_FEATURES_DIR / 'one_hot'                                # after one-hot encoding
TRAIN_ONE_HOT_CSV = ONE_HOT_DIR / 'train_one_hot.csv'
TEST_ONE_HOT_CSV = ONE_HOT_DIR / 'test_one_hot.csv'

# ---- final: the model-ready tables and the scalers that made them
FINAL_DIR = FEATURES_DIR / 'final'
STATIC_TRAIN_SCALED_CSV = FINAL_DIR / 'static_train_scaled.csv'
STATIC_TEST_SCALED_CSV = FINAL_DIR / 'static_test_scaled.csv'
TEMPORAL_RAW_CSV = FINAL_DIR / 'temporal_raw.csv'
TEMPORAL_SCALED_CSV = FINAL_DIR / 'temporal_scaled.csv'
SCALERS_DIR = FINAL_DIR / 'scalers'
STATIC_SCALER_PKL = SCALERS_DIR / 'static_numerical_scalers.pkl'
TEMPORAL_SCALER_PKL = SCALERS_DIR / 'temporal_numerical_scalers.pkl'

# ============================================================================================================================================ 01_data_processing
DBSCAN_MODELS_DIR = DATA_PROCESSING_DIR / '01_clustering_features' / 'models'    # fitted DBSCAN objects (eps_<m>m.pkl, dbscan_object.pkl, dbscan_output.npz)
FEATURE_ANALYSIS_DIR = DATA_PROCESSING_DIR / '03_feature_analysis'          # 00_visualise_all_features, 01_feature_ablation, 02/03_*_correlations
PCA_STUDY_DIR = DATA_PROCESSING_DIR / '04_pca_study'                         # pca_temporal_features.ipynb (it reads nn_utils.py from MODELLING_DIR)
ABLATION_RESULT_DIR = FEATURE_ANALYSIS_DIR / 'result'
ABLATION_STUDY_CSV = ABLATION_RESULT_DIR / 'ablation_study.csv'
ABLATION_BY_CATEGORY_CSV = ABLATION_RESULT_DIR / 'ablation_by_category.csv'
ABLATION_LOCATION_ENCODINGS_CSV = ABLATION_RESULT_DIR / 'ablation_location_encodings.csv'

# ============================================================================================================================================ 02_modelling
# Each folder below holds models/ and tables/<model>/ written by the notebook next to it (see nn_utils.use_results_dir).
SINGLE_STAGE_NN_RESULTS_DIR = MODELLING_DIR / 'single_stage_models' / 'neural_network' / 'results'
TWO_STAGE_NN_RESULTS_DIR = MODELLING_DIR / 'two_stage_models' / 'neural_network' / 'results'
TWO_STAGE_RESULTS_DIR = MODELLING_DIR / 'two_stage_models' / 'results'      # the Lasso + random forest model
NN_RETRAINING_RESULTS_DIR = MODELLING_DIR / 'neural_network_retraining' / 'results'

# ============================================================================================================================================ 03_evaluation
PREDICTIONS_DIR = EVALUATION_DIR / 'predictions'                            # {train,val,test,levels,submission}/<model>.csv
CHARTS_DIR = EVALUATION_DIR / 'charts'
TABLES_DIR = EVALUATION_DIR / 'tables'
OVERALL_OUTPUT_DIR = EVALUATION_DIR / 'overall_model_outputs'               # written by overall_model_evaluation.ipynb

# Folders searched for prediction bundles (a bundle has train/ val/ test/ subfolders) by overall_model_evaluation.ipynb, most important first (the notebook
# puts the folders of CS5228_PREDICTION_BUNDLES in front of these). The project's own predictions come last; the others only count if they exist on the
# machine running the notebook.
_DOWNLOADS = Path.home() / 'Downloads' / 'Telegram Desktop'
PREDICTION_BUNDLE_SEARCH_DIRS = [
    _DOWNLOADS / 'lasso' / 'lasso',
    _DOWNLOADS / 'predictions' / 'predictions',
    _DOWNLOADS / 'predictions-MarkLim' / 'predictions-MarkLim',
    _DOWNLOADS / 'predictions-CatBoost' / 'predictions-CatBoost',
    PREDICTIONS_DIR,
]
# Folders searched for train.csv + test.csv (the labels), most important first (the notebook puts CS5228_LABEL_DATA_ROOT in front of these).
LABEL_DATA_SEARCH_DIRS = [
    _DOWNLOADS / 'final_report' / 'final_report' / 'data' / 'original_dataset',
    ORIGINAL_DIR,
]


# ============================================================================================================================================ helpers
def paths():
    """{name: Path} of every path defined above (templates and search lists excluded)."""
    return {k: v for k, v in globals().items() if k.isupper() and not k.startswith('_') and isinstance(v, Path)}


def report():
    """Print every path with a tick if it exists on this machine. Run `python data/config.py`."""
    for name, path in paths().items():
        print(f"{'ok     ' if path.exists() else 'missing'}  {name:<42} {path.relative_to(ROOT) if path != ROOT and ROOT in path.parents else path}")


if __name__ == '__main__':
    print('project root:', ROOT)
    report()
