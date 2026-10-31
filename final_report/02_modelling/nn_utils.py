"""Shared helpers for the neural-network experiments (`single_stage_models/neural_network`, `two_stage_models/neural_network`): data loading, the networks, and saving the
outputs of training and inference. A notebook picks where its model and training statistics go with `use_results_dir`.
Scoring, the growth benchmark and the charts live in `03_evaluation` (they only read the saved predictions).

Data: the final feature tables of `final_report/data/features/final`, the same ones `02_modelling/baseline_features.ipynb` queries.

* load_temporal: the 177 temporal features (stock + COE + macro, `data/feature_config_temporal.py`) of the 67 months, from `temporal_scaled.csv`: already scaled by the
  data pipeline (a StandardScaler fitted on the train months 2021-01..2024-03), so the networks do no scaling of their own.
* load_static: the scaled static features (`data/feature_config_static.py`) of every train and test row, with `ROW_ID` (the position of the row in
  `data/original_dataset/train.csv` / `test.csv`) and the month. One feature row per rental: nothing has to be joined back.

Months are counted from 2021-01 = 0, with the split of the final feature tables: train 0..38 (to 2024-03), validation 39..50 (2024-04..2025-03), test 50..66
(2025-03..2026-07). The models fit on 0..38 and score 39..50; they are not refitted on the validation months.
"""
import copy
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]                   # final_report
sys.path.insert(0, str(ROOT))
from data.feature_config_static import CATEGORICAL_COL, HOT_COL, ID_COL, LABEL, NUMERICAL_COL      # noqa: E402
from data.feature_config_temporal import COE_FEATURE, MACRO_FEATURE, STOCK_FEATURE         # noqa: E402

FINAL_PATH = ROOT / 'data' / 'features' / 'final'
ORIGINAL_PATH = ROOT / 'data' / 'original_dataset'
TEMPORAL_FEATURES = STOCK_FEATURE + COE_FEATURE + MACRO_FEATURE
N_MONTHS = 67
TRAIN_LAST, VALIDATION_LAST, TEST_FIRST = 38, 50, 50
Q1_2025 = [48, 49, 50]                                       # month indices 2025-01..03: the train flats of these months are the fixed flats of the level tables


def month_index(year, month):
    return (np.asarray(year) - 2021) * 12 + np.asarray(month) - 1


def month_label(m):
    """'2025-03' for month index 50 (an array in, an array out)."""
    return pd.Series(np.atleast_1d(m)).map(lambda k: f'{2021 + k // 12}-{k % 12 + 1:02d}').to_numpy()


# ------------------------------------------------------------------------------------------------------------------------------------ data
def load_temporal():
    """The 177 temporal features of the months 0..66, scaled by the data pipeline (`temporal_scaled.csv`: a StandardScaler fitted on the train months, the same window as the static
    scaler, so no validation or test month enters the scaling). A stock's month-over-month return, missing in 2021-01, is set to 0, the scaled mean."""
    df = pd.read_csv(FINAL_PATH / 'temporal_scaled.csv')
    df.index = month_index(df['RENT_APPROVAL_YEAR'], df['RENT_APPROVAL_MONTH'])
    assert list(df.index) == list(range(N_MONTHS))
    return df[TEMPORAL_FEATURES].fillna(0.0)


def load_static():
    """(train, test, features): the static feature tables of the final feature set and the list of model input columns.

    train holds the train and validation rows (`sample_type` 'train' / 'val'); both frames get ROW_ID, the month, and the features as float32.
    The numeric columns and the one-hot columns are used as they are (the numeric columns were scaled on 2021-01..2024-03 by the data pipeline). The ordinal
    columns (`CATEGORICAL_COL`, unscaled counts such as MAX_FLOOR) are standardised with the mean and standard deviation of the 'train' rows; a column that is constant
    (`station_is_open`) carries no information and is dropped. Identifier columns (town, block, coordinates, ...) are not inputs.
    """
    train = pd.read_csv(FINAL_PATH / 'static_train_scaled.csv')
    test = pd.read_csv(FINAL_PATH / 'static_test_scaled.csv')
    raw_train = pd.read_csv(ORIGINAL_PATH / 'train.csv', usecols=['RENT_APPROVAL_DATE', 'LEASE_COMMENCE_DATE', LABEL])
    raw_test = pd.read_csv(ORIGINAL_PATH / 'test.csv', usecols=['RENT_APPROVAL_DATE'])
    # the pipeline dropped the train rows whose lease starts after the year of the rental approval (one row): ROW_ID is the position in the original file
    keep = raw_train['RENT_APPROVAL_DATE'].str[:4].astype(int) >= raw_train['LEASE_COMMENCE_DATE'].astype(int)
    train.insert(0, 'ROW_ID', np.flatnonzero(keep))
    test.insert(0, 'ROW_ID', np.arange(len(test)))
    for frame, raw in ((train, raw_train.loc[keep].reset_index(drop=True)), (test, raw_test)):
        frame['month'] = month_index(frame['RENT_APPROVAL_YEAR'], frame['RENT_APPROVAL_MONTH'])
        assert (month_label(frame['month'].to_numpy()) == raw['RENT_APPROVAL_DATE'].to_numpy()).all(), 'rows not aligned with the original file'
    assert len(train) == keep.sum() and len(test) == len(raw_test)
    assert (train[LABEL].to_numpy() == raw_train.loc[keep, LABEL].to_numpy()).all()
    assert train.loc[train['sample_type'] == 'train', 'month'].max() == TRAIN_LAST and train['month'].max() == VALIDATION_LAST

    fit = train['sample_type'] == 'train'
    ordinal = [c for c in CATEGORICAL_COL if train[c].nunique() > 1]
    for c in ordinal:
        mean, std = train.loc[fit, c].mean(), train.loc[fit, c].std()
        train[c], test[c] = (train[c] - mean) / std, (test[c] - mean) / std
    features = NUMERICAL_COL + ordinal + HOT_COL
    for frame in (train, test):
        frame[features] = frame[features].astype(np.float32)
        assert not frame[features].isna().any().any()
    return train, test, features


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((np.asarray(actual, dtype=np.float64) - np.asarray(predicted, dtype=np.float64)) ** 2)))


# ---------------------------------------------------------------------------------------------------------------------------------- network
def make_network(n_in, hidden, activation):
    """Feed-forward network n_in -> hidden[0] -> hidden[1] -> ... -> 1: LeakyReLU or Tanh after each hidden layer, a plain linear output."""
    act = {'leaky_relu': lambda: nn.LeakyReLU(0.01), 'tanh': nn.Tanh}[activation]
    layers, size = [], n_in
    for width in hidden:
        layers += [nn.Linear(size, width), act()]
        size = width
    return nn.Sequential(*layers, nn.Linear(size, 1))


def _inputs(Z_t, month_t, X_t=None):
    """Network inputs of the rows: the static features of the row (if any) followed by the temporal features of its month."""
    z = Z_t[month_t]
    return z if X_t is None else torch.cat([X_t, z], dim=1)


def _predict(net, y_mean, y_std, Z_t, month_t, X_t=None, chunk=20000):
    net.eval()
    with torch.no_grad():
        out = [net(_inputs(Z_t, month_t[i:i + chunk], None if X_t is None else X_t[i:i + chunk])).squeeze(1).numpy() for i in range(0, len(month_t), chunk)]
    return np.concatenate(out).astype(np.float64) * y_std + y_mean


def fit_ffn(Z, months, y, X=None, valid=None, hidden=(64, 16), activation='leaky_relu', epochs=100, batch_size=2048, learning_rate=1e-3, weight_decay=0.0, seed=0, patience=None):
    """Feed-forward network from the scaled temporal features of a row's month (Z: 67 months x features) and, if X is given, its static features
    (rows x features) to the raw monthly rent. MSE in dollars: the target is standardised inside the loop.

    months: month of every row. valid: optional (months, X, y) of held-out rows, scored after every epoch (never trained on).
    patience: early stopping on valid (required with it): training stops once the validation RMSE has not improved for `patience` epochs (`epochs` is then the maximum) and the
    weights of the best epoch are kept. None: all `epochs` are trained and the last weights are kept.
    Returns a model dict (see predict_ffn / save_ffn) with the epoch losses in model['history'].
    """
    torch.manual_seed(seed)
    Z_t = torch.tensor(Z.to_numpy(), dtype=torch.float32)
    X_t = None if X is None else torch.tensor(np.asarray(X), dtype=torch.float32)
    month_t = torch.tensor(np.asarray(months), dtype=torch.long)
    y = np.asarray(y, dtype=np.float64)
    y_mean, y_std = y.mean(), y.std()
    target = torch.tensor((y - y_mean) / y_std, dtype=torch.float32)
    n_static = 0 if X_t is None else X_t.shape[1]
    net = make_network(n_static + Z_t.shape[1], list(hidden), activation)
    optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    if valid is not None:
        v_month = torch.tensor(np.asarray(valid[0]), dtype=torch.long)
        v_X = None if valid[1] is None else torch.tensor(np.asarray(valid[1]), dtype=torch.float32)
        v_y = np.asarray(valid[2], dtype=np.float64)
    assert patience is None or valid is not None, 'early stopping needs validation rows'
    history, best_rmse, best_epoch, best_state = [], np.inf, None, None
    for epoch in range(1, epochs + 1):
        net.train()
        order = torch.randperm(len(month_t))
        total = 0.0
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            loss = loss_fn(net(_inputs(Z_t, month_t[idx], None if X_t is None else X_t[idx])).squeeze(1), target[idx])
            optimiser.zero_grad(); loss.backward(); optimiser.step()
            total += loss.item() * len(idx)
        row = {'epoch': epoch, 'train_mse_std': total / len(order), 'train_rmse_dollar': float(np.sqrt(total / len(order)) * y_std)}
        if valid is not None:
            row['valid_rmse_dollar'] = float(np.sqrt(np.mean((_predict(net, y_mean, y_std, Z_t, v_month, v_X) - v_y) ** 2)))
            if patience is not None and row['valid_rmse_dollar'] < best_rmse:
                best_rmse, best_epoch, best_state = row['valid_rmse_dollar'], epoch, copy.deepcopy(net.state_dict())
        history.append(row)
        if patience is not None and epoch - (best_epoch or 0) >= patience:
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    config = {'n_static': n_static, 'n_temporal': Z_t.shape[1], 'hidden': list(hidden), 'activation': activation, 'epochs': epochs, 'batch_size': batch_size,
              'learning_rate': learning_rate, 'weight_decay': weight_decay, 'seed': seed, 'patience': patience,
              'epochs_run': len(history), 'best_epoch': len(history) if best_epoch is None else best_epoch}
    return {'net': net, 'y_mean': y_mean, 'y_std': y_std, 'config': config, 'history': pd.DataFrame(history)}


def predict_ffn(model, Z, months, X=None):
    """Predicted rent (dollars) of rows approved in `months` (an array, one month per row) with static features X (None for a temporal-only network)."""
    Z_t = torch.tensor(Z.to_numpy(), dtype=torch.float32)
    X_t = None if X is None else torch.tensor(np.asarray(X), dtype=torch.float32)
    return _predict(model['net'], model['y_mean'], model['y_std'], Z_t, torch.tensor(np.asarray(months), dtype=torch.long), X_t)


def month_levels(model, Z):
    """Predicted rent of a temporal-only network for each of the months of Z (a Series indexed by month)."""
    return pd.Series(predict_ffn(model, Z, Z.index.to_numpy()), index=Z.index)


def fixed_flat_levels(model, Z, X_ref, months):
    """Mean predicted rent of the same flats (static features X_ref) as if they were approved in each of `months`: a level with the flat mix held fixed."""
    return pd.Series({m: predict_ffn(model, Z, np.full(len(X_ref), m), X_ref).mean() for m in months})


def save_ffn(model, path, **meta):
    torch.save({'state_dict': model['net'].state_dict(), 'target_mean': model['y_mean'], 'target_std': model['y_std'], 'config': model['config'], **meta}, path)


def load_ffn(path):
    """Rebuild a model saved by save_ffn: (model dict without the history, the checkpoint with the metadata given to save_ffn)."""
    ckpt = torch.load(path, weights_only=False)
    c = ckpt['config']
    net = make_network(c['n_static'] + c['n_temporal'], c['hidden'], c['activation'])
    net.load_state_dict(ckpt['state_dict']); net.eval()
    return {'net': net, 'y_mean': ckpt['target_mean'], 'y_std': ckpt['target_std'], 'config': c}, ckpt


# ------------------------------------------------------------------------------------------------------------------------------- output folders
RESULTS = MODEL_DIR = TABLE_DIR = None                                          # set by use_results_dir: <results>/models, <results>/tables/<model type>/ (the training statistics)
EVAL_PATH = ROOT / '03_evaluation'
PRED_DIR = EVAL_PATH / 'predictions'                                           # predictions/{train,val,test}/<model>.csv, predictions/levels/<model>.csv
SPLITS = ('train', 'val', 'test')


def use_results_dir(path):
    """Where this notebook saves its models (path/models) and training statistics (path/tables/<model type>/); the predictions always go to 03_evaluation."""
    global RESULTS, MODEL_DIR, TABLE_DIR
    RESULTS = Path(path).resolve()
    MODEL_DIR, TABLE_DIR = RESULTS / 'models', RESULTS / 'tables'


def ensure_output_dirs(model):
    assert RESULTS is not None, 'call use_results_dir first'
    for folder in (MODEL_DIR, TABLE_DIR / model, *(PRED_DIR / d for d in (*SPLITS, 'levels'))):
        os.makedirs(folder, exist_ok=True)


def save_table(table, model, filename, **kwargs):
    """results/tables/<model>/<filename>: a training statistic of the model type (epoch losses, tuning grids, ...)."""
    path = TABLE_DIR / model / filename
    table.to_csv(path, **kwargs)
    return path


def save_predictions(model, split, frame, prediction):
    """03_evaluation/predictions/<split>/<model>.csv: the prediction joined (on ROW_ID) to the identifier columns (ID_COL) of the static feature rows it was made for.

    frame: the rows of the static feature table (ROW_ID and ID_COL columns); split: 'train' (in-sample), 'val' (held-out) or 'test'."""
    assert split in SPLITS
    predicted = pd.DataFrame({'ROW_ID': frame['ROW_ID'].to_numpy(), 'prediction': np.asarray(prediction, dtype=np.float64)})
    out = frame[['ROW_ID'] + ID_COL].merge(predicted, on='ROW_ID', how='left', validate='one_to_one')
    assert len(out) == len(frame) and out['prediction'].notna().all()
    path = PRED_DIR / split / f'{model}.csv'
    out.to_csv(path, index=False)
    return path


def save_levels(model, levels):
    """03_evaluation/predictions/levels/<model>.csv: the model's rent level for each month 0..66 with the flat mix held fixed (a Series by month)."""
    assert list(levels.index) == list(range(N_MONTHS))
    path = PRED_DIR / 'levels' / f'{model}.csv'
    pd.DataFrame({'RENT_APPROVAL_DATE': month_label(levels.index.to_numpy()), 'level': levels.to_numpy(dtype=np.float64)}).to_csv(path, index=False)
    return path
