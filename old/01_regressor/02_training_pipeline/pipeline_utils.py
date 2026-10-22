"""Shared helpers for the 03 training pipeline.

Four model families, all on log rent:

* two_stage_rf / two_stage_rf_ridge     log rent_i = alpha_m + g(static features of flat i)
    - alpha_m (trend): the monthly level of the average flat, decomposed from the data with a Ridge that has one dummy per year-month plus all
      the static features. A Lasso (two_stage_rf) or a Ridge (two_stage_rf_ridge) on the selected temporal (stock + economy) features learns
      to forecast it.
    - g (static model): a Random Forest fit to log rent minus alpha_m on the static features, optionally with interaction features
      (numeric static features x selected temporal features).
* single_stage_lasso / single_stage_ridge   one Lasso (or Ridge) on everything at once: static features, the selected temporal features and all
  the interactions between them.

Months are counted from 2021-01 = 0: train 0..44 (to 2024-09), validation 45..50 (2024-10..2025-03), test 50..66 (2025-03..2026-07).
A model key is '<window>/<family>', e.g. 'from_2023/two_stage_rf'. Windows: full_history (from 2021-01), from_2023 (from 2023-01), from_2023_06 (only the
months after May 2023, i.e. from 2023-06 = month 29).
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

ROOT = '/Users/keltonn/Desktop/Projects/housing_price_regressor'
DATA = f'{ROOT}/data'
STATIC_DIR = f'{ROOT}/00_static_features/01_encoding/data'
for _path in (ROOT, DATA):
    if _path not in sys.path:
        sys.path.append(_path)
from stock_selection import near_duplicate_filter  # noqa: E402

KEYS = ['RENT_APPROVAL_YEAR', 'RENT_APPROVAL_MONTH', 'TOWN', 'BLOCK', 'STREET']
WINDOW_START = {'full_history': 0, 'from_2023': 24, 'from_2023_06': 29}      # first month used for fitting in each setup (2021-01, 2023-01, 2023-06)
FAMILIES = ('two_stage_rf', 'two_stage_rf_ridge', 'single_stage_lasso', 'single_stage_ridge')
TRAIN_LAST, VALIDATION_FIRST, VALIDATION_LAST, TEST_LAST = 44, 45, 50, 66


# ------------------------------------------------------------------------------------------------------------------------------ data
def month_index(df):
    return ((df['RENT_APPROVAL_YEAR'] - 2021) * 12 + df['RENT_APPROVAL_MONTH'] - 1).to_numpy()


def load_static():
    """Processed static features from 00_static_features/01_encoding: {'train', 'validation', 'test'} frames (+ a `month` column)."""
    meta = json.load(open(f'{STATIC_DIR}/preprocessing_metadata.json'))
    numeric, onehot = meta['numeric_features'], meta['encoded_columns']
    dtypes = {c: 'int8' for c in onehot} | {c: 'float32' for c in numeric}
    frames = {}
    for name in ('train', 'validation', 'test'):
        df = pd.read_csv(f'{STATIC_DIR}/{name}_processed.csv', dtype=dtypes)
        df['month'] = month_index(df)
        frames[name] = df
    return frames, numeric, onehot


def design(df, numeric, onehot):
    return df[numeric + onehot].to_numpy(dtype=np.float32)


def log_rent(df):
    return np.log(df['MONTHLY_RENT'].to_numpy(dtype=np.float64))


def load_temporal():
    """Stock aggregates, economy features and COE (car licence) features, one row per month (index = months since 2021-01).

    COE: quota premium (log), quota and bids received for categories A, B, C and E (the prevailing premium is not used: it depends on the licence age).
    ANNUAL_*: flats reaching their minimum occupation period (MOP) and the mid-year population (total, citizens, permanent residents,
    non-residents), one value per calendar year repeated in each month of that year."""
    stock = pd.read_csv(f'{DATA}/temporal_features/stock_features.csv')
    stock.index = (stock['RENT_APPROVAL_YEAR'] - 2021) * 12 + stock['RENT_APPROVAL_MONTH'] - 1
    stock = stock.drop(columns=['RENT_APPROVAL_YEAR', 'RENT_APPROVAL_MONTH'])
    macro = pd.read_csv(f'{DATA}/temporal_features/sg_macro_features_monthly.csv')
    date = pd.to_datetime(macro['RENT_APPROVAL_DATE'])
    macro.index = (date.dt.year - 2021) * 12 + date.dt.month - 1
    macro = macro.drop(columns=['RENT_APPROVAL_DATE'])
    coe = pd.read_csv(f'{DATA}/temporal_features/coe_features_monthly.csv')
    coe.index = (coe['RENT_APPROVAL_YEAR'] - 2021) * 12 + coe['RENT_APPROVAL_MONTH'] - 1
    coe = coe.drop(columns=['RENT_APPROVAL_YEAR', 'RENT_APPROVAL_MONTH'])
    for column in [c for c in coe if c.endswith('_quota_premium')]:       # the premium (price) enters as a log, like rent
        coe[column.replace('_quota_premium', '_log_quota_premium')] = np.log(coe.pop(column))
    annual = pd.read_csv(f'{DATA}/temporal_features/annual_mop_population.csv')        # one value per year, repeated in each of its months
    annual = annual.set_index('YEAR')[['MOP_UNITS', 'TOTAL_POPULATION', 'CITIZENS', 'PERMANENT_RESIDENTS', 'NON_RESIDENTS']].add_prefix('ANNUAL_')
    months = pd.Index(sorted(set(stock.index) | set(macro.index) | set(coe.index)))
    annual = annual.reindex(2021 + months // 12).set_axis(months)
    return pd.concat([stock, macro, coe, annual], axis=1)


# ------------------------------------------------------------------------------------------------------------ trend and selection
def decompose_trend(X, y, month, months, ridge_alpha=1.0):
    """alpha_m for each month in `months`: Ridge of log rent on one dummy per month plus all (centred) static features."""
    months = np.asarray(sorted(months))
    rows = np.flatnonzero(np.isin(month, months))
    dummies = np.zeros((len(rows), len(months)), dtype=np.float32)
    dummies[np.arange(len(rows)), np.searchsorted(months, month[rows])] = 1.0
    model = Ridge(alpha=ridge_alpha).fit(np.hstack([dummies, X[rows] - X[rows].mean(axis=0)]), y[rows])
    return pd.Series(model.intercept_ + model.coef_[:len(months)], index=months)


def trend_stability(features, trend, top_ks=(10,), min_months=12):
    """Stability selection with L2 of the temporal features against the decomposed trend.

    features: months x candidate features (complete over the months of `trend`); trend: alpha_m by month.
    For each expanding window of months, a RidgeCV (penalty chosen by forward chaining) is fit to the standardised features; the top_k by absolute
    coefficient score a point. Returns the share of windows in which each (de-duplicated) feature scored, one column per top_k.
    """
    months = list(trend.index)
    features = features.loc[months]
    kept = near_duplicate_filter(features, trend)
    X = features[kept]
    counts = {k: np.zeros(len(kept)) for k in top_ks}
    n_windows = 0
    for end in range(min_months, len(months) + 1):
        Xa = StandardScaler().fit_transform(X.iloc[:end])
        ridge = RidgeCV(alphas=np.logspace(-1, 4, 30), cv=TimeSeriesSplit(4), scoring='neg_mean_squared_error')
        ridge.fit(Xa, trend.iloc[:end].values)
        order = np.argsort(-np.abs(ridge.coef_))
        for k in top_ks:
            counts[k][order[:k]] += 1
        n_windows += 1
    return pd.DataFrame({k: counts[k] / n_windows for k in top_ks}, index=kept)


def pick(stability, threshold=0.5):
    """Features at or above the threshold; never empty (falls back to the 3 most stable)."""
    chosen = stability.index[stability >= threshold]
    return list(chosen if len(chosen) else stability.nlargest(3).index)


def zscore(features, selected, fit_months):
    """Selected temporal features standardised with the mean/std of `fit_months`; months before a stock existed (NaN) become 0 (the mean)."""
    ref = features.loc[fit_months, selected]
    return ((features[selected] - ref.mean()) / ref.std()).fillna(0.0)


# --------------------------------------------------------------------------------------------------------------- design matrices
def interaction_block(X_numeric, Z_rows):
    """Every numeric static feature times every selected temporal feature: rows x (n_numeric * n_temporal), numeric-major order."""
    Z_rows = np.asarray(Z_rows, dtype=np.float32)
    return (X_numeric[:, :, None] * Z_rows[:, None, :]).reshape(len(X_numeric), -1).astype(np.float32)


def interaction_names(numeric, selected):
    return [f'{n} x {t}' for n in numeric for t in selected]


def static_design(X, n_numeric, Z, months, use_interactions):
    """Static features, plus the interactions with the temporal features of each row's month when requested (two-stage static model)."""
    if not use_interactions:
        return X
    return np.hstack([X, interaction_block(X[:, :n_numeric], Z.loc[months].to_numpy())])


def single_stage_design(X, n_numeric, Z, months):
    """Everything for the single-stage Lasso: static features, the temporal features of each row's month, and their interactions."""
    Z_rows = Z.loc[months].to_numpy(dtype=np.float32)
    return np.hstack([X, Z_rows, interaction_block(X[:, :n_numeric], Z_rows)])


def single_stage_names(numeric, onehot, selected):
    return numeric + onehot + list(selected) + interaction_names(numeric, selected)


def ridge_path(Xs, y_centered, alphas):
    """Ridge coefficients for every penalty (sklearn convention: ||y - Xw||^2 + alpha ||w||^2) on centred columns, from one eigendecomposition."""
    Xs = np.asarray(Xs, dtype=np.float64)
    w, V = np.linalg.eigh(Xs.T @ Xs)
    Vb = V.T @ (Xs.T @ np.asarray(y_centered, dtype=np.float64))
    return np.column_stack([V @ (Vb / (np.clip(w, 0, None) + a)) for a in alphas])


# ------------------------------------------------------------------------------------------------------------------------- models
def make_static_model(params, seed=0):
    return RandomForestRegressor(n_jobs=-1, random_state=seed, **params)


def fit_model(key, setup, X, y, month, temporal, numeric, onehot, selected):
    """Fit one setup on the rows given. `month` holds each row's month; its maximum is the last month of the fitting data."""
    window, family = key.split('/')
    start = WINDOW_START[window]
    months = np.arange(start, int(month.max()) + 1)
    s1_months = months[months >= 1]                       # 2021-01 has no month-over-month stock return
    n_numeric = len(numeric)
    Z = zscore(temporal, selected, s1_months)             # temporal features standardised on the fitting months
    rows = np.flatnonzero(month >= start)
    fitted = {'key': key, 'window': window, 'family': family, 'setup': setup, 'selected': list(selected), 'Z': Z,
              'n_numeric': n_numeric, 'numeric': numeric, 'onehot': onehot}
    if family in ('two_stage_rf', 'two_stage_rf_ridge'):
        alpha_m = decompose_trend(X, y, month, months)
        fitted['alpha_m'] = alpha_m
        trend_model = Lasso(alpha=setup['lasso_alpha'], max_iter=100_000) if family == 'two_stage_rf' else Ridge(alpha=setup['ridge_alpha'])
        fitted['linear'] = trend_model.fit(Z.loc[s1_months], alpha_m.loc[s1_months])      # the trend model (Lasso or Ridge)
        fitted['static_model'] = make_static_model(setup['static_params']).fit(
            static_design(X[rows], n_numeric, Z, month[rows], setup['interactions']), y[rows] - alpha_m.loc[month[rows]].values)
    elif family in ('single_stage_lasso', 'single_stage_ridge'):
        D = single_stage_design(X[rows], n_numeric, Z, month[rows])
        fitted['scaler'] = StandardScaler().fit(D)
        model = Lasso(alpha=setup['lasso_alpha'], precompute=True, max_iter=5000) if family == 'single_stage_lasso' else Ridge(alpha=setup['ridge_alpha'])
        fitted['linear'] = model.fit(fitted['scaler'].transform(D), y[rows])
    else:
        raise ValueError(family)
    return fitted


def predict_model(fitted, X, months):
    """Predicted log rent of flats X as if approved in `months` (an array with one month per row)."""
    Z = fitted['Z']
    if fitted['family'].startswith('two_stage'):
        design_matrix = static_design(X, fitted['n_numeric'], Z, months, fitted['setup']['interactions'])
        return fitted['linear'].predict(Z.loc[months]) + fitted['static_model'].predict(design_matrix)
    return fitted['linear'].predict(fitted['scaler'].transform(single_stage_design(X, fitted['n_numeric'], Z, months)))


def fixed_flat_levels(fitted, X_ref, months):
    """Mean predicted rent of the same flats (X_ref) as if they were approved in each of `months`: a level with the flat mix held fixed."""
    return {m: float(np.exp(predict_model(fitted, X_ref, np.full(len(X_ref), m))).mean()) for m in months}


def score(y_log, pred_log):
    """log R2, dollar R2, log RMSE and dollar RMSE of predictions made on the log-rent scale."""
    rent, pred_rent = np.exp(y_log), np.exp(pred_log)
    resid = y_log - pred_log
    return {
        'log_r2': 1 - np.sum(resid ** 2) / np.sum((y_log - y_log.mean()) ** 2),
        'dollar_r2': 1 - np.sum((rent - pred_rent) ** 2) / np.sum((rent - rent.mean()) ** 2),
        'log_rmse': float(np.sqrt(np.mean(resid ** 2))),
        'dollar_rmse': float(np.sqrt(np.mean((rent - pred_rent) ** 2))),
    }
