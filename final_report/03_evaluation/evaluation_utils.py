"""Helpers of `evaluation.ipynb`: reading the predictions of any model, the metrics, the growth benchmark and the charts.

A model takes part in the evaluation by saving, under `predictions/`:

* `train/<model>.csv`, `val/<model>.csv`, `test/<model>.csv`: ROW_ID, the identifier columns of the static features (`ID_COL`) and `prediction` (the monthly rent in dollars).
  ROW_ID is the position of the row in `data/original_dataset/train.csv` / `test.csv`, which is where the actual rents are read from. `val` holds the held-out rows
  (months 2024-04..2025-03) predicted by a model fitted on the months before; `train` the train + validation rows predicted by the model fitted on all of them (in-sample);
  `test` the 50,000 test rows.
* optionally `levels/<model>.csv`: RENT_APPROVAL_DATE ('2025-01') and `level` for the months 2021-01..2026-07, the rent of a fixed set of flats (the train flats of 2025-01..03) as
  predicted for each month. It separates the growth of the market from the change in the flat mix, and feeds the growth benchmark.
"""
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent                                     # final_report
PRED_DIR, CHART_DIR, TABLE_DIR = EVAL_DIR / 'predictions', EVAL_DIR / 'charts', EVAL_DIR / 'tables'
ORIGINAL_PATH = ROOT / 'data' / 'original_dataset'
LABEL = 'MONTHLY_RENT'
Q1_2025, MID_2026, MONTHS_2026 = [48, 49, 50], 65, list(range(60, 67))      # month indices from 2021-01 = 0: 2025-01..03, 2026-06, 2026-01..07
GROWTH_BAND = (0.005, 0.03)                                                  # government rental growth by mid-2026 relative to the 2025 level


def month_index(year, month):
    return (np.asarray(year) - 2021) * 12 + np.asarray(month) - 1


def month_label(m):
    """'2025-03' for month index 50 (an array in, an array out)."""
    return pd.Series(np.atleast_1d(m)).map(lambda k: f'{2021 + k // 12}-{k % 12 + 1:02d}').to_numpy()


# ------------------------------------------------------------------------------------------------------------------------------------ data
def list_models():
    """The models with a test prediction file."""
    return sorted(p.stem for p in (PRED_DIR / 'test').glob('*.csv'))


def load_actuals():
    """Rent, month and flat type of the rows of `data/original_dataset/train.csv`, indexed by ROW_ID."""
    df = pd.read_csv(ORIGINAL_PATH / 'train.csv', usecols=['RENT_APPROVAL_DATE', 'FLAT_TYPE', LABEL])
    df.index.name = 'ROW_ID'
    df['month'] = month_index(df['RENT_APPROVAL_DATE'].str[:4].astype(int), df['RENT_APPROVAL_DATE'].str[5:7].astype(int))
    df['FLAT_TYPE'] = df['FLAT_TYPE'].str.lower().str.replace('-', ' ')
    return df


def load_predictions(model, split):
    """The prediction file of a model and split, with the month of each row added."""
    df = pd.read_csv(PRED_DIR / split / f'{model}.csv', low_memory=False)
    df['month'] = month_index(df['RENT_APPROVAL_YEAR'], df['RENT_APPROVAL_MONTH'])
    return df


def load_levels(model):
    """The fixed-flat level of a model by month (a Series indexed by month 0..66), or None when the model has no levels file."""
    path = PRED_DIR / 'levels' / f'{model}.csv'
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return pd.Series(df['level'].to_numpy(), index=month_index(df['RENT_APPROVAL_DATE'].str[:4].astype(int), df['RENT_APPROVAL_DATE'].str[5:7].astype(int)))


def monthly_mean(values, months):
    """Mean of `values` by month, as a Series with a monthly PeriodIndex."""
    s = pd.Series(np.asarray(values, dtype=np.float64)).groupby(np.asarray(months)).mean()
    return pd.Series(s.to_numpy(), index=pd.PeriodIndex(month_label(s.index.to_numpy()), freq='M'))


# ---------------------------------------------------------------------------------------------------------------------------------- metrics
def regression_metrics(actual, predicted):
    """MAE, RMSE, R2 and bias (mean prediction - mean actual) in dollars."""
    actual, predicted = np.asarray(actual, dtype=np.float64), np.asarray(predicted, dtype=np.float64)
    return {'MAE': float(np.mean(np.abs(actual - predicted))), 'RMSE': float(np.sqrt(np.mean((actual - predicted) ** 2))),
            'R2': float(1 - np.sum((actual - predicted) ** 2) / np.sum((actual - actual.mean()) ** 2)), 'BIAS': float(np.mean(predicted - actual))}


def growth_benchmark(levels):
    """Growth of a rent level (a Series by month 0..66) against the government band: mid-2026 (2026-06) and 2026 average over the 2025-01..03 level.
    Distance to the band is 0 for a growth inside it."""
    lo, hi = GROWTH_BAND
    base = levels.loc[Q1_2025].mean()
    g_mid, g_avg = levels.loc[MID_2026] / base - 1, levels.loc[MONTHS_2026].mean() / base - 1
    return {'level 2025-01..03': base, 'level 2026-06': levels.loc[MID_2026], 'growth at mid-2026': g_mid, 'avg 2026 growth': g_avg,
            'distance to the 0.5-3% band': max(0.0, lo - g_mid, g_mid - hi)}


# ------------------------------------------------------------------------------------------------------------------------------------ charts
SURFACE, INK, INK2, MUTED, GRID, AXIS, BAND = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#f0efec'
HUE = {'blue': '#2a78d6', 'orange': '#eb6834', 'aqua': '#1baf7a', 'magenta': '#e87ba4', 'red': '#e34948', 'violet': '#4a3aa7', 'green': '#008300'}
HUE_CYCLE = [HUE[k] for k in ('orange', 'aqua', 'violet', 'magenta', 'red', 'green')]       # colours handed out to models without a fixed one
DOTTED = (0, (0.01, 2.6))
SCENARIO_GROWTH = [0.005, 0.01, 0.02, 0.03]                        # government rental growth by mid-2026 relative to the 2025 level
SCENARIO_COLOURS = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab']    # one-hue ramp, darker = faster growth
LINE = dict(lw=1.5, solid_capstyle='round', solid_joinstyle='round', dash_capstyle='round', dash_joinstyle='round')
TRAIN_LABEL = 'train + validation (fitting months)'


def _style():
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans']})


def _axes_style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis='y', color=GRID, lw=0.75); ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0, labelleft=True)


def models_chart(models, actual, title, out_path=None):
    """models: list of dicts {'label', 'fit', 'forecast', 'colour'}: monthly mean predicted rent (PeriodIndex) for the train rows and the test rows.
    actual: monthly mean actual rent of the train rows.

    Scenarios: only 2025-01..03 have rents, so their average actual rent is the 2025 level; each scenario grows geometrically from it at 2025-03 to
    level x (1 + g) at 2026-06 (mid-2026).
    """
    _style()
    split = min(m['forecast'].index.min() for m in models)
    level_2025 = actual.loc['2025-01':'2025-03'].mean()
    s_months = pd.period_range('2025-03', '2026-06', freq='M')
    steps = np.arange(len(s_months))
    scenarios = {g: pd.Series(level_2025 * (1 + g) ** (steps / steps[-1]), index=s_months) for g in SCENARIO_GROWTH}

    everything = np.concatenate([actual.to_numpy()] + [m[k].to_numpy() for m in models for k in ('fit', 'forecast')] + [s.to_numpy() for s in scenarios.values()])
    ymin = 500 * (min(everything.min(), 2000) // 500)
    top = 500 * -(-max(everything.max(), 4000) // 500)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), facecolor=SURFACE)
    axes[1].sharey(axes[0])
    x = lambda s: s.index.to_timestamp()
    full = (pd.Timestamp('2021-01-01'), pd.Timestamp('2026-12-31'))
    zoom = (pd.Timestamp('2024-01-01'), full[1])

    def frame(ax, subtitle, xlim):
        _axes_style(ax)
        ax.axvspan(max(xlim[0], full[0]), split.to_timestamp(), color=BAND, zorder=0, lw=0)
        ax.axvline(split.to_timestamp(), color=AXIS, lw=0.75, zorder=1)
        ax.set_title(subtitle, loc='left', fontsize=12, color=INK, pad=10, fontweight='semibold')
        ax.set_xlim(*xlim); ax.set_ylim(ymin, top)
        ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(500)); ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))
        ytxt = ymin + 0.01 * (top - ymin)
        shaded_from = max(xlim[0], full[0])
        ax.text(shaded_from + (split.to_timestamp() - shaded_from) / 2, ytxt, TRAIN_LABEL, color=MUTED, fontsize=9, va='bottom', ha='center')
        ax.text(split.to_timestamp() + pd.Timedelta(days=20), ytxt, 'test forecast', color=MUTED, fontsize=9, va='bottom')

    def draw(ax, end_in_legend):
        for m in models:
            end = m['forecast'].iloc[-1]
            label = f"{m['label']} (2026-07: ${end:,.0f})" if end_in_legend else m['label']
            ax.plot(x(m['fit']), m['fit'], color=m['colour'], label=label, zorder=4, **LINE)
            ax.plot(x(m['forecast']), m['forecast'], color=m['colour'], zorder=4, **LINE)
            ax.plot(x(m['forecast'])[-1:], m['forecast'].iloc[-1:], 'o', ms=6, color=m['colour'], mec=SURFACE, mew=1.5, zorder=5)
        ax.plot(x(actual), actual, color=INK, lw=1.0, label='Actual rent (train flats)', zorder=6, solid_capstyle='round')

    frame(axes[0], 'Whole period', full)
    draw(axes[0], end_in_legend=False)
    last = None                                                # label an end value only if it clears the one above by 2.5% of the axis range
    for end, m in sorted(((m['forecast'].iloc[-1], m) for m in models), key=lambda t: -t[0]):
        if last is None or last - end >= 0.025 * (top - ymin):
            axes[0].annotate(f'{end:,.0f}', (x(m['forecast'])[-1], end), xytext=(8, 0), textcoords='offset points', va='center', fontsize=9, color=INK2)
            last = end
    axes[0].legend(loc='upper left', frameon=False, fontsize=9, labelcolor=INK2, handlelength=1.8)

    frame(axes[1], 'Models vs government growth scenarios (time axis from 2024)', zoom)
    draw(axes[1], end_in_legend=True)
    ends = []
    for g, c in zip(SCENARIO_GROWTH, SCENARIO_COLOURS):
        s = scenarios[g]
        axes[1].plot(x(s), s, color=c, linestyle=DOTTED, label=f'Government scenario: +{g * 100:g}% by mid-2026', zorder=4, **LINE)
        axes[1].plot(x(s)[-1:], s.iloc[-1:], 'o', ms=5, color=c, mec=SURFACE, mew=1.5, zorder=5)
        ends.append((s.iloc[-1], c, g))
    gap, centre = 0.04 * (top - ymin), np.mean([v for v, _, _ in ends])          # stacked labels with leader lines: the scenarios nearly coincide
    for k, (value, c, g) in enumerate(sorted(ends)):
        axes[1].annotate(f'+{g * 100:g}%', xy=(s_months[-1].to_timestamp(), value),
                         xytext=(s_months[-1].to_timestamp() + pd.Timedelta(days=75), centre + (k - (len(ends) - 1) / 2) * gap),
                         va='center', fontsize=9, color=INK2, arrowprops=dict(arrowstyle='-', color=c, lw=0.9, shrinkA=0, shrinkB=3))
    axes[1].legend(loc='upper left', frameon=False, fontsize=8.5, labelcolor=INK2, handlelength=2.4, columnspacing=1.2)

    fig.suptitle(title, x=0.05, ha='left', fontsize=14, color=INK, fontweight='semibold', y=0.995)
    fig.text(0.05, 0.925, 'Monthly mean predicted rent vs. actual mean rent, one dollar axis for both panels. Shaded = months the models were fitted on (2021-01..2025-03); right of the '
             'divider is a forecast for the test months.\nDotted blue lines are government growth scenarios (+g% by mid-2026 from the 2025 level, the average actual rent of 2025-01..03).',
             fontsize=9, color=INK2, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    if out_path:
        fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    return fig


def validation_chart(models, actual, first_validation, title, out_path=None):
    """Held-out run. models: list of dicts {'label', 'monthly', 'colour'}: monthly mean predicted rent (PeriodIndex) of the held-out predictions;
    actual: monthly mean actual rent, from a few months before `first_validation` (a Period, shaded as fitted months) to the last validation month."""
    _style()
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=SURFACE)
    _axes_style(ax)
    split = first_validation.to_timestamp()
    ax.axvspan(actual.index.min().to_timestamp(), split, color=BAND, zorder=0, lw=0)
    ax.axvline(split, color=AXIS, lw=0.75, zorder=1)
    for m in models:
        ax.plot(m['monthly'].index.to_timestamp(), m['monthly'], color=m['colour'], label=m['label'], zorder=4, **LINE)
    ax.plot(actual.index.to_timestamp(), actual, color=INK, lw=1.0, label='Actual rent', zorder=6, solid_capstyle='round')
    ymin = 250 * (min(actual.min(), min(m['monthly'].min() for m in models)) // 250)
    top = 250 * -(-max(actual.max(), max(m['monthly'].max() for m in models)) // 250) + 250
    ax.set_ylim(ymin, top)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(250)); ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3)); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ytxt = ymin + 0.01 * (top - ymin)
    ax.text(split - pd.Timedelta(days=10), ytxt, 'fitted months', color=MUTED, fontsize=9, va='bottom', ha='right')
    ax.text(split + pd.Timedelta(days=10), ytxt, 'validation (held out)', color=MUTED, fontsize=9, va='bottom')
    ax.legend(loc='upper left', frameon=False, fontsize=9, labelcolor=INK2, handlelength=1.8)
    ax.set_title(title, loc='left', fontsize=12, color=INK, pad=10, fontweight='semibold')
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    return fig


def loss_chart(curves, title, floor=None, out_path=None):
    """curves: list of dicts {'label', 'train', 'valid' (optional), 'colour'}: dollar RMSE by epoch (Series, index = epoch). floor: dotted reference line (a dollar RMSE, label)."""
    _style()
    fig, ax = plt.subplots(figsize=(8, 3.6), facecolor=SURFACE)
    _axes_style(ax)
    for c in curves:
        ax.plot(c['train'].index, c['train'], color=c['colour'], label=f"{c['label']}: train", **LINE)
        if 'valid' in c:
            ax.plot(c['valid'].index, c['valid'], color=c['colour'], linestyle=(0, (5, 2.5)), label=f"{c['label']}: held-out validation", **LINE)
    if floor is not None:
        ax.axhline(floor[0], color=MUTED, lw=0.8, ls=':'); ax.annotate(floor[1], (ax.get_xlim()[1], floor[0]), xytext=(-4, 4), textcoords='offset points', ha='right', fontsize=8.5, color=MUTED)
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))
    ax.set_xlabel('epoch', color=MUTED, fontsize=9); ax.set_ylabel('RMSE ($)', color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2)
    ax.set_title(title, loc='left', fontsize=11, color=INK, pad=8, fontweight='semibold')
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    return fig
