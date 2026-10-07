"""Charts for the 03 training pipeline.

* prediction_chart / prediction_charts: monthly mean predicted rent against actual rent, one chart per training window (full history, 2023 onward,
  after May 2023): three panels on ONE shared dollar axis, shaded training period, divider at the first test month, government growth scenarios as dotted lines.
* bar_panel: horizontal bars for coefficients / importances.
Reference palette (light), one fixed colour per model; text stays in neutral ink.
"""
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

SURFACE, INK, INK2, MUTED, GRID, AXIS, BAND = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#f0efec'
HUE = {'blue': '#2a78d6', 'orange': '#eb6834', 'aqua': '#1baf7a', 'magenta': '#e87ba4', 'red': '#e34948', 'violet': '#4a3aa7', 'green': '#008300'}
DOTTED = (0, (0.01, 2.6))
FAMILY_STYLE = {   # family -> (label, colour): one colour per model family
    'two_stage_rf': ('Two-stage, Lasso trend', HUE['orange']),
    'two_stage_rf_ridge': ('Two-stage, Ridge trend', HUE['violet']),
    'single_stage_lasso': ('Single-stage Lasso', HUE['aqua']),
    'single_stage_ridge': ('Single-stage Ridge', HUE['magenta']),
}
WINDOW_STYLE = {'full_history': ('full history', '-'), 'from_2023': ('2023 onward', (0, (5, 2.5))),            # solid, dashed, dash-dot
                'from_2023_06': ('after May 2023', (0, (7, 2, 1.5, 2)))}
WINDOW_START_DATE = {'full_history': None, 'from_2023': pd.Timestamp('2023-01-01'), 'from_2023_06': pd.Timestamp('2023-06-01')}
MODELS = {f'{w}/{f}': (f'{fl}, {wl}', colour, linestyle)
          for f, (fl, colour) in FAMILY_STYLE.items() for w, (wl, linestyle) in WINDOW_STYLE.items()}
SCENARIO_GROWTH = [0.005, 0.01, 0.02, 0.03]                        # government rental growth by mid-2026 relative to the 2025 level
SCENARIO_COLOURS = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab']    # one-hue ramp, darker = faster growth
LINE = dict(lw=1.5, solid_capstyle='round', solid_joinstyle='round', dash_capstyle='round', dash_joinstyle='round')


def _style():
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans']})


def prediction_chart(fit, forecast, actual, window, out_path=None):
    """One chart for the models of one training window.

    fit / forecast: monthly mean predicted rent (PeriodIndex x model key) for the flats the models were fitted on (train + validation) and for
    the test flats; actual: monthly mean actual rent of the fitted flats; window: 'full_history', 'from_2023' or 'from_2023_06'.

    Scenarios: only 2025-01..03 have rents, so their average actual rent is the 2025 level; each scenario grows geometrically from it at 2025-03 to
    level x (1 + g) at 2026-06 (mid-2026).
    """
    _style()
    window_label = WINDOW_STYLE[window][0]
    window_start = WINDOW_START_DATE[window]
    split = forecast.index.min()                                   # first test month
    keys = [k for k in MODELS if k.startswith(window + '/') and k in fit.columns]
    if window_start is not None:                                   # months before the window were not used for fitting: do not draw the models there
        fit = fit.loc[pd.Period(window_start, freq='M'):]
    fit_from = window_start if window_start is not None else pd.Timestamp('2021-01-01')
    panels = [('Two-stage models (static model: random forest)', [k for k in keys if '/two_stage' in k]),
              ('Single-stage models (all features + interactions)', [k for k in keys if '/single_stage' in k])]

    level_2025 = actual.loc['2025-01':'2025-03'].mean()
    s_months = pd.period_range('2025-03', '2026-06', freq='M')
    steps = np.arange(len(s_months))
    scenarios = {g: pd.Series(level_2025 * (1 + g) ** (steps / steps[-1]), index=s_months) for g in SCENARIO_GROWTH}

    everything = np.concatenate([fit[keys].to_numpy().ravel(), forecast[keys].to_numpy().ravel(), actual.to_numpy()]
                                + [s.to_numpy() for s in scenarios.values()])
    ymin = max(1000, 500 * (min(everything.min(), 2000) // 500))
    top = 500 * -(-max(everything.max(), 4000) // 500)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.8), facecolor=SURFACE)
    axes[1].sharey(axes[0]); axes[2].sharey(axes[0])               # one dollar axis for all three panels
    x = lambda s: s.index.to_timestamp()
    full = (pd.Timestamp('2021-01-01'), pd.Timestamp('2026-12-31'))
    zoom = (pd.Timestamp('2024-01-01'), full[1])

    def frame(ax, title, xlim):
        ax.set_facecolor(SURFACE)
        ax.axvspan(max(xlim[0], fit_from), split.to_timestamp(), color=BAND, zorder=0, lw=0)       # shade only the months the models were fitted on
        ax.axvline(split.to_timestamp(), color=AXIS, lw=0.75, zorder=1)
        ax.set_title(title, loc='left', fontsize=12, color=INK, pad=10, fontweight='semibold')
        ax.grid(axis='y', color=GRID, lw=0.75); ax.set_axisbelow(True)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=9, length=0, labelleft=True)
        ax.set_xlim(*xlim); ax.set_ylim(ymin, top)
        ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(500)); ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))
        ytxt = ymin + 0.01 * (top - ymin)
        shaded_from = max(xlim[0], fit_from)
        ax.text(shaded_from + (split.to_timestamp() - shaded_from) / 2, ytxt, 'train + validation (fitting months)', color=MUTED, fontsize=9, va='bottom', ha='center')
        ax.text(split.to_timestamp() + pd.Timedelta(days=20), ytxt, 'test forecast', color=MUTED, fontsize=9, va='bottom')
        if window_start is not None and xlim[0] < window_start:
            ax.axvline(window_start, color=AXIS, lw=0.75, zorder=1)
            ax.text(xlim[0] + (window_start - xlim[0]) / 2, ytxt, 'not used to fit these models', color=MUTED, fontsize=9, va='bottom', ha='center')

    def draw(ax, key, lw=1.5):
        label, colour = FAMILY_STYLE[key.split('/')[1]]
        ax.plot(x(fit), fit[key], color=colour, label=label, zorder=4, **{**LINE, 'lw': lw})
        ax.plot(x(forecast), forecast[key], color=colour, zorder=4, **{**LINE, 'lw': lw})
        ax.plot(x(forecast)[-1:], forecast[key].iloc[-1:], 'o', ms=6, color=colour, mec=SURFACE, mew=1.5, zorder=5)
        return forecast[key].iloc[-1]

    def actual_line(ax):
        if window_start is not None:                               # actual rent before the window is context only: drawn faint
            before = actual.loc[:pd.Period(window_start, freq='M')]
            ax.plot(x(before), before, color=MUTED, lw=1.0, zorder=6, solid_capstyle='round', label='Actual rent before the window (not fitted)')
        fitted_actual = actual if window_start is None else actual.loc[pd.Period(window_start, freq='M'):]
        ax.plot(x(fitted_actual), fitted_actual, color=INK, lw=1.0, label='Actual rent (fitted flats)', zorder=6, solid_capstyle='round')

    for ax, (title, panel_keys) in zip(axes[:2], panels):
        frame(ax, title, full)
        ends = [draw(ax, key) for key in panel_keys]
        actual_line(ax)
        last = None                                                # label an end value only if it clears the one above by 2.5% of the axis range
        for value in sorted(ends, reverse=True):
            if last is None or last - value >= 0.025 * (top - ymin):
                ax.annotate(f'{value:,.0f}', (x(forecast)[-1], value), xytext=(8, 0), textcoords='offset points', va='center', fontsize=9, color=INK2)
                last = value
        ax.legend(loc='upper left', frameon=False, fontsize=9, labelcolor=INK2, handlelength=1.8)

    ax3 = axes[2]
    frame(ax3, 'Models vs government growth scenarios (time axis from 2024)', zoom)
    for key in keys:
        draw(ax3, key, lw=1.3)
    scenario_ends = []
    for g, colour in zip(SCENARIO_GROWTH, SCENARIO_COLOURS):
        s = scenarios[g]
        ax3.plot(x(s), s, color=colour, linestyle=DOTTED, label=f'Government scenario: +{g * 100:g}% by mid-2026', zorder=4, **LINE)
        ax3.plot(x(s)[-1:], s.iloc[-1:], 'o', ms=5, color=colour, mec=SURFACE, mew=1.5, zorder=5)
        scenario_ends.append((s.iloc[-1], colour, g))
    gap, centre = 0.04 * (top - ymin), np.mean([v for v, _, _ in scenario_ends])    # stacked labels with leader lines: the scenarios nearly coincide
    for k, (value, colour, g) in enumerate(sorted(scenario_ends)):
        ax3.annotate(f'+{g * 100:g}%', xy=(s_months[-1].to_timestamp(), value),
                     xytext=(s_months[-1].to_timestamp() + pd.Timedelta(days=75), centre + (k - (len(scenario_ends) - 1) / 2) * gap),
                     va='center', fontsize=9, color=INK2, arrowprops=dict(arrowstyle='-', color=colour, lw=0.9, shrinkA=0, shrinkB=3))
    actual_line(ax3)
    ax3.legend(loc='upper left', frameon=False, fontsize=8.5, labelcolor=INK2, handlelength=2.4, ncol=2, columnspacing=1.2)

    fig.suptitle(f'Training window: {window_label}. Models retrained on train + validation: predictions for all months', x=0.05, ha='left',
                 fontsize=14, color=INK, fontweight='semibold', y=0.995)
    fig.text(0.05, 0.925, 'Monthly mean predicted rent vs. actual mean rent, one dollar axis for all panels. Shaded = months the models were fitted on; '
             'right of the divider is a forecast for the test flats.\nThe models are fitted only on the shaded months (the window start to 2025-03). Dotted blue lines are government growth scenarios (+g% by mid-2026 from the 2025 level). '
             'Each chart shows the models of one training window; the dollar axis is scaled to that chart.', fontsize=9, color=INK2, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    if out_path:
        fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    return fig


def prediction_charts(fit, forecast, actual, out_dir):
    """One chart per training window, saved as <out_dir>/prediction_chart_<window>.png. Returns {window: figure}."""
    return {window: prediction_chart(fit, forecast, actual, window, out_path=f'{out_dir}/prediction_chart_{window}.png') for window in WINDOW_STYLE}


def bar_panel(ax, values, title, colour=HUE['orange'], signed=False, top=20, fmt='{:.3f}'):
    """Horizontal bars of the `top` largest |values|, largest at the top. Signed values use the diverging blue / red pair."""
    _style()
    s = values.reindex(values.abs().sort_values(ascending=False).index).head(top)[::-1]
    colours = [HUE['blue'] if v >= 0 else HUE['red'] for v in s.values] if signed else colour
    ax.barh(range(len(s)), s.values, color=colours, height=0.62)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(list(s.index), fontsize=8.5, color=INK2)
    span = float(np.abs(s.values).max()) if len(s) else 1.0
    for i, v in enumerate(s.values):
        ax.annotate(fmt.format(v), (v, i), xytext=(4 if v >= 0 else -4, 0), textcoords='offset points', ha='left' if v >= 0 else 'right',
                    va='center', fontsize=8, color=INK2)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc='left', fontsize=11, color=INK, pad=8, fontweight='semibold')
    ax.grid(axis='x', color=GRID, lw=0.75); ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(AXIS)
    if signed:
        ax.axvline(0, color=AXIS, lw=1.0)
    ax.tick_params(axis='x', colors=MUTED, labelsize=8.5, length=0)
    ax.tick_params(axis='y', length=0)
    ax.set_xlim((-1.3 if signed else 0) * span, 1.3 * span)
