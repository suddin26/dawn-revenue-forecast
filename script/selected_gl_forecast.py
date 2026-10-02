# %% [markdown]
# ## Reproducible forecasts for the selected GL codes
# Run these cells from top to bottom. Change assumptions only in **Settings**. Tables, exports, and each GL chart have separate cells, so changing a plot does not refit the models.
# 
# | Method | How it forecasts |
# |---|---|
# | Linear | Straight-line trend fitted to recent annual revenue |
# | Holt | Exponential smoothing with a damped trend |
# | ML (Random forest) | Learns from two prior revenue values and time; forecasts recursively |
# | Monte Carlo | Simulates annual dollar changes sampled from recent history |
# | Naive baseline | Repeats the last annual revenue |
# 
# Use the same input file, settings, Python version, and package versions to reproduce a run. The export cell records the input SHA-256 hash, model settings, and package versions. Fixed seeds control both stochastic methods.

# %% [markdown]
# ### 1. Settings and imports

# %%
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from IPython.display import display, Markdown
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.holtwinters import Holt
import plotly.graph_objects as go

LAST_COMPLETE_FY = 2026
FORECAST_HORIZON = 3
LOOKBACK = 10  # Most recent available years, applied separately within each backtest.
N_SIMULATIONS = 10000
SEED = 2026
MIN_TRAIN_YEARS = 4
BACKTEST_ORIGINS = 3
RF_TREES = 100
RF_MAX_DEPTH = 3
MC_CHANGE_WINDOW = 5
MODELS = ['Linear', 'Holt', 'ML (Random forest)', 'Monte Carlo', 'Naive baseline']
selected_gl_codes = [3113, 3052, 3130, 3066, 3068, 3050, 3073, 3053, 3152,
                     3105, 3129, 3101, 3004, 3005, 3114, 3002, 3047, 3044,
                     3043, 3046, 3003, 3042, 3276, 3075, 4621, 3290]

# Find the project from either its root or the notebook directory.
project_root = next(p for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents]
                    if (p / 'data' / 'dawn_revenue_accounts_combined_history.csv').is_file())
input_csv = project_root / 'data' / 'dawn_revenue_accounts_combined_history.csv'
result_dir = project_root / 'result'
plot_dir = result_dir / 'forecast_plots'


# %% [markdown]
# ### 2. Prepare annual revenue and check the available history
# Exclude MAJOR revenue and GL codes 42, 4860, and 4870. Missing years are not converted to zero: each model uses the latest uninterrupted history.

# %%
# Read the source afresh so forecasts do not depend on changes in earlier cells.
forecast_source = pd.read_csv(input_csv)
is_major_forecast = forecast_source['Type'].fillna('').str.strip().str.upper().eq('MAJOR')
forecast_history = (
    forecast_source.loc[
        ~is_major_forecast
        & forecast_source['fy'].le(LAST_COMPLETE_FY)
        & ~forecast_source['GL Number'].isin([42, 4860, 4870])
        & forecast_source['GL Number'].isin(selected_gl_codes)
    ]
    .groupby(['GL Number', 'fy'], as_index=False)
    .agg(revenue=('Amount', 'sum'), description=('description', 'first'))
    .sort_values(['GL Number', 'fy'])
)
last_fy = LAST_COMPLETE_FY
future_fys = np.arange(last_fy + 1, last_fy + 1 + FORECAST_HORIZON)
histories = {}
history_notes = []
for gl in selected_gl_codes:
    history = forecast_history.loc[forecast_history['GL Number'].eq(gl)].copy()
    if history.empty or history['fy'].iloc[-1] != last_fy:
        raise ValueError(f'GL {gl} needs observations through FY {last_fy}.')
    breaks = np.flatnonzero(np.diff(history['fy'].to_numpy()) != 1)
    if len(breaks):
        history = history.iloc[breaks[-1] + 1:]
    values = history['revenue'].to_numpy(dtype=float)
    if len(values) < MIN_TRAIN_YEARS + 1 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f'GL {gl}: need at least {MIN_TRAIN_YEARS + 1} contiguous, finite, nonnegative annual totals.')
    histories[gl] = history
    origins = list(range(MIN_TRAIN_YEARS, len(values) - FORECAST_HORIZON + 1))[-BACKTEST_ORIGINS:]
    history_notes.append({
        'GL Number': gl, 'History start FY': int(history['fy'].iloc[0]),
        'History end FY': last_fy, 'Contiguous years': len(values),
        'Final training years': min(len(values), LOOKBACK),
        'Backtest origins': len(origins) if origins else 1,
        'Validation scope': 'Three-year rolling backtest' if origins else 'LIMITED: one-year holdout only',
    })
data_quality_table = pd.DataFrame(history_notes)
display(data_quality_table)


# %% [markdown]
# ### 3. Model functions
# Each function receives only past observations. The common wrapper scales large dollar values for numerical stability and floors forecast revenues at zero.

# %%
def forecast_linear(z, horizon):
    time = np.arange(len(z)).reshape(-1, 1)
    model = LinearRegression().fit(time, z)
    future_time = np.arange(len(z), len(z) + horizon).reshape(-1, 1)
    return model.predict(future_time)


def forecast_holt(z, horizon, context):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model = Holt(z, damped_trend=True, initialization_method='estimated').fit(optimized=True)
    for warning in caught:
        fit_diagnostics.append({'Context': context, 'Message': str(warning.message)})
    return np.asarray(model.forecast(horizon))


def forecast_random_forest(z, horizon, seed):
    # Predict this year from last year's revenue, two years ago, and time.
    features = np.array([[z[i-1], z[i-2], i] for i in range(2, len(z))])
    model = RandomForestRegressor(n_estimators=RF_TREES, max_depth=RF_MAX_DEPTH,
                                   min_samples_leaf=1, random_state=seed, n_jobs=1)
    model.fit(features, z[2:])
    extended = z.tolist()
    for _ in range(horizon):
        next_value = model.predict([[extended[-1], extended[-2], len(extended)]])[0]
        extended.append(float(next_value))
    return np.asarray(extended[-horizon:])


def forecast_monte_carlo(y, horizon, seed):
    # Sample recent annual dollar changes with replacement.
    rng = np.random.default_rng(seed)
    changes = np.diff(y)[-MC_CHANGE_WINDOW:]
    draws = rng.choice(changes, size=(N_SIMULATIONS, horizon), replace=True)
    paths = np.empty_like(draws)
    previous = np.full(N_SIMULATIONS, y[-1])
    for h in range(horizon):
        previous = np.maximum(previous + draws[:, h], 0)
        paths[:, h] = previous
    return paths.mean(axis=0), np.quantile(paths, 0.1, axis=0), np.quantile(paths, 0.9, axis=0)


def fit_forecasts(values, horizon, seed, context):
    y = np.asarray(values, dtype=float)[-LOOKBACK:]
    scale = max(float(np.max(np.abs(y))), 1.0)
    z = y / scale
    mc_mean, mc_p10, mc_p90 = forecast_monte_carlo(y, horizon, seed)
    predictions = {
        'Linear': np.maximum(forecast_linear(z, horizon) * scale, 0),
        'Holt': np.maximum(forecast_holt(z, horizon, context) * scale, 0),
        'ML (Random forest)': np.maximum(forecast_random_forest(z, horizon, seed) * scale, 0),
        'Monte Carlo': mc_mean,
        'Naive baseline': np.repeat(y[-1], horizon),
    }
    if not all(np.isfinite(values).all() for values in predictions.values()):
        raise ValueError(f'Non-finite forecast: {context}')
    return predictions, mc_p10, mc_p90


# %% [markdown]
# ### 4. Run historical backtests
# Use up to three recent training cutoffs. Forecast the next three years without seeing their actual revenue. Overlapping targets count as separate forecast cases. GL 3075 only supports a one-year holdout; its three-year accuracy is unvalidated.

# %%
# Reset outputs so rerunning this cell does not accumulate old results.
fit_diagnostics = []
backtest_rows = []
for gl, history in histories.items():
    years = history['fy'].to_numpy(dtype=int)
    values = history['revenue'].to_numpy(dtype=float)
    origins = list(range(MIN_TRAIN_YEARS, len(values) - FORECAST_HORIZON + 1))[-BACKTEST_ORIGINS:]
    if not origins:
        origins = [len(values) - 1]
    for cutoff in origins:
        origin_fy = int(years[cutoff - 1])
        horizon = min(FORECAST_HORIZON, len(values) - cutoff)
        predictions, low, high = fit_forecasts(
            values[:cutoff], horizon, SEED + gl + origin_fy,
            f'GL {gl}, origin FY {origin_fy}',
        )
        for model, predicted in predictions.items():
            for h in range(horizon):
                backtest_rows.append({
                    'GL Number': gl, 'Model': model, 'Origin FY': origin_fy,
                    'Target FY': int(years[cutoff + h]), 'Horizon': h + 1,
                    'Actual': values[cutoff + h], 'Predicted': predicted[h],
                    'MC P10': low[h] if model == 'Monte Carlo' else np.nan,
                    'MC P90': high[h] if model == 'Monte Carlo' else np.nan,
                })
backtest_predictions = pd.DataFrame(backtest_rows)
print(f'Backtesting complete: {len(backtest_predictions):,} forecast cases.')


# %% [markdown]
# ### 5. Validation tables
# MAE is the average absolute dollar error; RMSE penalizes large errors; WAPE is absolute error divided by total actual revenue. Lower is better. Positive skill beats the naive baseline. The lowest-MAE label is descriptive and is not independently validated. Pooled dollar errors emphasize larger GLs.

# %%
def validation_metrics(group):
    actual = group['Actual'].to_numpy()
    error = group['Predicted'].to_numpy() - actual
    denom = np.abs(actual).sum()
    return {
        'Forecast cases': len(group), 'MAE ($)': np.abs(error).mean(),
        'RMSE ($)': np.sqrt(np.mean(error ** 2)),
        'WAPE (%)': 100 * np.abs(error).sum() / denom if denom else np.nan,
        'Bias ($)': error.mean(),
        'MC 80% coverage (%)': (
            100 * ((actual >= group['MC P10']) & (actual <= group['MC P90'])).mean()
            if group['Model'].iloc[0] == 'Monte Carlo' else np.nan
        ),
    }


validation_table = pd.DataFrame([
    {'GL Number': code, 'Model': model, **validation_metrics(group)}
    for (code, model), group in backtest_predictions.groupby(['GL Number', 'Model'], sort=False)
])
validation_by_horizon = pd.DataFrame([
    {'GL Number': code, 'Model': model, 'Horizon': horizon, **validation_metrics(group)}
    for (code, model, horizon), group in backtest_predictions.groupby(['GL Number', 'Model', 'Horizon'], sort=False)
])
baseline_mae = validation_table.loc[validation_table['Model'].eq('Naive baseline')].set_index('GL Number')['MAE ($)']
validation_table['MAE skill vs naive (%)'] = 100 * (
    1 - validation_table['MAE ($)'] / validation_table['GL Number'].map(baseline_mae).replace(0, np.nan)
)
validation_table = validation_table.merge(
    data_quality_table[['GL Number', 'Validation scope']], on='GL Number', how='left'
)
# Descriptive winner on these backtests; not a separately validated model-selection rule.
best = validation_table.loc[validation_table.groupby('GL Number')['MAE ($)'].idxmin(), ['GL Number', 'Model']]
best = best.set_index('GL Number')['Model']
model_comparison = pd.DataFrame([
    {'Model': model, **validation_metrics(group)}
    for model, group in backtest_predictions.groupby('Model', sort=False)
])
model_comparison['GLs with lowest MAE'] = model_comparison['Model'].map(best.value_counts()).fillna(0).astype(int)


display(validation_table.style.format(precision=2, na_rep='?'))


# %% [markdown]
# #### Overall model comparison

# %%
display(model_comparison.style.format(precision=2, na_rep='?'))


# %% [markdown]
# #### Validation by forecast horizon
# Select a GL to compare its one-, two-, and three-year errors. The export contains all GL codes.

# %%
validation_gl = 3113
display(validation_by_horizon.loc[validation_by_horizon['GL Number'].eq(validation_gl)]
        .style.format(precision=2, na_rep='?'))


# %% [markdown]
# ### 6. Fit final models and calculate the next three fiscal years

# %%
forecast_rows = []
# Preserve backtest diagnostics, but remove final-fit diagnostics from a previous rerun.
fit_diagnostics = [item for item in fit_diagnostics if ', final FY ' not in item['Context']]
for gl, history in histories.items():
    values = history['revenue'].to_numpy(dtype=float)
    predictions, low, high = fit_forecasts(values, FORECAST_HORIZON, SEED + gl,
                                          f'GL {gl}, final FY {last_fy}')
    for h, year in enumerate(future_fys):
        forecast_rows.append({
            'GL Number': gl, 'Description': history['description'].iloc[-1],
            'FY': int(year), 'Last actual': values[-1],
            **{model: predicted[h] for model, predicted in predictions.items()},
            'MC P10': low[h], 'MC P90': high[h],
        })
forecast_table = pd.DataFrame(forecast_rows)
forecast_table['Best backtest model'] = forecast_table['GL Number'].map(best)
forecast_table['Best-model forecast'] = [row[row['Best backtest model']] for _, row in forecast_table.iterrows()]
forecast_table = forecast_table.merge(data_quality_table[['GL Number', 'Validation scope']], on='GL Number', how='left')
print(f'Calculated {len(forecast_table)} GL/year forecasts for FY {future_fys[0]}?{future_fys[-1]}.')


# %% [markdown]
# ### 7. Forecast table
# Dollar amounts are shown for every method. MC P10?P90 is the middle 80% of simulated outcomes under the Monte Carlo assumptions, not a guaranteed interval for actual revenue.

# %%
money_columns = ['Last actual'] + MODELS + ['MC P10', 'MC P90', 'Best-model forecast']
display(forecast_table.style.format({column: '${:,.2f}' for column in money_columns}))


# %% [markdown]
# ### 8. Export tables and record the inputs needed to reproduce this run

# %%
import hashlib
import importlib.metadata
import json
import platform

result_dir.mkdir(parents=True, exist_ok=True)
plot_dir.mkdir(parents=True, exist_ok=True)
exported_files = {}
for filename, table in {
    'selected_gl_forecasts.csv': forecast_table,
    'selected_gl_validation.csv': validation_table,
    'selected_gl_validation_by_horizon.csv': validation_by_horizon,
    'selected_gl_backtest_predictions.csv': backtest_predictions,
    'selected_gl_model_comparison.csv': model_comparison,
    'selected_gl_data_quality.csv': data_quality_table,
    'selected_gl_fit_diagnostics.csv': pd.DataFrame(fit_diagnostics, columns=['Context', 'Message']),
}.items():
    destination = result_dir / filename
    try:
        table.to_csv(destination, index=False)
    except PermissionError:
        # A CSV open in Excel may be locked. Keep it and save a new copy.
        destination = destination.with_name(destination.stem + '_updated.csv')
        table.to_csv(destination, index=False)
        print(f'Original CSV is locked; saved {destination.name} instead.')
    exported_files[filename] = destination.name
packages = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'statsmodels', 'plotly', 'ipython', 'jinja2']
versions = {package: importlib.metadata.version(package) for package in packages}
run_settings = {
    'input_file': str(input_csv.relative_to(project_root)),
    'input_sha256': hashlib.sha256(input_csv.read_bytes()).hexdigest(),
    'last_complete_fy': LAST_COMPLETE_FY, 'forecast_horizon': FORECAST_HORIZON,
    'lookback_years': LOOKBACK, 'seed': SEED, 'n_simulations': N_SIMULATIONS,
    'min_train_years': MIN_TRAIN_YEARS, 'backtest_origins': BACKTEST_ORIGINS,
    'rf_trees': RF_TREES, 'rf_max_depth': RF_MAX_DEPTH, 'rf_min_samples_leaf': 1,
    'rf_n_jobs': 1, 'mc_change_window': MC_CHANGE_WINDOW,
    'holt_damped_trend': True, 'holt_initialization': 'estimated',
    'forecast_floor': 0, 'selected_gl_codes': selected_gl_codes,
    'excluded_gl_codes': [42, 4860, 4870], 'excluded_type': 'MAJOR',
    'python_version': platform.python_version(), 'package_versions': versions,
    'exported_files': exported_files,
}
(result_dir / 'selected_gl_run_settings.json').write_text(json.dumps(run_settings, indent=2), encoding='utf-8')
(result_dir / 'forecast_requirements.txt').write_text(
    '\n'.join(f'{package}=={version}' for package, version in versions.items()) + '\n', encoding='utf-8')
if fit_diagnostics:
    display(pd.DataFrame(fit_diagnostics))
print(f'Tables and reproducibility records saved in {result_dir}')


# %% [markdown]
# ### 9. Interactive chart function
# Each chart shows one GL, all forecast methods, and its Monte Carlo range. Click legend entries to hide methods. The dotted vertical line separates history from forecasts. Each plot cell saves an individual HTML file under `result/forecast_plots`.

# %%
MODEL_COLORS = {'Linear': '#0072B2', 'Holt': '#D55E00', 'ML (Random forest)': '#009E73',
                'Monte Carlo': '#CC79A7', 'Naive baseline': '#888888'}


def plot_gl_forecast(gl):
    history = histories[gl]
    forecast = forecast_table.loc[forecast_table['GL Number'].eq(gl)]
    description = history['description'].iloc[-1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history['fy'], y=history['revenue'], name='Actual',
                            mode='lines+markers', line=dict(color='#222222')))
    fig.add_trace(go.Scatter(x=forecast['FY'], y=forecast['MC P10'], mode='lines',
                            line=dict(width=0), legendgroup='MC range', showlegend=False,
                            hoverinfo='skip'))
    fig.add_trace(go.Scatter(
        x=forecast['FY'], y=forecast['MC P90'], mode='lines', line=dict(width=0),
        fill='tonexty', fillcolor='rgba(204,121,167,0.17)', name='MC P10?P90',
        legendgroup='MC range', customdata=forecast[['MC P10']],
        hovertemplate='FY %{x}<br>MC range: $%{customdata[0]:,.0f}?$%{y:,.0f}<extra></extra>',
    ))
    for model in MODELS:
        fig.add_trace(go.Scatter(
            x=[last_fy, *forecast['FY']], y=[history['revenue'].iloc[-1], *forecast[model]],
            name=model, mode='lines+markers', line=dict(color=MODEL_COLORS[model], dash='dash'),
            hovertemplate=f'FY %{{x}}<br>{model}: $%{{y:,.0f}}<extra></extra>',
        ))
    fig.add_vline(x=last_fy + 0.5, line_dash='dot', line_color='gray')
    fig.update_layout(
        title=f'GL {gl}: {description}<br><sup>Forecast FY {future_fys[0]}?{future_fys[-1]}</sup>',
        template='plotly_white', height=550, hovermode='x unified',
        xaxis=dict(title='Fiscal year (FY)', tickformat='d', dtick=2),
        yaxis=dict(title='Revenue ($)', tickprefix='$', tickformat='~s'),
        legend=dict(orientation='h', y=-0.2, groupclick='togglegroup'),
        margin=dict(t=100, b=120),
    )
    plot_dir.mkdir(parents=True, exist_ok=True)
    # Share one local Plotly library across HTML charts; no internet is needed.
    fig.write_html(plot_dir / f'gl_{gl}_forecast.html', include_plotlyjs='directory')
    return fig


# %% [markdown]
# ### GL 3113 forecast

# %%
fig_gl_3113 = plot_gl_forecast(3113)
fig_gl_3113.show()


# %% [markdown]
# ### GL 3052 forecast

# %%
fig_gl_3052 = plot_gl_forecast(3052)
fig_gl_3052.show()


# %% [markdown]
# ### GL 3130 forecast

# %%
fig_gl_3130 = plot_gl_forecast(3130)
fig_gl_3130.show()


# %% [markdown]
# ### GL 3066 forecast

# %%
fig_gl_3066 = plot_gl_forecast(3066)
fig_gl_3066.show()


# %% [markdown]
# ### GL 3068 forecast

# %%
fig_gl_3068 = plot_gl_forecast(3068)
fig_gl_3068.show()


# %% [markdown]
# ### GL 3050 forecast

# %%
fig_gl_3050 = plot_gl_forecast(3050)
fig_gl_3050.show()


# %% [markdown]
# ### GL 3073 forecast

# %%
fig_gl_3073 = plot_gl_forecast(3073)
fig_gl_3073.show()


# %% [markdown]
# ### GL 3053 forecast

# %%
fig_gl_3053 = plot_gl_forecast(3053)
fig_gl_3053.show()


# %% [markdown]
# ### GL 3152 forecast

# %%
fig_gl_3152 = plot_gl_forecast(3152)
fig_gl_3152.show()


# %% [markdown]
# ### GL 3105 forecast

# %%
fig_gl_3105 = plot_gl_forecast(3105)
fig_gl_3105.show()


# %% [markdown]
# ### GL 3129 forecast

# %%
fig_gl_3129 = plot_gl_forecast(3129)
fig_gl_3129.show()


# %% [markdown]
# ### GL 3101 forecast

# %%
fig_gl_3101 = plot_gl_forecast(3101)
fig_gl_3101.show()


# %% [markdown]
# ### GL 3004 forecast

# %%
fig_gl_3004 = plot_gl_forecast(3004)
fig_gl_3004.show()


# %% [markdown]
# ### GL 3005 forecast

# %%
fig_gl_3005 = plot_gl_forecast(3005)
fig_gl_3005.show()


# %% [markdown]
# ### GL 3114 forecast

# %%
fig_gl_3114 = plot_gl_forecast(3114)
fig_gl_3114.show()


# %% [markdown]
# ### GL 3002 forecast

# %%
fig_gl_3002 = plot_gl_forecast(3002)
fig_gl_3002.show()


# %% [markdown]
# ### GL 3047 forecast

# %%
fig_gl_3047 = plot_gl_forecast(3047)
fig_gl_3047.show()


# %% [markdown]
# ### GL 3044 forecast

# %%
fig_gl_3044 = plot_gl_forecast(3044)
fig_gl_3044.show()


# %% [markdown]
# ### GL 3043 forecast

# %%
fig_gl_3043 = plot_gl_forecast(3043)
fig_gl_3043.show()


# %% [markdown]
# ### GL 3046 forecast

# %%
fig_gl_3046 = plot_gl_forecast(3046)
fig_gl_3046.show()


# %% [markdown]
# ### GL 3003 forecast

# %%
fig_gl_3003 = plot_gl_forecast(3003)
fig_gl_3003.show()


# %% [markdown]
# ### GL 3042 forecast

# %%
fig_gl_3042 = plot_gl_forecast(3042)
fig_gl_3042.show()


# %% [markdown]
# ### GL 3276 forecast

# %%
fig_gl_3276 = plot_gl_forecast(3276)
fig_gl_3276.show()


# %% [markdown]
# ### GL 3075 forecast

# %%
fig_gl_3075 = plot_gl_forecast(3075)
fig_gl_3075.show()


# %% [markdown]
# ### GL 4621 forecast

# %%
fig_gl_4621 = plot_gl_forecast(4621)
fig_gl_4621.show()


# %% [markdown]
# ### GL 3290 forecast

# %%
fig_gl_3290 = plot_gl_forecast(3290)
fig_gl_3290.show()


# %% [markdown]
# ### Chart index

# %%
from html import escape
links = ''.join(f'<li><a href="gl_{gl}_forecast.html">GL {gl}</a></li>' for gl in selected_gl_codes)
(plot_dir / 'index.html').write_text(
    '<!doctype html><html><head><meta charset="utf-8"><title>GL forecasts</title></head>'
    '<body><h1>Selected GL forecasts</h1><ul>' + links + '</ul></body></html>', encoding='utf-8')
print(f'Individual chart index: {plot_dir / "index.html"}')

