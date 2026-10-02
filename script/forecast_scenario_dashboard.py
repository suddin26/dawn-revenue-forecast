# %% [markdown]
# # Revenue forecast and scenario dashboard
# This notebook creates a self-contained dashboard that opens in a browser without a Python kernel.
# Run the cells in order. Model results are cached, so changing the dashboard design does not refit them.
#
# | Control | What it changes |
# |---|---|
# | BA | Budget account to aggregate; All BAs includes every account |
# | GL | GL code within the chosen BA; All GLs forecasts their aggregated revenue |
# | FY from / FY through | Visible years only; these controls do not refit models |
# | Model | The base projection: Linear, Holt, random forest, Monte Carlo, or naive |
# | Scenario % | A percentage increase/decrease applied only to the future projection |
# | Scenario mode | Same percentage each year, or compounded annually |
#
# Forecast origin is FY 2026 by default. Change LAST_COMPLETE_FY below and rerun to use another origin.
# All revenue Types are included. Codes 42, 4860, and 4870 are excluded.
# No absent year is filled with zero. A selection without records in the origin FY has no supported forecast.
# Model verdicts use chronological backtesting, not the user-created scenario adjustments.

# %% [markdown]
# ## 1. Imports and reproducible settings
# Random seeds are fixed. The export records the data hash, code hash, settings, and package versions.

# %%
from pathlib import Path
from html import escape
import hashlib
import ast
import inspect
import importlib.metadata
import json
import platform
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.holtwinters import Holt
from IPython.display import display, IFrame, FileLink

LAST_COMPLETE_FY = 2026
FORECAST_HORIZON = 3
EXCLUDED_GL_CODES = [42, 4860, 4870]
LOOKBACK = 10
MIN_TRAIN_YEARS = 4
BACKTEST_ORIGINS = 3
N_SIMULATIONS = 10000
MC_CHANGE_WINDOW = 5
RF_TREES = 100
RF_MAX_DEPTH = 3
SEED = 2026
MODELS = ['Linear', 'Holt', 'ML (Random forest)', 'Monte Carlo', 'Naive baseline']

# %% [markdown]
# ## 2. Read and filter the original transaction data
# BA filtering happens before annual aggregation and fitting. A BA/GL forecast is not a fraction of an all-BA forecast.

# %%
working_dir = Path.cwd().resolve()
candidate_paths = [
    working_dir / 'data' / 'dawn_revenue_accounts_combined_history.csv',
    working_dir.parent / 'data' / 'dawn_revenue_accounts_combined_history.csv',
    working_dir.parent / 'dawn_aggregate' / 'data' / 'dawn_revenue_accounts_combined_history.csv',
    working_dir / 'data' / 'processed' / 'dawn_revenue_accounts_combined_history.csv',
]
input_csv = next((path.resolve() for path in candidate_paths if path.is_file()), None)
if input_csv is None:
    raise FileNotFoundError(f'Data file not found. Checked {candidate_paths}')
project_root = next(parent for parent in input_csv.parents if (parent / 'notebook').is_dir())
output_dir = project_root / 'result' / 'forecast_dashboard'
output_dir.mkdir(parents=True, exist_ok=True)
raw_data = pd.read_csv(input_csv)
for column in ['BA Number', 'GL Number', 'fy', 'Amount']:
    raw_data[column] = pd.to_numeric(raw_data[column], errors='raise')
    if raw_data[column].isna().any():
        raise ValueError(f'Missing {column} values require review before aggregation.')
data = raw_data.loc[
    ~raw_data['GL Number'].isin(EXCLUDED_GL_CODES) & raw_data['fy'].le(LAST_COMPLETE_FY)
].copy()
annual_by_ba_gl = (
    data.groupby(['BA Number', 'GL Number', 'fy'], as_index=False)
    .agg(revenue=('Amount', 'sum'), description=('description', 'first'))
    .sort_values(['BA Number', 'GL Number', 'fy'])
)
display(annual_by_ba_gl.head())
print(f"Loaded {len(data):,} records; {data['BA Number'].nunique()} BAs and {data['GL Number'].nunique()} GLs.")

# %% [markdown]
# ## 3. Forecast model functions
# Same modeling choices as allGLcode_forecast.ipynb: ten-year lookback, damped Holt, two-lag random forest,
# and Monte Carlo resampling of recent dollar changes. Models receive past data only.
# Negative revenue is preserved when it occurs in a training window; otherwise predictions have a zero floor.
# Insufficient-history methods return missing values rather than invented estimates.

# %%
fit_diagnostics = []
def fit_forecasts(values, horizon, seed, context):
    y = np.asarray(values, dtype=float)[-LOOKBACK:]
    scale = max(float(np.max(np.abs(y))), 1.0)
    z = y / scale
    floor_at_zero = not (y < 0).any()

    def apply_floor(values):
        values = np.asarray(values, dtype=float)
        return np.maximum(values, 0) if floor_at_zero else values

    predictions = {model: np.full(horizon, np.nan) for model in MODELS}
    predictions['Naive baseline'] = np.repeat(y[-1], horizon)
    low, high = np.full(horizon, np.nan), np.full(horizon, np.nan)
    minimum = {'Linear': 2, 'Holt': MIN_TRAIN_YEARS,
               'ML (Random forest)': MIN_TRAIN_YEARS, 'Monte Carlo': 2}
    for model, needed in minimum.items():
        if len(y) < needed:
            continue
        try:
            if np.all(y == y[0]):
                predictions[model] = np.repeat(y[-1], horizon)
                if model == 'Monte Carlo':
                    low, high = predictions[model].copy(), predictions[model].copy()
                continue
            if model == 'Linear':
                reg = LinearRegression().fit(np.arange(len(y)).reshape(-1, 1), z)
                pred = reg.predict(np.arange(len(y), len(y) + horizon).reshape(-1, 1)) * scale
            elif model == 'Holt':
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    fit = Holt(z, damped_trend=True, initialization_method='estimated').fit(optimized=True)
                for warning in caught:
                    fit_diagnostics.append({'Context': context, 'Model': model, 'Message': str(warning.message)})
                pred = np.asarray(fit.forecast(horizon)) * scale
            elif model == 'ML (Random forest)':
                features = np.array([[z[i-1], z[i-2], i] for i in range(2, len(z))])
                fit = RandomForestRegressor(n_estimators=RF_TREES, max_depth=RF_MAX_DEPTH,
                    min_samples_leaf=1, random_state=seed, n_jobs=1).fit(features, z[2:])
                extended = z.tolist()
                for _ in range(horizon):
                    extended.append(float(fit.predict([[extended[-1], extended[-2], len(extended)]])[0]))
                pred = np.asarray(extended[-horizon:]) * scale
            else:
                rng = np.random.default_rng(seed)
                changes = np.diff(y)[-MC_CHANGE_WINDOW:]
                draws = rng.choice(changes, size=(N_SIMULATIONS, horizon), replace=True)
                paths = np.empty_like(draws)
                previous = np.full(N_SIMULATIONS, y[-1])
                for h in range(horizon):
                    previous = apply_floor(previous + draws[:, h])
                    paths[:, h] = previous
                pred = paths.mean(axis=0)
                low, high = np.quantile(paths, 0.1, axis=0), np.quantile(paths, 0.9, axis=0)
            if not np.isfinite(pred).all():
                raise ValueError('Model returned non-finite forecasts.')
            predictions[model] = apply_floor(pred)
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
            fit_diagnostics.append({'Context': context, 'Model': model, 'Message': str(error)})
    return predictions, low, high

# %% [markdown]
# ## 4. Build the BA/GL selections
# Include each existing BA/GL pair, each BA total, each GL total across BAs, and the overall total.
# "All GLs" is one model of aggregated revenue, not a sum of separately fitted GL forecasts.

# %%
selections = {}
gl_descriptions = data.dropna(subset=['description']).groupby('GL Number')['description'].first().to_dict()


def add_selection(ba, gl, frame):
    annual = frame.groupby('fy', as_index=False)['revenue'].sum().sort_values('fy')
    description = 'All included GL codes' if gl == 'ALL' else str(gl_descriptions.get(int(gl), ''))
    selections[f'{ba}|{gl}'] = {'ba': str(ba), 'gl': str(gl), 'description': description, 'annual': annual}


add_selection('ALL', 'ALL', annual_by_ba_gl)
for ba, group in annual_by_ba_gl.groupby('BA Number'):
    add_selection(str(int(ba)), 'ALL', group)
    for gl, subgroup in group.groupby('GL Number'):
        add_selection(str(int(ba)), str(int(gl)), subgroup)
for gl, group in annual_by_ba_gl.groupby('GL Number'):
    add_selection('ALL', str(int(gl)), group)
print(f'{len(selections)} BA/GL selections will be available in the dashboard.')

# %% [markdown]
# ## 5. Backtest and summarize one selection
# At most three recent origins forecast the next three unseen years. Short series use a limited one-year holdout.
# Verdicts compare MAE against repeating the last observed value. Scenario changes are never fitted or validated.
# Stale selections display actual history, but no forecast. Historical gaps are not filled with zero.

# %%
def summarize_selection(selection):
    global fit_diagnostics
    fit_diagnostics = []
    actual = selection['annual']
    history = actual.copy()
    breaks = np.flatnonzero(np.diff(history['fy'].to_numpy()) != 1)
    if len(breaks):
        history = history.iloc[breaks[-1] + 1:]
    years = history['fy'].to_numpy(dtype=int)
    values = history['revenue'].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Annual revenue must be finite.')
    is_current = int(years[-1]) == LAST_COMPLETE_FY
    ba_seed = 0 if selection['ba'] == 'ALL' else int(selection['ba']) * 100000
    gl_seed = 0 if selection['gl'] == 'ALL' else int(selection['gl'])
    selection_seed = SEED + ba_seed + gl_seed
    backtests = []
    if is_current:
        origins = list(range(MIN_TRAIN_YEARS, len(values) - FORECAST_HORIZON + 1))[-BACKTEST_ORIGINS:]
        if not origins and len(values) >= 2:
            origins = [len(values) - 1]
        for cutoff in origins:
            origin = int(years[cutoff - 1])
            horizon = min(FORECAST_HORIZON, len(values) - cutoff)
            predictions, low, high = fit_forecasts(
                values[:cutoff], horizon, selection_seed + origin, f'origin FY {origin}')
            for model, predicted in predictions.items():
                for h in range(horizon):
                    backtests.append({'model': model, 'origin': origin, 'horizon': h + 1,
                                      'actual': float(values[cutoff + h]), 'predicted': float(predicted[h])})
        predictions, low, high = fit_forecasts(values, FORECAST_HORIZON, selection_seed, 'final forecast')
    else:
        predictions = {model: np.full(FORECAST_HORIZON, np.nan) for model in MODELS}
        low, high = np.full(FORECAST_HORIZON, np.nan), np.full(FORECAST_HORIZON, np.nan)
    tests = pd.DataFrame(backtests, columns=['model', 'origin', 'horizon', 'actual', 'predicted'])
    baseline = tests.loc[tests['model'].eq('Naive baseline')]
    validation = []
    for model in MODELS:
        valid = tests.loc[tests['model'].eq(model)].dropna(subset=['predicted'])
        paired = valid.merge(baseline[['origin', 'horizon', 'predicted']],
                             on=['origin', 'horizon'], suffixes=('', '_baseline'))
        errors = valid['predicted'] - valid['actual']
        mae = errors.abs().mean()
        base_mae = (paired['predicted_baseline'] - paired['actual']).abs().mean()
        actual_total = valid['actual'].abs().sum()
        same_cases = len(valid) > 0 and len(valid) == len(baseline)
        enough = same_cases and valid['origin'].nunique() >= 2 and valid['horizon'].max() == FORECAST_HORIZON
        verdict = 'No validation' if valid.empty else 'Limited validation'
        if enough:
            if model == 'Naive baseline':
                verdict = 'Baseline benchmark'
            elif mae < base_mae and not np.isclose(mae, base_mae):
                verdict = 'Improves on baseline'
            else:
                verdict = 'Does not improve on baseline'
        if any(item['Model'] == model for item in fit_diagnostics):
            verdict = 'Review model fitting warnings'
        if not is_current:
            verdict = 'Unavailable: no cutoff-year records'
        validation.append({
            'model': model, 'cases': len(valid), 'origins': valid['origin'].nunique(),
            'mae': mae, 'rmse': np.sqrt(np.mean(errors ** 2)) if len(valid) else np.nan,
            'wape': 100 * errors.abs().sum() / actual_total if actual_total else np.nan,
            'skill': 100 * (1 - mae / base_mae) if base_mae > 0 else np.nan,
            'verdict': verdict, 'comparable': same_cases,
        })
    candidates = [row for row in validation if row['comparable'] and np.isfinite(row['mae'])
                  and np.isfinite(predictions[row['model']]).all()
                  and row['verdict'] != 'Review model fitting warnings']
    # Prefer the simpler naive model when errors tie.
    chosen = min(candidates, key=lambda row: (row['mae'], row['model'] != 'Naive baseline'))['model'] if candidates else (
        'Naive baseline' if is_current else None)
    status = 'Current' if is_current else f'No records in FY {LAST_COMPLETE_FY}; forecast unavailable'
    if is_current and len(values) < MIN_TRAIN_YEARS:
        status = 'Short history; some models unavailable'
    return {
        'ba': selection['ba'], 'gl': selection['gl'], 'description': selection['description'],
        'history': [[int(row.fy), float(row.revenue)] for row in actual.itertuples()],
        'future_years': list(range(LAST_COMPLETE_FY + 1, LAST_COMPLETE_FY + 1 + FORECAST_HORIZON)),
        'forecasts': {model: predicted.tolist() for model, predicted in predictions.items()},
        'mc_p10': low.tolist(), 'mc_p90': high.tolist(), 'validation': validation,
        'backtests': backtests, 'chosen_model': chosen, 'status': status,
        'training_start': int(years[max(0, len(years) - LOOKBACK)]), 'training_years': min(len(values), LOOKBACK),
        'last_actual_fy': int(years[-1]), 'last_actual': float(values[-1]), 'diagnostics': fit_diagnostics.copy(),
    }

# %% [markdown]
# ## 6. Calculate or reuse reproducible cached forecasts
# This is the slowest cell on the first run. A cache is reusable only when the input data, Python code,
# model settings, Python version, and package versions match. Scenario adjustments do not require a refit.

# %%
packages = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'statsmodels', 'plotly', 'ipython']
versions = {package: importlib.metadata.version(package) for package in packages}
# Hash the functions currently defined in the notebook as well as the companion script.
# This invalidates cached fits after a user edits a model cell directly in the notebook.
function_source = inspect.getsource(fit_forecasts) + '\n' + inspect.getsource(summarize_selection)
function_signature = ast.dump(ast.parse(function_source), include_attributes=False)
run_settings = {
    'input_sha256': hashlib.sha256(input_csv.read_bytes()).hexdigest(),
    'code_sha256': hashlib.sha256((project_root / 'script' / 'forecast_scenario_dashboard.py').read_bytes()).hexdigest(),
    'function_sha256': hashlib.sha256(function_signature.encode()).hexdigest(),
    'last_complete_fy': LAST_COMPLETE_FY, 'horizon': FORECAST_HORIZON, 'excluded_gl_codes': EXCLUDED_GL_CODES,
    'lookback': LOOKBACK, 'min_train_years': MIN_TRAIN_YEARS, 'backtest_origins': BACKTEST_ORIGINS,
    'simulations': N_SIMULATIONS, 'mc_change_window': MC_CHANGE_WINDOW,
    'rf_trees': RF_TREES, 'rf_max_depth': RF_MAX_DEPTH, 'seed': SEED,
    'python_version': platform.python_version(), 'versions': versions,
    'revenue_types': 'ALL', 'missing_year_policy': 'unavailable',
}
cache_key = hashlib.sha256(json.dumps(run_settings, sort_keys=True).encode()).hexdigest()
cache_file = output_dir / 'model_cache.json'
cache = json.loads(cache_file.read_text(encoding='utf-8')) if cache_file.exists() else {}
if cache.get('cache_key') != cache_key:
    cache = {'cache_key': cache_key, 'series': {}}


def json_safe(value):
    """JSON uses null for unavailable values, never non-standard NaN/Infinity."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


for index, (key, selection) in enumerate(selections.items(), start=1):
    if key not in cache['series']:
        cache['series'][key] = json_safe(summarize_selection(selection))
    if index % 20 == 0 or index == len(selections):
        cache_file.write_text(json.dumps(cache, allow_nan=False), encoding='utf-8')
        print(f'Prepared {index}/{len(selections)} BA/GL selections.', flush=True)
series_results = {key: cache['series'][key] for key in selections}

# %% [markdown]
# ## 7. Scenario arithmetic
# The browser uses the same formulas. Historical observations never change.
# Uniform: adjusted = base * (1 + percent/100).
# Compounded: adjusted = base * (1 + percent/100) ** horizon, where horizons are 1, 2, and 3.
# Compounding is based on the original forecast horizon, not the first visible year.
# Multiplying signed revenue means a positive percentage makes a negative amount more negative.

# %%
def apply_scenario(base_forecast, percent=0.0, mode='uniform'):
    if not np.isfinite(percent) or percent < -100:
        raise ValueError('Enter a finite adjustment of at least -100%.')
    if mode not in {'uniform', 'compound'}:
        raise ValueError('Scenario mode must be uniform or compound.')
    base = np.asarray(base_forecast, dtype=float)
    horizons = np.arange(1, len(base) + 1)
    factors = np.repeat(1 + percent / 100, len(base)) if mode == 'uniform' else (1 + percent / 100) ** horizons
    return pd.DataFrame({'Horizon': horizons, 'Base forecast': base,
                         'Scenario forecast': base * factors, 'Difference': base * (factors - 1),
                         'Applied adjustment (%)': 100 * (factors - 1)})


display(apply_scenario([100, 100, 100], percent=10, mode='compound'))

# %% [markdown]
# ## 8. Export model forecasts and validation tables
# The dashboard's Download buttons export the currently selected scenario and validation rows.
# These Python exports contain the unadjusted forecasts and validation for every BA/GL selection.

# %%
forecast_rows, validation_rows, backtest_rows = [], [], []
for result in series_results.values():
    identity = {'BA': result['ba'], 'GL': result['gl'], 'Description': result['description']}
    for index, year in enumerate(result['future_years']):
        forecast_rows.append({**identity, 'FY': year,
            **{model: result['forecasts'][model][index] for model in MODELS},
            'MC P10': result['mc_p10'][index], 'MC P90': result['mc_p90'][index],
            'Best backtest model': result['chosen_model'], 'Status': result['status']})
    validation_rows.extend({**identity, **row} for row in result['validation'])
    backtest_rows.extend({**identity, **row} for row in result['backtests'])
base_forecast_table = pd.DataFrame(forecast_rows)
validation_table = pd.DataFrame(validation_rows)
for filename, table in {
    'base_forecasts.csv': base_forecast_table, 'model_validation.csv': validation_table,
    'backtest_predictions.csv': pd.DataFrame(backtest_rows),
}.items():
    destination = output_dir / filename
    try:
        table.to_csv(destination, index=False)
    except PermissionError:
        destination = destination.with_name(destination.stem + '_updated.csv')
        table.to_csv(destination, index=False)
    print(f'Saved {destination.name}')
(output_dir / 'run_settings.json').write_text(json.dumps(run_settings, indent=2), encoding='utf-8')
(output_dir / 'requirements.txt').write_text(
    '\n'.join(f'{package}=={version}' for package, version in versions.items()) + '\n', encoding='utf-8')
display(base_forecast_table.head())

# %% [markdown]
# ## 9. Build the standalone dashboard report
# The HTML template holds the layout and commented browser-side controls.
# All required data and the Plotly library are embedded: no network or Python kernel is needed to use the report.

# %%
payload = {'settings': run_settings, 'models': MODELS, 'series': series_results}
template_file = project_root / 'script' / 'forecast_dashboard_template.html'
template = template_file.read_text(encoding='utf-8')
# Escape '<' so descriptions cannot accidentally end the embedded JSON script element.
payload_json = json.dumps(payload, allow_nan=False).replace('<', '\\u003c')
report_html = template.replace('__PLOTLY_LIBRARY__', get_plotlyjs()).replace('__DASHBOARD_DATA__', payload_json)
report_file = output_dir / 'revenue_forecast_dashboard.html'
report_file.write_text(report_html, encoding='utf-8')
print(f'Open this report in your browser: {report_file}')

# %% [markdown]
# ## 10. Open the dashboard
# Open the HTML file directly in a browser if the notebook viewer blocks embedded scripts.
# Controls work in the browser without rerunning Python. Rerun this notebook only to change data or model settings.

# %%
import os
report_relative_path = Path(os.path.relpath(report_file, Path.cwd())).as_posix()
display(FileLink(report_relative_path))
display(IFrame(src=report_relative_path, width='100%', height=1350))

