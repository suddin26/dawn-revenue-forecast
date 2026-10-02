# %% [markdown]
# # All GL code revenue forecasts
# Run from top to bottom. Settings, models, forecasts, validation, exports, and charts are separate.
# All revenue Types are included. GL codes 42, 4860, and 4870 are excluded.
# Every remaining GL appears in the output, including codes with insufficient or stale data.
#
# | Method | Description |
# |---|---|
# | Linear | Straight-line regression on recent annual revenue |
# | Holt | Exponential smoothing with a damped trend |
# | ML (Random forest) | Two annual revenue lags and time; recursive predictions |
# | Monte Carlo | 10,000 paths using sampled recent annual dollar changes |
# | Naive baseline | Last recorded annual revenue repeated |
#
# Monte Carlo P10/P90 describe the middle 80% of simulated outcomes, not a guaranteed revenue range.

# %% [markdown]
# ## 1. Settings and imports
# Keep the same data, settings, Python version, and package versions to reproduce a run.

# %%
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import warnings
from html import escape
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.holtwinters import Holt
import plotly.graph_objects as go

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
# Alternatives: 'zero_fill' or 'carry_forward'. Do not change without a data assumption.
MISSING_YEAR_POLICY = 'unavailable'
MODELS = ['Linear', 'Holt', 'ML (Random forest)', 'Monte Carlo', 'Naive baseline']

# %% [markdown]
# ## 2. Read the source data
# Use the same candidate-path approach as the earlier notebooks. All Types, including MAJOR, are retained.

# %%
PROJECT_ROOT = Path.cwd().resolve()
RAW_DIR = PROJECT_ROOT / 'data'
PROCESSED_DIR = RAW_DIR / 'processed'
candidate_paths = [
    RAW_DIR / 'dawn_revenue_accounts_combined_history.csv',
    PROJECT_ROOT.parent / 'data' / 'dawn_revenue_accounts_combined_history.csv',
    PROJECT_ROOT.parent / 'dawn_aggregate' / 'data' / 'dawn_revenue_accounts_combined_history.csv',
    PROCESSED_DIR / 'dawn_revenue_accounts_combined_history.csv',
]
input_csv = next((path for path in candidate_paths if path.exists()), None)
if input_csv is None:
    raise FileNotFoundError(f'Data file not found. Checked: {candidate_paths}')
input_csv = input_csv.resolve()
project_root = next(parent for parent in input_csv.parents if (parent / 'notebook').is_dir())
result_dir = project_root / 'result'
plot_dir = result_dir / 'allGLcode_forecast_plots'
input_sha256 = hashlib.sha256(input_csv.read_bytes()).hexdigest()
df_all = pd.read_csv(input_csv)
required_columns = ['GL Number', 'fy', 'Amount', 'description', 'Type']
if not set(required_columns).issubset(df_all.columns):
    raise ValueError(f'Required columns: {required_columns}')
for column in ['GL Number', 'fy', 'Amount']:
    df_all[column] = pd.to_numeric(df_all[column], errors='raise')
if df_all[['GL Number', 'fy', 'Amount']].isna().any().any():
    raise ValueError('Missing GL, fiscal year, or revenue values require review.')
df_all_filtered = df_all.loc[
    ~df_all['GL Number'].isin(EXCLUDED_GL_CODES) & df_all['fy'].le(LAST_COMPLETE_FY)
].copy()
annual_revenue = (
    df_all_filtered.groupby(['GL Number', 'fy'], as_index=False)
    .agg(revenue=('Amount', 'sum'), description=('description', 'first'))
    .sort_values(['GL Number', 'fy'])
)
all_gl_codes = sorted(int(gl) for gl in annual_revenue['GL Number'].unique())
future_fys = np.arange(LAST_COMPLETE_FY + 1, LAST_COMPLETE_FY + 1 + FORECAST_HORIZON)
print(f'{len(all_gl_codes)} GL codes; forecast FY {future_fys[0]}-{future_fys[-1]}.')
display(annual_revenue.head())

# %% [markdown]
# ## 3. Available history and data status
# By default, missing years are not treated as zero. Models use the latest uninterrupted history.
# A GL without FY 2026 records remains in the CSV with unavailable forecasts and a review verdict.
# Current GLs with short histories get only methods that have enough observations.
# Negative net revenue is preserved: a model trained on negative values has no zero floor.

# %%
if MISSING_YEAR_POLICY not in {'unavailable', 'zero_fill', 'carry_forward'}:
    raise ValueError('Unknown MISSING_YEAR_POLICY.')
histories = {}
quality_rows = []
for gl in all_gl_codes:
    actual = annual_revenue.loc[annual_revenue['GL Number'].eq(gl)].copy()
    recorded_end = int(actual['fy'].iloc[-1])
    years = actual['fy'].to_numpy(dtype=int)
    missing = sorted(set(range(int(years[0]), LAST_COMPLETE_FY + 1)) - set(years))
    history = actual.copy()
    if MISSING_YEAR_POLICY == 'zero_fill':
        history = history.set_index('fy').reindex(range(int(years[0]), LAST_COMPLETE_FY + 1))
        history['revenue'] = history['revenue'].fillna(0)
        history['GL Number'] = gl
        history['description'] = actual['description'].iloc[-1]
        history = history.reset_index(names='fy')
    else:
        breaks = np.flatnonzero(np.diff(years) != 1)
        if len(breaks):
            history = history.iloc[breaks[-1] + 1:]
    if not np.isfinite(history['revenue']).all():
        raise ValueError(f'GL {gl}: non-finite annual revenue.')
    histories[gl] = history
    stale = recorded_end < LAST_COMPLETE_FY
    status = 'CURRENT'
    if stale:
        status = 'STALE: no records in cutoff FY'
    elif len(history) < MIN_TRAIN_YEARS:
        status = 'SHORT HISTORY: some methods unavailable'
    quality_rows.append({
        'GL Number': gl, 'Description': actual['description'].iloc[-1],
        'First recorded FY': int(years[0]), 'Last recorded FY': recorded_end,
        'Recorded years': len(actual), 'Model history start FY': int(history['fy'].iloc[0]),
        'Model history years': len(history), 'Final training years': min(len(history), LOOKBACK),
        'Missing fiscal years': ', '.join(map(str, missing)),
        'Negative annual totals': int((actual['revenue'] < 0).sum()),
        'Data status': status, 'Missing-year policy': MISSING_YEAR_POLICY,
    })
data_quality_table = pd.DataFrame(quality_rows)
quality_lookup = data_quality_table.set_index('GL Number')
display(data_quality_table)

# %% [markdown]
# ## 4. Model functions
# Hyperparameters and seeds match the selected-GL workflow. Each fit receives only past data.
# Linear needs 2 observations, Holt and random forest need 4, Monte Carlo needs 2, and the baseline needs 1.
# Constant series use an exact constant forecast, avoiding unnecessary optimizer warnings.

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
# ## 5. Historical backtests
# Use up to three recent origins with three unseen years each. If too short, use the last year
# as a limited holdout. No future actuals enter training. Stale GLs can have historical tests,
# but those tests do not establish current forecasting suitability.

# %%
fit_diagnostics = []
backtest_rows = []
for position, (gl, history) in enumerate(histories.items(), start=1):
    years = history['fy'].to_numpy(dtype=int)
    values = history['revenue'].to_numpy(dtype=float)
    origins = list(range(MIN_TRAIN_YEARS, len(values) - FORECAST_HORIZON + 1))[-BACKTEST_ORIGINS:]
    if not origins and len(values) >= 2:
        origins = [len(values) - 1]
    for cutoff in origins:
        origin = int(years[cutoff - 1])
        horizon = min(FORECAST_HORIZON, len(values) - cutoff)
        predictions, low, high = fit_forecasts(values[:cutoff], horizon, SEED + gl + origin,
                                              f'GL {gl}, origin FY {origin}')
        for model, predicted in predictions.items():
            for h in range(horizon):
                backtest_rows.append({
                    'GL Number': gl, 'Model': model, 'Origin FY': origin,
                    'Target FY': int(years[cutoff+h]), 'Horizon': h+1,
                    'Actual': values[cutoff+h], 'Predicted': predicted[h],
                    'MC P10': low[h] if model == 'Monte Carlo' else np.nan,
                    'MC P90': high[h] if model == 'Monte Carlo' else np.nan,
                })
    if position % 25 == 0:
        print(f'Backtested {position}/{len(histories)} GL codes.')
backtest_predictions = pd.DataFrame(backtest_rows, columns=[
    'GL Number', 'Model', 'Origin FY', 'Target FY', 'Horizon', 'Actual', 'Predicted', 'MC P10', 'MC P90'])
print('Backtesting complete.')

# %% [markdown]
# ## 6. Forecast FY 2027-2029
# Blank forecast values mean the method could not be supported by the available data.
# A stale-data carry-forward option supplies only the baseline, never a disguised fitted model.

# %%
forecast_rows = []
fit_diagnostics = [item for item in fit_diagnostics if ', final FY ' not in item['Context']]
for gl, history in histories.items():
    stale = quality_lookup.loc[gl, 'Last recorded FY'] < LAST_COMPLETE_FY
    values = history['revenue'].to_numpy(dtype=float)
    if stale and MISSING_YEAR_POLICY != 'zero_fill':
        predictions = {model: np.full(FORECAST_HORIZON, np.nan) for model in MODELS}
        low, high = np.full(FORECAST_HORIZON, np.nan), np.full(FORECAST_HORIZON, np.nan)
        if MISSING_YEAR_POLICY == 'carry_forward':
            predictions['Naive baseline'] = np.repeat(values[-1], FORECAST_HORIZON)
    else:
        predictions, low, high = fit_forecasts(values, FORECAST_HORIZON, SEED + gl,
                                              f'GL {gl}, final FY {LAST_COMPLETE_FY}')
    for h, year in enumerate(future_fys):
        forecast_rows.append({
            'GL Number': gl, 'Description': quality_lookup.loc[gl, 'Description'], 'FY': int(year),
            'Last actual FY': int(quality_lookup.loc[gl, 'Last recorded FY']),
            'Last actual': annual_revenue.loc[annual_revenue['GL Number'].eq(gl), 'revenue'].iloc[-1],
            **{model: predicted[h] for model, predicted in predictions.items()},
            'MC P10': low[h], 'MC P90': high[h], 'Data status': quality_lookup.loc[gl, 'Data status'],
        })
forecast_table = pd.DataFrame(forecast_rows)
print(f'Created {len(forecast_table)} GL/year rows, including explicitly unavailable forecasts.')

# %% [markdown]
# ## 7. Validation metrics and verdicts
# MAE/RMSE are dollar errors; WAPE is total absolute error / total absolute actual revenue.
# Positive MAE skill means improvement over the baseline on the same forecast cases.
#
# | Verdict | Meaning |
# |---|---|
# | IMPROVES ON BASELINE | Lower MAE, at least two origins, all three horizons, and no missing model fits |
# | DOES NOT IMPROVE ON BASELINE | Sufficient backtest coverage, but no MAE improvement |
# | LIMITED VALIDATION | Too few origins/horizons to assess three-year accuracy |
# | NO VALIDATION | No supported out-of-sample forecasts |
# | REVIEW FIT WARNINGS | Model fitting emitted a warning or failure |
# | REVIEW STALE DATA | No actual record in the cutoff FY; overrides a historical model verdict |
#
# These are evidence-based comparisons, not a guarantee of forecast accuracy or formal pass/fail thresholds.

# %%
def metrics(group):
    valid = group.dropna(subset=['Actual', 'Predicted'])
    if valid.empty:
        return {'Forecast cases': 0, 'Origins': 0, 'Max horizon': 0, 'MAE ($)': np.nan,
                'RMSE ($)': np.nan, 'WAPE (%)': np.nan, 'Bias ($)': np.nan, 'MC 80% coverage (%)': np.nan}
    errors = valid['Predicted'] - valid['Actual']
    denominator = valid['Actual'].abs().sum()
    return {
        'Forecast cases': len(valid), 'Origins': valid['Origin FY'].nunique(),
        'Max horizon': int(valid['Horizon'].max()), 'MAE ($)': errors.abs().mean(),
        'RMSE ($)': np.sqrt(np.mean(errors ** 2)),
        'WAPE (%)': 100 * errors.abs().sum() / denominator if denominator else np.nan,
        'Bias ($)': errors.mean(),
        'MC 80% coverage (%)': (100 * valid['Actual'].between(valid['MC P10'], valid['MC P90']).mean()
                               if valid['Model'].eq('Monte Carlo').all() else np.nan),
    }


validation_rows = []
for gl in all_gl_codes:
    baseline = backtest_predictions.loc[backtest_predictions['GL Number'].eq(gl)
                                        & backtest_predictions['Model'].eq('Naive baseline')]
    for model in MODELS:
        group = backtest_predictions.loc[backtest_predictions['GL Number'].eq(gl)
                                         & backtest_predictions['Model'].eq(model)]
        measured = metrics(group)
        matched = group.dropna(subset=['Predicted']).merge(
            baseline[['Origin FY', 'Target FY', 'Horizon', 'Predicted']],
            on=['Origin FY', 'Target FY', 'Horizon'], suffixes=('', '_baseline'))
        base_mae = (matched['Predicted_baseline'] - matched['Actual']).abs().mean()
        skill = 100 * (1 - measured['MAE ($)'] / base_mae) if base_mae > 0 else np.nan
        enough = (measured['Origins'] >= 2 and measured['Max horizon'] == FORECAST_HORIZON
                  and measured['Forecast cases'] == len(baseline))
        warned = any(item['Context'].startswith(f'GL {gl},') and item['Model'] == model
                     for item in fit_diagnostics)
        if measured['Forecast cases'] == 0:
            verdict = 'NO VALIDATION'
        elif not enough:
            verdict = 'LIMITED VALIDATION'
        elif model == 'Naive baseline':
            verdict = 'BASELINE BENCHMARK'
        elif measured['MAE ($)'] < base_mae and not np.isclose(measured['MAE ($)'], base_mae):
            verdict = 'IMPROVES ON BASELINE'
        else:
            verdict = 'DOES NOT IMPROVE ON BASELINE'
        if warned:
            verdict = 'REVIEW FIT WARNINGS'
        if quality_lookup.loc[gl, 'Last recorded FY'] < LAST_COMPLETE_FY:
            verdict = 'REVIEW STALE DATA'
        validation_rows.append({'GL Number': gl, 'Model': model, **measured,
            'Baseline MAE on matched cases ($)': base_mae, 'MAE skill vs naive (%)': skill,
            'Comparable cases': measured['Forecast cases'] > 0 and measured['Forecast cases'] == len(baseline),
            'Data status': quality_lookup.loc[gl, 'Data status'], 'Verdict': verdict})
validation_table = pd.DataFrame(validation_rows)
validation_by_horizon = pd.DataFrame([
    {'GL Number': gl, 'Model': model, 'Horizon': horizon, **metrics(group)}
    for (gl, model, horizon), group in backtest_predictions.groupby(['GL Number', 'Model', 'Horizon'])
])
gl_verdict_rows = []
for gl in all_gl_codes:
    available = forecast_table.loc[forecast_table['GL Number'].eq(gl)].iloc[0]
    candidates = validation_table.loc[validation_table['GL Number'].eq(gl)
        & validation_table['Comparable cases'] & validation_table['MAE ($)'].notna()].copy()
    candidates = candidates.loc[candidates['Model'].map(lambda model: pd.notna(available[model]))]
    candidates = candidates.loc[~candidates['Verdict'].eq('REVIEW FIT WARNINGS')]
    if candidates.empty:
        chosen = 'Naive baseline' if pd.notna(available['Naive baseline']) else None
        verdict = 'UNVALIDATED BASELINE ONLY' if chosen else 'UNAVAILABLE: resolve missing cutoff-year data'
    else:
        winner = candidates.sort_values('MAE ($)', kind='stable').iloc[0]
        chosen, verdict = winner['Model'], winner['Verdict']
    if quality_lookup.loc[gl, 'Last recorded FY'] < LAST_COMPLETE_FY:
        verdict = 'REVIEW STALE DATA: ' + ('scenario only' if chosen else 'forecast unavailable')
    gl_verdict_rows.append({'GL Number': gl, 'Selected model': chosen, 'GL verdict': verdict})
gl_verdict_table = data_quality_table.merge(pd.DataFrame(gl_verdict_rows), on='GL Number')
forecast_table = forecast_table.merge(pd.DataFrame(gl_verdict_rows), on='GL Number')
forecast_table['Selected-model forecast'] = [row[row['Selected model']] if pd.notna(row['Selected model'])
                                            else np.nan for _, row in forecast_table.iterrows()]
display(validation_table.style.format(precision=2, na_rep='Unavailable'))

# %% [markdown]
# ## 8. Per-GL verdict table
# Model selection is descriptive on these backtests, not independently tested. A baseline can win.

# %%
display(gl_verdict_table)

# %% [markdown]
# ## 9. Forecast values
# Each row is one GL and future fiscal year. All five methods and the Monte Carlo bounds are retained.

# %%
money_columns = ['Last actual'] + MODELS + ['MC P10', 'MC P90', 'Selected-model forecast']
display(forecast_table.style.format({column: '${:,.2f}' for column in money_columns}, na_rep='Unavailable'))

# %% [markdown]
# ## 10. Model comparison on common cases
# Pooled metrics use only GL/origin/horizon cases where all five models produced a forecast.
# Large revenue sources have more influence on pooled dollar errors. Use the per-GL verdicts too.

# %%
keys = ['GL Number', 'Origin FY', 'Target FY', 'Horizon']
complete = backtest_predictions.groupby(keys)['Predicted'].apply(lambda values: values.notna().sum() == len(MODELS))
common_keys = complete.loc[complete].reset_index()[keys]
common_cases = backtest_predictions.merge(common_keys, on=keys)
model_comparison = pd.DataFrame([
    {'Model': model, **metrics(common_cases.loc[common_cases['Model'].eq(model)])} for model in MODELS])
display(model_comparison.style.format(precision=2, na_rep='Unavailable'))

# %% [markdown]
# ## 11. Export CSVs and reproducibility records
# Files go under `result`. If a CSV is open and locked, save an `_updated.csv` copy and record its name.

# %%
result_dir.mkdir(parents=True, exist_ok=True)
plot_dir.mkdir(parents=True, exist_ok=True)
exported_files = {}
for filename, table in {
    'allGLcode_forecasts.csv': forecast_table,
    'allGLcode_model_validation.csv': validation_table,
    'allGLcode_verdicts.csv': gl_verdict_table,
    'allGLcode_validation_by_horizon.csv': validation_by_horizon,
    'allGLcode_backtest_predictions.csv': backtest_predictions,
    'allGLcode_model_comparison.csv': model_comparison,
    'allGLcode_data_quality.csv': data_quality_table,
    'allGLcode_fit_diagnostics.csv': pd.DataFrame(fit_diagnostics, columns=['Context', 'Model', 'Message']),
}.items():
    destination = result_dir / filename
    try:
        table.to_csv(destination, index=False)
    except PermissionError:
        destination = destination.with_name(destination.stem + '_updated.csv')
        table.to_csv(destination, index=False)
    exported_files[filename] = destination.name
packages = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'statsmodels', 'plotly', 'ipython', 'jinja2']
versions = {package: importlib.metadata.version(package) for package in packages}
run_settings = {
    'input_file': str(input_csv.relative_to(project_root)), 'input_sha256': input_sha256,
    'last_complete_fy': LAST_COMPLETE_FY, 'forecast_horizon': FORECAST_HORIZON,
    'excluded_gl_codes': EXCLUDED_GL_CODES, 'revenue_types': 'ALL',
    'missing_year_policy': MISSING_YEAR_POLICY, 'lookback': LOOKBACK,
    'min_train_years': MIN_TRAIN_YEARS, 'backtest_origins': BACKTEST_ORIGINS,
    'seed': SEED, 'n_simulations': N_SIMULATIONS, 'mc_change_window': MC_CHANGE_WINDOW,
    'rf_trees': RF_TREES, 'rf_max_depth': RF_MAX_DEPTH, 'rf_min_samples_leaf': 1, 'rf_n_jobs': 1,
    'holt_damped_trend': True, 'holt_initialization': 'estimated',
    'forecast_floor': 'Zero only when the training window contains no negative revenue',
    'python_version': platform.python_version(), 'package_versions': versions,
    'exported_files': exported_files, 'gl_codes': all_gl_codes,
}
(result_dir / 'allGLcode_run_settings.json').write_text(json.dumps(run_settings, indent=2), encoding='utf-8')
(result_dir / 'allGLcode_requirements.txt').write_text(
    '\n'.join(f'{package}=={version}' for package, version in versions.items()) + '\n', encoding='utf-8')
print(f'CSV files saved in {result_dir}')
display(pd.DataFrame(exported_files.items(), columns=['Table', 'Saved filename']))

# %% [markdown]
# ## 12. Historical-versus-forecast chart function
# One interactive chart per GL. Actual observations include earlier history; gaps are displayed as gaps.
# Forecasts use the latest continuous block. An annotation explains unavailable forecasts.

# %%
MODEL_COLORS = {'Linear': '#0072B2', 'Holt': '#D55E00', 'ML (Random forest)': '#009E73',
                'Monte Carlo': '#CC79A7', 'Naive baseline': '#888888'}


def plot_gl_forecast(gl):
    actual = annual_revenue.loc[annual_revenue['GL Number'].eq(gl)]
    forecast = forecast_table.loc[forecast_table['GL Number'].eq(gl)]
    observed = actual.set_index('fy')['revenue'].reindex(range(int(actual['fy'].min()), LAST_COMPLETE_FY + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=observed.index, y=observed, name='Actual', mode='lines+markers',
                            connectgaps=False, line=dict(color='#222222')))
    if forecast['MC P10'].notna().any():
        fig.add_trace(go.Scatter(x=forecast['FY'], y=forecast['MC P10'], mode='lines',
            line=dict(width=0), legendgroup='MC range', showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=forecast['FY'], y=forecast['MC P90'], mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor='rgba(204,121,167,0.17)', name='MC P10-P90', legendgroup='MC range',
            customdata=forecast[['MC P10']],
            hovertemplate='FY %{x}<br>MC range: $%{customdata[0]:,.0f} to $%{y:,.0f}<extra></extra>'))
    for model in MODELS:
        if forecast[model].isna().all():
            continue
        anchor = observed.loc[LAST_COMPLETE_FY]
        if pd.isna(anchor) and MISSING_YEAR_POLICY == 'zero_fill':
            anchor = 0
        fig.add_trace(go.Scatter(x=[LAST_COMPLETE_FY, *forecast['FY']],
            y=[anchor, *forecast[model]], name=model, mode='lines+markers',
            line=dict(color=MODEL_COLORS[model], dash='dash'),
            hovertemplate=f'FY %{{x}}<br>{model}: $%{{y:,.0f}}<extra></extra>'))
    verdict = forecast['GL verdict'].iloc[0]
    fig.add_annotation(x=0, y=1.05, xref='paper', yref='paper', text=escape(verdict),
                       showarrow=False, xanchor='left', font=dict(size=11))
    fig.add_vline(x=LAST_COMPLETE_FY + 0.5, line_dash='dot', line_color='gray')
    fig.update_layout(title=f'GL {gl}: {escape(str(actual["description"].iloc[-1]))}',
        template='plotly_white', height=550, hovermode='x unified',
        xaxis=dict(title='Fiscal year (FY)', tickformat='d', dtick=2,
                   range=[int(actual['fy'].min()) - 0.5, int(future_fys[-1]) + 0.5]),
        yaxis=dict(title='Revenue ($)', tickprefix='$', tickformat='~s'),
        legend=dict(orientation='h', y=-0.2, groupclick='togglegroup'), margin=dict(t=110, b=120))
    return fig

# %% [markdown]
# ## 13. Export a chart for every GL
# HTML charts share a local Plotly library and work offline. The index lists every GL and its verdict.

# %%
chart_links = []
for gl in all_gl_codes:
    figure = plot_gl_forecast(gl)
    filename = f'gl_{gl}_forecast.html'
    figure.write_html(plot_dir / filename, include_plotlyjs='directory')
    row = gl_verdict_table.loc[gl_verdict_table['GL Number'].eq(gl)].iloc[0]
    chart_links.append(f'<li><a href="{filename}">GL {gl}: {escape(str(row["Description"]))}</a>'
                       f' - {escape(row["GL verdict"])}</li>')
(plot_dir / 'index.html').write_text(
    '<!doctype html><html><head><meta charset="utf-8"><title>All GL forecasts</title></head>'
    '<body><h1>All GL historical revenue and forecasts</h1><ul>' + ''.join(chart_links) + '</ul></body></html>',
    encoding='utf-8')
print(f'Saved {len(all_gl_codes)} interactive charts: {plot_dir / "index.html"}')
