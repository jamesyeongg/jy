"""
Full evaluation of the saved GBR model on the held-out test set.
Prints MAE, RMSE, and MAPE for the model and the naïve baseline.
"""
import pathlib
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

ROOT       = pathlib.Path(__file__).parent.parent
DATA_PATH  = ROOT / 'data' / 'one_meter.csv'
MODELS_DIR = ROOT / 'models'

# ── Rebuild features (identical pipeline) ───────────────────────────────────
raw = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
raw = raw.sort_values('timestamp').reset_index(drop=True)

df = raw[['timestamp', 'electricity_kwh']].copy()
for lag in [1, 2, 3, 24, 168]:
    df[f'lag_{lag}h'] = df['electricity_kwh'].shift(lag)
df['hour']      = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.day_of_week
df = df.dropna().reset_index(drop=True)

FEATURES   = ['lag_1h', 'lag_2h', 'lag_3h', 'lag_24h', 'lag_168h', 'hour', 'dayofweek']
SPLIT_DATE = '2016-02-16'

test     = df[df['timestamp'] >= SPLIT_DATE]
X_test   = test[FEATURES]
y_test   = test['electricity_kwh']

# ── Predict ──────────────────────────────────────────────────────────────────
model      = joblib.load(MODELS_DIR / 'gbr_model.joblib')
y_pred     = model.predict(X_test)
y_baseline = X_test['lag_1h'].values

# ── Metrics ──────────────────────────────────────────────────────────────────
def evaluate(label, actual, predicted):
    mae  = mean_absolute_error(actual, predicted)
    rmse = root_mean_squared_error(actual, predicted)
    mape = float(np.mean(np.abs((actual.values - predicted) / actual.values)) * 100)
    return label, mae, rmse, mape

results = [
    evaluate('Naïve baseline (lag_1h)',    y_test, y_baseline),
    evaluate('GradientBoostingRegressor',  y_test, y_pred),
]

print(f"\n{'Model':<35}  {'MAE (kWh)':>10}  {'RMSE (kWh)':>11}  {'MAPE (%)':>9}")
print('-' * 72)
for label, mae, rmse, mape in results:
    print(f"{label:<35}  {mae:>10.3f}  {rmse:>11.3f}  {mape:>9.2f}")

b_mae = results[0][1]; m_mae = results[1][1]
b_rmse= results[0][2]; m_rmse= results[1][2]
print(f"\nMAE  improvement over baseline: {(1 - m_mae / b_mae)*100:.1f}%")
print(f"RMSE improvement over baseline: {(1 - m_rmse/ b_rmse)*100:.1f}%")

# ── Worst hours ──────────────────────────────────────────────────────────────
errors = test[['timestamp']].copy().reset_index(drop=True)
errors['actual']    = y_test.values
errors['predicted'] = y_pred
errors['abs_error'] = np.abs(y_pred - y_test.values)

print("\nTop 10 largest absolute errors:")
print(errors.nlargest(10, 'abs_error')[['timestamp','actual','predicted','abs_error']].to_string(index=False))
