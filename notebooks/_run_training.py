"""
Standalone version of the notebook logic.
Run once to verify everything works and to save artefacts.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend — no window needed
import matplotlib.pyplot as plt
import json, joblib

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT       = pathlib.Path(__file__).parent.parent
DATA_PATH  = ROOT / 'data' / 'one_meter.csv'
MODELS_DIR = ROOT / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load & inspect ────────────────────────────────────────────────────────
raw = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
raw = raw.sort_values('timestamp').reset_index(drop=True)

print("=== 1. Data overview ===")
print(f"Shape : {raw.shape}")
print(f"Range : {raw['timestamp'].min()}  →  {raw['timestamp'].max()}")
print(f"Missing values:\n{raw.isnull().sum()}")
print(f"\nConsumption statistics (kWh):")
print(raw['electricity_kwh'].describe().round(2))

diffs = raw['timestamp'].diff().dropna()
gaps  = diffs[diffs != pd.Timedelta('1h')]
print(f"\nTimestamp gaps / irregular steps: {len(gaps)}")
print(f"Duplicate timestamps            : {raw['timestamp'].duplicated().sum()}")
# → Clean data: 1 440 rows, no nulls, no gaps, no duplicates.

# ── 2. Build features ────────────────────────────────────────────────────────
print("\n=== 2. Building features ===")
df = raw[['timestamp', 'electricity_kwh']].copy()

for lag in [1, 2, 3, 24, 168]:
    df[f'lag_{lag}h'] = df['electricity_kwh'].shift(lag)

df['hour']      = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek

df = df.dropna().reset_index(drop=True)
print(f"Rows after dropping first 168 (incomplete lag_168h): {len(df)}")

# ── 3. Train / test split ────────────────────────────────────────────────────
SPLIT_DATE = '2016-02-16'
FEATURES   = ['lag_1h', 'lag_2h', 'lag_3h', 'lag_24h', 'lag_168h', 'hour', 'dayofweek']
TARGET     = 'electricity_kwh'

train = df[df['timestamp'] <  SPLIT_DATE]
test  = df[df['timestamp'] >= SPLIT_DATE]

X_train, y_train = train[FEATURES], train[TARGET]
X_test,  y_test  = test[FEATURES],  test[TARGET]

print(f"\n=== 3. Train/test split ===")
print(f"Train : {len(train):>4} rows  ({train['timestamp'].min().date()} → {train['timestamp'].max().date()})")
print(f"Test  : {len(test):>4} rows  ({test['timestamp'].min().date()} → {test['timestamp'].max().date()})")

# ── 4. Train ─────────────────────────────────────────────────────────────────
print("\n=== 4. Training GradientBoostingRegressor ===")
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)
print("Training complete.")

importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nFeature importances:")
print(importances.round(3).to_string())

# ── 5. Evaluate ──────────────────────────────────────────────────────────────
print("\n=== 5. Evaluation (test set) ===")
y_pred     = model.predict(X_test)
y_baseline = X_test['lag_1h'].values

m_mae  = mean_absolute_error(y_test, y_pred)
m_rmse = root_mean_squared_error(y_test, y_pred)
b_mae  = mean_absolute_error(y_test, y_baseline)
b_rmse = root_mean_squared_error(y_test, y_baseline)

print(f"{'Model':<30}  MAE = {m_mae:.3f} kWh   RMSE = {m_rmse:.3f} kWh")
print(f"{'Naïve baseline (lag_1h)':<30}  MAE = {b_mae:.3f} kWh   RMSE = {b_rmse:.3f} kWh")
print(f"\nMAE  improvement over baseline: {(1 - m_mae/b_mae)*100:.1f}%")
print(f"RMSE improvement over baseline: {(1 - m_rmse/b_rmse)*100:.1f}%")

# ── 6. Plot & save ───────────────────────────────────────────────────────────
print("\n=== 6. Saving artefacts ===")
results = test[['timestamp']].copy().reset_index(drop=True)
results['actual']    = y_test.values
results['predicted'] = y_pred
results['error']     = y_pred - y_test.values

fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

axes[0].plot(results['timestamp'], results['actual'],    label='Actual',    lw=1.0, color='steelblue')
axes[0].plot(results['timestamp'], results['predicted'], label='Predicted', lw=1.0, color='darkorange', ls='--')
axes[0].set_title('METER_01 — Actual vs Predicted (test: last two weeks)')
axes[0].set_ylabel('kWh')
axes[0].legend()

axes[1].fill_between(results['timestamp'], results['error'], 0,
                     where=results['error'] >= 0, color='darkorange', alpha=0.4, label='Over-predicted')
axes[1].fill_between(results['timestamp'], results['error'], 0,
                     where=results['error'] <  0, color='steelblue',  alpha=0.4, label='Under-predicted')
axes[1].axhline(0, color='black', lw=0.8)
axes[1].set_title('Prediction error (predicted − actual)')
axes[1].set_ylabel('kWh')
axes[1].set_xlabel('Date')
axes[1].legend()

plt.tight_layout()
plot_path = MODELS_DIR / 'test_predictions.png'
plt.savefig(plot_path, dpi=120)
print(f"Plot saved  → {plot_path}")

# Save model and feature order
model_path   = MODELS_DIR / 'gbr_model.joblib'
feature_path = MODELS_DIR / 'feature_order.json'

joblib.dump(model, model_path)
print(f"Model saved → {model_path}")

with open(feature_path, 'w') as f:
    json.dump(FEATURES, f, indent=2)
print(f"Features saved → {feature_path}")
print("\nDone.")
