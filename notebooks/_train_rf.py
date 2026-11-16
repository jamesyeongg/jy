"""
Train a RandomForestRegressor on the same features and split used for the
GradientBoostingRegressor, then compare both against the naïve baseline.
"""
import pathlib, json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import joblib

ROOT       = pathlib.Path(__file__).parent.parent
DATA_PATH  = ROOT / 'data' / 'one_meter.csv'
MODELS_DIR = ROOT / 'models'

# ── Build features (identical to training script) ───────────────────────────
raw = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
raw = raw.sort_values('timestamp').reset_index(drop=True)

df = raw[['timestamp', 'electricity_kwh']].copy()
for lag in [1, 2, 3, 24, 168]:
    df[f'lag_{lag}h'] = df['electricity_kwh'].shift(lag)
df['hour']      = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.day_of_week
df = df.dropna().reset_index(drop=True)

FEATURES   = ['lag_1h', 'lag_2h', 'lag_3h', 'lag_24h', 'lag_168h', 'hour', 'dayofweek']
TARGET     = 'electricity_kwh'
SPLIT_DATE = '2016-02-16'

train = df[df['timestamp'] <  SPLIT_DATE]
test  = df[df['timestamp'] >= SPLIT_DATE]
X_train, y_train = train[FEATURES], train[TARGET]
X_test,  y_test  = test[FEATURES],  test[TARGET]

# ── Train Random Forest ──────────────────────────────────────────────────────
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print("Random Forest training complete.")

imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nFeature importances:")
print(imp.round(3).to_string())

# ── Load GBR for comparison ──────────────────────────────────────────────────
gbr = joblib.load(MODELS_DIR / 'gbr_model.joblib')

# ── Evaluate all three ───────────────────────────────────────────────────────
y_rf       = rf.predict(X_test)
y_gbr      = gbr.predict(X_test)
y_baseline = X_test['lag_1h'].values

def row(label, actual, predicted):
    mae  = mean_absolute_error(actual, predicted)
    rmse = root_mean_squared_error(actual, predicted)
    return label, mae, rmse

rows = [
    row('Naïve baseline (lag_1h)',         y_test, y_baseline),
    row('Random Forest  (n=100)',           y_test, y_rf),
    row('Gradient Boosting (default)',      y_test, y_gbr),
]

print(f"\n{'Model':<35}  {'MAE (kWh)':>10}  {'RMSE (kWh)':>11}")
print('-' * 60)
for label, mae, rmse in rows:
    print(f"{label:<35}  {mae:>10.3f}  {rmse:>11.3f}")

# ── Comparison plot ──────────────────────────────────────────────────────────
timestamps = test['timestamp'].reset_index(drop=True)

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(timestamps, y_test.values, lw=1.0, color='steelblue',  label='Actual')
ax.plot(timestamps, y_rf,          lw=1.0, color='seagreen',   label='Random Forest',     linestyle='--')
ax.plot(timestamps, y_gbr,         lw=1.0, color='darkorange', label='Gradient Boosting', linestyle=':')
ax.set_title('METER_01 — Random Forest vs Gradient Boosting (test period)')
ax.set_ylabel('kWh')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
p = MODELS_DIR / 'rf_vs_gbr.png'
plt.savefig(p, dpi=130); plt.close()
print(f"\nPlot saved → {p}")

# ── Save RF model ────────────────────────────────────────────────────────────
rf_path = MODELS_DIR / 'rf_model.joblib'
joblib.dump(rf, rf_path)
print(f"RF model saved → {rf_path}")
