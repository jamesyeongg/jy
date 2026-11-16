"""
Exploration plots for one_meter.csv.
Saves three PNGs to models/ for review:
  explore_full_series.png   – the complete 60-day series
  explore_daily_profile.png – average hourly profile by weekday vs weekend
  explore_weekly_heatmap.png – hour-of-day × day-of-week heatmap
"""
import pathlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT      = pathlib.Path(__file__).parent.parent
DATA_PATH = ROOT / 'data' / 'one_meter.csv'
OUT_DIR   = ROOT / 'models'
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)
df['hour']      = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek          # 0=Mon … 6=Sun
df['is_weekend'] = df['dayofweek'] >= 5
df['date']      = df['timestamp'].dt.date

# ── Plot 1: Full 60-day series ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 3.5))
ax.plot(df['timestamp'], df['electricity_kwh'], lw=0.6, color='steelblue')
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
ax.set_title('METER_01 — Full 60-day hourly series', fontsize=12)
ax.set_ylabel('kWh')
ax.grid(axis='y', alpha=0.3)
# Shade weekends
for _, grp in df.groupby((df['is_weekend'] != df['is_weekend'].shift()).cumsum()):
    if grp['is_weekend'].iloc[0]:
        ax.axvspan(grp['timestamp'].iloc[0], grp['timestamp'].iloc[-1],
                   color='lightyellow', alpha=0.6, zorder=0)
ax.text(0.01, 0.92, 'Yellow bands = weekends', transform=ax.transAxes,
        fontsize=8, color='goldenrod')
plt.tight_layout()
p = OUT_DIR / 'explore_full_series.png'
plt.savefig(p, dpi=130); plt.close()
print(f'Saved {p}')

# ── Plot 2: Average daily profile — weekday vs weekend ───────────────────────
profile = df.groupby(['is_weekend', 'hour'])['electricity_kwh'].mean().reset_index()
weekday = profile[~profile['is_weekend']]
weekend = profile[ profile['is_weekend']]

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(weekday['hour'], weekday['electricity_kwh'],
        marker='o', ms=4, lw=1.8, color='steelblue', label='Weekday (Mon–Fri)')
ax.plot(weekend['hour'], weekend['electricity_kwh'],
        marker='s', ms=4, lw=1.8, color='tomato',    label='Weekend (Sat–Sun)')
ax.set_xticks(range(0, 24))
ax.set_title('METER_01 — Average hourly profile: weekday vs weekend', fontsize=12)
ax.set_xlabel('Hour of day')
ax.set_ylabel('Mean kWh')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
p = OUT_DIR / 'explore_daily_profile.png'
plt.savefig(p, dpi=130); plt.close()
print(f'Saved {p}')

# ── Plot 3: Hour-of-day × day-of-week heatmap ────────────────────────────────
pivot = df.groupby(['dayofweek', 'hour'])['electricity_kwh'].mean().unstack('hour')
day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

fig, ax = plt.subplots(figsize=(14, 4))
im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd')
ax.set_yticks(range(7)); ax.set_yticklabels(day_labels)
ax.set_xticks(range(0, 24)); ax.set_xticklabels(range(0, 24), fontsize=8)
ax.set_xlabel('Hour of day')
ax.set_title('METER_01 — Mean consumption by day × hour (kWh)', fontsize=12)
plt.colorbar(im, ax=ax, label='Mean kWh', shrink=0.8)
plt.tight_layout()
p = OUT_DIR / 'explore_weekly_heatmap.png'
plt.savefig(p, dpi=130); plt.close()
print(f'Saved {p}')

print('All plots saved.')
