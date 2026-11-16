"""
Upload three_meters.csv to Neon Postgres via the HTTP SQL API.
Uses the Neon-Connection-String header so port 5432 is never needed.
"""
import pathlib, os, json, re
import pandas as pd
import requests

# ── Load DATABASE_URL from .env ──────────────────────────────────────────────
ENV_PATH = pathlib.Path(__file__).parent.parent / '.env'
db_url = None
for line in ENV_PATH.read_text().splitlines():
    line = line.strip()
    if line.startswith('DATABASE_URL'):
        db_url = line.split('=', 1)[1].strip().strip('"')
        break

if not db_url:
    raise RuntimeError("DATABASE_URL not found in .env")

# Strip channel_binding=require (unsupported by HTTP API)
db_url = re.sub(r'[&?]channel_binding=require', '', db_url)
print(f"Connecting to: {db_url.split('@')[1]}")   # host/db only — hide credentials

# ── Derive HTTP SQL endpoint from the connection string ──────────────────────
# postgresql://user:pass@ep-xxx.region.aws.neon.tech/dbname?sslmode=require
# → https://ep-xxx.region.aws.neon.tech/sql
host_match = re.search(r'@([^/]+)/', db_url)
if not host_match:
    raise RuntimeError("Could not parse host from DATABASE_URL")
host = host_match.group(1)
sql_url = f"https://{host}/sql"
print(f"HTTP SQL endpoint: {sql_url}")

HEADERS = {
    "Content-Type": "application/json",
    "Neon-Connection-String": db_url,
}

def run_sql(sql, params=None):
    body = {"query": sql}
    if params:
        body["params"] = params
    r = requests.post(sql_url, headers=HEADERS, json=body, timeout=30)
    r.raise_for_status()
    return r.json()

# ── 1. Create table ──────────────────────────────────────────────────────────
print("\nCreating table meter_readings_source ...")
run_sql("""
CREATE TABLE IF NOT EXISTS meter_readings_source (
    timestamp       TIMESTAMPTZ  NOT NULL,
    meter_id        TEXT         NOT NULL,
    building_id     TEXT         NOT NULL,
    electricity_kwh NUMERIC(10,4) NOT NULL,
    CONSTRAINT meter_readings_source_pk PRIMARY KEY (meter_id, timestamp)
)
""")
print("Table ready.")

# ── 2. Load CSV ──────────────────────────────────────────────────────────────
CSV_PATH = pathlib.Path(__file__).parent.parent / 'data' / 'three_meters.csv'
df = pd.read_csv(CSV_PATH, parse_dates=['timestamp'])
df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S+00:00')
print(f"\nLoaded {len(df)} rows from {CSV_PATH.name}")

# ── 3. Upload in batches ─────────────────────────────────────────────────────
BATCH_SIZE = 500
total_inserted = 0

for start in range(0, len(df), BATCH_SIZE):
    batch = df.iloc[start:start + BATCH_SIZE]

    # Build a single INSERT with one VALUES row per record
    placeholders = ',\n'.join(
        f"(${ (i*4)+1 },${ (i*4)+2 },${ (i*4)+3 },${ (i*4)+4 })"
        for i in range(len(batch))
    )
    sql = f"""
INSERT INTO meter_readings_source (timestamp, meter_id, building_id, electricity_kwh)
VALUES {placeholders}
ON CONFLICT (meter_id, timestamp) DO NOTHING
"""
    params = []
    for _, row in batch.iterrows():
        params += [row['timestamp'], row['meter_id'], row['building_id'], float(row['electricity_kwh'])]

    run_sql(sql, params)
    total_inserted += len(batch)
    print(f"  Uploaded rows {start+1}–{start+len(batch)} ...")

print(f"\nUpload complete — {total_inserted} rows sent.")

# ── 4. Summary query ─────────────────────────────────────────────────────────
print("\nRow count, earliest and latest timestamp per meter:")
result = run_sql("""
SELECT
    meter_id,
    COUNT(*)                          AS row_count,
    MIN(timestamp)::TIMESTAMPTZ       AS earliest,
    MAX(timestamp)::TIMESTAMPTZ       AS latest
FROM meter_readings_source
GROUP BY meter_id
ORDER BY meter_id
""")

rows    = result.get('rows', [])
columns = result.get('columns', [])
col_names = [c['name'] for c in columns]

summary = pd.DataFrame(rows, columns=col_names)
print(summary.to_string(index=False))
