"""
Query Neon for per-meter summary and print it clearly.
"""
import pathlib, re, json
import requests

ENV_PATH = pathlib.Path(__file__).parent.parent / '.env'
db_url = None
for line in ENV_PATH.read_text().splitlines():
    line = line.strip()
    if line.startswith('DATABASE_URL'):
        db_url = line.split('=', 1)[1].strip().strip('"')
        break

db_url = re.sub(r'[&?]channel_binding=require', '', db_url)
host   = re.search(r'@([^/]+)/', db_url).group(1)
sql_url = f"https://{host}/sql"

HEADERS = {
    "Content-Type": "application/json",
    "Neon-Connection-String": db_url,
}

def run_sql(sql):
    r = requests.post(sql_url, headers=HEADERS, json={"query": sql}, timeout=30)
    r.raise_for_status()
    return r.json()

# Debug: inspect raw API response shape
result = run_sql("""
SELECT
    meter_id,
    COUNT(*)           AS row_count,
    MIN(timestamp)     AS earliest,
    MAX(timestamp)     AS latest
FROM meter_readings_source
GROUP BY meter_id
ORDER BY meter_id
""")

print("Raw API response:")
print(json.dumps(result, indent=2, default=str))
