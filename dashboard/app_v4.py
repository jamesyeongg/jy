"""
FM Energy Dashboard — v4 (live simulation + LLM Analysis)
-----------------------------------------------------------
Based on v2. All simulation behaviour is unchanged.

New in v4:
  • LLM Analysis panel replaces the placeholder.
  • Fixed read-only prompt: "Give me a summary of the energy usage for
    METER_02 for the past 2 days"
  • Send button: builds a JSON context from the latest 48 hours of METER_02
    data (total, average, min, max, hourly readings) and calls the OpenAI
    Chat Completions API.
  • Response is kept in session state — the one-minute replay and page
    refreshes do NOT trigger new API calls.
  • Shows "Based on readings up to <timestamp>", a session request counter,
    and a collapsed expander containing the JSON that was sent.

Run from the project root:
    c:\\temp\\workshop\\vibe\\Scripts\\streamlit.exe run dashboard/app_v4.py --server.headless true
"""

import json
import os
import re
import time
import pathlib

import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

# ── Configuration ─────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

METER_COLORS = {
    "METER_01": "#3B82F6",
    "METER_02": "#10B981",
    "METER_03": "#F59E0B",
}
TOTAL_COLOR = "#8B5CF6"

SIM_START    = pd.Timestamp("2016-01-01 00:00:00", tz="UTC")
SIM_INIT_END = pd.Timestamp("2016-01-31 23:00:00", tz="UTC")
SIM_END      = pd.Timestamp("2016-02-29 23:00:00", tz="UTC")

REFRESH_SECONDS = 60
LLM_PROMPT      = "Give me a summary of the energy usage for METER_02 for the past 2 days"


# ── Neon helper ───────────────────────────────────────────────────────────────
def _neon_connection():
    raw_url = os.getenv("DATABASE_URL", "")
    if not raw_url:
        st.error("DATABASE_URL not found in .env")
        st.stop()
    db_url = re.sub(r"[&?]channel_binding=require", "", raw_url)
    host = re.search(r"@([^/]+)/", db_url).group(1)
    return f"https://{host}/sql", db_url


def run_sql(query: str, params: list | None = None) -> pd.DataFrame:
    sql_url, db_url = _neon_connection()
    headers = {
        "Content-Type": "application/json",
        "Neon-Connection-String": db_url,
    }
    body = {"query": query}
    if params:
        body["params"] = params
    resp = requests.post(sql_url, headers=headers, json=body, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    cols = [f["name"] for f in payload["fields"]]
    df = pd.DataFrame(payload["rows"], columns=cols)
    if not df.empty:
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        if "next_ts" in df.columns:
            df["next_ts"] = pd.to_datetime(df["next_ts"], utc=True)
        if "electricity_kwh" in df.columns:
            df["electricity_kwh"] = df["electricity_kwh"].astype(float)
    return df


@st.cache_data(show_spinner=False)
def fetch_january() -> pd.DataFrame:
    return run_sql("""
        SELECT timestamp, meter_id, building_id, electricity_kwh
        FROM meter_readings_source
        WHERE timestamp >= $1 AND timestamp <= $2
        ORDER BY meter_id, timestamp
    """, [SIM_START.isoformat(), SIM_INIT_END.isoformat()])


def fetch_next_hour(after: pd.Timestamp) -> pd.DataFrame:
    next_ts = run_sql("""
        SELECT MIN(timestamp) AS next_ts
        FROM meter_readings_source
        WHERE timestamp > $1
    """, [after.isoformat()])
    if next_ts.empty or next_ts["next_ts"].iloc[0] is None:
        return pd.DataFrame()
    ts = pd.to_datetime(next_ts["next_ts"].iloc[0], utc=True)
    return run_sql("""
        SELECT timestamp, meter_id, building_id, electricity_kwh
        FROM meter_readings_source
        WHERE timestamp = $1
        ORDER BY meter_id
    """, [ts.isoformat()])


# ── LLM helper ────────────────────────────────────────────────────────────────
def build_llm_payload(df: pd.DataFrame) -> tuple[str, dict]:
    """
    Take the latest 48 hours of METER_02 from df.
    Return (json_string, stats_dict).
    """
    m02 = (
        df[df["meter_id"] == "METER_02"]
        .sort_values("timestamp")
        .tail(48)
        .reset_index(drop=True)
    )

    readings = [
        {
            "timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M UTC"),
            "electricity_kwh": round(row["electricity_kwh"], 3),
        }
        for _, row in m02.iterrows()
    ]

    kwh_values = m02["electricity_kwh"]
    stats = {
        "meter":      "METER_02",
        "from":       m02["timestamp"].iloc[0].strftime("%Y-%m-%d %H:%M UTC"),
        "to":         m02["timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M UTC"),
        "total_kwh":  round(float(kwh_values.sum()), 3),
        "average_kwh": round(float(kwh_values.mean()), 3),
        "min_kwh":    round(float(kwh_values.min()), 3),
        "max_kwh":    round(float(kwh_values.max()), 3),
        "hourly_readings": readings,
    }
    return json.dumps(stats, indent=2), stats


def call_openai(prompt: str, data_json: str) -> str:
    """Send prompt + JSON context to OpenAI and return the reply text."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    model   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if not api_key:
        return "⚠️ OPENAI_API_KEY not set in .env"

    # Swap out broken model name from .env if present
    if model == "gpt-6-luna":
        model = "gpt-4o-mini"

    user_message = (
        f"{prompt}\n\n"
        f"Here is the data as JSON:\n```json\n{data_json}\n```"
    )

    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an energy analyst. Provide a concise, plain-English "
                        "summary of the electricity consumption data you receive. "
                        "Highlight trends, peaks, and anything notable."
                    ),
                },
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.3,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ── Chart helpers ─────────────────────────────────────────────────────────────
def make_meter_chart(df: pd.DataFrame, meter_id: str) -> go.Figure:
    mdf = df[df["meter_id"] == meter_id].sort_values("timestamp")
    building = mdf["building_id"].iloc[0] if len(mdf) else meter_id
    color = METER_COLORS.get(meter_id, "#6B7280")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mdf["timestamp"], y=mdf["electricity_kwh"],
        mode="lines", line=dict(color=color, width=1.5),
        hovertemplate="%{x|%d %b %H:%M}<br>%{y:.1f} kWh<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"{meter_id}<br><sup>{building}</sup>", font=dict(size=13)),
        xaxis=dict(showgrid=False, tickformat="%d %b", range=[SIM_START, SIM_END]),
        yaxis=dict(title="kWh", showgrid=True, gridcolor="#F3F4F6"),
        margin=dict(l=40, r=10, t=55, b=30), height=260,
        plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
    )
    return fig


def make_total_chart(df: pd.DataFrame) -> go.Figure:
    total = (
        df.groupby("timestamp", as_index=False)["electricity_kwh"]
        .sum().sort_values("timestamp")
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=total["timestamp"], y=total["electricity_kwh"],
        mode="lines", fill="tozeroy",
        line=dict(color=TOTAL_COLOR, width=2),
        fillcolor="rgba(139,92,246,0.10)",
        hovertemplate="%{x|%d %b %H:%M}<br>%{y:.1f} kWh<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Total — all meters combined", font=dict(size=13)),
        xaxis=dict(showgrid=False, tickformat="%d %b", range=[SIM_START, SIM_END]),
        yaxis=dict(title="kWh", showgrid=True, gridcolor="#F3F4F6"),
        margin=dict(l=40, r=10, t=45, b=30), height=280,
        plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
    )
    return fig


# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("df",                  None),
    ("last_fetched_ts",     None),
    ("last_refresh_wall",   None),
    ("sim_complete",        False),
    ("llm_response",        None),   # cached LLM reply text
    ("llm_data_json",       None),   # JSON that was sent
    ("llm_based_on_ts",     None),   # timestamp at time of last Send
    ("llm_request_count",   0),      # total API calls this session
]:
    if key not in st.session_state:
        st.session_state[key] = default


def reset_simulation():
    fetch_january.clear()
    st.session_state.df               = None
    st.session_state.last_fetched_ts  = None
    st.session_state.last_refresh_wall = None
    st.session_state.sim_complete     = False
    # LLM state is preserved across restarts intentionally


# ── Page ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="FM Energy Dashboard v4", page_icon="⚡", layout="wide")
st.title("⚡ FM Energy Dashboard")
st.caption("Live simulation — Neon Postgres · METER_01 / METER_02 / METER_03")

# Initial data load
if st.session_state.df is None:
    with st.spinner("Loading January 2016 data from Neon …"):
        jan_df = fetch_january()
    st.session_state.df               = jan_df
    st.session_state.last_fetched_ts  = jan_df["timestamp"].max()
    st.session_state.last_refresh_wall = time.time()

df        = st.session_state.df
latest_ts = st.session_state.last_fetched_ts
meters    = sorted(df["meter_id"].unique())

# Status bar
status_col, btn_col = st.columns([6, 1])
with status_col:
    if st.session_state.sim_complete:
        st.success("✅ Simulation complete — all records loaded (Jan–Feb 2016).")
    else:
        rows_loaded = len(df) // len(meters) if meters else 0
        total_hours = int((SIM_END - SIM_START).total_seconds() // 3600) + 1
        pct = int(rows_loaded / total_hours * 100)
        st.info(
            f"🕐 Showing data up to **{latest_ts.strftime('%d %b %Y %H:%M UTC')}** "
            f"— {rows_loaded}/{total_hours} hours loaded ({pct}%). "
            f"Next fetch in ~{REFRESH_SECONDS}s."
        )
with btn_col:
    if st.button("🔄 Restart simulation", use_container_width=True):
        reset_simulation()
        st.rerun()

st.divider()

# Charts
st.subheader("Hourly electricity consumption by meter")
cols = st.columns(3)
for col, meter in zip(cols, meters):
    with col:
        st.plotly_chart(make_meter_chart(df, meter), use_container_width=True)

st.subheader("Total electricity consumption (all meters)")
st.plotly_chart(make_total_chart(df), use_container_width=True)

st.divider()

# ── Bottom panels ─────────────────────────────────────────────────────────────
pred_col, llm_col = st.columns(2)

with pred_col:
    st.subheader("🔮 Prediction")
    st.info(
        "**Coming soon** — next-hour consumption forecast for each meter "
        "using a Gradient Boosting model trained on lag and calendar features.",
        icon="🕐",
    )

with llm_col:
    st.subheader("🤖 LLM Analysis")

    # Read-only prompt display
    st.text_area(
        label="Prompt (read-only)",
        value=LLM_PROMPT,
        height=68,
        disabled=True,
        label_visibility="collapsed",
    )

    # Check METER_02 has at least 48 hours of data
    m02_rows = df[df["meter_id"] == "METER_02"]
    has_48h  = len(m02_rows) >= 48

    send_clicked = st.button(
        "📨 Send",
        disabled=not has_48h,
        help="Needs at least 48 hours of METER_02 data" if not has_48h else "",
    )

    if send_clicked:
        with st.spinner("Calling OpenAI …"):
            data_json, _ = build_llm_payload(df)
            try:
                reply = call_openai(LLM_PROMPT, data_json)
            except Exception as e:
                reply = f"⚠️ API error: {e}"
        st.session_state.llm_response      = reply
        st.session_state.llm_data_json     = data_json
        st.session_state.llm_based_on_ts   = latest_ts
        st.session_state.llm_request_count += 1

    # Show cached response
    if st.session_state.llm_response:
        based_on = st.session_state.llm_based_on_ts
        count    = st.session_state.llm_request_count
        st.caption(
            f"Based on readings up to **{based_on.strftime('%d %b %Y %H:%M UTC')}** "
            f"· {count} LLM request{'s' if count != 1 else ''} this session"
        )
        st.markdown(st.session_state.llm_response)
        with st.expander("📄 Data sent to the LLM"):
            st.code(st.session_state.llm_data_json, language="json")
    elif not has_48h:
        st.caption(
            f"METER_02 has {len(m02_rows)} hours of data so far. "
            "Send becomes active once 48 hours are loaded."
        )

# ── Auto-refresh ──────────────────────────────────────────────────────────────
if not st.session_state.sim_complete:
    elapsed = time.time() - st.session_state.last_refresh_wall
    if elapsed >= REFRESH_SECONDS:
        next_df = fetch_next_hour(st.session_state.last_fetched_ts)
        if next_df.empty:
            st.session_state.sim_complete = True
        else:
            st.session_state.df = pd.concat(
                [st.session_state.df, next_df], ignore_index=True
            )
            st.session_state.last_fetched_ts  = next_df["timestamp"].max()
            st.session_state.last_refresh_wall = time.time()
        st.rerun()
    else:
        time.sleep(min(REFRESH_SECONDS - elapsed, 5))
        st.rerun()
