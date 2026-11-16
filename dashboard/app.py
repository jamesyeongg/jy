"""
FM Energy Dashboard
-------------------
Reads historical meter readings from Neon Postgres and displays:
  • Three individual hourly electricity charts (one per meter) across the top
  • A combined total chart below them
  • Placeholders for Prediction and LLM Analysis at the bottom

Run from the project root:
    vibe\\Scripts\\streamlit run dashboard/app.py --server.headless true
"""

import os
import re
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
    "METER_01": "#3B82F6",   # blue
    "METER_02": "#10B981",   # green
    "METER_03": "#F59E0B",   # amber
}
TOTAL_COLOR = "#8B5CF6"      # purple

# ── Neon HTTP SQL helper ───────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Fetch all rows from meter_readings_source via the Neon HTTP SQL API."""
    raw_url = os.getenv("DATABASE_URL", "")
    if not raw_url:
        st.error("DATABASE_URL not found in .env")
        st.stop()

    # Strip channel_binding=require — not supported by the HTTP API
    db_url = re.sub(r"[&?]channel_binding=require", "", raw_url)

    # Derive HTTPS endpoint from the host in the connection string
    host_match = re.search(r"@([^/]+)/", db_url)
    if not host_match:
        st.error("Could not parse host from DATABASE_URL")
        st.stop()
    sql_url = f"https://{host_match.group(1)}/sql"

    headers = {
        "Content-Type": "application/json",
        "Neon-Connection-String": db_url,
    }

    query = """
        SELECT timestamp, meter_id, building_id, electricity_kwh
        FROM meter_readings_source
        ORDER BY meter_id, timestamp
    """

    resp = requests.post(sql_url, headers=headers, json={"query": query}, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    cols = [f["name"] for f in payload["fields"]]
    df = pd.DataFrame(payload["rows"], columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["electricity_kwh"] = df["electricity_kwh"].astype(float)
    return df


def make_meter_chart(df: pd.DataFrame, meter_id: str) -> go.Figure:
    """Line chart for a single meter."""
    mdf = df[df["meter_id"] == meter_id].sort_values("timestamp")
    building = mdf["building_id"].iloc[0] if len(mdf) else meter_id
    color = METER_COLORS.get(meter_id, "#6B7280")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mdf["timestamp"],
        y=mdf["electricity_kwh"],
        mode="lines",
        line=dict(color=color, width=1.5),
        name=meter_id,
        hovertemplate="%{x|%d %b %H:%M}<br>%{y:.1f} kWh<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"{meter_id}<br><sup>{building}</sup>", font=dict(size=13)),
        xaxis=dict(showgrid=False, tickformat="%d %b"),
        yaxis=dict(title="kWh", showgrid=True, gridcolor="#F3F4F6"),
        margin=dict(l=40, r=10, t=55, b=30),
        height=260,
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
    )
    return fig


def make_total_chart(df: pd.DataFrame) -> go.Figure:
    """Line chart for the summed total across all meters."""
    total = (
        df.groupby("timestamp", as_index=False)["electricity_kwh"]
        .sum()
        .sort_values("timestamp")
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=total["timestamp"],
        y=total["electricity_kwh"],
        mode="lines",
        fill="tozeroy",
        line=dict(color=TOTAL_COLOR, width=2),
        fillcolor="rgba(139,92,246,0.10)",
        name="Total",
        hovertemplate="%{x|%d %b %H:%M}<br>%{y:.1f} kWh<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Total — all meters combined", font=dict(size=13)),
        xaxis=dict(showgrid=False, tickformat="%d %b"),
        yaxis=dict(title="kWh", showgrid=True, gridcolor="#F3F4F6"),
        margin=dict(l=40, r=10, t=45, b=30),
        height=280,
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
    )
    return fig


# ── Page layout ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FM Energy Dashboard",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ FM Energy Dashboard")
st.caption("Historical data — Neon Postgres · METER_01 / METER_02 / METER_03")
st.divider()

# Load data
with st.spinner("Loading data from Neon …"):
    df = load_data()

meters = sorted(df["meter_id"].unique())

# ── Row 1: three meter charts ─────────────────────────────────────────────────
st.subheader("Hourly electricity consumption by meter")
cols = st.columns(3)
for col, meter in zip(cols, meters):
    with col:
        st.plotly_chart(make_meter_chart(df, meter), use_container_width=True)

# ── Row 2: total chart ────────────────────────────────────────────────────────
st.subheader("Total electricity consumption (all meters)")
st.plotly_chart(make_total_chart(df), use_container_width=True)

st.divider()

# ── Row 3: placeholders ───────────────────────────────────────────────────────
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
    st.info(
        "**Coming soon** — an LLM-powered plain-English analysis of recent "
        "consumption patterns, anomalies, and recommendations.",
        icon="💬",
    )
