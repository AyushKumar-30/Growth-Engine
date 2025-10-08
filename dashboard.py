# ------------------------------------------------------------
# CareYaya Growth Engine Dashboard v2
# ------------------------------------------------------------
# Run with:
#   streamlit run dashboard.py
# ------------------------------------------------------------

import os
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime

# -------------------------------
# Layout & Page Configuration
# -------------------------------
st.set_page_config(
    page_title="CareYaya Growth Engine Dashboard",
    page_icon="📈",
    layout="wide",
)

st.title("📊 CareYaya Growth Engine Dashboard")
st.caption("Real-time performance view for YouTube, Reddit, and Trends discovery")

DATA_PATH = os.path.join("output", "comments_posted.csv")

# -------------------------------
# Data Loading
# -------------------------------
@st.cache_data
def load_data(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Normalize
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["Keyword"] = df["Keyword"].astype(str)
    df["Platform"] = df["Platform"].astype(str)
    df["Posted"] = df["Posted"].astype(str)
    # Identify dynamic (Trends) keywords
    static_keywords = [
        "elder care",
        "caregiver burnout",
        "home health care",
        "dementia caregiving",
        "AI caregiving",
    ]
    df["Keyword Type"] = df["Keyword"].apply(
        lambda x: "Dynamic (Trends)" if x.lower() not in [kw.lower() for kw in static_keywords] else "Static"
    )
    return df

df = load_data(DATA_PATH)

# -------------------------------
# Empty State
# -------------------------------
if df.empty:
    st.warning("⚠️ No data available yet. Run the engine to generate output.")
    st.stop()

# -------------------------------
# Summary Metrics
# -------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Entries", len(df))
col2.metric("Unique Keywords", df["Keyword"].nunique())
col3.metric("Platforms", ", ".join(sorted(df["Platform"].unique())))
col4.metric("Last Update", df["Timestamp"].max().strftime("%b %d, %H:%M"))

st.markdown("---")

# -------------------------------
# Filters
# -------------------------------
with st.expander("🔍 Filters", expanded=True):
    platform_filter = st.multiselect(
        "Filter by Platform",
        options=sorted(df["Platform"].unique()),
        default=sorted(df["Platform"].unique()),
    )
    keyword_type_filter = st.multiselect(
        "Keyword Type",
        options=df["Keyword Type"].unique(),
        default=list(df["Keyword Type"].unique()),
    )

filtered = df[df["Platform"].isin(platform_filter)]
filtered = filtered[filtered["Keyword Type"].isin(keyword_type_filter)]

# -------------------------------
# Keyword Distribution
# -------------------------------
st.subheader("Keyword Distribution")

colA, colB = st.columns([2, 1])

with colA:
    kw_counts = (
        filtered.groupby(["Keyword", "Keyword Type"])
        .size()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
    )
    fig_kw = px.bar(
        kw_counts,
        x="Keyword",
        y="Count",
        color="Keyword Type",
        title="Frequency of Keywords",
        text="Count",
    )
    fig_kw.update_layout(xaxis_tickangle=-45, showlegend=True)
    st.plotly_chart(fig_kw, use_container_width=True)

with colB:
    pie_kw = (
        filtered.groupby("Keyword Type")
        .size()
        .reset_index(name="Count")
    )
    fig_pie = px.pie(
        pie_kw,
        values="Count",
        names="Keyword Type",
        title="Static vs Dynamic Keywords",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# -------------------------------
# Platform Distribution
# -------------------------------
st.subheader("Platform Distribution")
platform_counts = (
    filtered.groupby("Platform")
    .size()
    .reset_index(name="Count")
    .sort_values("Count", ascending=False)
)
fig_platform = px.bar(
    platform_counts,
    x="Platform",
    y="Count",
    color="Platform",
    title="Posts / Comments per Platform",
    text="Count",
)
fig_platform.update_layout(showlegend=False)
st.plotly_chart(fig_platform, use_container_width=True)

# -------------------------------
# Activity Over Time
# -------------------------------
st.subheader("Activity Over Time")
time_series = (
    filtered.groupby(filtered["Timestamp"].dt.date)
    .size()
    .reset_index(name="Count")
)
fig_time = px.line(
    time_series,
    x="Timestamp",
    y="Count",
    markers=True,
    title="Number of Entries per Day",
)
st.plotly_chart(fig_time, use_container_width=True)

# -------------------------------
# Recent Activity Table
# -------------------------------
st.subheader("Recent Entries")
st.dataframe(
    filtered.sort_values("Timestamp", ascending=False)[
        ["Timestamp", "Platform", "Keyword Type", "Keyword", "Topic", "URL", "Generated Comment", "Posted"]
    ],
    use_container_width=True,
    hide_index=True,
)

st.markdown("---")
st.caption("Built with ❤️ by Ayush — CareYaya Growth Engine Prototype")
