# ------------------------------------------------------------
# CareYaya Growth Engine Dashboard v3 (YouTube + Reddit only)
# ------------------------------------------------------------
# Run with:
#   streamlit run dashboard.py
# ------------------------------------------------------------

import os
import pandas as pd
import streamlit as st
import plotly.express as px

# -------------------------------
# Layout & Page Configuration
# -------------------------------
st.set_page_config(
    page_title="CareYaya Growth Engine Dashboard",
    page_icon="📈",
    layout="wide",
)

st.title("📊 CareYaya Growth Engine Dashboard")
st.caption("Real-time overview of YouTube & Reddit engagement")

DATA_PATH = os.path.join("output", "comments_posted.csv")

# -------------------------------
# Data Loading
# -------------------------------
@st.cache_data(ttl=5)
def load_data(path):
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Ensure consistent column casing
    expected_cols = [
        "Timestamp", "Platform", "Keyword", "Topic",
        "URL", "Generated Comment", "Posted", "Posted At"
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        st.error(f"Missing columns in CSV: {missing}")
        st.stop()

    # Normalize datetime
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["Posted At"] = pd.to_datetime(df["Posted At"], errors="coerce")

    # Keep only Reddit + YouTube
    df = df[df["Platform"].isin(["youtube", "reddit"])]

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
# Handle missing or NaT timestamps safely
last_update = df["Posted At"].max()
if pd.isna(last_update):
    last_update_str = "No data"
else:
    last_update_str = last_update.strftime("%b %d, %H:%M")



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
    keyword_filter = st.multiselect(
        "Filter by Keyword",
        options=sorted(df["Keyword"].unique()),
        default=sorted(df["Keyword"].unique()),
    )

filtered = df[df["Platform"].isin(platform_filter)]
filtered = filtered[filtered["Keyword"].isin(keyword_filter)]

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
# Keyword Frequency
# -------------------------------
st.subheader("Keyword Frequency")
kw_counts = (
    filtered.groupby("Keyword")
    .size()
    .reset_index(name="Count")
    .sort_values("Count", ascending=False)
)
fig_kw = px.bar(
    kw_counts,
    x="Keyword",
    y="Count",
    title="Frequency of Keywords Used",
    text="Count",
)
fig_kw.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig_kw, use_container_width=True)

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
# Recent Entries Table
# -------------------------------
st.subheader("Recent Entries")

df_display = filtered.sort_values("Posted At", ascending=False).copy()
df_display["Posted At"] = df_display["Posted At"].dt.strftime("%Y-%m-%d %H:%M")

st.dataframe(
    df_display[["Platform", "Keyword", "Topic", "Generated Comment", "URL", "Posted", "Posted At"]],
    use_container_width=True,
    height=600,
)

st.markdown("---")
st.caption("Built with ❤️ by Ayush — CareYaya Growth Engine Prototype")
