# Recreate the two files you need to download: the Streamlit app and the CSV template
from textwrap import dedent
import pandas as pd

# 1) Streamlit app tailored to your CSV schema
app_code = dedent("""
import streamlit as st
import pandas as pd
import numpy as np

# ---------------------------------------
# WFP Campaign Dashboard — District Attempts & Support IDs
# ---------------------------------------

st.set_page_config(page_title="WFP Campaign Dashboard", layout="wide")
st.title("WFP Campaign Dashboard — District Attempts & Support IDs")

st.caption(
    "This app expects a CSV with columns: "
    "`district, tier, attempts, target_doors, 1, 2, 3, 4, 5`. "
    "Replace the CSV in the repo (named `dashboard_wfp_template.csv`) **or** upload one below."
)

uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])

DEFAULT_PATH = "dashboard_wfp_template.csv"  # Keep this filename constant in your repo

def coerce_int_series(s):
    # strip commas and spaces, coerce to numeric
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")

@st.cache_data
def load_data(file_or_path):
    df = pd.read_csv(file_or_path)
    required = ["district", "tier", "attempts", "target_doors", "1", "2", "3", "4", "5"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    # Types & cleaning
    df["district"] = coerce_int_series(df["district"]).astype("Int64")
    df["tier"] = coerce_int_series(df["tier"]).astype("Int64")
    df["attempts"] = coerce_int_series(df["attempts"]).astype("Int64")
    df["target_doors"] = coerce_int_series(df["target_doors"]).astype("Int64")
    for c in ["1","2","3","4","5"]:
        df[c] = coerce_int_series(df[c]).astype("Int64")
    # Derived
    df["tier_label"] = df["tier"].map({1: "Tier 1", 2: "Tier 2"}).fillna("Unassigned")
    df["penetration"] = (df["attempts"] / df["target_doors"]).replace([np.inf, -np.inf], np.nan).clip(upper=1.0)
    return df

# Try uploader first; else default file
try:
    if uploaded is not None:
        df = load_data(uploaded)
    else:
        df = load_data(DEFAULT_PATH)
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    all_districts = sorted([int(x) for x in df["district"].dropna().unique().tolist()])
    sel_districts = st.multiselect("Districts (1–12)", options=all_districts, default=all_districts)
    comp_thresh = st.slider("Completion threshold (penetration)", 0.0, 1.0, 0.8, 0.05)

f = df[df["district"].isin(sel_districts)].copy()
if f.empty:
    st.warning("No rows match your filters."); st.stop()

# KPIs
total_attempts = int(f["attempts"].sum(skipna=True))
total_targets  = int(f["target_doors"].sum(skipna=True))
avg_pen = float(f["penetration"].mean(skipna=True)) if not f.empty else np.nan

c1, c2, c3 = st.columns(3)
c1.metric("Total Attempts (filtered)", f"{total_attempts:,}")
c2.metric("Total Target Doors (filtered)", f"{total_targets:,}")
c3.metric("Avg. Penetration", f"{avg_pen:.1%}" if pd.notnull(avg_pen) else "—")

# District summary table
st.markdown("### District Summary")
summary = (
    f.sort_values("district")
     .assign(penetration_pct=(f["penetration"]*100).round(1).astype(str) + "%",
             status=np.where(f["penetration"] >= comp_thresh, "Complete", "In Progress"))
     [["district","tier_label","attempts","target_doors","penetration_pct","status","1","2","3","4","5"]]
     .rename(columns={"1":"SID 1","2":"SID 2","3":"SID 3","4":"SID 4","5":"SID 5"})
)
st.dataframe(summary, use_container_width=True)

# Support distribution (stacked) — simple bar chart
st.markdown("### Support Distribution (1–5)")
support_melt = f.melt(id_vars=["district"], value_vars=["1","2","3","4","5"], var_name="SID", value_name="count").fillna(0)
pivot = support_melt.pivot_table(index="district", columns="SID", values="count", aggfunc="sum").fillna(0)
st.bar_chart(pivot)

# Downloads
st.download_button(
    "Download District Summary (CSV)",
    data=summary.to_csv(index=False).encode("utf-8"),
    file_name="wfp_district_summary_with_support.csv",
    mime="text/csv"
)
st.download_button(
    "Download Cleaned Source (CSV)",
    data=f.to_csv(index=False).encode("utf-8"),
    file_name="wfp_cleaned_source.csv",
    mime="text/csv"
)

st.caption("Replace the file named `dashboard_wfp_template.csv` in your repo (or use the uploader) to update this dashboard.")
""")

with open("/mnt/data/wfp_campaign_dashboard_app.py", "w") as f:
    f.write(app_code)

# 2) CSV template with the expected header (empty rows for you to fill)
cols = ["district","tier","attempts","target_doors","1","2","3","4","5"]
example = pd.DataFrame([
    [2,1, "1,000", "3,200", 120, 200, 300, 250, 130],
    [5,1, "850",  "2,500", 100, 180, 260, 220, 90],
    [7,1, "1,150","2,800", 140, 210, 310, 280, 110],
    [8,1, "900",  "3,000", 110, 190, 270, 240, 100],
    [3,2, "700",  "2,600",  80, 160, 230, 210, 75],
], columns=cols)

template_path = "/mnt/data/dashboard_wfp_template.csv"
example.to_csv(template_path, index=False)

print("/mnt/data/wfp_campaign_dashboard_app.py")
print(template_path)
