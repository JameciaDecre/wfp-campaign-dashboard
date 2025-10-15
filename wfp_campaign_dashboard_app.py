import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------
# WFP Campaign Dashboard — District Attempts & Support IDs
# ---------------------------------------

st.set_page_config(page_title="WFP Campaign Dashboard", layout="wide")
st.title("WFP Campaign Dashboard — District Attempts & Support IDs")

st.caption(
    "This app reads a CSV committed in the repo (no uploads needed).\n"
    "Expected columns: `district, tier, attempts, target_doors, 1, 2, 3, 4, 5`."
)

CSV_PATH = Path("dashboard_wfp_template.csv")  # keep this filename at repo root

def coerce_int_series(s: pd.Series) -> pd.Series:
    # strip commas/spaces, coerce to number
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")

@st.cache_data
def load_data(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"CSV not found at '{path}'. Commit a file named 'dashboard_wfp_template.csv' to the repo root."
        )
    df = pd.read_csv(path)
    required = ["district", "tier", "attempts", "target_doors", "1", "2", "3", "4", "5"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # clean types
    df["district"] = coerce_int_series(df["district"]).astype("Int64")
    df["tier"] = coerce_int_series(df["tier"]).astype("Int64")
    df["attempts"] = coerce_int_series(df["attempts"]).astype("Int64")
    df["target_doors"] = coerce_int_series(df["target_doors"]).astype("Int64")
    for c in ["1","2","3","4","5"]:
        df[c] = coerce_int_series(df[c]).astype("Int64")

    # derived
    df["tier_label"] = df["tier"].map({1: "Tier 1", 2: "Tier 2"}).fillna("Unassigned")
    df["penetration"] = (df["attempts"] / df["target_doors"]).replace([np.inf, -np.inf], np.nan).clip(upper=1.0)
    return df

# Load once per session
try:
    df = load_data(CSV_PATH)
except Exception as e:
    st.error(str(e))
    st.stop()

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    all_districts = sorted([int(x) for x in df["district"].dropna().unique().tolist()])
    sel_districts = st.multiselect("Districts (1–12)", options=all_districts, default=all_districts)
    comp_thresh = st.slider("Completion threshold (penetration)", 0.0, 1.0, 0.8, 0.05)

f = df[df["district"].isin(sel_districts)].copy()
if f.empty:
    st.warning("No rows match your filters.")
    st.stop()

# KPIs
total_attempts = int(f["attempts"].sum(skipna=True))
total_targets  = int(f["target_doors"].sum(skipna=True))
avg_pen = float(f["penetration"].mean(skipna=True)) if not f.empty else np.nan

c1, c2, c3 = st.columns(3)
c1.metric("Total Attempts (filtered)", f"{total_attempts:,}")
c2.metric("Total Target Doors (filtered)", f"{total_targets:,}")
c3.metric("Avg. Penetration", f"{avg_pen:.1%}" if pd.notnull(avg_pen) else "—")

# District summary
st.markdown("### District Summary")
summary = (
    f.sort_values("district")
     .assign(
         penetration_pct=(f["penetration"]*100).round(1).astype(str) + "%",
         status=np.where(f["penetration"] >= comp_thresh, "Complete", "In Progress")
     )
     [["district","tier_label","attempts","target_doors","penetration_pct","status","1","2","3","4","5"]]
     .rename(columns={"1":"SID 1","2":"SID 2","3":"SID 3","4":"SID 4","5":"SID 5"})
)
st.dataframe(summary, use_container_width=True)

# Support distribution
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

st.caption("To update the dashboard, commit a new 'dashboard_wfp_template.csv' (same columns) to this repo.")
