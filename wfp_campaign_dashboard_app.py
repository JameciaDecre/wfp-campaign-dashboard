# Write a drop-in replacement Streamlit app WITHOUT campaign_name
# and generate a matching CSV template.

import pandas as pd

app_code = """
import streamlit as st
import pandas as pd
import numpy as np

# ------------------------------
# WFP Campaign Dashboard (no campaign_name, no precincts)
# ------------------------------

st.set_page_config(page_title="WFP Campaign Dashboard", layout="wide")
st.title("WFP Campaign Dashboard")

st.caption(
    "Upload a tidy CSV with columns: "
    "`date, district, pass_number, contact_method, attempts, target_attempts`. "
    "Dates should be YYYY-MM-DD."
)

uploaded = st.file_uploader("Upload data CSV", type=["csv"])

# Tier mapping by district (two tiers)
TIER1_SET = {2, 5, 7, 8}
TIER2_SET = {3, 4, 6, 9, 10, 11, 12}

def map_tier(d):
    try:
        d_int = int(d)
    except Exception:
        return "Unassigned"
    if d_int in TIER1_SET:
        return "Tier 1"
    if d_int in TIER2_SET:
        return "Tier 2"
    return "Unassigned"

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    required = [
        "date", "district", "pass_number",
        "contact_method", "attempts", "target_attempts"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    # Parse types
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["district", "pass_number", "attempts", "target_attempts"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["tier"] = df["district"].apply(map_tier)
    df["contact_method"] = (
        df["contact_method"]
          .astype("string")
          .str.strip()
          .str.title()
          .fillna("Unknown")
    )
    return df

if uploaded is None:
    st.info("⬆️ Upload your CSV to get started.")
    st.stop()

try:
    df = load_data(uploaded)
except Exception as e:
    st.error(str(e))
    st.stop()

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    all_districts = sorted([int(d) for d in df["district"].dropna().unique().tolist()])
    sel_districts = st.multiselect(
        "Districts (1–12)",
        options=list(range(1, 13)),
        default=all_districts or list(range(1, 13))
    )
    all_passes = sorted([int(p) for p in df["pass_number"].dropna().unique().tolist()])
    sel_passes = st.multiselect("Pass Number", options=all_passes, default=all_passes)
    all_methods = sorted(df["contact_method"].dropna().unique().tolist())
    sel_methods = st.multiselect("Contact Method", options=all_methods, default=all_methods)
    comp_thresh = st.slider("Completion threshold (penetration)", 0.0, 1.0, 0.8, 0.05)

mask = (
    df["district"].isin(sel_districts) &
    df["pass_number"].isin(sel_passes) &
    df["contact_method"].isin(sel_methods)
)
f = df.loc[mask].copy()
if f.empty:
    st.warning("No rows match your filters.")
    st.stop()

def penetration_status(frame, group_cols, comp_threshold):
    agg = (
        frame.groupby(group_cols, dropna=False)
             .agg(attempts=("attempts", "sum"),
                  target_attempts=("target_attempts", "max"))
             .reset_index()
    )
    agg["penetration"] = (agg["attempts"] / agg["target_attempts"]).replace([np.inf, -np.inf], np.nan)
    agg["penetration"] = agg["penetration"].clip(upper=1.0)
    agg["status"] = np.where(agg["penetration"] >= comp_threshold, "Complete", "In Progress")
    return agg

# KPI row (by method grain)
agg_method = penetration_status(f, ["district", "tier", "pass_number", "contact_method"], comp_thresh)
total_attempts = int(agg_method["attempts"].sum())
total_targets = int(agg_method["target_attempts"].sum())
avg_pen = agg_method["penetration"].mean()

c1, c2, c3 = st.columns(3)
c1.metric("Total Attempts (filtered)", f"{total_attempts:,}")
c2.metric("Total Targets (max by group)", f"{total_targets:,}")
c3.metric("Avg. Penetration", f"{avg_pen:.1%}" if pd.notnull(avg_pen) else "—")

# District × Pass × Method
st.markdown("### Status by District × Pass × Method")
view_method = (
    agg_method.sort_values(["district", "pass_number", "contact_method"])
              .assign(penetration_pct=(agg_method["penetration"]*100).round(1).astype(str) + "%")
              [["district", "tier", "pass_number", "contact_method",
                "attempts", "target_attempts", "penetration_pct", "status"]]
)
st.dataframe(view_method, use_container_width=True)

# District Summary (roll up across methods)
st.markdown("### District Summary (all methods combined)")
agg_district = penetration_status(f, ["district", "tier", "pass_number"], comp_thresh)
view_district = (
    agg_district.sort_values(["district", "pass_number"])
                .assign(penetration_pct=(agg_district["penetration"]*100).round(1).astype(str) + "%")
                [["district", "tier", "pass_number", "attempts",
                  "target_attempts", "penetration_pct", "status"]]
)
st.dataframe(view_district, use_container_width=True)

# Downloads
st.download_button(
    "Download Status by District × Pass × Method (CSV)",
    data=agg_method.to_csv(index=False).encode("utf-8"),
    file_name="wfp_status_by_district_pass_method.csv",
    mime="text/csv"
)
st.download_button(
    "Download District Summary (CSV)",
    data=agg_district.to_csv(index=False).encode("utf-8"),
    file_name="wfp_district_summary.csv",
    mime="text/csv"
)

st.caption("Tiers auto-assigned from districts — Tier 1: {2,5,7,8}; Tier 2: {3,4,6,9,10,11,12}; others → Unassigned.")
"""

with open("/mnt/data/wfp_campaign_dashboard_app.py", "w") as f:
    f.write(app_code)

# Create a matching CSV template (no campaign_name)
rows = [
    ["2025-10-01",2,1,"Door",45,320],
    ["2025-10-01",2,1,"Phone",35,320],
    ["2025-10-02",5,1,"Text",28,250],
    ["2025-10-03",7,2,"Door",60,280],
    ["2025-10-03",7,2,"Phone",25,280],
    ["2025-10-04",8,2,"Text",30,300],
    ["2025-10-01",3,1,"Door",20,260],
    ["2025-10-02",3,1,"Text",15,260],
    ["2025-10-02",4,1,"Door",22,240],
    ["2025-10-02",6,1,"Phone",18,275],
    ["2025-10-03",9,2,"Door",34,310],
    ["2025-10-03",10,2,"Phone",26,290],
    ["2025-10-04",11,2,"Text",19,305],
    ["2025-10-04",12,2,"Door",33,315],
]
cols = ["date","district","pass_number","contact_method","attempts","target_attempts"]
df = pd.DataFrame(rows, columns=cols)
csv_path = "/mnt/data/wfp_dashboard_template_no_campaign.csv"
df.to_csv(csv_path, index=False)

csv_path
