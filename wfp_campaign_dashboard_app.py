
import streamlit as st
import pandas as pd
import numpy as np

# ------------------------------
# WFP Campaign Dashboard
# ------------------------------

st.set_page_config(page_title="WFP Campaign Dashboard", layout="wide")

st.title("WFP Campaign Dashboard")

st.caption(
    "Upload a tidy CSV with columns: "
    "`date, campaign_name, district, precinct_id, precinct_name, pass_number, contact_method, attempts, target_attempts`. "
    "Dates should be YYYY-MM-DD."
)

# File uploader
uploaded = st.file_uploader("Upload data CSV", type=["csv"])

# Tier mapping by district (only two tiers)
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
    # Basic schema checks
    required = [
        "date", "campaign_name", "district", "precinct_id", "precinct_name",
        "pass_number", "contact_method", "attempts", "target_attempts"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    # Parse types
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # Coerce numeric
    for c in ["district", "pass_number", "attempts", "target_attempts"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Map Tier
    df["tier"] = df["district"].apply(map_tier)
    # Clean contact_method
    df["contact_method"] = df["contact_method"].astype(str).str.strip().str.title()
    # Clean precinct fields
    df["precinct_id"] = df["precinct_id"].astype(str).str.strip()
    df["precinct_name"] = df["precinct_name"].astype(str).str.strip()
    return df

if uploaded is None:
    st.info("⬆️ Upload your CSV to get started.")
    st.stop()

# Load
try:
    df = load_data(uploaded)
except Exception as e:
    st.error(str(e))
    st.stop()

# Filters (Districts, Pass, Method, Precinct)
with st.sidebar:
    st.header("Filters")
    all_districts = sorted([d for d in df["district"].dropna().unique().tolist()])
    sel_districts = st.multiselect(
        "Districts (1–12)", 
        options=list(range(1,13)), 
        default=all_districts or list(range(1,13))
    )
    all_passes = sorted([int(p) for p in df["pass_number"].dropna().unique().tolist()])
    sel_passes = st.multiselect("Pass Number", options=all_passes, default=all_passes)
    all_methods = sorted(df["contact_method"].dropna().unique().tolist())
    sel_methods = st.multiselect("Contact Method", options=all_methods, default=all_methods)
    # Precinct filter choices depend on selected districts
    precinct_opts = (
        df.loc[df["district"].isin(sel_districts), ["precinct_id", "precinct_name"]]
          .drop_duplicates()
          .assign(label=lambda x: x["precinct_id"] + " — " + x["precinct_name"])
          .sort_values("label")
    )
    sel_precinct_labels = st.multiselect(
        "Precincts (optional)",
        options=precinct_opts["label"].tolist(),
        default=[]
    )
    comp_thresh = st.slider("Completion threshold (penetration)", 0.0, 1.0, 0.8, 0.05)

# Build mask
base_mask = (
    df["district"].isin(sel_districts) &
    df["pass_number"].isin(sel_passes) &
    df["contact_method"].isin(sel_methods)
)

# If any precincts selected, filter those
if len(sel_precinct_labels) > 0:
    # Parse back to id + name
    split_map = dict(zip(precinct_opts["label"], zip(precinct_opts["precinct_id"], precinct_opts["precinct_name"])))
    selected_pairs = {split_map[label] for label in sel_precinct_labels}
    p_mask = df[["precinct_id", "precinct_name"]].apply(tuple, axis=1).isin(selected_pairs)
    mask = base_mask & p_mask
else:
    mask = base_mask

f = df.loc[mask].copy()

if f.empty:
    st.warning("No rows match your filters.")
    st.stop()

# --------------------------
# Aggregation helpers
# --------------------------
def penetration_status(frame, group_cols, comp_threshold):
    agg = (
        frame.groupby(group_cols, dropna=False)
             .agg(attempts=("attempts", "sum"), target_attempts=("target_attempts", "max"))
             .reset_index()
    )
    agg["penetration"] = (agg["attempts"] / agg["target_attempts"]).replace([np.inf, -np.inf], np.nan)
    agg["penetration"] = agg["penetration"].clip(upper=1.0)
    agg["status"] = np.where(agg["penetration"] >= comp_threshold, "Complete", "In Progress")
    return agg

# --------------------------
# KPI row
# --------------------------
group_cols_method = ["district", "tier", "pass_number", "contact_method"]
agg_method = penetration_status(f, group_cols_method, comp_thresh)

total_attempts = int(agg_method["attempts"].sum())
total_targets = int(agg_method["target_attempts"].sum())
avg_pen = agg_method["penetration"].mean()

c1, c2, c3 = st.columns(3)
c1.metric("Total Attempts (filtered)", f"{total_attempts:,}")
c2.metric("Total Targets (max by group)", f"{total_targets:,}")
c3.metric("Avg. Penetration", f"{avg_pen:.1%}" if pd.notnull(avg_pen) else "—")

# --------------------------
# District x Pass x Method
# --------------------------
st.markdown("### Status by District × Pass × Method")
st.dataframe(
    agg_method.sort_values(["district", "pass_number", "contact_method"])
              .assign(penetration_pct=(agg_method["penetration"]*100).round(1).astype(str) + "%")
              [["district", "tier", "pass_number", "contact_method", "attempts", "target_attempts", "penetration_pct", "status"]],
    use_container_width=True
)

# --------------------------
# District Summary (roll up across methods)
# --------------------------
st.markdown("### District Summary (all methods combined)")
agg_district = penetration_status(f, ["district", "tier", "pass_number"], comp_thresh)
st.dataframe(
    agg_district.sort_values(["district", "pass_number"])
                .assign(penetration_pct=(agg_district["penetration"]*100).round(1).astype(str) + "%")
                [["district", "tier", "pass_number", "attempts", "target_attempts", "penetration_pct", "status"]],
    use_container_width=True
)

# --------------------------
# NEW: Precinct Summary (all methods combined)
# --------------------------
st.markdown("### Precinct Summary (all methods combined)")
agg_precinct = penetration_status(f, ["district", "tier", "pass_number", "precinct_id", "precinct_name"], comp_thresh)
st.dataframe(
    agg_precinct.sort_values(["district", "pass_number", "precinct_id"])
                .assign(penetration_pct=(agg_precinct["penetration"]*100).round(1).astype(str) + "%")
                [["district", "tier", "pass_number", "precinct_id", "precinct_name", "attempts", "target_attempts", "penetration_pct", "status"]],
    use_container_width=True
)

# --------------------------
# NEW: Precinct × Method (granular view)
# --------------------------
st.markdown("### Precinct × Method (granular view)")
agg_precinct_method = penetration_status(f, ["district", "tier", "pass_number", "precinct_id", "precinct_name", "contact_method"], comp_thresh)
st.dataframe(
    agg_precinct_method.sort_values(["district", "pass_number", "precinct_id", "contact_method"])
                       .assign(penetration_pct=(agg_precinct_method["penetration"]*100).round(1).astype(str) + "%")
                       [["district", "tier", "pass_number", "precinct_id", "precinct_name", "contact_method", "attempts", "target_attempts", "penetration_pct", "status"]],
    use_container_width=True
)

# --------------------------
# Downloads
# --------------------------
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

st.download_button(
    "Download Precinct Summary (CSV)",
    data=agg_precinct.to_csv(index=False).encode("utf-8"),
    file_name="wfp_precinct_summary.csv",
    mime="text/csv"
)

st.download_button(
    "Download Precinct × Method (CSV)",
    data=agg_precinct_method.to_csv(index=False).encode("utf-8"),
    file_name="wfp_precinct_by_method.csv",
    mime="text/csv"
)

st.caption("Tiers auto-assigned from districts — Tier 1: {2,5,7,8}; Tier 2: {3,4,6,9,10,11,12}; others → Unassigned.")
