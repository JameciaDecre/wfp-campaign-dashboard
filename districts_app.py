import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Rohit - Districts", layout="wide")
st.title("WFP Campaign — Districts")

# ---------------- file location ----------------
APP_DIR = Path(__file__).resolve().parent
def first_existing(*paths: Path) -> Path:
    for p in paths:
        if p.exists():
            return p
    return paths[0]

DISTRICT_CSV = first_existing(
    APP_DIR / "dashboard_wfp_template.csv",
    APP_DIR.parent / "dashboard_wfp_template.csv"
)

# ---------------- helpers ----------------
TIER1 = {2, 5, 7, 8}
TIER2 = {3, 4, 6, 9, 10, 11, 12}

def tier_from_district(d):
    try:
        d = int(d)
    except Exception:
        return "Unassigned"
    if d in TIER1:
        return "Tier 1"
    if d in TIER2:
        return "Tier 2"
    return "Unassigned"

def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace("\uFEFF", "", regex=True)
        .str.lower()
        .str.replace(" ", "_")
    )
    alias = {
        "targetdoors": "target_doors",
        "sid_1": "1", "sid1": "1",
        "sid_2": "2", "sid2": "2",
        "sid_3": "3", "sid3": "3",
        "sid_4": "4", "sid4": "4",
        "sid_5": "5", "sid5": "5",
    }
    df.rename(columns={k: v for k, v in alias.items() if k in df.columns}, inplace=True)
    return df

def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")

def kpis(frame: pd.DataFrame):
    total_attempts = int(frame["attempts"].sum(skipna=True))
    total_targets = int(frame["target_doors"].sum(skipna=True))
    avg_pen = float(frame["penetration"].mean(skipna=True)) if not frame.empty else np.nan
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Attempts", f"{total_attempts:,}")
    c2.metric("Total Target Doors", f"{total_targets:,}")
    c3.metric("Avg. Penetration", f"{avg_pen:.1%}" if pd.notnull(avg_pen) else "—")

def support_bar(frame: pd.DataFrame):
    st.markdown("### Support Distribution (1–5)")
    melt = frame.melt(id_vars=["district"], value_vars=["1","2","3","4","5"], var_name="SID", value_name="count").fillna(0)
    pivot = melt.pivot_table(index="district", columns="SID", values="count", aggfunc="sum").fillna(0)
    st.bar_chart(pivot)

# ---------------- loader ----------------
@st.cache_data
def load_districts_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"District CSV not found at '{path}'. Commit 'dashboard_wfp_template.csv' next to this app or at repo root."
        )
    df = pd.read_csv(path)
    df = normalize_headers(df)
    required = ["district","tier","attempts","target_doors","1","2","3","4","5"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"District CSV missing columns: {missing}")

    df["district"] = to_num(df["district"]).astype("Int64")
    df["tier"] = to_num(df["tier"]).astype("Int64")
    df["attempts"] = to_num(df["attempts"]).astype("Int64")
    df["target_doors"] = to_num(df["target_doors"]).astype("Int64")
    for c in ["1","2","3","4","5"]:
        df[c] = to_num(df[c]).astype("Int64")

    df["tier_label"] = df["district"].apply(tier_from_district)  # derive from district
    df["penetration"] = (df["attempts"] / df["target_doors"]).replace([np.inf,-np.inf], np.nan).clip(upper=1.0)
    return df

# ---------------- UI ----------------
st.caption("Reads a committed CSV (no uploads). File: dashboard_wfp_template.csv")

try:
    df = load_districts_csv(DISTRICT_CSV)
except Exception as e:
    st.error(str(e))
    st.stop()

with st.sidebar:
    st.header("Filters — Districts")
    dopts = sorted([int(x) for x in df["district"].dropna().unique().tolist()])
    sel_districts = st.multiselect("Districts (1–12)", options=dopts, default=dopts)
    comp_thresh = st.slider("Completion threshold (penetration)", 0.0, 1.0, 0.8, 0.05)

f = df[df["district"].isin(sel_districts)].copy()
if f.empty:
    st.warning("No rows match your filters.")
    st.stop()

kpis(f)

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
support_bar(f)

st.download_button(
    "Download District Summary (CSV)",
    data=summary.to_csv(index=False).encode("utf-8"),
    file_name="wfp_district_summary_with_support.csv",
    mime="text/csv"
)

st.caption(f"Reading district data from: {DISTRICT_CSV}")
