
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import os

st.set_page_config(page_title="WFP Campaign Dashboard", layout="wide")
st.title("WFP Campaign Dashboard — District Attempts & Support IDs")

st.caption(
    "This app reads a CSV committed in the repo (no uploads). "
    "Expected columns (case/spacing tolerant): "
    "`district, tier, attempts, target_doors, 1, 2, 3, 4, 5`."
)
# --- add these near your other paths ---
NATE_CSV = (APP_DIR / "dashboard_wfp_nate.csv") if (APP_DIR / "dashboard_wfp_nate.csv").exists() \
           else (APP_DIR.parent / "dashboard_wfp_nate.csv")

@st.cache_data
def load_precincts_csv_generic(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Nate race CSV not found at '{path}'. Commit 'dashboard_wfp_nate.csv' "
            "to the repo (root or next to the app)."
        )
    df = pd.read_csv(path)
    # normalize headers + coerce
    df.columns = (df.columns.astype(str).str.strip()
                              .str.replace("\uFEFF", "", regex=True)
                              .str.lower().str.replace(" ", "_"))
    required = ["precinct","attempts","target_doors","1","2","3","4","5"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Nate CSV missing columns: {missing}")

    def to_num(s: pd.Series) -> pd.Series:
        return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")

    df["precinct"]     = df["precinct"].astype("string").str.strip()
    df["attempts"]     = to_num(df["attempts"]).astype("Int64")
    df["target_doors"] = to_num(df["target_doors"]).astype("Int64")
    for c in ["1","2","3","4","5"]:
        df[c] = to_num(df[c]).astype("Int64")
    df["penetration"]  = (df["attempts"] / df["target_doors"]).replace([np.inf,-np.inf], np.nan).clip(upper=1.0)
    return df

APP_DIR = Path(__file__).resolve().parent
DEFAULT_NAME = "dashboard_wfp_template.csv"
CSV_PATH = Path(__file__).resolve().parent / "dashboard_wfp_template.csv"


env_path = os.getenv("WFP_CSV_PATH", "").strip()
if env_path:
    p = Path(env_path)
    CSV_PATH = p if p.is_absolute() else (APP_DIR / p)

if not CSV_PATH.exists():
    CSV_PATH_ALT = APP_DIR.parent / DEFAULT_NAME
    if CSV_PATH_ALT.exists():
        CSV_PATH = CSV_PATH_ALT

def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
          .str.strip()
          .str.replace("\uFEFF", "", regex=True)
          .str.lower()
          .str.replace(" ", "_")
    )
    alias = {"targetdoors":"target_doors","sid_1":"1","sid1":"1","sid_2":"2","sid2":"2","sid_3":"3","sid3":"3","sid_4":"4","sid4":"4","sid_5":"5","sid5":"5"}
    df.rename(columns={k:v for k,v in alias.items() if k in df.columns}, inplace=True)
    return df

def coerce_int_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",","",regex=False).str.strip(), errors="coerce")

@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found at '{path}'. Commit '{path.name}' next to the app file ({APP_DIR}) or set env WFP_CSV_PATH.")
    df = pd.read_csv(path)
    df = normalize_headers(df)
    required = ["district","tier","attempts","target_doors","1","2","3","4","5"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df["district"] = coerce_int_series(df["district"]).astype("Int64")
    df["tier"] = coerce_int_series(df["tier"]).astype("Int64")
    df["attempts"] = coerce_int_series(df["attempts"]).astype("Int64")
    df["target_doors"] = coerce_int_series(df["target_doors"]).astype("Int64")
    for c in ["1","2","3","4","5"]:
        df[c] = coerce_int_series(df[c]).astype("Int64")
    df["tier_label"] = df["tier"].map({1:"Tier 1",2:"Tier 2"}).fillna("Unassigned")
    df["penetration"] = (df["attempts"]/df["target_doors"]).replace([np.inf,-np.inf], np.nan).clip(upper=1.0)
    return df

try:
    df = load_data(CSV_PATH)
except Exception as e:
    st.error(str(e))
    st.stop()

with st.sidebar:
    st.header("Filters")
    districts = sorted([int(x) for x in df["district"].dropna().unique().tolist()])
    sel_districts = st.multiselect("Districts (1–12)", options=districts, default=districts)
    comp_thresh = st.slider("Completion threshold (penetration)", 0.0, 1.0, 0.8, 0.05)

f = df[df["district"].isin(sel_districts)].copy()
if f.empty:
    st.warning("No rows match your filters.")
    st.stop()

total_attempts = int(f["attempts"].sum(skipna=True))
total_targets  = int(f["target_doors"].sum(skipna=True))
avg_pen = float(f["penetration"].mean(skipna=True)) if not f.empty else np.nan

c1, c2, c3 = st.columns(3)
c1.metric("Total Attempts", f"{total_attempts:,}")
c2.metric("Total Target Doors", f"{total_targets:,}")
c3.metric("Avg. Penetration", f"{avg_pen:.1%}" if pd.notnull(avg_pen) else "—")

st.markdown("### District Summary")
summary = (
    f.sort_values("district")
     .assign(penetration_pct=(f["penetration"]*100).round(1).astype(str) + "%",
             status=np.where(f["penetration"]>=comp_thresh,"Complete","In Progress"))
     [["district","tier_label","attempts","target_doors","penetration_pct","status","1","2","3","4","5"]]
     .rename(columns={"1":"SID 1","2":"SID 2","3":"SID 3","4":"SID 4","5":"SID 5"})
)
st.dataframe(summary, use_container_width=True)

st.markdown("### Support Distribution (1–5)")
support_melt = f.melt(id_vars=["district"], value_vars=["1","2","3","4","5"], var_name="SID", value_name="count").fillna(0)
pivot = support_melt.pivot_table(index="district", columns="SID", values="count", aggfunc="sum").fillna(0)
st.bar_chart(pivot)

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

st.caption(f"Reading data from: {CSV_PATH} — Commit a new file with the same name to update.")
