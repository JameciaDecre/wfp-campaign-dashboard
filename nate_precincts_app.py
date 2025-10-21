# nate_precincts_app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="WFP — Precincts (Nate Race)", layout="wide")
st.title("WFP Campaign — Precincts (Nate Race)")

# ------------------------------------------------------------
# File resolution (next to app, then repo root)
# ------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent

def first_existing(*paths: Path) -> Path:
    for p in paths:
        if p.exists():
            return p
    return paths[0]

NATE_CSV = first_existing(
    APP_DIR / "dashboard_wfp_nate.csv",
    APP_DIR.parent / "dashboard_wfp_nate.csv"
)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
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
        "targets": "target_doors",
        "door_targets": "target_doors",
        "precinctcode": "precinct",
        "precinct_code": "precinct",
        "precinctid": "precinct_id",
        "precinctname": "precinct_name",
        "phone": "phone_attempts",
        "calls": "phone_attempts",
        "text": "text_attempts",
        "door": "door_attempts",
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

def pick_display_precinct(df: pd.DataFrame) -> pd.Series:
    cols = df.columns.tolist()
    if "precinct" in cols:
        return df["precinct"].astype("string").str.strip()
    id_col = "precinct_id" if "precinct_id" in cols else None
    name_col = "precinct_name" if "precinct_name" in cols else None
    if id_col and name_col:
        return (df[id_col].astype("string").str.strip() + " — " + df[name_col].astype("string").str.strip())
    fallback = next((c for c in cols if "precinct" in c), None)
    if fallback:
        return df[fallback].astype("string").str.strip()
    return pd.Series([f"Precinct {i+1}" for i in range(len(df))], index=df.index, dtype="string")

def available_support_columns(df: pd.DataFrame) -> list:
    return [c for c in ["1", "2", "3", "4", "5"] if c in df.columns]

def available_attempt_metrics(df: pd.DataFrame) -> list:
    candidates = []
    for c in ["attempts", "phone_attempts", "door_attempts", "text_attempts", "total_attempts"]:
        if c in df.columns:
            candidates.append(c)
    return candidates

# ------------------------------------------------------------
# Loader
# ------------------------------------------------------------
@st.cache_data
def load_precincts(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found at '{path}'. Commit 'dashboard_wfp_nate.csv' next to this app or at repo root.")
    df = pd.read_csv(path)
    df = normalize_headers(df)
    for col in ["attempts", "phone_attempts", "door_attempts", "text_attempts", "total_attempts", "target_doors", "1", "2", "3", "4", "5"]:
        if col in df.columns:
            df[col] = to_num(df[col]).astype("Float64")
    df["__precinct_display"] = pick_display_precinct(df)
    metrics = available_attempt_metrics(df)
    if not metrics:
        raise ValueError("No attempts metric found. Include one of: attempts, phone_attempts, door_attempts, text_attempts, total_attempts.")
    return df

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.caption("Reads a committed CSV (no uploads). File: dashboard_wfp_nate.csv")

try:
    df = load_precincts(NATE_CSV)
except Exception as e:
    st.error(str(e))
    st.stop()

with st.sidebar:
    st.header("Filters & Settings — Precincts (Nate)")
    metric_options = available_attempt_metrics(df)
    default_metric = "phone_attempts" if "phone_attempts" in metric_options else ("attempts" if "attempts" in metric_options else metric_options[0])
    attempts_metric = st.selectbox("Attempts metric", options=metric_options, index=metric_options.index(default_metric))
    precinct_opts = df["__precinct_display"].dropna().astype(str).unique().tolist()
    sel_precincts = st.multiselect("Precincts", options=sorted(precinct_opts), default=sorted(precinct_opts))
    comp_thresh = st.slider("Completion threshold (penetration)", 0.0, 1.0, 0.8, 0.05)

f = df[df["__precinct_display"].isin(sel_precincts)].copy()
if f.empty:
    st.warning("No rows match your filters.")
    st.stop()

if "target_doors" in f.columns:
    f["__attempts"] = f[attempts_metric].fillna(0)
    f["__penetration"] = (f["__attempts"] / f["target_doors"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).clip(upper=1.0)
else:
    f["__attempts"] = f[attempts_metric].fillna(0)
    f["__penetration"] = np.nan

total_attempts = int(pd.to_numeric(f["__attempts"], errors="coerce").fillna(0).sum())
c1, c2, c3 = st.columns(3)
c1.metric("Total Attempts", f"{total_attempts:,}")

if "target_doors" in f.columns:
    total_targets = int(pd.to_numeric(f["target_doors"], errors="coerce").fillna(0).sum())
    avg_pen = float(f["__penetration"].mean(skipna=True)) if not f["__penetration"].dropna().empty else np.nan
    c2.metric("Total Target Doors", f"{total_targets:,}")
    c3.metric("Avg. Penetration", f"{avg_pen:.1%}" if pd.notnull(avg_pen) else "—")
else:
    c2.metric("Total Target Doors", "—")
    c3.metric("Avg. Penetration", "—")

st.caption(f"Attempts metric in use: {attempts_metric}")

cols_base = ["__precinct_display", "__attempts"]
if "target_doors" in f.columns:
    cols_base.append("target_doors")

sid_cols = available_support_columns(f)
cols_sids = sid_cols.copy()

if "target_doors" in f.columns:
    f["__penetration_pct"] = (f["__penetration"] * 100).round(1)
    f["__status"] = np.where(f["__penetration"] >= comp_thresh, "Complete", "In Progress")
    cols_base.extend(["__penetration_pct", "__status"])

summary_cols = cols_base + cols_sids
summary = (
    f.sort_values("__precinct_display")[summary_cols].rename(
        columns={
            "__precinct_display": "precinct",
            "__attempts": "attempts",
            "__penetration_pct": "penetration_pct",
            "__status": "status",
            "1": "SID 1", "2": "SID 2", "3": "SID 3", "4": "SID 4", "5": "SID 5"
        }
    )
)

if "penetration_pct" in summary.columns:
    summary["penetration_pct"] = summary["penetration_pct"].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "—")

st.markdown("### Precinct Summary — Nate Race")
st.dataframe(summary, use_container_width=True)

if sid_cols:
    st.markdown("### Support Distribution (1–5)")
    melt = f.melt(id_vars=["__precinct_display"], value_vars=sid_cols, var_name="SID", value_name="count").fillna(0)
    pivot = melt.pivot_table(index="__precinct_display", columns="SID", values="count", aggfunc="sum").fillna(0)
    pivot.index.name = "precinct"
    st.bar_chart(pivot)

st.download_button(
    "Download Nate Precinct Summary (CSV)",
    data=summary.to_csv(index=False).encode("utf-8"),
    file_name="wfp_nate_precinct_summary.csv",
    mime="text/csv"
)

clean_export_cols = ["__precinct_display", attempts_metric]
if "target_doors" in f.columns:
    clean_export_cols.append("target_doors")
clean_export_cols += sid_cols
clean_export = f[clean_export_cols].rename(columns={"__precinct_display": "precinct"})
st.download_button(
    "Download Cleaned Source (CSV)",
    data=clean_export.to_csv(index=False).encode("utf-8"),
    file_name="wfp_nate_cleaned_source.csv",
    mime="text/csv"
)

st.caption(f"Reading Nate data from: {NATE_CSV}")
