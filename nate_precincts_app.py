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
        .str.replace("\uFEFF", "", regex=True)  # BOM
        .str.lower()
        .str.replace(" ", "_")
    )
    # alias map for common variants
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
    # Allow commas/spaces; keep NaN
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")

def pick_display_precinct(df: pd.DataFrame) -> pd.Series:
    # prefer explicit "precinct"; else combine id+name; else fallback to any column containing 'precinct'
    cols = df.columns.tolist()
    if "precinct" in cols:
        return df["precinct"].astype("string").str.strip()
    id_col = "precinct_id" if "precinct_id" in cols else None
    name_col = "precinct_name" if "precinct_name" in cols else None
    if id_col and name_col:
        return (df[id_col].astype("string").str.strip() + " — " + df[name_col].astype("string").str.strip())
    # last resort: find first column containing 'precinct'
    fallback = next((c for c in cols if "precinct" in c), None)
    if fallback:
        return df[fallback].astype("string").str.strip()
    # if nothing, synthesize index
    return pd.Series([f"Precinct {i+1}" for i in range(len(df))], index=df.index, dtype="string")

def available_support_columns(df: pd.DataFrame) -> list:
    return [c for c in ["1","2","3","4","5"] if c in df.columns]

def available_attempt_metrics(df: pd.DataFrame) -> list:
    # any of these will be offered in the sidebar
    candidates = []
    for c in ["attempts", "phone_attempts", "door_attempts", "text_attempts", "total_attempts"]:
        if c in df.columns:
            candidates.append(c)
    return candidates

# ------------------------------------------------------------
# Loader (schema-flexible)
# ------------------------------------------------------------
@st.cache_data
def load_precincts(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"CSV not found at '{path}'. Commit 'dashboard_wfp_nate.csv' next to this app or at repo root."
        )
    df = pd.read_csv(path)
    df = normalize_headers(df)

    # Coerce numeric fields if present
    for col in ["attempts", "phone_attempts", "door_attempts", "text_attempts", "total_attempts",
                "target_doors", "1","2","3","4","5"]:
        if col in df.columns:
            df[col] = to_num(df[col]).astype("Float64")  # allow NaN; cast to float; display as Int where needed

    # Compute a display precinct column
    df["__precinct_display"] = pick_display_precinct(df)

    # Determine default metric for attempts
    metrics = available_attempt_metrics(df)
    if not metrics:
        # minimal requirement: we need at least one attempts-like column
        raise ValueError(
            "No attempts metric found. Include one of: attempts, phone_attempts, door_attempts, text_attempts, total_attempts."
        )

    # Penetration will be computed later based on selected metric & target_doors (if present)
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

# Sidebar — choose metric and filters
with st.sidebar:
    st.header("Filters & Settings — Precincts (Nate)")
    # let user pick which metric counts as "Attempts"
    metric_options = available_attempt_metrics(df)
    # prefer phone_attempts if available, else attempts, else first
    default_metric = "phone_attempts" if "phone_attempts" in metric_options else ("attempts" if "attempts" in metric_options else metric_options[0])
    attempts_metric = st.selectbox("Attempts metric", options=metric_options, index=metric_options.index(default_metric))
    # precinct selection
    precinct_opts = df["__precinct_display"].dropna().astype(str).unique().tolist()
    sel_precincts = st.multiselect("Precincts", options=sorted(precinct_opts), default=sorted(precinct_opts))
    comp_thresh = st.slider("Completion threshold (penetration)", 0.0, 1.0, 0.8, 0.05)

# Filter
f = df[df["__precinct_display"].isin(sel_precincts)].copy()
if f.empty:
    st.warning("No rows match your filters.")
    st.stop()

# Compute Penetration using selected attempts metric and target_doors if present
if "target_doors" in f.columns:
    f["__attempts"] = f[attempts_metric].fillna(0)
    # avoid division by zero
    f["__penetration"] = (f["__attempts"] / f["target_doors"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).clip(upper=1.0)
else:
    f["__attempts"] = f[attempts_metric].fillna(0)
    f["__penetration"] = np.nan  # will show N/A if no target_doors present

# ---------------- KPIs ----------------
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
    c3.metric("Avg. Pen
