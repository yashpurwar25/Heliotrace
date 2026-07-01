"""
HELIOTRACE — Solar Flare Nowcasting Operational Dashboard
Run with: streamlit run heliotrace_ops_dashboard.py

Edit DATA_PATH / CATALOG_PATH below to match your files.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# CONFIG — edit these
# ---------------------------------------------------------------------------
DATA_PATH = "/Users/yashpurwar/data/aligned/aligned_all_days_final_v3.parquet"
CATALOG_PATH = "/Users/yashpurwar/heliotrace_master_flare_catalog.csv"

R_LOG_THRESHOLD = 1.743
TSS_VALUE = 0.844
H3_MAE_MIN = 18.1          # honest current value — edit if you retrain
H4_MAE_LOG = 0.886         # Model D isolated-head value
LATENCY_MS = 2.26

st.set_page_config(
    page_title="HELIOTRACE | Aditya-L1 Solar Flare Nowcasting",
    page_icon="\u2600\ufe0f",
    layout="wide",
)

# ---------------------------------------------------------------------------
# STYLE — dark ops-console theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp { background-color: #0a0e17; }
    .block-container { padding-top: 1rem; padding-bottom: 1rem; max-width: 100%; }
    h1, h2, h3, p, span, label { color: #e6e9ef !important; }

    .header-box {
        display: flex; justify-content: space-between; align-items: center;
        border-bottom: 3px solid #f5a623; padding-bottom: 12px; margin-bottom: 18px;
    }
    .badge {
        background-color: #1b7a3d; color: white; padding: 8px 18px;
        border-radius: 8px; font-weight: 600; font-size: 14px;
    }
    .alert-purple {
        background-color: #3a2159; border-left: 5px solid #a259ff;
        padding: 12px 20px; border-radius: 6px; margin-bottom: 8px; font-weight: 500;
    }
    .alert-red {
        background-color: #4a1e1e; border-left: 5px solid #ff4d4d;
        padding: 12px 20px; border-radius: 6px; margin-bottom: 8px; font-weight: 500;
    }
    .alert-purple-dim {
        background-color: #241733; border-left: 5px solid #7c4dff;
        padding: 10px 20px; border-radius: 6px; margin-bottom: 8px; font-size: 14px; opacity: 0.9;
    }
    .fsm-box {
        background-color: #0f1420; border: 1px solid #2a3040; border-radius: 10px;
        padding: 20px; text-align: center;
    }
    .fsm-state {
        display: inline-block; padding: 10px 30px; border-radius: 8px;
        border: 2px solid #4caf50; color: #4caf50 !important; font-weight: 700;
        font-size: 20px; letter-spacing: 2px;
    }
    .metric-box {
        background-color: #0f1420; border: 1px solid #2a3040; border-radius: 10px;
        padding: 14px; text-align: center;
    }
    .metric-label { color: #9aa4b5 !important; font-size: 12px; }
    .metric-value { font-size: 24px; font-weight: 700; }
    .green-dot { color: #4caf50 !important; }
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
    header[data-testid="stHeader"] { background-color: #0a0e17; }
    header[data-testid="stHeader"] { display: none; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
@st.cache_data
def load_data(data_path, catalog_path):
    df = pd.read_parquet(data_path) if data_path.endswith(".parquet") else pd.read_csv(data_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    try:
        catalog = pd.read_csv(catalog_path, parse_dates=["start_time", "peak_time", "stop_time"])
    except FileNotFoundError:
        catalog = pd.DataFrame()
    return df, catalog

try:
    df, catalog = load_data(DATA_PATH, CATALOG_PATH)
except FileNotFoundError as e:
    st.error(f"Could not load data files: {e}\n\nEdit DATA_PATH / CATALOG_PATH at the top of this script.")
    st.stop()

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
h1, h2 = st.columns([5, 1])
with h1:
    st.markdown("""
    <div class="header-box">
        <div>
            <span style="font-size:32px;">\u2600\ufe0f</span>
            <span style="font-size:32px; font-weight:800; color:#f5a623 !important;"> HELIOTRACE</span>
            <span style="font-size:16px; color:#9aa4b5 !important;"> | Aditya-L1 Solar Flare Nowcasting</span>
            <br>
            <span style="font-size:12px; color:#6b7386 !important;">
                Bharatiya Antariksh Hackathon 2025 &middot; PS-15 &middot; SoLEXS SDD2 + HEL1OS
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
with h2:
    st.markdown('<div class="badge">\u2699\ufe0f Neural Net (H1+H3+H4)</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# CONTROLS
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns([1.2, 3, 1, 1])
with c1:
    available_dates = sorted(df["datetime"].dt.date.unique())
    sel_date = st.selectbox("Date", available_dates, index=len(available_dates) - 1)
with c2:
    window_pct = st.slider("Time window", 0, 100, (0, 100))
with c3:
    h1_threshold = st.slider("H1 threshold", 0.1, 0.9, 0.5, 0.05)
with c4:
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("\u25b6 Run Inference", use_container_width=True)

# ---------------------------------------------------------------------------
# FILTER TO SELECTED DATE + WINDOW
# ---------------------------------------------------------------------------
day_df = df[df["datetime"].dt.date == sel_date].reset_index(drop=True)
n = len(day_df)
if n == 0:
    st.warning("No data for selected date.")
    st.stop()

lo = int(n * window_pct[0] / 100)
hi = max(lo + 1, int(n * window_pct[1] / 100))
view = day_df.iloc[lo:hi]

# ---------------------------------------------------------------------------
# COMPUTE ALERT SIGNALS
# ---------------------------------------------------------------------------
has_hardness = "hardness_ratio" in view.columns
if has_hardness:
    r_log = np.log10(view["hardness_ratio"].clip(lower=1e-6))
else:
    r_log = pd.Series(np.zeros(len(view)), index=view.index)

occ_mask = r_log > np.log10(R_LOG_THRESHOLD)
n_occ = int(occ_mask.sum())

if "forecast_5min" in view.columns:
    p_flare_series = view["forecast_5min"]
else:
    p_flare_series = pd.Series(np.zeros(len(view)), index=view.index)
p_peak = float(p_flare_series.max()) if len(p_flare_series) else 0.0
alert_triggered = p_peak >= 0.75

if "flare_label" in view.columns and len(view):
    current_label = view["flare_label"].iloc[-1]
else:
    current_label = "quiet"
fsm_color = {"quiet": "#4caf50", "uncertain": "#ffb300", "weak": "#ff7043", "high_confidence": "#ff4d4d"}.get(current_label, "#4caf50")

n_imp = int((view["hxr_derivative_smooth"] > view["hxr_derivative_smooth"].quantile(0.9)).sum()) if "hxr_derivative_smooth" in view.columns else 0

# ---------------------------------------------------------------------------
# ALERT BANNERS
# ---------------------------------------------------------------------------
if n_occ > 0:
    st.markdown(f'<div class="alert-purple">\U0001f4e1 OCCULTATION CANDIDATE DETECTED — HXR signal without SXR counterpart ({n_occ} windows)</div>', unsafe_allow_html=True)

if alert_triggered:
    st.markdown(f'<div class="alert-red">\u26a0\ufe0f ALERT STATE REACHED — P(flare) \u2265 0.75</div>', unsafe_allow_html=True)

st.markdown(f'<div class="alert-purple-dim">\u25cf OLR triggered {n_occ} rows &middot; IMP_PRECURSOR {n_imp} rows &middot; R_log &gt; {R_LOG_THRESHOLD}</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# FSM STATE + VALIDATION METRICS
# ---------------------------------------------------------------------------
mc1, mc2 = st.columns([1, 4])

with mc1:
    st.markdown(f"""
    <div class="fsm-box">
        <div style="color:#9aa4b5 !important; font-size:13px; margin-bottom:10px;">Current FSM State</div>
        <div class="fsm-state" style="border-color:{fsm_color}; color:{fsm_color} !important;">
            {current_label.upper().replace('_', ' ')}
        </div>
    </div>
    """, unsafe_allow_html=True)

with mc2:
    st.markdown('<div style="color:#9aa4b5 !important; font-size:13px; margin-bottom:6px;">Validation Metrics (Oct 10\u201311 2024)</div>', unsafe_allow_html=True)
    mcols = st.columns(6)
    metrics = [
        ("TSS", f"{TSS_VALUE:.3f}", True),
        ("P(flare) peak", f"{p_peak:.3f}", p_peak < 0.75),
        ("H4 MAE", f"{H4_MAE_LOG:.3f} log", H4_MAE_LOG < 1.0),
        ("H3 MAE", f"{H3_MAE_MIN:.1f} min", H3_MAE_MIN < 5),
        ("OCC triggers", f"{n_occ}", True),
        ("IMP triggers", f"{n_imp}", True),
    ]
    for col, (label, value, is_good) in zip(mcols, metrics):
        dot = "\U0001f7e2" if is_good else "\U0001f7e0"
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value} {dot}</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# TIME SERIES PANELS
# ---------------------------------------------------------------------------
fig = make_subplots(
    rows=4, cols=1, shared_xaxes=True,
    subplot_titles=(
        "SoLEXS SDD2 — Soft X-Ray (log counts)",
        "HEL1OS CdTe 2\u201390 keV — Hard X-Ray (log counts)",
        f"R_log Hardness Ratio (CdTe/SoLEXS)  threshold={R_LOG_THRESHOLD}",
        "H1 Nowcast — P(flare in progress)",
    ),
    vertical_spacing=0.06,
    row_heights=[1, 1, 1, 0.8],
)

if "pseudo_flux_solexs_log10" in view.columns:
    fig.add_trace(go.Scatter(x=view["datetime"], y=view["pseudo_flux_solexs_log10"],
                               line=dict(color="#4fc3f7", width=0.8), fill="tozeroy",
                               fillcolor="rgba(79,195,247,0.25)", showlegend=False), row=1, col=1)

if "hel1os_broadband_czt" in view.columns:
    hel1os_log = np.log10(view["hel1os_broadband_czt"].clip(lower=1))
    fig.add_trace(go.Scatter(x=view["datetime"], y=hel1os_log,
                               line=dict(color="#ff9800", width=0.8), fill="tozeroy",
                               fillcolor="rgba(255,152,0,0.25)", showlegend=False), row=2, col=1)

fig.add_trace(go.Scatter(x=view["datetime"], y=r_log,
                           line=dict(color="#a259ff", width=0.8), fill="tozeroy",
                           fillcolor="rgba(162,89,255,0.25)", showlegend=False), row=3, col=1)
fig.add_hline(y=np.log10(R_LOG_THRESHOLD), line=dict(color="#ff4d4d", dash="dash", width=1),
              annotation_text=f"R_log threshold ({R_LOG_THRESHOLD})", row=3, col=1)

fig.add_trace(go.Scatter(x=view["datetime"], y=p_flare_series,
                           line=dict(color="#66bb6a", width=1.2), showlegend=False), row=4, col=1)
fig.add_hline(y=h1_threshold, line=dict(color="#ffb300", dash="dot", width=1),
              annotation_text=f"H1 threshold ({h1_threshold})", row=4, col=1)

fig.update_layout(
    height=900, template="plotly_dark",
    plot_bgcolor="#0a0e17", paper_bgcolor="#0a0e17",
    margin=dict(l=10, r=10, t=40, b=10),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# MASTER FLARE CATALOG
# ---------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("Master Flare Catalog")
if len(catalog) > 0:
    display_cols = [c for c in ["event_id", "start_time", "peak_time", "stop_time",
                                  "duration_min", "severity", "intensity_class", "peak_log_flux"]
                     if c in catalog.columns]
    st.dataframe(catalog[display_cols].sort_values("start_time", ascending=False),
                 use_container_width=True, height=280)
else:
    st.info("No catalog loaded — run build_master_catalog() and point CATALOG_PATH at the CSV.")

st.caption("HELIOTRACE — PA-MAST-Lite-TL &middot; ISRO Bharatiya Antariksh Hackathon 2025 (PS-15) "
           f"&middot; Inference latency: {LATENCY_MS} ms")
