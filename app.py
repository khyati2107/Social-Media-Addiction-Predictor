"""
app.py  —  Social Media Addiction Risk Screener
Rebuilt from scratch. Architecture:
  raw user inputs → feature engineering (matching 04_feature_engineering_shared.py)
  → scaler → clf_xgb.pkl → prediction + probabilities + z-score driver chart.

Model: XGBoost classifier (clf_xgb.pkl) — trained on Nusratt dataset.
6 engineered features: behavioral_intensity_index, psychological_distress_index,
  sleep_disturbance_proxy, age_norm, is_student, usage_hours_sq.

Subscores (sections A–E) are a UX diagnostic layer only.
They are NOT injected into the feature engineering pipeline.
The only subscore used in feature construction is behavioural_intensity
as a proxy for the `conflicts` variable (flagged explicitly to the user).

Class-conditional stats (for driver chart) are computed from features_train.csv
if that file is present; otherwise fall back to conservative hardcoded estimates.
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import pickle
import os
import warnings
import io

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Social Media Addiction Risk Screener",
    page_icon="📱",
    layout="centered",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg-primary:    #1a1033;
    --bg-secondary:  #231744;
    --bg-card:       #2a1d52;
    --accent:        #c084fc;
    --accent-glow:   rgba(192,132,252,0.25);
    --border:        rgba(192,132,252,0.18);
    --border-strong: rgba(192,132,252,0.4);
    --text-primary:  #f3f0ff;
    --text-secondary:#b8a8d8;
    --text-muted:    #7c6da0;
    --green:  #34d399;
    --yellow: #fbbf24;
    --red:    #f87171;
}

html, body, [data-testid="stApp"], .stApp {
    background: var(--bg-primary) !important;
    background-image:
        radial-gradient(ellipse 80% 50% at 15% 10%, rgba(124,58,237,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 85% 80%, rgba(168,85,247,0.12) 0%, transparent 55%) !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
}

#MainMenu, footer, header {visibility: hidden;}
.block-container { padding-top: 2.5rem; padding-bottom: 3rem; max-width: 820px; }
h1 a, h2 a, h3 a, h4 a { display: none !important; }

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span { color: var(--text-secondary) !important; }
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4,
[data-testid="stMarkdownContainer"] strong {
    color: var(--text-primary) !important;
    font-family: 'Syne', sans-serif !important;
}

[data-testid="stSelectbox"] > div > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
}
[data-testid="stSlider"] > div > div > div > div { background: var(--accent) !important; }

[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, #7c3aed, #a855f7) !important;
    border: none !important; color: white !important;
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
    letter-spacing: 0.04em !important; border-radius: 10px !important;
    padding: 0.6rem 1.6rem !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.4) !important;
}
[data-testid="stButton"] button[kind="primary"]:hover {
    box-shadow: 0 6px 28px rgba(124,58,237,0.65) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stButton"] button[kind="secondary"],
[data-testid="stButton"] button:not([kind]) {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-strong) !important;
    color: var(--accent) !important;
    font-family: 'DM Sans', sans-serif !important;
    border-radius: 10px !important;
}

.pill {
    display: inline-block; background: rgba(192,132,252,0.15); color: var(--accent);
    border: 1px solid var(--border-strong); font-size: 12px; font-weight: 600;
    padding: 5px 16px; border-radius: 99px; margin-bottom: 1.2rem; letter-spacing: 0.05em;
}
.section-divider { border: none; border-top: 1px solid var(--border); margin: 2rem 0; }
.section-header {
    background: rgba(192,132,252,0.08); border-left: 4px solid var(--accent);
    border-radius: 0 10px 10px 0; padding: 12px 18px; margin: 1.8rem 0 1rem;
    font-family: 'Syne', sans-serif; font-weight: 700; font-size: .95rem;
    color: var(--accent); letter-spacing: 0.02em;
}

/* Bubble radio — use actual visible number labels, no hidden circles */
div.bubble-wrap div[data-testid="stRadio"] > label { display: none !important; }
div.bubble-wrap div[role="radiogroup"] {
    display: flex !important;
    flex-direction: row !important;
    justify-content: center !important;
    align-items: center !important;
    gap: 12px !important;
    background: transparent !important;
    width: 100% !important;
    padding: 4px 0 !important;
}
div.bubble-wrap div[role="radiogroup"] > label {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    margin: 0 !important;
    padding: 0 !important;
    flex: 0 0 auto !important;
}
/* Hide the default Streamlit radio dot */
div.bubble-wrap div[role="radiogroup"] > label > div:first-child {
    display: none !important;
}
/* Style the label text as a bubble */
div.bubble-wrap div[role="radiogroup"] > label > div:last-child > p {
    width: 48px !important;
    height: 48px !important;
    border-radius: 50% !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    margin: 0 !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    font-family: 'Syne', sans-serif !important;
    cursor: pointer !important;
    box-sizing: border-box !important;
    transition: transform 0.15s ease, background 0.2s ease, box-shadow 0.2s ease !important;
    color: var(--text-primary) !important;
}
div.bubble-wrap div[role="radiogroup"] > label:nth-child(1) > div:last-child > p { background: rgba(52,211,153,0.15) !important; border: 2.5px solid #34d399 !important; }
div.bubble-wrap div[role="radiogroup"] > label:nth-child(2) > div:last-child > p { background: rgba(96,165,250,0.15) !important; border: 2.5px solid #60a5fa !important; }
div.bubble-wrap div[role="radiogroup"] > label:nth-child(3) > div:last-child > p { background: rgba(192,132,252,0.15) !important; border: 2.5px solid #c084fc !important; }
div.bubble-wrap div[role="radiogroup"] > label:nth-child(4) > div:last-child > p { background: rgba(244,114,182,0.15) !important; border: 2.5px solid #f472b6 !important; }
div.bubble-wrap div[role="radiogroup"] > label:nth-child(5) > div:last-child > p { background: rgba(248,113,113,0.15) !important; border: 2.5px solid #f87171 !important; }
div.bubble-wrap div[role="radiogroup"] > label:has(input:checked):nth-child(1) > div:last-child > p
    { background: #34d399 !important; border-color: #34d399 !important; color: #1a1033 !important; box-shadow: 0 0 14px rgba(52,211,153,0.5) !important; }
div.bubble-wrap div[role="radiogroup"] > label:has(input:checked):nth-child(2) > div:last-child > p
    { background: #60a5fa !important; border-color: #60a5fa !important; color: #1a1033 !important; box-shadow: 0 0 14px rgba(96,165,250,0.5) !important; }
div.bubble-wrap div[role="radiogroup"] > label:has(input:checked):nth-child(3) > div:last-child > p
    { background: #c084fc !important; border-color: #c084fc !important; color: #1a1033 !important; box-shadow: 0 0 14px rgba(192,132,252,0.5) !important; }
div.bubble-wrap div[role="radiogroup"] > label:has(input:checked):nth-child(4) > div:last-child > p
    { background: #f472b6 !important; border-color: #f472b6 !important; color: #1a1033 !important; box-shadow: 0 0 14px rgba(244,114,182,0.5) !important; }
div.bubble-wrap div[role="radiogroup"] > label:has(input:checked):nth-child(5) > div:last-child > p
    { background: #f87171 !important; border-color: #f87171 !important; color: #1a1033 !important; box-shadow: 0 0 14px rgba(248,113,113,0.5) !important; }

.bubble-end-labels {
    display: flex !important; justify-content: space-between !important;
    padding: 4px 0 14px !important;
    font-size: 11px !important; font-weight: 700 !important;
    letter-spacing: .08em !important; text-transform: uppercase !important;
    color: var(--text-muted) !important; font-family: 'Syne', sans-serif !important;
}

.result-banner {
    border-radius: 14px; padding: 22px 26px; margin-bottom: 1.5rem;
    border: 1px solid var(--border);
}
.result-banner h2 { margin: 0 0 6px; font-size: 1.5rem; font-family: 'Syne', sans-serif; }
.result-banner p  { margin: 0; font-size: .95rem; color: var(--text-secondary); }

.rec-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 12px; padding: 16px; height: 100%; font-size: .85rem;
}
.rec-title { font-weight: 700; font-size: .9rem; font-family: 'Syne', sans-serif; color: var(--text-primary); }
.rec-desc  { color: var(--text-secondary); margin-top: 6px; }

.footer-note {
    font-size: .72rem; color: var(--text-muted); text-align: center;
    margin-top: 2rem; padding-top: 1rem; border-top: 1px solid var(--border);
}

.q-text {
    font-family: 'DM Sans', sans-serif; font-size: 1rem; font-weight: 500;
    color: var(--text-primary) !important; text-align: center;
    margin: 20px 0 6px; line-height: 1.45;
}

.subscore-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 16px; margin-bottom: 10px;
}
.subscore-title {
    font-family: 'Syne', sans-serif; font-size: .82rem; font-weight: 700;
    color: var(--text-primary); margin-bottom: 6px;
}

.proxy-note {
    background: rgba(251,191,36,0.08); border: 1px solid rgba(251,191,36,0.25);
    border-radius: 8px; padding: 10px 14px; font-size: .78rem;
    color: #b8a8d8; margin: 8px 0 16px; line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
LABEL_NAMES  = ['Low', 'Moderate', 'High']
LABEL_EMOJI  = ['🟢', '🟡', '🔴']
LABEL_COLORS = ['#34d399', '#fbbf24', '#f87171']
LABEL_BG_CSS = [
    'background: rgba(52,211,153,0.12); border-color: #34d399;',
    'background: rgba(251,191,36,0.12);  border-color: #fbbf24;',
    'background: rgba(248,113,113,0.12); border-color: #f87171;',
]

FEAT_FRIENDLY = {
    'behavioral_intensity_index':   '📱 Behavioural Intensity',
    'psychological_distress_index': '🧠 Psychological Distress',
    'sleep_disturbance_proxy':      '😴 Sleep Disturbance',
    'age_norm':                     '👤 Age (normalised)',
    'is_student':                   '🎓 Student Status',
    'usage_hours_sq':               '⏰ Usage Pattern',
}

# Chart styling
CHART_BG        = "#1a1033"
CHART_CARD_BG   = "#2a1d52"
CHART_TEXT      = "#f3f0ff"
CHART_MUTED     = "#7c6da0"
CHART_ACCENT    = "#c084fc"
CHART_GRID      = (0.753, 0.518, 0.988, 0.08)
CHART_SPINE     = (0.753, 0.518, 0.988, 0.15)

# ── Artifact loading ──────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    errors = []
    art = {}

    model_path = "models/clf_xgb.pkl"
    if not os.path.exists(model_path):
        errors.append(f"XGBoost model not found at {model_path}. Run 05_train_classification.py first.")
    else:
        with open(model_path, "rb") as f:
            art["model"] = pickle.load(f)

    for key, path in [
        ("scaler",    "models/scaler_clf.pkl"),
        ("le",        "models/label_encoder.pkl"),
        ("fe_params", "models/fe_params.pkl"),
        ("feat_cols", "models/feature_cols.pkl"),
        ("imputer",   "models/imputer.pkl"),
        ("reg_model",  "models/reg_xgb.pkl"),  
        ("scaler_reg", "models/scaler_reg.pkl"),
    ]:
        if os.path.exists(path):
            with open(path, "rb") as f:
                art[key] = pickle.load(f)
        else:
            errors.append(f"Missing: {path}")

    return art, errors


@st.cache_data
def load_class_cond_stats():
    """
    Load class-conditional feature means and stds from features_train.csv.
    Used for the z-score driver chart.
    Returns dict: {feature: {class_label: (mean, std)}} or None if file missing.
    """
    path = "outputs/features_train.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c not in ("addiction_label", "addicted_score")]
    stats = {}
    for feat in feat_cols:
        stats[feat] = {}
        for label in ["Low", "Moderate", "High"]:
            subset = df.loc[df["addiction_label"] == label, feat].dropna()
            if len(subset) > 0:
                stats[feat][label] = (float(subset.mean()), max(float(subset.std()), 1e-6))
    return stats


# Conservative fallback if features_train.csv is absent.
FALLBACK_CLASS_COND_STATS = {
    'behavioral_intensity_index': {
        'High': (0.720, 0.110), 'Moderate': (0.590, 0.095), 'Low': (0.390, 0.100),
    },
    'psychological_distress_index': {
        'High': (0.680, 0.115), 'Moderate': (0.530, 0.100), 'Low': (0.310, 0.105),
    },
    'sleep_disturbance_proxy': {
        'High': (0.560, 0.120), 'Moderate': (0.460, 0.105), 'Low': (0.330, 0.100),
    },
    'usage_hours_sq': {
        'High': (0.530, 0.130), 'Moderate': (0.420, 0.115), 'Low': (0.270, 0.110),
    },
}

artifacts, LOAD_ERRORS = load_artifacts()
class_cond_stats = load_class_cond_stats()
USING_COMPUTED_STATS = class_cond_stats is not None


# ═════════════════════════════════════════════════════════════════════════════
# HONEST FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════

def build_feature_row(
    age: float,
    avg_daily_usage_hours: float,
    mental_health_score: float,
    sleep_hours: float,
    academic_level: str,
    conflicts_proxy: float,
    params: dict,
    imputer,
    feat_cols: list,
) -> pd.DataFrame:
    # 1. behavioral_intensity_index
    usage_norm     = np.clip(avg_daily_usage_hours, 0, params["usage_max"]) / params["usage_max"]
    conflicts_norm = np.clip(conflicts_proxy, 0, params["conflicts_max"]) / params["conflicts_max"]
    behavioral_intensity_index = (usage_norm + conflicts_norm) / 2

    # 2. psychological_distress_index
    mh_range = params["mh_max"] - params["mh_min"] or 1
    psychological_distress_index = 1 - (
        (np.clip(mental_health_score, params["mh_min"], params["mh_max"]) - params["mh_min"]) / mh_range
    )

    # 3. sleep_disturbance_proxy
    sleep_range = params["sleep_max"] - params["sleep_min"] or 1
    sleep_disturbance_proxy = 1 - (
        (np.clip(sleep_hours, params["sleep_min"], params["sleep_max"]) - params["sleep_min"]) / sleep_range
    )

    # 4. age_norm
    age_range = params["age_max"] - params["age_min"] or 1
    age_norm = (np.clip(age, params["age_min"], params["age_max"]) - params["age_min"]) / age_range

    # 5. is_student
    student_levels = {"High School", "Undergraduate", "Graduate"}
    is_student = float(academic_level in student_levels)

    # 6. usage_hours_sq
    clipped_usage  = np.clip(avg_daily_usage_hours, 0, params["usage_max"])
    usage_hours_sq = np.clip(clipped_usage ** 2, 0, params["usage_sq_max"]) / params["usage_sq_max"]

    row = pd.DataFrame([{
        "behavioral_intensity_index":   behavioral_intensity_index,
        "psychological_distress_index": psychological_distress_index,
        "sleep_disturbance_proxy":      sleep_disturbance_proxy,
        "age_norm":                     age_norm,
        "is_student":                   is_student,
        "usage_hours_sq":               usage_hours_sq,
    }])

    row = row[feat_cols]
    row_imputed = pd.DataFrame(imputer.transform(row), columns=feat_cols)
    return row_imputed


def run_inference(age, usage_hours, mental_health_score, sleep_hours,
                  academic_level, conflicts_proxy):
    params    = artifacts["fe_params"]
    imputer   = artifacts["imputer"]
    feat_cols = artifacts["feat_cols"]
    scaler    = artifacts["scaler"]
    le        = artifacts["le"]
    model     = artifacts["model"]

    feat_row    = build_feature_row(
        age, usage_hours, mental_health_score, sleep_hours,
        academic_level, conflicts_proxy, params, imputer, feat_cols,
    )
    feat_scaled = scaler.transform(feat_row.values)

    pred_idx   = model.predict(feat_scaled)[0]
    pred_label = le.inverse_transform([pred_idx])[0]

    probs = {}
    if hasattr(model, "predict_proba"):
        prob_arr = model.predict_proba(feat_scaled)[0]
        for i, cls in enumerate(le.classes_):
            probs[cls] = float(prob_arr[i])
    else:
        for cls in le.classes_:
            probs[cls] = 1.0 if cls == pred_label else 0.0

    # ── Regression branch (continuous addiction score 1–10) ───────────────────
    reg_score = None
    if "reg_model" in artifacts and "scaler_reg" in artifacts:
        feat_scaled_reg = artifacts["scaler_reg"].transform(feat_row.values)
        reg_raw = float(artifacts["reg_model"].predict(feat_scaled_reg)[0])
        reg_score = round(float(np.clip(reg_raw, 1.0, 10.0)), 1)

    return pred_label, probs, feat_row, reg_score


# ═════════════════════════════════════════════════════════════════════════════
# SUBSCORE CALCULATORS
# ═════════════════════════════════════════════════════════════════════════════

def calc_subscore(*vals: int) -> float:
    return (np.mean(list(vals)) - 1) / 4.0


# ── Bubble scale widget ───────────────────────────────────────────────────────
def bubble_scale(label: str, key: str,
                 lo_label="Never", hi_label="Always", default: int = 2) -> int:
    if key not in st.session_state:
        st.session_state[key] = default
    st.markdown(
        f'<div class="bubble-wrap"><p class="q-text">{label}</p>',
        unsafe_allow_html=True,
    )
    val = st.radio(
        label, options=[1, 2, 3, 4, 5],
        index=st.session_state[key] - 1,
        horizontal=True, key=f"_radio_{key}",
        label_visibility="collapsed",
    )
    st.markdown(
        f'<div class="bubble-end-labels"><span>{lo_label}</span><span>{hi_label}</span></div></div>',
        unsafe_allow_html=True,
    )
    st.session_state[key] = val
    return val


def subscore_bar(emoji: str, label: str, score: float):
    pct = int(score * 100)
    col = "#34d399" if pct < 35 else ("#fbbf24" if pct < 65 else "#f87171")
    st.markdown(f"""
    <div class="subscore-card">
      <div class="subscore-title">{emoji} {label}: <span style="color:{col}">{pct}%</span></div>
      <div style="background:rgba(255,255,255,0.07);border-radius:4px;height:8px;">
        <div style="width:{pct}%;height:8px;background:{col};border-radius:4px;opacity:.85;"></div>
      </div>
    </div>""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
for _k, _v in [("page", "about"), ("result", None), ("scroll_top", False)]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

st.markdown('<div id="page-top"></div>', unsafe_allow_html=True)
if st.session_state.scroll_top:
    components.html("""
    <script>
      function su() {
        var a = window.parent.document.getElementById('page-top');
        if (a) a.scrollIntoView({behavior:'instant'});
        else window.parent.scrollTo({top:0,behavior:'instant'});
      }
      su(); setTimeout(su,80); setTimeout(su,300);
    </script>""", height=0)
    st.session_state.scroll_top = False


def go(page):
    st.session_state.page = page
    st.session_state.scroll_top = True
    st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _fig_base(figsize=(7, 3.5)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_CARD_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(CHART_SPINE)
        spine.set_linewidth(0.8)
    ax.tick_params(colors=CHART_MUTED, labelsize=8)
    ax.xaxis.label.set_color(CHART_MUTED)
    ax.yaxis.label.set_color(CHART_MUTED)
    return fig, ax


def _to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=CHART_BG)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def chart_probability_bars(probs: dict) -> bytes:
    ordered  = [l for l in LABEL_NAMES if l in probs]
    values   = [probs[l] * 100 for l in ordered]
    colors   = [LABEL_COLORS[LABEL_NAMES.index(l)] for l in ordered]

    fig, ax = _fig_base(figsize=(6, 2.6))
    bars = ax.barh(ordered, values, color=colors, height=0.5, edgecolor="none", alpha=0.88)
    for bar, val in zip(bars, values):
        ax.text(min(val + 1.5, 98), bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", ha="left",
                color=CHART_TEXT, fontsize=9, fontweight="bold")
    ax.set_xlim(0, 110)
    ax.set_xlabel("Probability (%)", fontsize=8)
    ax.set_title("Model Class Probabilities", color=CHART_TEXT, fontsize=10, fontweight="bold", pad=10)
    ax.axvline(50, color=CHART_ACCENT, linewidth=0.6, linestyle="--", alpha=0.4)
    ax.grid(axis="x", color=CHART_GRID, linewidth=0.5)
    ax.invert_yaxis()
    plt.tight_layout(pad=1.0)
    return _to_png(fig)


def chart_driver_analysis(feat_vals: dict, pred_label: str,
                          stats_source: dict) -> tuple[bytes, bool]:
    DRIVER_FEATS = [
        "behavioral_intensity_index",
        "psychological_distress_index",
        "sleep_disturbance_proxy",
        "usage_hours_sq",
    ]
    FRIENDLY = {
        "behavioral_intensity_index":   "Behavioural Intensity",
        "psychological_distress_index": "Psychological Distress",
        "sleep_disturbance_proxy":      "Sleep Disturbance",
        "usage_hours_sq":               "Usage Pattern",
    }

    entries = []
    for k in DRIVER_FEATS:
        if k not in feat_vals:
            continue
        val  = float(feat_vals[k])
        s    = stats_source.get(k, {}).get(pred_label)
        if s:
            mu, std = s
        else:
            mu, std = 0.5, 0.12
        std = max(std, 1e-6)
        z   = (val - mu) / std
        entries.append((k, val, z, mu))

    entries.sort(key=lambda x: abs(x[2]), reverse=True)
    feat_labels = [FRIENDLY.get(e[0], e[0]) for e in entries]
    raw_vals    = [e[1] for e in entries]
    z_scores    = [e[2] for e in entries]
    class_means = [e[3] for e in entries]
    bar_colors  = ["#ef4444" if z > 0 else "#22c55e" for z in z_scores]

    n = len(entries)
    fig, ax = _fig_base(figsize=(6.5, max(3.0, 1.0 + n * 0.8)))
    y = np.arange(n)

    bars = ax.barh(y, z_scores, left=0.0, color=bar_colors,
                   height=0.50, edgecolor="none", alpha=0.90)

    for bar, z, val, mu in zip(bars, z_scores, raw_vals, class_means):
        ha     = "left" if z >= 0 else "right"
        offset = 0.04 if z >= 0 else -0.04
        ax.text(z + offset, bar.get_y() + bar.get_height() / 2,
                f"val={val:.3f}  z={z:+.2f}",
                va="center", ha=ha, color=CHART_TEXT, fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(feat_labels, fontsize=9)
    ax.axvline(0, color=CHART_ACCENT, linewidth=1.0, alpha=0.6)

    x_lim = max(abs(z) for z in z_scores) * 1.55 if z_scores else 3.0
    ax.set_xlim(-x_lim, x_lim)
    ax.set_xlabel("Z-score from class mean", fontsize=8)
    ax.set_title(f"Feature Drivers — {pred_label} Class", color=CHART_TEXT,
                 fontsize=10, fontweight="bold", pad=10)

    red_p  = mpatches.Patch(color="#ef4444", alpha=0.85, label="Above class mean (↑ risk signal)")
    grn_p  = mpatches.Patch(color="#22c55e", alpha=0.85, label="Below class mean (↓ risk signal)")
    base_l = plt.Line2D([0], [0], color=CHART_ACCENT, linewidth=1.2,
                         linestyle="--", label=f"{pred_label} class mean")
    ax.legend(handles=[red_p, grn_p, base_l], fontsize=7.5,
              facecolor=CHART_CARD_BG, edgecolor=CHART_ACCENT,
              labelcolor=CHART_TEXT, loc="lower right", framealpha=0.85)

    ax.grid(axis="x", color=CHART_GRID, linewidth=0.5)
    ax.invert_yaxis()
    plt.tight_layout(pad=1.0)
    used_computed = (stats_source is not FALLBACK_CLASS_COND_STATS)
    return _to_png(fig), used_computed

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics import renderPDF
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import datetime

def generate_diagnostic_pdf(
    pred_label: str,
    probs: dict,
    feat_vals: dict,
    subscores: dict,
    core_inputs: dict,
) -> bytes:
    buf = io.BytesIO()
    # FIX 4: Set PDF title metadata to classification label so it shows correctly
    pdf_title = f"{pred_label} Risk Diagnostic Report"
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
        title=pdf_title,
        author="Social Media Addiction Risk Screener",
        subject="Diagnostic Report",
    )

    # ── Colour palette ──────────────────────────────────────────────────────
    RISK_COLOR = {
        "Low":      colors.HexColor("#34d399"),
        "Moderate": colors.HexColor("#fbbf24"),
        "High":     colors.HexColor("#f87171"),
    }
    TEAL       = colors.HexColor("#1a7d6e")
    DARK_BG    = colors.HexColor("#1a1033")
    CARD_BG    = colors.HexColor("#f5f3ff")
    BORDER     = colors.HexColor("#c084fc")
    risk_col   = RISK_COLOR.get(pred_label, colors.grey)

    styles = getSampleStyleSheet()

    def style(name, **kw):
        return ParagraphStyle(name, parent=styles["Normal"], **kw)

    title_style   = style("Title",   fontName="Helvetica-Bold", fontSize=18, textColor=TEAL, spaceAfter=4)
    sub_style     = style("Sub",     fontName="Helvetica",      fontSize=10, textColor=colors.grey, spaceAfter=12)
    heading_style = style("Heading", fontName="Helvetica-Bold", fontSize=11, textColor=colors.white,
                          backColor=TEAL, borderPadding=(6,8,6,8), spaceAfter=6, spaceBefore=14)
    body_style    = style("Body",    fontName="Helvetica",      fontSize=9,  textColor=colors.HexColor("#333333"), leading=14)
    label_style   = style("Label",   fontName="Helvetica-Bold", fontSize=9,  textColor=colors.HexColor("#555555"))
    value_style   = style("Value",   fontName="Helvetica",      fontSize=9,  textColor=colors.HexColor("#222222"))
    risk_style    = style("Risk",    fontName="Helvetica-Bold", fontSize=28, textColor=risk_col, alignment=TA_CENTER)
    caption_style = style("Caption", fontName="Helvetica-Oblique", fontSize=8, textColor=colors.grey,
                          alignment=TA_CENTER)
    warn_style    = style("Warn",    fontName="Helvetica-Oblique", fontSize=8,
                          textColor=colors.HexColor("#92400e"), leading=13)

    # ── Table style helper ──────────────────────────────────────────────────
    def base_table_style():
        return TableStyle([
            ("BACKGROUND", (0,0), (-1,0), TEAL),
            ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
            ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",   (0,0), (-1,0), 9),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, CARD_BG]),
            ("FONTNAME",   (0,1), (-1,-1), "Helvetica"),
            ("FONTSIZE",   (0,1), (-1,-1), 9),
            ("GRID",       (0,0), (-1,-1), 0.4, colors.HexColor("#d1d5db")),
            ("TOPPADDING", (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ("LEFTPADDING",   (0,0), (-1,-1), 8),
            ("RIGHTPADDING",  (0,0), (-1,-1), 8),
        ])

    story = []
    PAGE_W = A4[0] - 4*cm   # usable width

    # ── Header ─────────────────────────────────────────────────────────────
    story.append(Paragraph("Social Media Addiction Risk", title_style))
    story.append(Paragraph("Diagnostic Report — Research Prototype", sub_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=TEAL, spaceAfter=10))

    date_str = datetime.date.today().strftime("%d %B %Y")

    inp_items = list(core_inputs.items())
    half = (len(inp_items) + 1) // 2
    left_inp  = "  |  ".join(f"{k}: {v}" for k, v in inp_items[:half])
    right_inp = "  |  ".join(f"{k}: {v}" for k, v in inp_items[half:])

    meta_data = [
        ["Report Date", date_str,   "Model",   "XGBoost Classifier"],
        ["Classification", pred_label, "Tool", "Social Media Addiction Risk Screener"],
        ["", "", "", ""],
        [Paragraph("<b>User Inputs</b>", label_style),
         Paragraph(left_inp,  value_style),
         Paragraph("",        value_style),
         Paragraph(right_inp, value_style)],
    ]
    meta_table = Table(meta_data, colWidths=[3*cm, 6.5*cm, 2.5*cm, 5.5*cm])
    meta_table.setStyle(TableStyle([
        ("FONTNAME",  (0,0), (0,1), "Helvetica-Bold"),
        ("FONTNAME",  (2,0), (2,1), "Helvetica-Bold"),
        ("FONTNAME",  (0,3), (0,3), "Helvetica-Bold"),
        ("FONTSIZE",  (0,0), (-1,-1), 8.5),
        ("TEXTCOLOR", (0,0), (-1,-1), colors.HexColor("#374151")),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("LINEBELOW", (0,0), (-1,1), 0.3, colors.HexColor("#e5e7eb")),
        ("SPAN",      (1,3), (3,3)),
        ("BACKGROUND",(0,3), (-1,3), colors.HexColor("#f5f3ff")),
        ("TOPPADDING",    (0,3), (-1,3), 5),
        ("BOTTOMPADDING", (0,3), (-1,3), 5),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 12))

    # ── Section 1 : Risk Classification Result ──────────────────────────────
    story.append(Paragraph("① RISK CLASSIFICATION RESULT", heading_style))

    EMOJI = {"Low": "🟢", "Moderate": "🟡", "High": "🔴"}
    INTERP_BODY = {
        "Low":      "Your feature profile sits in the low-risk zone across behavioural intensity, psychological distress, and sleep disturbance. Your usage hours are within a healthy range relative to the training population.",
        "Moderate": "One or more features — likely behavioural intensity, psychological distress, or sleep disturbance — are elevated relative to the Low-risk class mean. Taking proactive steps now is the most effective way to prevent escalation.",
        "High":     "Multiple engineered features are significantly elevated relative to healthy-class benchmarks. Your daily usage hours, sleep hours, and/or mental health score are driving a High-risk classification. Immediate and sustained habit change is strongly recommended.",
    }

    result_data = [
        [Paragraph(f"<b>Predicted Risk Level</b>", label_style),
         Paragraph(f"<b>{EMOJI.get(pred_label,'')} {pred_label}</b>", style("R", fontName="Helvetica-Bold", fontSize=14, textColor=risk_col))],
        [Paragraph("<b>Interpretation</b>", label_style),
         Paragraph(INTERP_BODY.get(pred_label, ""), body_style)],
    ]
    rt = Table(result_data, colWidths=[4*cm, PAGE_W - 4*cm])
    rt.setStyle(TableStyle([
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("LINEBELOW",     (0,0), (-1,-1), 0.3, colors.HexColor("#e5e7eb")),
        ("BACKGROUND",    (0,0), (-1,-1), CARD_BG),
        ("BOX",           (0,0), (-1,-1), 1, risk_col),
    ]))
    story.append(rt)
    story.append(Spacer(1, 10))

    # ── Section 2 : Class Probabilities ────────────────────────────────────
    story.append(Paragraph("② CLASS PROBABILITIES", heading_style))

    BAR_W  = 120
    BAR_H  = 10

    def make_conf_bar(probability: float, hex_color: str) -> Drawing:
        d = Drawing(BAR_W, BAR_H + 2)
        track = Rect(0, 1, BAR_W, BAR_H,
                     fillColor=colors.HexColor("#e5e7eb"),
                     strokeColor=None, strokeWidth=0)
        d.add(track)
        fill_w = max(BAR_W * probability, 2.0)
        bar = Rect(0, 1, fill_w, BAR_H,
                   fillColor=colors.HexColor(hex_color),
                   strokeColor=None, strokeWidth=0)
        d.add(bar)
        return d

    PROB_BAR_COLORS = {"Low": "#34d399", "Moderate": "#fbbf24", "High": "#f87171"}

    prob_header = ["Risk Class", "Probability", "Confidence Bar"]
    prob_rows   = []
    for lbl in ["Low", "Moderate", "High"]:
        p        = probs.get(lbl, 0.0)
        bar_draw = make_conf_bar(p, PROB_BAR_COLORS.get(lbl, "#c084fc"))
        prob_rows.append([lbl, f"{p*100:.1f}%", bar_draw])

    prob_table = Table([prob_header] + prob_rows, colWidths=[4*cm, 3*cm, PAGE_W - 7*cm])
    ps = base_table_style()
    for i, lbl in enumerate(["Low", "Moderate", "High"]):
        c = RISK_COLOR.get(lbl, colors.grey)
        ps.add("TEXTCOLOR", (0, i+1), (0, i+1), c)
        ps.add("FONTNAME",  (0, i+1), (0, i+1), "Helvetica-Bold")
        ps.add("VALIGN",    (2, i+1), (2, i+1), "MIDDLE")
    prob_table.setStyle(ps)
    story.append(prob_table)
    story.append(Spacer(1, 10))

    # ── Section 3 : Engineered Features ────────────────────────────────────
    story.append(Paragraph("③ ENGINEERED FEATURE VECTOR", heading_style))

    FEAT_FRIENDLY_PDF = {
        'behavioral_intensity_index':   'Behavioural Intensity Index',
        'psychological_distress_index': 'Psychological Distress Index',
        'sleep_disturbance_proxy':      'Sleep Disturbance Proxy',
        'usage_hours_sq':               'Usage Pattern',
    }

    feat_header = ["Feature", "Value", "Risk Signal"]
    feat_rows   = []
    FEAT_EXCLUDE_PDF = {"age_norm", "is_student"}
    for k, v in feat_vals.items():
        if k in FEAT_EXCLUDE_PDF:
            continue
        signal = "[HIGH]" if float(v) > 0.6 else ("[MOD]" if float(v) > 0.35 else "[OK]")
        feat_rows.append([FEAT_FRIENDLY_PDF.get(k, k), f"{float(v):.4f}", signal])

    feat_table = Table([feat_header] + feat_rows,
                       colWidths=[7*cm, 3*cm, PAGE_W - 10*cm])
    fts = base_table_style()
    SIGNAL_COLORS = {
        "[HIGH]": colors.HexColor("#f87171"),
        "[MOD]":  colors.HexColor("#fbbf24"),
        "[OK]":   colors.HexColor("#34d399"),
    }
    for i, row in enumerate(feat_rows):
        sig = row[2]
        sc = SIGNAL_COLORS.get(sig, colors.grey)
        fts.add("TEXTCOLOR", (2, i+1), (2, i+1), sc)
        fts.add("FONTNAME",  (2, i+1), (2, i+1), "Helvetica-Bold")
    feat_table.setStyle(fts)
    story.append(feat_table)
    story.append(Spacer(1, 10))

    # ── Section 4 : Questionnaire Subscores ────────────────────────────────
    story.append(Paragraph("④ QUESTIONNAIRE SUBSCORES", heading_style))
    story.append(Paragraph(
        "Derived from your questionnaire responses. Shown for self-awareness — "
        "NOT used as model input features (except Behavioural Intensity as a proxy for the conflicts variable).",
        style("SN", fontName="Helvetica-Oblique", fontSize=8, textColor=colors.grey, spaceAfter=6)
    ))

    sub_header = ["Dimension", "Score", "Level"]
    sub_rows   = []
    for lbl, val in subscores.items():
        pct   = int(val * 100)
        level = "Low" if pct < 35 else ("Moderate" if pct < 65 else "High")
        sub_rows.append([lbl, f"{pct}%", level])

    sub_table = Table([sub_header] + sub_rows,
                      colWidths=[7*cm, 3*cm, PAGE_W - 10*cm])
    sub_table.setStyle(base_table_style())
    story.append(sub_table)
    story.append(Spacer(1, 10))

    # ── Section 5 : Recommendations ────────────────────────────────────────
    RECS_PDF = {
        "Low": [
            ("Stay Consistent", "Healthy habits are fragile. Set a monthly reminder to re-take this screener and track any changes over time."),
            ("Keep Monitoring", "Use built-in screen time tools (iOS Screen Time / Android Digital Wellbeing) for passive weekly awareness."),
            ("Invest Offline",  "Prioritise offline hobbies and face-to-face connections — these are your strongest long-term buffers."),
        ],
        "Moderate": [
            ("Set Daily App Limits",  "Enforce a 45-minute daily social media budget using app timers. Resist overriding for the first three days."),
            ("Phone-Free Bedtime",    "No social media 60 minutes before sleep. Replace scrolling with reading or journaling to wind down properly."),
            ("Mindful Opening",       "Before opening any app, ask: 'Why am I opening this?' If you can't answer, close it immediately."),
            ("Weekly Detox Day",      "Choose one day per week with zero social media. Fill it with a planned offline activity — structure is key."),
        ],
        "High": [
            ("Hard Limit — Start Today", "Set a strict 1-hour daily cap using both your phone's built-in limits AND a third-party app (Opal / Cold Turkey) for redundancy."),
            ("Remove App Shortcuts",     "Delete social media apps from your home screen. Access via browser only — this single change reduces usage 20–30%."),
            ("Grayscale Mode Evenings",  "Enable phone grayscale after 8pm. Removing colour significantly reduces the dopamine reward from scrolling."),
            ("Replace the Habit Loop",   "Identify your top 3 triggers and write a concrete replacement action for each. Post it where you can see it daily."),
            ("Seek Professional Support","CBT has strong evidence for behavioural addiction. Contact your university counselling service or a digital-wellness therapist."),
        ],
    }
    story.append(Paragraph("⑤ RECOMMENDATIONS", heading_style))
    recs = RECS_PDF.get(pred_label, [])
    rec_data = [["#", "Action", "Detail"]]
    for i, (title, desc) in enumerate(recs, 1):
        rec_data.append([str(i), Paragraph(f"<b>{title}</b>", label_style), Paragraph(desc, body_style)])

    rec_table = Table(rec_data, colWidths=[1*cm, 5*cm, PAGE_W - 6*cm])
    rec_table.setStyle(base_table_style())
    story.append(rec_table)
    story.append(Spacer(1, 16))

    # ── Disclaimer footer ───────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey, spaceAfter=6))
    story.append(Paragraph(
        "⚠️  DISCLAIMER: This report is produced by a research prototype and is NOT a clinical diagnostic tool. "
        "Results do not replace professional mental health advice. If you are concerned about your wellbeing, "
        "please consult a qualified healthcare professional.",
        warn_style
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — LANDING
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "about":

    if LOAD_ERRORS:
        st.error("⚠️ Model artifacts missing — prediction will not work until resolved.")
        for e in LOAD_ERRORS:
            st.warning(e)

    components.html("""
    <!DOCTYPE html><html><head>
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');
    *{margin:0;padding:0;box-sizing:border-box;}
    body{background:transparent;font-family:'DM Sans',sans-serif;color:#f3f0ff;overflow:hidden;}
    .hero{display:grid;grid-template-columns:1fr 1fr;gap:24px;align-items:center;padding:12px 0 20px;}
    .hero-left{display:flex;flex-direction:column;gap:0;}
    .eyebrow{font-family:'Syne',sans-serif;font-size:.6rem;font-weight:800;letter-spacing:.22em;text-transform:uppercase;color:#c084fc;margin-bottom:12px;}
    h1{font-family:'Syne',sans-serif;font-size:2.4rem;font-weight:800;line-height:1.12;color:#f3f0ff;margin-bottom:14px;}
    h1 span{background:linear-gradient(135deg,#a855f7,#c084fc,#e879f9);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
    .body-text{font-size:.88rem;color:#7c6da0;line-height:1.7;margin-bottom:18px;}
    .body-text strong{color:#b8a8d8;}
    .meta{display:flex;align-items:center;gap:10px;font-size:.78rem;color:#4c3a7a;font-weight:600;}
    .meta-dot{width:3px;height:3px;border-radius:50%;background:#4c3a7a;}
    .hero-right{display:flex;justify-content:center;align-items:center;position:relative;padding:20px 0;min-height:320px;}
    .orb-bg{position:absolute;width:260px;height:260px;border-radius:50%;
        background:radial-gradient(circle at 38% 32%, #7c3aed 0%, #4c1d95 45%, #1a0a3c 100%);
        border:1.5px solid rgba(192,132,252,0.3);
        box-shadow:0 0 60px rgba(124,58,237,0.35), 0 0 120px rgba(124,58,237,0.15);
        left:50%;transform:translateX(-50%);}
    .phone-wrap{position:relative;z-index:2;margin:0 auto;}
    .float-card{position:absolute;background:rgba(28,16,60,0.95);border:1px solid rgba(192,132,252,0.3);
        border-radius:12px;padding:10px 14px;backdrop-filter:blur(12px);z-index:10;white-space:nowrap;}
    .card-top{top:8px;right:0px;}
    .card-bottom{bottom:8px;left:0px;}
    .card-label{display:flex;align-items:center;gap:6px;font-size:.58rem;font-weight:700;
        letter-spacing:.12em;text-transform:uppercase;color:#7c6da0;margin-bottom:4px;}
    .card-dot{width:6px;height:6px;border-radius:50%;background:#22c55e;flex-shrink:0;}
    .card-dot-purple{width:6px;height:6px;border-radius:50%;background:#c084fc;flex-shrink:0;}
    .card-value{font-family:'Syne',sans-serif;font-size:1.55rem;font-weight:800;line-height:1;}
    .card-sub{font-size:.6rem;color:#4c3a7a;margin-top:3px;}
    </style></head><body>
    <div class="hero">
      <div class="hero-left">
        <div class="eyebrow">— ML-Powered Screener</div>
        <h1>Are You <span>Addicted</span> to Social Media?</h1>
        <p class="body-text">Answer <strong>30+ deep questions</strong> across 5 dimensions. Our trained ML model predicts your <strong>addiction risk tier</strong> with detailed visual analytics.</p>
        <div class="meta"><span>⏱ ~5 min</span><div class="meta-dot"></div><span>Free</span><div class="meta-dot"></div><span>Nothing stored</span></div>
      </div>
      <div class="hero-right">
        <div class="orb-bg"></div>
        <div class="phone-wrap">
          <svg width="160" height="290" viewBox="0 0 160 290" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <linearGradient id="phonebody" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#2d1b69"/>
                <stop offset="100%" stop-color="#1a0a40"/>
              </linearGradient>
              <linearGradient id="screen" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#0f0630"/>
                <stop offset="100%" stop-color="#1a0a50"/>
              </linearGradient>
              <linearGradient id="approw" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stop-color="#c084fc"/>
                <stop offset="100%" stop-color="#a855f7"/>
              </linearGradient>
            </defs>
            <rect x="4" y="4" width="152" height="282" rx="24" ry="24" fill="url(#phonebody)" stroke="rgba(192,132,252,0.5)" stroke-width="1.5"/>
            <rect x="12" y="18" width="136" height="230" rx="14" ry="14" fill="url(#screen)"/>
            <rect x="56" y="22" width="48" height="10" rx="5" ry="5" fill="#0a0422"/>
            <rect x="24" y="50" width="26" height="26" rx="7" fill="#ef4444" opacity="0.9"/>
            <rect x="58" y="50" width="26" height="26" rx="7" fill="#3b82f6" opacity="0.9"/>
            <rect x="92" y="50" width="26" height="26" rx="7" fill="#22c55e" opacity="0.9"/>
            <rect x="110" y="48" width="18" height="18" rx="9" fill="#f97316" opacity="0.9"/>
            <circle cx="118" cy="48" r="7" fill="#f97316" opacity="0.95"/>
            <text x="118" y="52" font-size="8" text-anchor="middle" fill="white" font-weight="bold">3</text>
            <rect x="24" y="86" width="26" height="26" rx="7" fill="#8b5cf6" opacity="0.85"/>
            <rect x="58" y="86" width="26" height="26" rx="7" fill="#ec4899" opacity="0.85"/>
            <rect x="92" y="86" width="26" height="26" rx="7" fill="#f59e0b" opacity="0.85"/>
            <rect x="126" y="86" width="26" height="26" rx="7" fill="#14b8a6" opacity="0.85"/>
            <line x1="24" y1="126" x2="136" y2="126" stroke="rgba(192,132,252,0.15)" stroke-width="1"/>
            <rect x="24" y="136" width="88" height="8" rx="4" fill="rgba(192,132,252,0.2)"/>
            <rect x="24" y="152" width="60" height="8" rx="4" fill="rgba(192,132,252,0.12)"/>
            <rect x="24" y="168" width="74" height="8" rx="4" fill="rgba(192,132,252,0.16)"/>
            <text x="80" y="215" font-size="22" text-anchor="middle" fill="#ec4899" opacity="0.9">♥</text>
            <rect x="58" y="240" width="44" height="4" rx="2" fill="rgba(192,132,252,0.35)"/>
            <rect x="0" y="80" width="4" height="28" rx="2" fill="rgba(192,132,252,0.4)"/>
            <rect x="0" y="116" width="4" height="20" rx="2" fill="rgba(192,132,252,0.4)"/>
            <rect x="156" y="90" width="4" height="40" rx="2" fill="rgba(192,132,252,0.4)"/>
          </svg>
        </div>
        <div class="float-card card-top">
          <div class="card-label"><div class="card-dot"></div>Model Accuracy</div>
          <div class="card-value" style="color:#22c55e">97.9<span style="font-size:.9rem">%</span></div>
          <div class="card-sub">on Nusratt test set</div>
        </div>
        <div class="float-card card-bottom">
          <div class="card-label"><div class="card-dot-purple"></div>5 Dimensions</div>
          <div class="card-value" style="color:#c084fc">30<span style="font-size:.9rem">+</span></div>
          <div class="card-sub">deep questions</div>
        </div>
      </div>
    </div>
    </body></html>""", height=340, scrolling=False)

    st.markdown("""
    <style>
    .how-strip { background:linear-gradient(135deg,#2a1d52 0%,#1e1040 100%); border:1px solid rgba(192,132,252,0.15); border-radius:18px; padding:28px 32px; margin:4px 0 0; display:grid; grid-template-columns:auto 1fr; gap:28px; align-items:center; }
    .how-icon-box { width:50px; height:50px; background:rgba(192,132,252,0.12); border:1px solid rgba(192,132,252,0.25); border-radius:14px; display:flex; align-items:center; justify-content:center; font-size:1.5rem; flex-shrink:0; }
    .how-text-block h3 { font-family:'Syne',sans-serif; font-size:1.15rem; font-weight:800; color:#f3f0ff; margin:0 0 6px; }
    .how-text-block p { font-size:.84rem; color:#7c6da0; line-height:1.65; margin:0; }
    .how-text-block p strong { color:#b8a8d8; }
    </style>
    <div class="how-strip">
      <div class="how-icon-box">🧬</div>
      <div class="how-text-block">
        <h3>How the Prediction Works</h3>
        <p>Your raw inputs — <strong>daily usage hours, sleep hours, mental health score, age, and academic level</strong> — are passed through the same feature engineering pipeline used during model training. The 5 questionnaire sections generate diagnostic subscores shown alongside your results, but the <strong>6 model features</strong> are derived directly from your core inputs.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    .risk-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin:24px 0 32px; }
    .risk-card { background:rgba(42,29,82,0.55); border-radius:14px; padding:16px 18px; }
    .risk-card-top { display:flex; align-items:center; gap:8px; margin-bottom:7px; }
    .risk-dot { width:9px; height:9px; border-radius:50%; flex-shrink:0; }
    .risk-title { font-family:'Syne',sans-serif; font-size:.8rem; font-weight:700; color:#f3f0ff; }
    .risk-desc { font-size:.75rem; color:#6b5e8a; line-height:1.55; }
    </style>
    <br>
    <div style="font-family:'Syne',sans-serif;font-size:.68rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase;color:#c084fc;margin-bottom:14px;">The three risk tiers</div>
    <div class="risk-grid">
      <div class="risk-card" style="border-left:4px solid #34d399">
        <div class="risk-card-top"><div class="risk-dot" style="background:#34d399"></div><div class="risk-title">🟢 Low Risk</div></div>
        <div class="risk-desc">Addicted Score ≤ 4. Healthy usage. SM is not significantly impacting your wellbeing.</div>
      </div>
      <div class="risk-card" style="border-left:4px solid #fbbf24">
        <div class="risk-card-top"><div class="risk-dot" style="background:#fbbf24"></div><div class="risk-title">🟡 Moderate Risk</div></div>
        <div class="risk-desc">Addicted Score 5–6. Some dependency signals. Proactive steps can prevent escalation.</div>
      </div>
      <div class="risk-card" style="border-left:4px solid #f87171">
        <div class="risk-card-top"><div class="risk-dot" style="background:#f87171"></div><div class="risk-title">🔴 High Risk</div></div>
        <div class="risk-desc">Addicted Score ≥ 7. Strong dependency signals. Immediate habit change is recommended.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 3, 2])
    with col2:
        if st.button("Start the Screener →", type="primary", use_container_width=True):
            go("survey")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SURVEY
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "survey":

    st.markdown('<div class="pill">📋 Screener Survey</div>', unsafe_allow_html=True)
    st.markdown("## Social Media Addiction Risk Survey")
    st.markdown(
        '<p style="color:var(--text-muted);font-size:.85rem;margin-bottom:1.5rem">'
        'Complete all sections. Your responses are processed locally — nothing is stored.</p>',
        unsafe_allow_html=True,
    )

    # ── SECTION 0: Core inputs — FIX 1: all inputs stacked vertically ─────────
    st.markdown(
        '<div class="section-header">🎯 Section 0 — Core Inputs (Model Features)</div>',
        unsafe_allow_html=True,
    )
    st.markdown("""
    <div class="proxy-note">
    <strong style="color:#fbbf24">ℹ️ These inputs directly determine your prediction.</strong>
    They feed into the XGBoost model's 6 features via the same pipeline used during training.
    The questionnaire sections below add diagnostic subscores shown alongside your results.
    </div>
    """, unsafe_allow_html=True)

    # Row 1: Age + Academic Level side by side
    col_s0_a, col_s0_b = st.columns(2)
    with col_s0_a:
        age = st.number_input("Age", min_value=13, max_value=35, value=20, step=1)
    with col_s0_b:
        academic_level = st.selectbox(
            "Academic / Occupation Level",
            ["High School", "Undergraduate", "Graduate", "Other"],
            index=1,
        )

    # All sliders stacked full-width below
    avg_daily_usage_hours = st.slider(
        "Average daily social media usage (hours)", 0.5, 12.0, 3.5, 0.5,
    )
    mental_health_score = st.slider(
        "Mental health score (1 = very poor, 10 = excellent)", 1, 10, 6,
    )
    sleep_hours = st.slider("Average sleep hours per night", 3.0, 11.0, 7.0, 0.5)

    # ── SECTION A: Behavioural Intensity ──────────────────────────────────────
    st.markdown(
        '<div class="section-header">📱 Section A — Behavioural Intensity</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="font-size:.8rem;color:var(--text-muted);margin-bottom:.5rem">'
        'How compulsive is your social media engagement? (1 = Never, 5 = Always)</p>',
        unsafe_allow_html=True,
    )
    st.markdown("""
    <div class="proxy-note">
    ⚡ <strong style="color:#fbbf24">Model note:</strong>
    Your average score here is used as a proxy for the "conflicts over social media" variable
    (a 0–5 integer in the Nusratt training set with no direct UI equivalent). It contributes
    to <em>behavioral_intensity_index</em> alongside your daily usage hours.
    </div>
    """, unsafe_allow_html=True)

    beh_q1 = bubble_scale("How often do you open a social media app without a specific reason?",
                           "beh_q1", default=1)
    beh_q2 = bubble_scale("How often do you check notifications immediately, even when busy?",
                           "beh_q2", default=1)
    beh_q3 = bubble_scale("How often do you mindlessly scroll without reading or engaging?",
                           "beh_q3", default=1)
    beh_q4 = bubble_scale("How often do you post, like, or react impulsively?",
                           "beh_q4", default=1)

    beh_subscore = calc_subscore(beh_q1, beh_q2, beh_q3, beh_q4)
    subscore_bar("📱", "Behavioural Intensity", beh_subscore)

    # ── SECTION B: Psychological Distress ─────────────────────────────────────
    st.markdown(
        '<div class="section-header">🧠 Section B — Psychological Distress</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="font-size:.8rem;color:var(--text-muted);margin-bottom:.5rem">'
        'How much does social media affect your mental wellbeing? (1 = Never, 5 = Always)</p>',
        unsafe_allow_html=True,
    )
    st.markdown("""
    <div class="proxy-note">
    📊 <strong style="color:#b8a8d8">Diagnostic only.</strong>
    This subscore is shown for your insight but does not enter the model. Psychological
    distress in the model is derived from your <em>Mental Health Score</em> (Section 0).
    </div>
    """, unsafe_allow_html=True)

    psy_q1 = bubble_scale("How often do you feel low or empty after using social media?",
                           "psy_q1", default=1)
    psy_q2 = bubble_scale("How often do you feel anxious after browsing your feeds?",
                           "psy_q2", default=1)
    psy_q3 = bubble_scale("How often do you fear missing out if you are offline?",
                           "psy_q3", default=1)
    psy_q4 = bubble_scale("How often do others' posts make you feel inadequate?",
                           "psy_q4", default=1)
    psy_q5 = bubble_scale("How often are you irritable when unable to access social media?",
                           "psy_q5", default=1)

    psy_subscore = calc_subscore(psy_q1, psy_q2, psy_q3, psy_q4, psy_q5)
    subscore_bar("🧠", "Psychological Distress", psy_subscore)

    # ── SECTION C: Sleep Disturbance ──────────────────────────────────────────
    st.markdown(
        '<div class="section-header">😴 Section C — Sleep Disturbance</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="font-size:.8rem;color:var(--text-muted);margin-bottom:.5rem">'
        'How much does social media disrupt your sleep? (1 = Never, 5 = Always)</p>',
        unsafe_allow_html=True,
    )
    st.markdown("""
    <div class="proxy-note">
    📊 <strong style="color:#b8a8d8">Diagnostic only.</strong>
    Sleep disturbance in the model is computed from your <em>Sleep Hours</em> (Section 0).
    This subscore is shown for context only.
    </div>
    """, unsafe_allow_html=True)

    slp_q1 = bubble_scale("How often do you use social media in bed before sleep?",
                           "slp_q1", default=1)
    slp_q2 = bubble_scale("How often does browsing delay the time you fall asleep?",
                           "slp_q2", default=1)
    slp_q3 = bubble_scale("How often do you wake at night to check notifications?",
                           "slp_q3", default=1)
    slp_q4 = bubble_scale("How often is checking social media the first thing you do after waking?",
                           "slp_q4", default=1)

    slp_subscore = calc_subscore(slp_q1, slp_q2, slp_q3, slp_q4)
    subscore_bar("😴", "Sleep Disturbance", slp_subscore)

    # ── SECTION D: Usage Pattern ───────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">⏰ Section D — Usage Pattern</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="font-size:.8rem;color:var(--text-muted);margin-bottom:.5rem">'
        'How do you use social media across contexts? (1 = Never, 5 = Always)</p>',
        unsafe_allow_html=True,
    )
    st.markdown("""
    <div class="proxy-note">
    📊 <strong style="color:#b8a8d8">Diagnostic only.</strong>
    Usage pattern in the model is captured by <em>usage_hours_sq</em> (your daily hours squared),
    which encodes non-linear escalation. This subscore is shown for insight.
    </div>
    """, unsafe_allow_html=True)

    usg_q1 = bubble_scale("How much more do you use social media on weekends?",
                           "usg_q1", lo_label="No difference", hi_label="Much more", default=1)
    usg_q2 = bubble_scale("How often does stress or boredom increase your SM usage?",
                           "usg_q2", default=1)
    usg_q3 = bubble_scale("How much of your time is passive scrolling vs. active creation?",
                           "usg_q3", lo_label="Mostly active", hi_label="Mostly passive", default=2)
    usg_q4 = bubble_scale("How often do you hop between multiple platforms in one session?",
                           "usg_q4", default=1)

    usg_subscore = calc_subscore(usg_q1, usg_q2, usg_q3, usg_q4)
    subscore_bar("⏰", "Usage Pattern", usg_subscore)

    # ── SECTION E: Academic Impact ────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">📚 Section E — Academic & Concentration Impact</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="font-size:.8rem;color:var(--text-muted);margin-bottom:.5rem">'
        'How much does SM interfere with your studies and focus? (1 = Never, 5 = Always)</p>',
        unsafe_allow_html=True,
    )
    st.markdown("""
    <div class="proxy-note">
    📊 <strong style="color:#b8a8d8">Diagnostic only.</strong>
    Academic impact has no direct model feature (the training dataset uses a binary
    "affects_academic" flag that is not in the 6 shared features). This section
    is purely for your self-awareness.
    </div>
    """, unsafe_allow_html=True)

    acad_q1 = bubble_scale("How often does the urge to check SM distract you while studying?",
                            "acad_q1", default=1)
    acad_q2 = bubble_scale("How often do you use SM to procrastinate on deadlines?",
                            "acad_q2", default=1)
    acad_q3 = bubble_scale("How often do you lose concentration because of an SM urge?",
                            "acad_q3", default=1)
    acad_q4 = bubble_scale("How often have you submitted work late due to time on SM?",
                            "acad_q4", default=1)

    acad_subscore = calc_subscore(acad_q1, acad_q2, acad_q3, acad_q4)
    subscore_bar("📚", "Academic Impact", acad_subscore)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Submit ────────────────────────────────────────────────────────────────
    col_b1, col_b2, col_b3 = st.columns([1.5, 4, 1.5])
    with col_b2:
        submitted = st.button("🔍  Analyse My Risk", type="primary", use_container_width=True)

    if submitted:
        if LOAD_ERRORS:
            st.error("Cannot run inference — model artifacts are missing. See errors above.")
        else:
            params = artifacts["fe_params"]
            conflicts_proxy = beh_subscore * params.get("conflicts_max", 5)

            with st.spinner("Running XGBoost inference..."):
                try:
                    pred_label, probs, feat_row, reg_score = run_inference(
                        age=float(age),
                        usage_hours=float(avg_daily_usage_hours),
                        mental_health_score=float(mental_health_score),
                        sleep_hours=float(sleep_hours),
                        academic_level=academic_level,
                        conflicts_proxy=conflicts_proxy,
                    )
                    st.session_state.result = {
                        "pred_label":   pred_label,
                        "probs":        probs,
                        "feat_row":     feat_row.to_dict(orient="records")[0],
                        "reg_score":    reg_score,  
                        "subscores": {
                            "Behavioural Intensity":  beh_subscore,
                            "Psychological Distress": psy_subscore,
                            "Sleep Disturbance":      slp_subscore,
                            "Usage Pattern":          usg_subscore,
                            "Academic Impact":        acad_subscore,
                            
                        },
                        "core_inputs": {
                            "Age":                age,
                            "Daily Usage (hrs)":  avg_daily_usage_hours,
                            "Sleep Hours":        sleep_hours,
                            "Mental Health (1–10)": mental_health_score,
                            "Academic Level":     academic_level,
                        },
                    }
                    go("results")
                except Exception as ex:
                    st.error(f"Inference failed: {ex}")
                    st.exception(ex)

    st.markdown("<br>", unsafe_allow_html=True)
    col_back1, col_back2, col_back3 = st.columns([1.5, 4, 1.5])
    with col_back2:
        if st.button("← Back to Home", use_container_width=True):
            go("about")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RESULTS
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "results":

    result = st.session_state.get("result")
    if result is None:
        st.warning("No result found. Please complete the survey first.")
        if st.button("← Go to Survey"):
            go("survey")
        st.stop()

    pred_label  = result["pred_label"]
    probs       = result["probs"]
    feat_vals   = result["feat_row"]
    subscores   = result["subscores"]
    core_inputs = result["core_inputs"]
    reg_score = result.get("reg_score") 

    risk_idx   = LABEL_NAMES.index(pred_label) if pred_label in LABEL_NAMES else 0
    risk_color = LABEL_COLORS[risk_idx]
    risk_emoji = LABEL_EMOJI[risk_idx]
    risk_bg    = LABEL_BG_CSS[risk_idx]
    confidence = probs.get(pred_label, 1.0)


    # ── Result banner ─────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="result-banner" style="{risk_bg}">
      <h2 style="color:{risk_color}">{risk_emoji} {pred_label} Addiction Risk</h2>
      <p>Model confidence: <strong style="color:{risk_color}">{confidence*100:.1f}%</strong>
         &nbsp;·&nbsp; Model: <strong style="color:var(--text-secondary)">XGBoost (clf_xgb)</strong>
         &nbsp;·&nbsp; Features: <strong style="color:var(--text-secondary)">6 engineered</strong></p>
    </div>
    """, unsafe_allow_html=True)

    # ── Continuous addiction score (regression branch) ────────────────────────
    if reg_score is not None:
        score_color = "#34d399" if reg_score <= 4 else ("#fbbf24" if reg_score <= 6 else "#f87171")
        fill_pct = int((reg_score - 1) / 9 * 100)
        st.markdown(f"""
        <div style="background:var(--bg-card);border:1px solid var(--border);border-left:4px solid {score_color};
                    border-radius:12px;padding:16px 22px;margin-bottom:1.2rem;">
          <div style="font-family:'Syne',sans-serif;font-size:.78rem;font-weight:700;
                      color:var(--text-muted);letter-spacing:.08em;text-transform:uppercase;
                      margin-bottom:6px">📊 Continuous Addiction Score (Regression)</div>
          <div style="display:flex;align-items:baseline;gap:10px;margin-bottom:8px">
            <span style="font-family:'Syne',sans-serif;font-size:2.4rem;font-weight:800;
                         color:{score_color};line-height:1">{reg_score}</span>
            <span style="font-size:.9rem;color:var(--text-muted)">/ 10</span>
          </div>
          <div style="background:rgba(255,255,255,0.07);border-radius:4px;height:8px;margin-bottom:8px">
            <div style="width:{fill_pct}%;height:8px;background:{score_color};
                        border-radius:4px;transition:width .4s ease"></div>
          </div>
          <div style="font-size:.78rem;color:var(--text-muted)">
            Predicted by XGBoost Regressor on the same 6 engineered features.
            Complements the classifier — gives continuous severity within the
            <strong style="color:{score_color}">{pred_label}</strong> band.
            Scale: 1–4 Low · 5–6 Moderate · 7–10 High.
          </div>
        </div>
        """, unsafe_allow_html=True)
        
    # ── CHART 1: Class probabilities ──────────────────────────────────────────
    st.markdown("### 📈 Model Class Probabilities")
    st.markdown(
        '<p style="font-size:.82rem;color:var(--text-muted);margin-top:-.4rem;margin-bottom:.8rem">'
        'Probability assigned by the XGBoost classifier to each risk class.</p>',
        unsafe_allow_html=True,
    )
    st.image(chart_probability_bars(probs), use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── CHART 2: Feature driver (z-score) ─────────────────────────────────────
    st.markdown("### 🔎 What Is Driving Your Prediction")
    stats_source = class_cond_stats if USING_COMPUTED_STATS else FALLBACK_CLASS_COND_STATS
    driver_png, used_computed = chart_driver_analysis(feat_vals, pred_label, stats_source)

    source_note = (
        "Class-conditional statistics computed from <strong>features_train.csv</strong> (Nusratt training set)."
        if used_computed else
        "⚠️ <strong>features_train.csv</strong> not found — using conservative fallback estimates. "
        "Run the full pipeline to enable computed stats."
    )
    st.markdown(
        f'<p style="font-size:.82rem;color:var(--text-muted);margin-top:-.4rem;margin-bottom:.8rem">'
        f'Z-score distance from the <strong style="color:var(--accent)">{pred_label}</strong> class mean. '
        f'<span style="color:#ef4444">Red = above class mean (amplifies risk signal)</span>, '
        f'<span style="color:#22c55e">green = below (attenuates risk signal)</span>. '
        f'{source_note}</p>',
        unsafe_allow_html=True,
    )
    st.image(driver_png, use_container_width=True)

    # FIX 3: Feature value cards — exclude age_norm and is_student, 2-per-row layout
    st.markdown("<br>", unsafe_allow_html=True)

    # Only the 4 meaningful driver features
    DRIVER_FEAT_KEYS = [
        "behavioral_intensity_index",
        "psychological_distress_index",
        "sleep_disturbance_proxy",
        "usage_hours_sq",
    ]
    driver_feat_items = [(k, feat_vals[k]) for k in DRIVER_FEAT_KEYS if k in feat_vals]

    # Render in rows of 2
    for row_start in range(0, len(driver_feat_items), 2):
        row_items = driver_feat_items[row_start:row_start + 2]
        cols_f = st.columns(2)
        for col_idx, (fname, fval) in enumerate(row_items):
            friendly = FEAT_FRIENDLY.get(fname, fname)
            s = (class_cond_stats or FALLBACK_CLASS_COND_STATS).get(fname, {}).get(pred_label)
            if s:
                mu, std = s
                z_val = (float(fval) - mu) / max(std, 1e-6)
            else:
                z_val = float(fval) - 0.5
            bar_color = "#ef4444" if z_val > 0 else "#22c55e"
            pct = min(int(float(fval) * 100), 100)
            with cols_f[col_idx]:
                st.markdown(f"""
                <div style="margin-bottom:12px;background:var(--bg-card);border:1px solid var(--border);
                            border-radius:10px;padding:11px 14px;">
                  <div style="font-size:.74rem;color:var(--text-muted);margin-bottom:4px">{friendly}</div>
                  <div style="font-size:1.05rem;font-weight:700;color:var(--text-primary);
                              font-family:'Syne',sans-serif;margin-bottom:5px">
                    {float(fval):.3f}
                    <span style="font-size:.75rem;color:{'#ef4444' if z_val>0 else '#22c55e'};
                                 font-family:'DM Sans';font-weight:400;margin-left:6px">
                      z={z_val:+.2f}
                    </span>
                  </div>
                  <div style="background:rgba(255,255,255,0.07);border-radius:4px;height:6px;">
                    <div style="width:{pct}%;height:6px;background:{bar_color};border-radius:4px;opacity:.82;"></div>
                  </div>
                </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Diagnostic subscores ──────────────────────────────────────────────────
    st.markdown("### 🧮 Questionnaire Subscores")
    st.markdown(
        '<p style="font-size:.82rem;color:var(--text-muted);margin-top:-.4rem;margin-bottom:.8rem">'
        'Derived from your questionnaire responses. Shown for self-awareness — '
        '<strong>not used as model features</strong> (except Behavioural Intensity as '
        'a proxy for the conflicts variable in behavioral_intensity_index).</p>',
        unsafe_allow_html=True,
    )
    subscore_icons = ["📱", "🧠", "😴", "⏰", "📚"]
    cols_sc = st.columns(5)
    for i, (lbl, val) in enumerate(subscores.items()):
        pct = int(val * 100)
        sc_col = "#34d399" if pct < 35 else ("#fbbf24" if pct < 65 else "#f87171")
        with cols_sc[i]:
            st.markdown(f"""
            <div style="background:var(--bg-card);border:1px solid var(--border);border-radius:10px;
                        padding:10px;text-align:center;">
              <div style="font-size:1.3rem">{subscore_icons[i]}</div>
              <div style="font-size:.66rem;color:var(--text-muted);font-family:'Syne',sans-serif;
                          font-weight:700;letter-spacing:.05em;margin:4px 0 2px">
                {lbl}</div>
              <div style="font-size:1.05rem;font-weight:800;color:{sc_col};
                          font-family:'Syne',sans-serif">{pct}%</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Interpretation ────────────────────────────────────────────────────────
    INTERP = {
        "Low": (
            "🌱 Your social media habits appear healthy.",
            "Your feature profile sits in the low-risk zone across behavioural intensity, "
            "psychological distress, and sleep disturbance. Your usage hours are within a "
            "healthy range relative to the training population. Keep maintaining these habits."
        ),
        "Moderate": (
            "⚠️ Some warning signs detected.",
            "One or more features — likely behavioural intensity, psychological distress, "
            "or sleep disturbance — are elevated relative to the Low-risk class mean. "
            "Taking proactive steps now is the most effective way to prevent escalation."
        ),
        "High": (
            "🚨 Strong addiction risk signals detected.",
            "Multiple engineered features are significantly elevated relative to healthy-class "
            "benchmarks. Your daily usage hours, sleep hours, and/or mental health score are "
            "driving a High-risk classification. Immediate and sustained habit change is "
            "strongly recommended."
        ),
    }
    interp_title, interp_body = INTERP.get(pred_label, INTERP["Low"])
    st.markdown(f"""
    <div style="background:var(--bg-card);border:1px solid var(--border);border-left:4px solid {risk_color};
                border-radius:12px;padding:18px 20px;margin:1.5rem 0;">
      <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;
                  color:{risk_color};margin-bottom:8px">{interp_title}</div>
      <div style="font-size:.87rem;color:var(--text-secondary);line-height:1.7">{interp_body}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Recommendations ───────────────────────────────────────────────────────
    st.markdown("### 💡 Recommendations")
    RECS = {
        "Low": [
            ("🎯 Stay Consistent", "Healthy habits are fragile. Set a monthly reminder to re-take this screener and track any changes over time."),
            ("📱 Keep Monitoring", "Use built-in screen time tools (iOS Screen Time / Android Digital Wellbeing) for passive weekly awareness."),
            ("🌍 Invest Offline", "Prioritise offline hobbies and face-to-face connections — these are your strongest long-term buffers."),
        ],
        "Moderate": [
            ("⏱️ Set Daily App Limits", "Enforce a 45-minute daily social media budget using app timers. Resist overriding for the first three days."),
            ("🌙 Phone-Free Bedtime", "No social media 60 minutes before sleep. Replace scrolling with reading or journaling to wind down properly."),
            ("🧠 Mindful Opening", "Before opening any app, ask: 'Why am I opening this?' If you can't answer, close it immediately."),
            ("📅 Weekly Detox Day", "Choose one day per week with zero social media. Fill it with a planned offline activity — structure is key."),
        ],
        "High": [
            ("🚫 Hard Limit — Start Today", "Set a strict 1-hour daily cap using both your phone's built-in limits AND a third-party app (Opal / Cold Turkey) for redundancy."),
            ("🗑️ Remove App Shortcuts", "Delete social media apps from your home screen. Access via browser only — this single change reduces usage 20–30%."),
            ("🌑 Grayscale Mode Evenings", "Enable phone grayscale after 8pm. Removing colour significantly reduces the dopamine reward from scrolling."),
            ("📝 Replace the Habit Loop", "Identify your top 3 triggers and write a concrete replacement action for each. Post it where you can see it daily."),
            ("🩺 Seek Professional Support", "CBT has strong evidence for behavioural addiction. Contact your university counselling service or a digital-wellness therapist."),
        ],
    }

    recs = RECS.get(pred_label, RECS["Low"])
    cols_rec = st.columns(min(len(recs), 3))
    for i, (title, desc) in enumerate(recs):
        with cols_rec[i % len(cols_rec)]:
            st.markdown(f"""
            <div class="rec-card">
              <div class="rec-title">{title}</div>
              <div class="rec-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Inputs + features expander ────────────────────────────────────────────
    with st.expander("📋 View submitted inputs, engineered features, and subscores"):
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            st.markdown("**Core Inputs → Model Features**")
            st.table(pd.DataFrame(core_inputs.items(), columns=["Input", "Value"]))
            st.markdown("**Engineered Feature Vector**")
            feat_df = pd.DataFrame(
                [(k, f"{v:.4f}") for k, v in feat_vals.items()],
                columns=["Feature", "Value"],
            )
            st.table(feat_df)
        with col_exp2:
            st.markdown("**Questionnaire Subscores (diagnostic only)**")
            ss_df = pd.DataFrame(
                [(k, f"{int(v*100)}%") for k, v in subscores.items()],
                columns=["Dimension", "Score"],
            )
            st.table(ss_df)
            st.markdown("**Class Probabilities**")
            prob_df = pd.DataFrame(
                [(k, f"{v*100:.1f}%") for k, v in probs.items()],
                columns=["Class", "Probability"],
            )
            st.table(prob_df)

    # ── Navigation ────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2, 3, 2])
    with c1:
        if st.button("← Retake Survey", use_container_width=True):
            go("survey")
    with c3:
        if st.button("🏠 Home", use_container_width=True):
            go("about")

    # ── Download Diagnostic Report ────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📄 Download Your Diagnostic Report")
    st.markdown(
        '<p style="font-size:.83rem;color:var(--text-muted);margin-top:-.3rem;margin-bottom:.8rem">'
        'Generate a full PDF report of your results, features, and personalised recommendations.</p>',
        unsafe_allow_html=True,
    )

    pdf_bytes = generate_diagnostic_pdf(
        pred_label=pred_label,
        probs=probs,
        feat_vals=feat_vals,
        subscores=subscores,
        core_inputs=core_inputs,
    )

    pdf_filename = f"{pred_label}_Risk_Diagnostic_Report.pdf"

    st.download_button(
        label="⬇️ Download Diagnostic Report (PDF)",
        data=pdf_bytes,
        file_name=pdf_filename,
        mime="application/pdf",
        use_container_width=True,
        type="primary",
    )

    st.markdown(
        '<div class="footer-note">⚠️ Research prototype only — not a clinical diagnostic tool. '
        'Results do not replace professional mental health advice.</div>',
        unsafe_allow_html=True,
    )