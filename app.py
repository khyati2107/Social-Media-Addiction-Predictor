import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Social Media Addiction Risk Screener",
    page_icon="📱",
    layout="centered"
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg-primary:    #1a1033;
    --bg-secondary:  #231744;
    --bg-card:       #2a1d52;
    --bg-card-hover: #321f64;
    --accent:        #c084fc;
    --accent-yellow: #fbbf24;
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
h1 a, h2 a, h3 a, h4 a, h5 a, h6 a { display: none !important; }

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

[data-testid="stSelectbox"] > div > div,
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stNumberInput"] label { color: var(--text-secondary) !important; }
[data-testid="stSelectbox"] > div > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
}
[data-testid="stSlider"] > div > div > div > div { background: var(--accent) !important; }

[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary { color: var(--text-secondary) !important; }

[data-testid="stProgress"] > div > div { background: var(--accent) !important; }
[data-testid="stProgress"] > div { background: rgba(192,132,252,0.15) !important; }
[data-testid="stProgress"] p {
    color: var(--text-primary) !important;
    background: transparent !important;
    -webkit-text-fill-color: var(--text-primary) !important;
}

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

[data-testid="stDownloadButton"] button {
    background: linear-gradient(135deg, #7c3aed, #a855f7) !important;
    border: none !important; color: white !important;
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.4) !important;
}

[data-testid="stAlert"] { background: var(--bg-card) !important; border-radius: 10px !important; }

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

/* Bubble radio */
div.bubble-wrap div[data-testid="stRadio"] > label { display: none !important; }
div.bubble-wrap div[role="radiogroup"] {
    display: flex !important; flex-direction: row !important;
    justify-content: flex-start !important; align-items: center !important;
    gap: 20px !important; background: transparent !important;
}
div.bubble-wrap div[role="radiogroup"] > label {
    display: flex !important; flex-direction: column !important;
    align-items: center !important; cursor: pointer !important;
    margin: 0 !important; padding: 0 !important; gap: 0 !important;
}
div.bubble-wrap div[role="radiogroup"] > label > div:first-child { display: none !important; }
div.bubble-wrap div[role="radiogroup"] > label > div:last-child > p {
    width: 52px !important; height: 52px !important; border-radius: 50% !important;
    background: var(--bg-card) !important; margin: 0 !important; font-size: 0 !important;
    color: transparent !important; cursor: pointer !important; box-sizing: border-box !important;
    transition: transform 0.15s ease, background 0.2s ease, box-shadow 0.2s ease !important;
}
div.bubble-wrap div[role="radiogroup"] > label:hover > div:last-child > p
    { transform: scale(1.12) !important; box-shadow: 0 0 12px var(--accent-glow) !important; }
div.bubble-wrap div[role="radiogroup"] > label:nth-child(1) > div:last-child > p
    { border: 2.5px solid #34d399 !important; }
div.bubble-wrap div[role="radiogroup"] > label:nth-child(2) > div:last-child > p
    { border: 2.5px solid #60a5fa !important; }
div.bubble-wrap div[role="radiogroup"] > label:nth-child(3) > div:last-child > p
    { border: 2.5px solid #c084fc !important; }
div.bubble-wrap div[role="radiogroup"] > label:nth-child(4) > div:last-child > p
    { border: 2.5px solid #f472b6 !important; }
div.bubble-wrap div[role="radiogroup"] > label:nth-child(5) > div:last-child > p
    { border: 2.5px solid #f87171 !important; }
div.bubble-wrap div[role="radiogroup"] > label:has(input:checked):nth-child(1) > div:last-child > p
    { background: #34d399 !important; border-color: #34d399 !important; box-shadow: 0 0 14px rgba(52,211,153,0.5) !important; }
div.bubble-wrap div[role="radiogroup"] > label:has(input:checked):nth-child(2) > div:last-child > p
    { background: #60a5fa !important; border-color: #60a5fa !important; box-shadow: 0 0 14px rgba(96,165,250,0.5) !important; }
div.bubble-wrap div[role="radiogroup"] > label:has(input:checked):nth-child(3) > div:last-child > p
    { background: #c084fc !important; border-color: #c084fc !important; box-shadow: 0 0 14px rgba(192,132,252,0.5) !important; }
div.bubble-wrap div[role="radiogroup"] > label:has(input:checked):nth-child(4) > div:last-child > p
    { background: #f472b6 !important; border-color: #f472b6 !important; box-shadow: 0 0 14px rgba(244,114,182,0.5) !important; }
div.bubble-wrap div[role="radiogroup"] > label:has(input:checked):nth-child(5) > div:last-child > p
    { background: #f87171 !important; border-color: #f87171 !important; box-shadow: 0 0 14px rgba(248,113,113,0.5) !important; }

.bubble-end-labels {
    display: flex !important; justify-content: center !important;
    gap: 160px !important; padding: 4px 0 14px !important;
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

.prob-bar-wrap { background: rgba(255,255,255,0.07); border-radius: 6px; height: 22px; margin-bottom: 6px; }

.rec-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 12px; padding: 16px; height: 100%; font-size: .85rem;
    transition: border-color 0.2s, transform 0.2s;
}
.rec-card:hover { border-color: var(--border-strong); transform: translateY(-2px); }
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

.feat-bar-label {
    font-size: 0.78rem; color: var(--text-secondary);
    font-family: 'DM Sans', sans-serif; margin-bottom: 3px;
}

.subscore-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 16px; margin-bottom: 10px;
}
.subscore-title {
    font-family: 'Syne', sans-serif; font-size: .82rem; font-weight: 700;
    color: var(--text-primary); margin-bottom: 6px;
}
.subscore-bar-bg {
    background: rgba(255,255,255,0.07); border-radius: 4px; height: 8px;
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
    'age_norm':                     '🎂 Age (normalised)',
    'is_student':                   '🎓 Student Status',
    'usage_hours_sq':               '⏰ Usage² (non-linear)',
}

# ── BENCHMARK AVERAGES (replace with dataset means from features_train.csv) ──
# These are approximate population midpoints for the "your profile vs average" chart.
# TO REPLACE: compute train_feats.mean() from outputs/features_train.csv
BENCHMARK_FEAT_AVGS = {
    'behavioral_intensity_index':   0.48,   # placeholder — replace with train mean
    'psychological_distress_index': 0.45,   # placeholder
    'sleep_disturbance_proxy':      0.42,   # placeholder
    'age_norm':                     0.43,   # from domain shift report (Nusratt train)
    'is_student':                   1.00,   # Nusratt is all students
    'usage_hours_sq':               0.44,   # from domain shift report
}

# Subscore benchmarks (0–1 scale) — replace with actual dataset subscore means
BENCHMARK_SUBSCORE_AVGS = {
    'Behavioural Intensity':   0.48,
    'Psychological Distress':  0.45,
    'Sleep Disturbance':       0.42,
    'Usage Pattern':           0.50,
    'Academic Impact':         0.40,
}

# ── Artifact loading ──────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    errors = []
    artifacts = {}

    model_path = None
    for candidate in ["models/clf_xgb.pkl", "models/clf_rf.pkl",
                       "models/clf_lr.pkl", "models/clf_svm.pkl"]:
        if os.path.exists(candidate):
            model_path = candidate
            break
    if model_path is None:
        errors.append("No classification model found in models/clf_*.pkl")
    else:
        with open(model_path, "rb") as f:
            artifacts["model"] = pickle.load(f)
        artifacts["model_name"] = os.path.basename(model_path).replace(".pkl", "")

    for key, path in [
        ("scaler",    "models/scaler_clf.pkl"),
        ("le",        "models/label_encoder.pkl"),
        ("fe_params", "models/fe_params.pkl"),
        ("feat_cols", "models/feature_cols.pkl"),
        ("imputer",   "models/imputer.pkl"),
    ]:
        if os.path.exists(path):
            with open(path, "rb") as f:
                artifacts[key] = pickle.load(f)
        else:
            errors.append(f"Missing required artifact: {path}")

    return artifacts, errors

artifacts, LOAD_ERRORS = load_artifacts()

# ═════════════════════════════════════════════════════════════════════════════
# EXPANDED QUESTIONNAIRE SUBSCORE CALCULATORS
# Each function takes raw 1–5 Likert responses and returns a 0–1 normalised
# subscore. These subscores are then fused into the 6 model features.
# ═════════════════════════════════════════════════════════════════════════════

def calc_behavioural_intensity_subscore(
    purposeless_opening: int,     # How often do you open apps without intent?
    compulsive_checking: int,     # How often do you check notifs immediately?
    mindless_scrolling:  int,     # How often do you scroll past content without reading?
    impulse_posting:     int,     # How often do you post/react without thinking?
) -> float:
    """
    Behavioural intensity subscore (0–1).
    Higher = more compulsive, impulsive engagement.
    Maps into behavioral_intensity_index alongside usage hours.
    """
    raw = np.mean([purposeless_opening, compulsive_checking,
                   mindless_scrolling, impulse_posting])
    return (raw - 1) / 4.0   # scale 1–5 → 0–1


def calc_psychological_distress_subscore(
    low_mood_freq:       int,   # How often do you feel low/sad without clear reason?
    anxiety_freq:        int,   # How often do you feel anxious after using SM?
    fomo_freq:           int,   # How often do you fear missing out if offline?
    self_esteem_impact:  int,   # How often does SM make you feel inadequate?
    irritability_freq:   int,   # How often do you feel irritable when unable to use SM?
) -> float:
    """
    Psychological distress subscore (0–1).
    Higher = more psychological harm from SM use.
    Maps into psychological_distress_index.
    Note: mental_health_score from slider is ALSO used, giving a blended signal.
    """
    raw = np.mean([low_mood_freq, anxiety_freq, fomo_freq,
                   self_esteem_impact, irritability_freq])
    return (raw - 1) / 4.0


def calc_sleep_disturbance_subscore(
    late_night_use:      int,   # How often do you use SM in bed before sleep?
    sleep_delay:         int,   # How often does SM delay you falling asleep?
    night_wakeup_check:  int,   # How often do you wake up to check SM at night?
    morning_first_check: int,   # How often is SM the first thing you check?
) -> float:
    """
    Sleep disturbance subscore (0–1).
    Higher = more sleep disruption due to SM.
    Maps into sleep_disturbance_proxy alongside sleep_hours.
    """
    raw = np.mean([late_night_use, sleep_delay, night_wakeup_check, morning_first_check])
    return (raw - 1) / 4.0


def calc_usage_pattern_subscore(
    weekend_binge:       int,   # How much more do you use SM on weekends vs weekdays?
    usage_spike_stress:  int,   # How often does stress increase your SM use?
    passive_consume:     int,   # How much time do you spend just consuming (not creating)?
    multi_platform_hop:  int,   # How often do you switch between multiple SM platforms?
) -> float:
    """
    Usage pattern subscore (0–1).
    Higher = more problematic usage patterns (bingeing, stress-driven, passive).
    Used to modulate the usage_hours_sq feature.
    """
    raw = np.mean([weekend_binge, usage_spike_stress, passive_consume, multi_platform_hop])
    return (raw - 1) / 4.0


def calc_academic_impact_subscore(
    distraction_study:   int,   # How often does SM distract you while studying?
    deadline_procrastin: int,   # How often do you use SM to procrastinate on deadlines?
    concentration_loss:  int,   # How often do you lose focus during tasks due to SM urge?
    assignment_delay:    int,   # How often have you submitted work late due to SM?
) -> float:
    """
    Academic / concentration impact subscore (0–1).
    Higher = more academic disruption.
    Feeds into conflicts proxy (compulsive-engagement conflict signal).
    """
    raw = np.mean([distraction_study, deadline_procrastin,
                   concentration_loss, assignment_delay])
    return (raw - 1) / 4.0


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING — fuses expanded subscores + direct inputs into the
# same 6 model features as the original pipeline (04_feature_engineering_shared.py)
# ═════════════════════════════════════════════════════════════════════════════

def build_feature_row(user_inputs: dict, subscores: dict,
                      params: dict, imputer, feat_cols: list) -> pd.DataFrame:
    """
    Build the 6-feature vector that the trained model expects.

    Fusion strategy (preserves original FE math, augments with subscores):
      - behavioral_intensity_index : blends usage_norm + conflicts_norm,
            where conflicts_norm is now enriched by behavioural + academic subscores.
      - psychological_distress_index : inverts mental_health_score, further
            weighted by psychological distress subscore.
      - sleep_disturbance_proxy : blends inverted sleep_hours with sleep subscore.
      - age_norm : unchanged direct input.
      - is_student : unchanged from academic_level.
      - usage_hours_sq : squared clipped usage, scaled by usage pattern subscore.
    """
    age           = float(user_inputs["age"])
    usage_hours   = float(user_inputs["avg_daily_usage_hours"])
    mental_health = float(user_inputs["mental_health_score"])
    sleep_hours   = float(user_inputs["sleep_hours"])
    academic_level= user_inputs["academic_level"]

    beh_sub  = subscores["behavioural_intensity"]   # 0–1
    psy_sub  = subscores["psychological_distress"]  # 0–1
    slp_sub  = subscores["sleep_disturbance"]       # 0–1
    usg_sub  = subscores["usage_pattern"]           # 0–1
    acad_sub = subscores["academic_impact"]         # 0–1

    # ── 1. behavioral_intensity_index ────────────────────────────────────────
    # Original: (usage_norm + conflicts_norm) / 2
    # Enhanced: conflicts_norm = weighted blend of raw conflicts + behavioural
    #           + academic subscores (all reflect compulsive/conflict engagement)
    usage_norm     = np.clip(usage_hours / params["usage_max"], 0, 1)
    # Synthesise an enriched conflicts signal from subscores (0–1 range matches)
    enriched_conflicts_norm = np.clip(
        0.4 * beh_sub + 0.35 * acad_sub + 0.25 * usg_sub, 0, 1
    )
    behavioral_intensity_index = (usage_norm + enriched_conflicts_norm) / 2

    # ── 2. psychological_distress_index ──────────────────────────────────────
    # Original: 1 - normalised(mental_health_score)
    # Enhanced: blend direct MH score inversion with questionnaire distress subscore
    mh_range = params["mh_max"] - params["mh_min"] or 1
    mh_direct_distress = 1 - (
        (np.clip(mental_health, params["mh_min"], params["mh_max"]) - params["mh_min"]) / mh_range
    )
    # 70% direct MH score (ground truth), 30% questionnaire signal
    psychological_distress_index = np.clip(
        0.70 * mh_direct_distress + 0.30 * psy_sub, 0, 1
    )

    # ── 3. sleep_disturbance_proxy ───────────────────────────────────────────
    # Original: 1 - normalised(sleep_hours)
    # Enhanced: blend hours-based disturbance with behavioural sleep subscore
    sleep_range = params["sleep_max"] - params["sleep_min"] or 1
    sleep_hours_disturbance = 1 - (
        (np.clip(sleep_hours, params["sleep_min"], params["sleep_max"]) - params["sleep_min"]) / sleep_range
    )
    # 65% sleep hours (objective proxy), 35% self-reported sleep behaviour
    sleep_disturbance_proxy = np.clip(
        0.65 * sleep_hours_disturbance + 0.35 * slp_sub, 0, 1
    )

    # ── 4. age_norm ──────────────────────────────────────────────────────────
    age_range = params["age_max"] - params["age_min"] or 1
    age_norm = (np.clip(age, params["age_min"], params["age_max"]) - params["age_min"]) / age_range

    # ── 5. is_student ────────────────────────────────────────────────────────
    student_levels = {"High School", "Undergraduate", "Graduate"}
    is_student = float(academic_level in student_levels)

    # ── 6. usage_hours_sq ────────────────────────────────────────────────────
    # Original: (clipped_usage²) / usage_sq_max
    # Enhanced: scale by usage pattern subscore to capture binge/stress patterns
    clipped_usage  = np.clip(usage_hours, 0, params["usage_max"])
    base_usage_sq  = np.clip(clipped_usage ** 2, 0, params["usage_sq_max"]) / params["usage_sq_max"]
    # Modulate: heavy binge/stress patterns push the non-linear term up
    usage_hours_sq = np.clip(base_usage_sq * (0.75 + 0.25 * usg_sub), 0, 1)

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


def run_inference(user_inputs: dict, subscores: dict):
    """Full inference pipeline. Returns (pred_label, probs_dict, feat_row_df)."""
    params    = artifacts["fe_params"]
    imputer   = artifacts["imputer"]
    feat_cols = artifacts["feat_cols"]
    scaler    = artifacts["scaler"]
    le        = artifacts["le"]
    model     = artifacts["model"]

    feat_row    = build_feature_row(user_inputs, subscores, params, imputer, feat_cols)
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

    return pred_label, probs, feat_row

# ── Bubble scale widget ───────────────────────────────────────────────────────
def bubble_scale(label: str, key: str,
                 lo_label="Never", hi_label="Always", default: int = 2) -> int:
    if key not in st.session_state:
        st.session_state[key] = default
    st.markdown(
        f'<div class="bubble-wrap"><p class="q-text">{label}</p>',
        unsafe_allow_html=True
    )
    val = st.radio(
        label, options=[1, 2, 3, 4, 5],
        index=st.session_state[key] - 1,
        horizontal=True, key=f"_radio_{key}",
        label_visibility="collapsed",
    )
    st.markdown(
        f'<div class="bubble-end-labels"><span>{lo_label}</span><span>{hi_label}</span></div></div>',
        unsafe_allow_html=True
    )
    st.session_state[key] = val
    return val

# ── Session state ─────────────────────────────────────────────────────────────
for _k, _v in [("page", "about"), ("inputs", {}), ("subscores", {}),
               ("scroll_to_top", False)]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

def trigger_scroll():
    st.session_state.scroll_to_top = True

st.markdown('<div id="page-top"></div>', unsafe_allow_html=True)
if st.session_state.scroll_to_top:
    components.html("""
    <script>
      function su() {
        var a = window.parent.document.getElementById('page-top');
        if (a) a.scrollIntoView({behavior:'instant'});
        else window.parent.scrollTo({top:0,behavior:'instant'});
      }
      su(); setTimeout(su,80); setTimeout(su,300);
    </script>""", height=0)
    st.session_state.scroll_to_top = False

# ═════════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ═════════════════════════════════════════════════════════════════════════════

CHART_BG      = "#1a1033"
CHART_CARD_BG = "#2a1d52"
CHART_TEXT    = "#f3f0ff"
CHART_MUTED   = "#7c6da0"
CHART_ACCENT  = "#c084fc"
# Matplotlib-safe RGBA tuples (R, G, B, A) — CSS rgba() is not valid in Matplotlib
CHART_SPINE_COLOR = (0.753, 0.518, 0.988, 0.15)   # rgba(192,132,252,0.15) equivalent
CHART_GRID_COLOR  = (0.753, 0.518, 0.988, 0.08)   # rgba(192,132,252,0.08) equivalent

def _fig_base(figsize=(7, 3.5)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_CARD_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(CHART_SPINE_COLOR)
        spine.set_linewidth(0.8)
    ax.tick_params(colors=CHART_MUTED, labelsize=8)
    ax.xaxis.label.set_color(CHART_MUTED)
    ax.yaxis.label.set_color(CHART_MUTED)
    return fig, ax


def chart_probability_bars(probs: dict) -> bytes:
    """Horizontal bar chart of class probabilities."""
    fig, ax = _fig_base(figsize=(6, 2.6))
    labels = list(probs.keys())
    values = [probs[l] * 100 for l in labels]
    colors = [LABEL_COLORS[LABEL_NAMES.index(l)] if l in LABEL_NAMES else CHART_ACCENT
              for l in labels]
    bars = ax.barh(labels, values, color=colors, height=0.5,
                   edgecolor="none", alpha=0.88)
    for bar, val in zip(bars, values):
        ax.text(min(val + 1.5, 98), bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", ha="left",
                color=CHART_TEXT, fontsize=9, fontweight="bold",
                fontfamily="DejaVu Sans")
    ax.set_xlim(0, 108)
    ax.set_xlabel("Probability (%)", color=CHART_MUTED, fontsize=8)
    ax.set_title("Class Probabilities", color=CHART_TEXT, fontsize=10,
                 fontweight="bold", pad=10)
    ax.axvline(50, color=CHART_ACCENT, linewidth=0.6, linestyle="--", alpha=0.4)
    ax.grid(axis="x", color=CHART_GRID_COLOR, linewidth=0.5)
    ax.invert_yaxis()
    plt.tight_layout(pad=1.0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=CHART_BG)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def chart_subscore_bars(subscores: dict) -> bytes:
    """Vertical bar chart of the 5 subscore dimensions."""
    labels = list(subscores.keys())
    values = [subscores[l] for l in labels]
    colors = []
    for v in values:
        if v < 0.35:   colors.append("#34d399")
        elif v < 0.65: colors.append("#fbbf24")
        else:           colors.append("#f87171")

    fig, ax = _fig_base(figsize=(7, 3.4))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, width=0.55, edgecolor="none", alpha=0.88)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.2f}", ha="center", va="bottom",
                color=CHART_TEXT, fontsize=8, fontweight="bold")
    short_labels = [l.replace(" ", "\n") for l in labels]
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=7.5, color=CHART_TEXT)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score (0–1)", color=CHART_MUTED, fontsize=8)
    ax.set_title("Your Composite Subscore Profile", color=CHART_TEXT,
                 fontsize=10, fontweight="bold", pad=10)
    ax.axhline(0.5, color=CHART_ACCENT, linewidth=0.8, linestyle="--", alpha=0.5)
    ax.grid(axis="y", color=CHART_GRID_COLOR, linewidth=0.5)
    plt.tight_layout(pad=1.0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=CHART_BG)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def chart_driver_analysis(feat_vals: dict, pred_label: str) -> bytes:
    """
    Horizontal bar chart showing which features drive the prediction.
    Bar length = feature value (0–1). Coloured by risk direction.
    High-risk features (distress, intensity, sleep disturbance) are red-toned.
    Protective features (age, is_student) are blue-toned.
    """
    risk_feats = {
        "behavioral_intensity_index":   ("#f87171", "Risk ↑"),
        "psychological_distress_index": ("#f472b6", "Risk ↑"),
        "sleep_disturbance_proxy":      ("#fb923c", "Risk ↑"),
        "usage_hours_sq":               ("#fbbf24", "Risk ↑"),
        "age_norm":                     ("#60a5fa", "Contextual"),
        "is_student":                   ("#34d399", "Contextual"),
    }
    friendly = {
        "behavioral_intensity_index":   "Behavioural Intensity",
        "psychological_distress_index": "Psychological Distress",
        "sleep_disturbance_proxy":      "Sleep Disturbance",
        "usage_hours_sq":               "Usage² (non-linear)",
        "age_norm":                     "Age (normalised)",
        "is_student":                   "Student Status",
    }

    labels = [friendly.get(k, k) for k in feat_vals]
    values = [float(feat_vals[k]) for k in feat_vals]
    colors = [risk_feats.get(k, (CHART_ACCENT, ""))[0] for k in feat_vals]

    sorted_pairs = sorted(zip(values, labels, colors), reverse=True)
    values, labels, colors = zip(*sorted_pairs) if sorted_pairs else ([], [], [])

    fig, ax = _fig_base(figsize=(6.5, 3.5))
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=colors, height=0.52,
                   edgecolor="none", alpha=0.88)
    for bar, val in zip(bars, values):
        ax.text(min(val + 0.02, 0.98), bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left",
                color=CHART_TEXT, fontsize=8, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8, color=CHART_TEXT)
    ax.set_xlim(0, 1.2)
    ax.set_xlabel("Feature Value (0–1)", color=CHART_MUTED, fontsize=8)
    ax.set_title("What Is Driving Your Prediction", color=CHART_TEXT,
                 fontsize=10, fontweight="bold", pad=10)
    ax.axvline(0.5, color=CHART_ACCENT, linewidth=0.7, linestyle="--", alpha=0.45)

    risk_patch = mpatches.Patch(color="#f87171", label="Risk ↑ features", alpha=0.8)
    ctx_patch  = mpatches.Patch(color="#60a5fa", label="Contextual features", alpha=0.8)
    ax.legend(handles=[risk_patch, ctx_patch], fontsize=7,
              facecolor=CHART_CARD_BG, edgecolor=CHART_ACCENT, labelcolor=CHART_TEXT,
              loc="lower right")

    ax.grid(axis="x", color=CHART_GRID_COLOR, linewidth=0.5)
    ax.invert_yaxis()
    plt.tight_layout(pad=1.0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=CHART_BG)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def chart_vs_average(feat_vals: dict, subscores_named: dict) -> bytes:
    """
    Grouped bar chart: your subscore profile vs dataset benchmark averages.
    Benchmarks are in BENCHMARK_SUBSCORE_AVGS — replace with real train means.
    """
    labels    = list(subscores_named.keys())
    your_vals = [subscores_named[l] for l in labels]
    avg_vals  = [BENCHMARK_SUBSCORE_AVGS.get(l, 0.5) for l in labels]

    x     = np.arange(len(labels))
    width = 0.34

    fig, ax = _fig_base(figsize=(7, 3.6))

    your_colors = []
    for v in your_vals:
        if v < 0.35:   your_colors.append("#34d399")
        elif v < 0.65: your_colors.append("#fbbf24")
        else:           your_colors.append("#f87171")

    b1 = ax.bar(x - width / 2, your_vals, width, color=your_colors,
                alpha=0.88, edgecolor="none", label="You")
    b2 = ax.bar(x + width / 2, avg_vals,  width, color="#4c3a7a",
                alpha=0.75, edgecolor="none", label="Avg User*")

    for bar, val in zip(b1, your_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.2f}", ha="center", fontsize=7.5,
                color=CHART_TEXT, fontweight="bold")
    for bar, val in zip(b2, avg_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.2f}", ha="center", fontsize=7.5, color=CHART_MUTED)

    short_labels = [l.replace(" ", "\n") for l in labels]
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=7.5, color=CHART_TEXT)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score (0–1)", color=CHART_MUTED, fontsize=8)
    ax.set_title("Your Profile vs Average User*", color=CHART_TEXT,
                 fontsize=10, fontweight="bold", pad=10)
    ax.axhline(0.5, color=CHART_ACCENT, linewidth=0.7, linestyle="--", alpha=0.4)
    ax.grid(axis="y", color=CHART_GRID_COLOR, linewidth=0.5)

    leg = ax.legend(fontsize=8, facecolor=CHART_CARD_BG,
                    edgecolor=CHART_ACCENT, labelcolor=CHART_TEXT)
    ax.text(0.01, -0.18,
            "*Benchmark averages are placeholders — replace BENCHMARK_SUBSCORE_AVGS with real train means.",
            transform=ax.transAxes, fontsize=6, color=CHART_MUTED, style="italic")

    plt.tight_layout(pad=1.0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=CHART_BG)
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — LANDING
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "about":

    if LOAD_ERRORS:
        st.error("⚠️ Some model artifacts could not be loaded.")
        for e in LOAD_ERRORS:
            st.markdown(f"- `{e}`")
        st.stop()

    components.html("""
    <!DOCTYPE html><html><head>
    <link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
    <style>
      * { margin:0; padding:0; box-sizing:border-box; }
      body { background:transparent; font-family:'DM Sans',sans-serif; overflow:hidden; }
      .hero { display:grid; grid-template-columns:1fr 1fr; gap:24px; align-items:center; padding:8px 4px 20px; min-height:340px; }
      .hero-left { display:flex; flex-direction:column; justify-content:center; }
      .eyebrow { font-family:'Syne',sans-serif; font-size:.7rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; color:#fbbf24; margin-bottom:14px; display:flex; align-items:center; gap:10px; }
      .eyebrow-line { width:22px; height:2px; background:#fbbf24; border-radius:2px; }
      h1 { font-family:'Syne',sans-serif; font-size:2.6rem; font-weight:800; line-height:1.1; color:#f3f0ff; margin-bottom:16px; }
      h1 .accent { background:linear-gradient(120deg,#c084fc 0%,#a855f7 60%,#7c3aed 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
      .body-text { font-size:.92rem; color:#9d8ec0; line-height:1.75; margin-bottom:24px; }
      .body-text strong { color:#d4c8f0; }
      .meta { font-size:.75rem; color:#5a4e78; display:flex; align-items:center; gap:8px; }
      .meta-dot { width:4px; height:4px; border-radius:50%; background:#3d2d6e; }
      .hero-right { position:relative; height:340px; display:flex; align-items:center; justify-content:center; }
      .orb-wrap { position:relative; width:260px; height:260px; flex-shrink:0; }
      .orb-wrap svg.orb-svg { width:260px; height:260px; filter:drop-shadow(0 0 40px rgba(109,40,217,0.5)); }
      .orb-inner { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; }
      .orb-inner svg.phone-svg { width:100px; height:160px; filter:drop-shadow(0 4px 20px rgba(192,132,252,0.5)); }
      .float-card { position:absolute; background:rgba(26,14,54,0.92); border:1px solid rgba(192,132,252,0.3); border-radius:14px; padding:11px 15px; backdrop-filter:blur(16px); min-width:130px; }
      .card-label { font-size:.62rem; color:#6b5e8a; font-weight:600; letter-spacing:.08em; text-transform:uppercase; display:flex; align-items:center; gap:5px; margin-bottom:4px; }
      .card-dot { width:5px; height:5px; border-radius:50%; background:#fbbf24; flex-shrink:0; }
      .card-value { font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:800; color:#f3f0ff; line-height:1; margin-bottom:2px; }
      .card-sub { font-size:.68rem; color:#6b5e8a; }
      .card-top { top:12px; right:0px; } .card-bottom { bottom:28px; left:0px; }
      .mini-bars { display:flex; align-items:flex-end; gap:3px; margin-top:7px; height:26px; }
      .mb { width:9px; border-radius:2px 2px 0 0; }
    </style></head><body>
    <div class="hero">
      <div class="hero-left">
        <div class="eyebrow"><div class="eyebrow-line"></div>ML-Powered Screener</div>
        <h1>Are You <span class="accent">Addicted</span><br>to Social Media?</h1>
        <p class="body-text">Answer <strong>30+ deep questions</strong> across 5 dimensions. Our trained ML model predicts your <strong>addiction risk tier</strong> with detailed visual analytics.</p>
        <div class="meta"><span>⏱ ~5 min</span><div class="meta-dot"></div><span>Free</span><div class="meta-dot"></div><span>No account needed</span></div>
      </div>
      <div class="hero-right">
        <div class="orb-wrap">
          <svg class="orb-svg" viewBox="0 0 280 280" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <radialGradient id="orbGrad" cx="38%" cy="32%" r="65%">
                <stop offset="0%" stop-color="#7c3aed"/><stop offset="45%" stop-color="#4c1d95"/><stop offset="100%" stop-color="#1a0a3c"/>
              </radialGradient>
              <clipPath id="cc"><circle cx="140" cy="140" r="138"/></clipPath>
            </defs>
            <circle cx="140" cy="140" r="138" fill="url(#orbGrad)"/>
            <g clip-path="url(#cc)" opacity="0.07">
              <line x1="0" y1="70" x2="280" y2="70" stroke="#c084fc" stroke-width="1"/>
              <line x1="0" y1="140" x2="280" y2="140" stroke="#c084fc" stroke-width="1"/>
              <line x1="0" y1="210" x2="280" y2="210" stroke="#c084fc" stroke-width="1"/>
              <line x1="70" y1="0" x2="70" y2="280" stroke="#c084fc" stroke-width="1"/>
              <line x1="140" y1="0" x2="140" y2="280" stroke="#c084fc" stroke-width="1"/>
              <line x1="210" y1="0" x2="210" y2="280" stroke="#c084fc" stroke-width="1"/>
            </g>
            <circle cx="140" cy="140" r="138" fill="none" stroke="rgba(192,132,252,0.3)" stroke-width="1.5"/>
          </svg>
          <div class="orb-inner">
            <svg class="phone-svg" viewBox="0 0 100 170" xmlns="http://www.w3.org/2000/svg">
              <defs>
                <linearGradient id="pg" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#a855f7"/><stop offset="100%" stop-color="#7c3aed"/></linearGradient>
                <linearGradient id="sg" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#2a1d52"/><stop offset="100%" stop-color="#1a0a3c"/></linearGradient>
              </defs>
              <rect x="8" y="4" width="84" height="162" rx="14" fill="url(#pg)" opacity="0.9"/>
              <rect x="14" y="18" width="72" height="128" rx="6" fill="url(#sg)"/>
              <rect x="36" y="10" width="28" height="6" rx="3" fill="#1a0a3c" opacity="0.8"/>
              <rect x="22" y="30" width="16" height="16" rx="4" fill="#c084fc" opacity="0.7"/>
              <rect x="42" y="30" width="16" height="16" rx="4" fill="#f472b6" opacity="0.7"/>
              <rect x="62" y="30" width="16" height="16" rx="4" fill="#fbbf24" opacity="0.7"/>
              <rect x="22" y="52" width="16" height="16" rx="4" fill="#34d399" opacity="0.7"/>
              <rect x="42" y="52" width="16" height="16" rx="4" fill="#60a5fa" opacity="0.7"/>
              <rect x="22" y="80" width="56" height="3" rx="1.5" fill="#c084fc" opacity="0.3"/>
              <rect x="22" y="88" width="40" height="3" rx="1.5" fill="#c084fc" opacity="0.2"/>
              <text x="50" y="120" font-size="22" text-anchor="middle" fill="#f472b6" opacity="0.9">♥</text>
              <circle cx="76" cy="28" r="8" fill="#f87171"/>
              <text x="76" y="32" font-size="9" text-anchor="middle" fill="white" font-weight="bold">12</text>
              <rect x="38" y="154" width="24" height="3" rx="1.5" fill="#c084fc" opacity="0.4"/>
            </svg>
          </div>
        </div>
        <div class="float-card card-top">
          <div class="card-label"><div class="card-dot"></div>Model accuracy</div>
          <div class="card-value" style="color:#34d399">97.9%</div>
          <div class="card-sub">on Nusratt test set</div>
        </div>
        <div class="float-card card-bottom">
          <div class="card-label"><div class="card-dot"></div>5 dimensions</div>
          <div class="card-value" style="color:#c084fc">30+</div>
          <div class="card-sub">deep questions</div>
        </div>
      </div>
    </div>
    </body></html>""", height=400, scrolling=False)

    st.markdown("""
    <style>
    .how-strip { background:linear-gradient(135deg,#2a1d52 0%,#1e1040 100%); border:1px solid rgba(192,132,252,0.15); border-radius:18px; padding:28px 32px; margin:4px 0 0; display:grid; grid-template-columns:auto 1fr; gap:28px; align-items:center; }
    .how-icon-box { width:50px; height:50px; background:rgba(192,132,252,0.12); border:1px solid rgba(192,132,252,0.25); border-radius:14px; display:flex; align-items:center; justify-content:center; font-size:1.5rem; flex-shrink:0; }
    .how-text-block h3 { font-family:'Syne',sans-serif; font-size:1.15rem; font-weight:800; color:#f3f0ff; margin:0 0 6px; }
    .how-text-block p { font-size:.84rem; color:#7c6da0; line-height:1.65; margin:0; }
    .how-text-block p strong { color:#b8a8d8; }
    .feat-label { font-family:'Syne',sans-serif; font-size:.68rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; color:#c084fc; margin:36px 0 18px; display:flex; align-items:center; gap:14px; }
    .feat-label::after { content:''; flex:1; height:1px; background:linear-gradient(90deg,rgba(192,132,252,0.25),transparent); }
    .feat-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-bottom:32px; }
    .feat-card { background:rgba(42,29,82,0.55); border:1px solid rgba(192,132,252,0.12); border-radius:14px; padding:18px; transition:border-color .2s,transform .2s; }
    .feat-card:hover { border-color:rgba(192,132,252,0.32); transform:translateY(-3px); }
    .feat-num { font-family:'Syne',sans-serif; font-size:.6rem; font-weight:800; color:rgba(76,29,149,0.7); letter-spacing:.12em; margin-bottom:10px; }
    .feat-icon { width:38px; height:38px; background:rgba(192,132,252,0.1); border-radius:9px; display:flex; align-items:center; justify-content:center; font-size:1.1rem; margin-bottom:10px; }
    .feat-card h4 { font-family:'Syne',sans-serif; font-size:.85rem; font-weight:700; color:#f3f0ff; margin:0 0 5px; }
    .feat-card p { font-size:.75rem; color:#6b5e8a; margin:0; line-height:1.55; }
    .risk-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-bottom:32px; }
    .risk-card { background:rgba(42,29,82,0.55); border-radius:14px; padding:16px 18px; border-top:1px solid rgba(192,132,252,0.1); border-right:1px solid rgba(192,132,252,0.1); border-bottom:1px solid rgba(192,132,252,0.1); }
    .risk-card-top { display:flex; align-items:center; gap:8px; margin-bottom:7px; }
    .risk-dot { width:9px; height:9px; border-radius:50%; flex-shrink:0; }
    .risk-title { font-family:'Syne',sans-serif; font-size:.8rem; font-weight:700; color:#f3f0ff; }
    .risk-desc { font-size:.75rem; color:#6b5e8a; line-height:1.55; }
    </style>
    <div class="how-strip">
      <div class="how-icon-box">🧬</div>
      <div class="how-text-block">
        <h3>How We Analyse Your Risk</h3>
        <p>We ask <strong>30+ deep questions</strong> across 5 subtopic dimensions. Your answers are combined into <strong>5 subscores</strong>, then fused into the 6 model features (behavioural intensity, psychological distress, sleep disturbance, age, student status, usage²) that the trained ML classifier expects.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="feat-label">What you get</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="feat-grid">
      <div class="feat-card"><div class="feat-num">01</div><div class="feat-icon">📊</div><h4>Risk Prediction</h4><p>Low, Moderate, or High — with model confidence %</p></div>
      <div class="feat-card"><div class="feat-num">02</div><div class="feat-icon">📈</div><h4>Probability Chart</h4><p>Visual confidence bars for all three risk classes</p></div>
      <div class="feat-card"><div class="feat-num">03</div><div class="feat-icon">🧮</div><h4>5 Subscores</h4><p>Behavioural, distress, sleep, usage pattern, academic impact</p></div>
      <div class="feat-card"><div class="feat-num">04</div><div class="feat-icon">🔎</div><h4>Driver Analysis</h4><p>Which features are most responsible for your prediction</p></div>
      <div class="feat-card"><div class="feat-num">05</div><div class="feat-icon">👥</div><h4>vs Average User</h4><p>How your profile compares to the dataset benchmark</p></div>
      <div class="feat-card"><div class="feat-num">06</div><div class="feat-icon">💡</div><h4>Recommendations</h4><p>Targeted action steps matched to your risk level</p></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="feat-label">The three risk tiers</div>', unsafe_allow_html=True)
    st.markdown("""
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
        <div class="risk-desc">Addicted Score ≥ 7. Strong dependency. Immediate habit change recommended.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 3, 2])
    with col2:
        if st.button("Start the Screener →", type="primary", use_container_width=True):
            st.session_state.page = "survey"
            trigger_scroll()
            st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SURVEY (expanded, 30+ questions across 5 sections)
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "survey":

    if LOAD_ERRORS:
        st.error("Model artifacts missing.")
        for e in LOAD_ERRORS:
            st.code(e)
        st.stop()

    st.markdown('<div class="pill">📋 Deep Questionnaire — 5 Dimensions</div>', unsafe_allow_html=True)
    st.markdown("## Tell us about yourself")
    st.markdown(
        '<p style="color:var(--text-muted);font-size:.88rem;margin-top:-.5rem;margin-bottom:1.5rem">'
        'Rate each statement honestly on a 1–5 scale. All responses stay on your device.</p>',
        unsafe_allow_html=True
    )

    # ── SECTION 0: Core Demographics ─────────────────────────────────────────
    st.markdown('<div class="section-header">👤 Section 0 — About You</div>', unsafe_allow_html=True)
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        age = st.number_input("Age", min_value=10, max_value=80, value=20, step=1)
    with col_d2:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    col_d3, col_d4 = st.columns(2)
    with col_d3:
        academic_level = st.selectbox(
            "Academic Level",
            ["Undergraduate", "Graduate", "High School", "Other"]
        )
    with col_d4:
        relationship_status = st.selectbox(
            "Relationship Status",
            ["Single", "In Relationship", "Married", "Complicated"]
        )

    col_d5, col_d6 = st.columns(2)
    with col_d5:
        avg_daily_usage_hours = st.slider(
            "Average daily social media usage (hours)",
            min_value=0.0, max_value=12.0, value=3.0, step=0.5
        )
    with col_d6:
        sleep_hours = st.slider(
            "Average sleep hours per night",
            min_value=2.0, max_value=12.0, value=7.0, step=0.5
        )

    mental_health_score = st.slider(
        "Overall mental health score (1 = very poor, 10 = excellent)",
        min_value=1, max_value=10, value=6, step=1
    )

    most_used_platform = st.selectbox(
        "Most Used Platform",
        ["Instagram", "TikTok", "Facebook", "Twitter/X", "Snapchat",
         "YouTube", "LinkedIn", "WhatsApp", "Reddit", "Other"]
    )

    # ── SECTION A: Behavioural Intensity (4 questions) ────────────────────────
    st.markdown(
        '<div class="section-header">📱 Section A — Behavioural Intensity</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="font-size:.8rem;color:var(--text-muted);margin-bottom:.5rem">'
        'How compulsive and automatic is your social media behaviour? (1 = Never, 5 = Always)</p>',
        unsafe_allow_html=True
    )

    beh_q1 = bubble_scale(
        "How often do you open a social media app without a specific reason or intention?",
        key="beh_q1", lo_label="Never", hi_label="Always", default=2
    )
    beh_q2 = bubble_scale(
        "How often do you check notifications within seconds of hearing an alert?",
        key="beh_q2", lo_label="Never", hi_label="Always", default=2
    )
    beh_q3 = bubble_scale(
        "How often do you scroll past content mindlessly without actually reading it?",
        key="beh_q3", lo_label="Never", hi_label="Always", default=2
    )
    beh_q4 = bubble_scale(
        "How often do you like, share, or comment impulsively without thinking?",
        key="beh_q4", lo_label="Never", hi_label="Always", default=2
    )

    beh_subscore = calc_behavioural_intensity_subscore(beh_q1, beh_q2, beh_q3, beh_q4)
    beh_pct = int(beh_subscore * 100)
    beh_col = "#34d399" if beh_pct < 35 else ("#fbbf24" if beh_pct < 65 else "#f87171")
    st.markdown(f"""
    <div class="subscore-card">
      <div class="subscore-title">📱 Behavioural Intensity Subscore: <span style="color:{beh_col}">{beh_pct}%</span></div>
      <div class="subscore-bar-bg">
        <div style="width:{beh_pct}%;height:8px;background:{beh_col};border-radius:4px;opacity:.85;"></div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── SECTION B: Psychological Distress (5 questions) ───────────────────────
    st.markdown(
        '<div class="section-header">🧠 Section B — Psychological Distress</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="font-size:.8rem;color:var(--text-muted);margin-bottom:.5rem">'
        'How is social media affecting your emotional and mental wellbeing? (1 = Never, 5 = Always)</p>',
        unsafe_allow_html=True
    )

    psy_q1 = bubble_scale(
        "How often do you feel low or down in mood without a clear reason, linked to SM use?",
        key="psy_q1", lo_label="Never", hi_label="Always", default=2
    )
    psy_q2 = bubble_scale(
        "How often do you feel anxious, restless, or on-edge after using social media?",
        key="psy_q2", lo_label="Never", hi_label="Always", default=2
    )
    psy_q3 = bubble_scale(
        "How often do you fear missing out (FOMO) when you haven't checked social media?",
        key="psy_q3", lo_label="Never", hi_label="Always", default=2
    )
    psy_q4 = bubble_scale(
        "How often does comparing yourself to others on social media make you feel inadequate?",
        key="psy_q4", lo_label="Never", hi_label="Always", default=2
    )
    psy_q5 = bubble_scale(
        "How often do you feel irritable or agitated when you are unable to use social media?",
        key="psy_q5", lo_label="Never", hi_label="Always", default=2
    )

    psy_subscore = calc_psychological_distress_subscore(
        psy_q1, psy_q2, psy_q3, psy_q4, psy_q5
    )
    psy_pct = int(psy_subscore * 100)
    psy_col = "#34d399" if psy_pct < 35 else ("#fbbf24" if psy_pct < 65 else "#f87171")
    st.markdown(f"""
    <div class="subscore-card">
      <div class="subscore-title">🧠 Psychological Distress Subscore: <span style="color:{psy_col}">{psy_pct}%</span></div>
      <div class="subscore-bar-bg">
        <div style="width:{psy_pct}%;height:8px;background:{psy_col};border-radius:4px;opacity:.85;"></div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── SECTION C: Sleep Disturbance (4 questions) ────────────────────────────
    st.markdown(
        '<div class="section-header">😴 Section C — Sleep Disturbance</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="font-size:.8rem;color:var(--text-muted);margin-bottom:.5rem">'
        'How much does social media disrupt your sleep? (1 = Never, 5 = Always)</p>',
        unsafe_allow_html=True
    )

    slp_q1 = bubble_scale(
        "How often do you use social media while in bed just before trying to sleep?",
        key="slp_q1", lo_label="Never", hi_label="Always", default=2
    )
    slp_q2 = bubble_scale(
        "How often does social media scrolling delay the time you actually fall asleep?",
        key="slp_q2", lo_label="Never", hi_label="Always", default=2
    )
    slp_q3 = bubble_scale(
        "How often do you wake up in the middle of the night to check social media?",
        key="slp_q3", lo_label="Never", hi_label="Always", default=2
    )
    slp_q4 = bubble_scale(
        "How often is checking social media the very first thing you do after waking up?",
        key="slp_q4", lo_label="Never", hi_label="Always", default=2
    )

    slp_subscore = calc_sleep_disturbance_subscore(slp_q1, slp_q2, slp_q3, slp_q4)
    slp_pct = int(slp_subscore * 100)
    slp_col = "#34d399" if slp_pct < 35 else ("#fbbf24" if slp_pct < 65 else "#f87171")
    st.markdown(f"""
    <div class="subscore-card">
      <div class="subscore-title">😴 Sleep Disturbance Subscore: <span style="color:{slp_col}">{slp_pct}%</span></div>
      <div class="subscore-bar-bg">
        <div style="width:{slp_pct}%;height:8px;background:{slp_col};border-radius:4px;opacity:.85;"></div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── SECTION D: Usage Pattern (4 questions) ────────────────────────────────
    st.markdown(
        '<div class="section-header">⏰ Section D — Usage Pattern & Intensity</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="font-size:.8rem;color:var(--text-muted);margin-bottom:.5rem">'
        'How problematic is your pattern of use — not just how long, but how and when? (1 = Never/Rarely, 5 = Always/Very much)</p>',
        unsafe_allow_html=True
    )

    usg_q1 = bubble_scale(
        "How much more do you use social media on weekends compared to weekdays (bingeing)?",
        key="usg_q1", lo_label="Much less", hi_label="Much more", default=2
    )
    usg_q2 = bubble_scale(
        "How often does stress, boredom, or a bad mood cause you to increase SM usage?",
        key="usg_q2", lo_label="Never", hi_label="Always", default=2
    )
    usg_q3 = bubble_scale(
        "How much of your time on social media is passive consumption (scrolling, watching) vs. active creation?",
        key="usg_q3", lo_label="Mostly active", hi_label="Mostly passive", default=3
    )
    usg_q4 = bubble_scale(
        "How often do you hop between multiple social media platforms in a single session?",
        key="usg_q4", lo_label="Rarely", hi_label="Always", default=2
    )

    usg_subscore = calc_usage_pattern_subscore(usg_q1, usg_q2, usg_q3, usg_q4)
    usg_pct = int(usg_subscore * 100)
    usg_col = "#34d399" if usg_pct < 35 else ("#fbbf24" if usg_pct < 65 else "#f87171")
    st.markdown(f"""
    <div class="subscore-card">
      <div class="subscore-title">⏰ Usage Pattern Subscore: <span style="color:{usg_col}">{usg_pct}%</span></div>
      <div class="subscore-bar-bg">
        <div style="width:{usg_pct}%;height:8px;background:{usg_col};border-radius:4px;opacity:.85;"></div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── SECTION E: Academic / Concentration Impact (4 questions) ─────────────
    st.markdown(
        '<div class="section-header">📚 Section E — Academic & Concentration Impact</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="font-size:.8rem;color:var(--text-muted);margin-bottom:.5rem">'
        'How much does social media interfere with your studies and focus? (1 = Never, 5 = Always)</p>',
        unsafe_allow_html=True
    )

    acad_q1 = bubble_scale(
        "How often does the urge to check social media distract you while studying or working?",
        key="acad_q1", lo_label="Never", hi_label="Always", default=2
    )
    acad_q2 = bubble_scale(
        "How often do you use social media to procrastinate when you have a deadline or task?",
        key="acad_q2", lo_label="Never", hi_label="Always", default=2
    )
    acad_q3 = bubble_scale(
        "How often do you lose your train of thought or concentration during a task due to an SM urge?",
        key="acad_q3", lo_label="Never", hi_label="Always", default=2
    )
    acad_q4 = bubble_scale(
        "How often have you submitted work late or missed a commitment because of time spent on SM?",
        key="acad_q4", lo_label="Never", hi_label="Always", default=1
    )

    acad_subscore = calc_academic_impact_subscore(acad_q1, acad_q2, acad_q3, acad_q4)
    acad_pct = int(acad_subscore * 100)
    acad_col = "#34d399" if acad_pct < 35 else ("#fbbf24" if acad_pct < 65 else "#f87171")
    st.markdown(f"""
    <div class="subscore-card">
      <div class="subscore-title">📚 Academic Impact Subscore: <span style="color:{acad_col}">{acad_pct}%</span></div>
      <div class="subscore-bar-bg">
        <div style="width:{acad_pct}%;height:8px;background:{acad_col};border-radius:4px;opacity:.85;"></div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Submit ────────────────────────────────────────────────────────────────
    col_b1, col_b2, col_b3 = st.columns([1.5, 4, 1.5])
    with col_b2:
        submitted = st.button("🔍  Analyse My Risk", type="primary", use_container_width=True)

    if submitted:
        user_inputs = {
            "age":                   age,
            "gender":                gender,
            "academic_level":        academic_level,
            "relationship_status":   relationship_status,
            "avg_daily_usage_hours": avg_daily_usage_hours,
            "most_used_platform":    most_used_platform,
            "sleep_hours":           sleep_hours,
            "mental_health_score":   float(mental_health_score),
        }
        subscores = {
            "behavioural_intensity":  beh_subscore,
            "psychological_distress": psy_subscore,
            "sleep_disturbance":      slp_subscore,
            "usage_pattern":          usg_subscore,
            "academic_impact":        acad_subscore,
        }
        subscores_named = {
            "Behavioural\nIntensity":  beh_subscore,
            "Psychological\nDistress": psy_subscore,
            "Sleep\nDisturbance":      slp_subscore,
            "Usage\nPattern":         usg_subscore,
            "Academic\nImpact":       acad_subscore,
        }

        with st.spinner("Running model inference..."):
            try:
                pred_label, probs, feat_row = run_inference(user_inputs, subscores)
                st.session_state.inputs          = user_inputs
                st.session_state.subscores       = subscores
                st.session_state.subscores_named = subscores_named
                st.session_state.pred_label      = pred_label
                st.session_state.probs           = probs
                st.session_state.feat_row        = feat_row.to_dict(orient="records")[0]
                st.session_state.page            = "results"
                trigger_scroll()
                st.rerun()
            except Exception as ex:
                st.error(f"Inference failed: {ex}")
                st.exception(ex)

    st.markdown("<br>", unsafe_allow_html=True)
    col_back1, col_back2, col_back3 = st.columns([1.5, 4, 1.5])
    with col_back2:
        if st.button("← Back to Home", use_container_width=True):
            st.session_state.page = "about"
            trigger_scroll()
            st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RESULTS (with 4 charts)
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "results":

    pred_label      = st.session_state.get("pred_label", "Low")
    probs           = st.session_state.get("probs", {"Low": 1.0, "Moderate": 0.0, "High": 0.0})
    feat_vals       = st.session_state.get("feat_row", {})
    subscores       = st.session_state.get("subscores", {})
    subscores_named_raw = st.session_state.get("subscores_named", {})
    inputs          = st.session_state.get("inputs", {})

    risk_idx   = LABEL_NAMES.index(pred_label) if pred_label in LABEL_NAMES else 0
    risk_color = LABEL_COLORS[risk_idx]
    risk_emoji = LABEL_EMOJI[risk_idx]
    risk_bg    = LABEL_BG_CSS[risk_idx]
    confidence = probs.get(pred_label, 1.0)

    # Named subscores for charts (cleaner keys)
    subscores_named = {
        "Behavioural Intensity":  subscores.get("behavioural_intensity", 0),
        "Psychological Distress": subscores.get("psychological_distress", 0),
        "Sleep Disturbance":      subscores.get("sleep_disturbance", 0),
        "Usage Pattern":          subscores.get("usage_pattern", 0),
        "Academic Impact":        subscores.get("academic_impact", 0),
    }

    # ── Result banner ─────────────────────────────────────────────────────────
    st.markdown('<div class="pill">📊 Your Results</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="result-banner" style="{risk_bg}">
      <h2 style="color:{risk_color}">{risk_emoji} {pred_label} Addiction Risk</h2>
      <p>Model confidence: <strong style="color:{risk_color}">{confidence*100:.1f}%</strong>
         &nbsp;·&nbsp; Model: <strong style="color:var(--text-secondary)">{artifacts.get('model_name','clf')}</strong></p>
    </div>
    """, unsafe_allow_html=True)

    # ── CHART 1: Probability bars ─────────────────────────────────────────────
    st.markdown("### 📈 Risk Level Probabilities")
    chart1_bytes = chart_probability_bars(probs)
    st.image(chart1_bytes, use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── CHART 2: Subscore profile ─────────────────────────────────────────────
    st.markdown("### 🧮 Your Composite Subscore Profile")
    st.markdown(
        '<p style="font-size:.82rem;color:var(--text-muted);margin-top:-.4rem;margin-bottom:.8rem">'
        'Your scores across the 5 questionnaire dimensions (0 = no risk, 1 = maximum risk).</p>',
        unsafe_allow_html=True
    )
    chart2_bytes = chart_subscore_bars(subscores_named)
    st.image(chart2_bytes, use_container_width=True)

    # Live subscore cards below chart
    cols_sc = st.columns(5)
    subscore_icons = ["📱", "🧠", "😴", "⏰", "📚"]
    for i, (lbl, val) in enumerate(subscores_named.items()):
        pct = int(val * 100)
        sc_col = "#34d399" if pct < 35 else ("#fbbf24" if pct < 65 else "#f87171")
        with cols_sc[i]:
            st.markdown(f"""
            <div style="background:var(--bg-card);border:1px solid var(--border);border-radius:10px;
                        padding:10px;text-align:center;">
              <div style="font-size:1.3rem">{subscore_icons[i]}</div>
              <div style="font-size:.68rem;color:var(--text-muted);font-family:'Syne',sans-serif;
                          font-weight:700;letter-spacing:.05em;margin:4px 0 2px">
                {lbl.replace(chr(10),' ')}</div>
              <div style="font-size:1.05rem;font-weight:800;color:{sc_col};
                          font-family:'Syne',sans-serif">{pct}%</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── CHART 3: Driver analysis ──────────────────────────────────────────────
    st.markdown("### 🔎 What Is Driving Your Prediction")
    st.markdown(
        '<p style="font-size:.82rem;color:var(--text-muted);margin-top:-.4rem;margin-bottom:.8rem">'
        'The 6 engineered model features, sorted by magnitude. Red = risk-increasing, blue = contextual.</p>',
        unsafe_allow_html=True
    )
    chart3_bytes = chart_driver_analysis(feat_vals, pred_label)
    st.image(chart3_bytes, use_container_width=True)

    # Feature breakdown cards
    st.markdown("<br>", unsafe_allow_html=True)
    feat_col1, feat_col2 = st.columns(2)
    feat_items = list(feat_vals.items())
    half = (len(feat_items) + 1) // 2
    high_risk_feats = {"behavioral_intensity_index", "psychological_distress_index",
                       "sleep_disturbance_proxy", "usage_hours_sq"}

    for col_obj, items in [(feat_col1, feat_items[:half]), (feat_col2, feat_items[half:])]:
        with col_obj:
            for fname, fval in items:
                friendly = FEAT_FRIENDLY.get(fname, fname)
                pct = min(int(float(fval) * 100), 100)
                bar_color = "#f87171" if fname in high_risk_feats else "#60a5fa"
                st.markdown(f"""
                <div style="margin-bottom:12px;background:var(--bg-card);border:1px solid var(--border);
                            border-radius:10px;padding:11px 14px;">
                  <div style="font-size:.76rem;color:var(--text-secondary);margin-bottom:4px">{friendly}</div>
                  <div style="font-size:1.05rem;font-weight:700;color:var(--text-primary);
                              font-family:'Syne',sans-serif;margin-bottom:5px">{float(fval):.3f}</div>
                  <div style="background:rgba(255,255,255,0.07);border-radius:4px;height:6px;">
                    <div style="width:{pct}%;height:6px;background:{bar_color};border-radius:4px;opacity:.82;"></div>
                  </div>
                </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── CHART 4: Your profile vs average user ────────────────────────────────
    st.markdown("### 👥 Your Profile vs Average User")
    st.markdown(
        '<p style="font-size:.82rem;color:var(--text-muted);margin-top:-.4rem;margin-bottom:.8rem">'
        'Coloured bars = your subscores. Grey bars = dataset benchmark average. '
        '*Benchmarks are placeholders — see <code>BENCHMARK_SUBSCORE_AVGS</code> in code to update.</p>',
        unsafe_allow_html=True
    )
    chart4_bytes = chart_vs_average(feat_vals, subscores_named)
    st.image(chart4_bytes, use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Interpretation block ──────────────────────────────────────────────────
    INTERP = {
        "Low": (
            "🌱 Your social media habits appear healthy.",
            "Your behavioural intensity, psychological distress, and sleep disturbance scores "
            "are all in the healthy zone. Your usage pattern and academic impact are within "
            "normal bounds. Keep maintaining these habits."
        ),
        "Moderate": (
            "⚠️ Some warning signs detected across multiple dimensions.",
            "One or more of your subscores — likely behavioural intensity, psychological "
            "distress, or sleep disturbance — are in the elevated range. Taking proactive "
            "steps now is the most effective way to prevent escalation to high risk."
        ),
        "High": (
            "🚨 Strong addiction risk signals detected across multiple dimensions.",
            "Multiple subscores are in the high-risk range. Your daily usage pattern, "
            "psychological distress, sleep disruption, and/or academic impact are significantly "
            "elevated. Immediate and sustained habit change is strongly recommended."
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
    st.markdown("### 💡 Personalised Recommendations")
    RECS = {
        "Low": [
            ("🎯 Stay Consistent", "Healthy habits are fragile — reinforce them. Set a monthly reminder to re-take this screener and track changes over time."),
            ("📱 Keep Monitoring", "Use built-in screen time tools (iOS Screen Time / Android Digital Wellbeing) for weekly awareness without effort."),
            ("🌍 Invest Offline", "Continue prioritising offline hobbies and face-to-face connections — these are your strongest long-term buffers against addiction."),
        ],
        "Moderate": [
            ("⏱️ Set Daily App Limits", "Enforce a 45-minute daily social media budget using app timers. Resist overriding for the first 3 days — each refusal rewires the habit loop."),
            ("🌙 Phone-Free Bedtime", "No social media 60 minutes before sleep. Replace scrolling with reading, journaling, or light stretching to wind down."),
            ("🧠 Mindful Scrolling", "Before opening any app, ask: 'Why am I opening this?' If you can't answer, close it immediately. Track purposeless opens and aim to halve them in 2 weeks."),
            ("📅 Weekly Detox Day", "Choose one day per week (e.g. Sunday) with zero social media. Fill it with a planned offline activity — the structure is key."),
        ],
        "High": [
            ("🚫 Hard Limit — Start Today", "Set a strict 1-hour daily total cap using both your phone's built-in limits AND a third-party app (Opal / Cold Turkey) for redundancy."),
            ("🗑️ Remove App Triggers", "Delete social media apps from your home screen. Access via browser only — this single change reduces usage by 20–30% within one week."),
            ("🌑 Grayscale Mode Evenings", "Enable phone grayscale after 8pm (iOS: Accessibility → Color Filters; Android: Developer Options). Removing colour reduces dopamine reward from scrolling."),
            ("📝 Replace the Habit Loop", "Write your top 3 triggers (boredom, anxiety, loneliness) and a concrete replacement action for each. Post it where you can see it daily."),
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

    # ── Input summary expander ────────────────────────────────────────────────
    with st.expander("📋 View your submitted inputs & subscores"):
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            st.markdown("**Core Inputs**")
            summary_data = {
                "Age":                 inputs.get("age"),
                "Gender":              inputs.get("gender"),
                "Academic Level":      inputs.get("academic_level"),
                "Relationship Status": inputs.get("relationship_status"),
                "Daily Usage (hrs)":   inputs.get("avg_daily_usage_hours"),
                "Sleep Hours":         inputs.get("sleep_hours"),
                "Mental Health (1–10)":inputs.get("mental_health_score"),
                "Platform":            inputs.get("most_used_platform"),
            }
            st.table(pd.DataFrame(summary_data.items(), columns=["Field", "Value"]))
        with col_exp2:
            st.markdown("**Questionnaire Subscores**")
            ss_data = {k: f"{int(v*100)}%" for k, v in subscores_named.items()}
            st.table(pd.DataFrame(ss_data.items(), columns=["Dimension", "Score"]))

    # ── Navigation ────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2, 3, 2])
    with c1:
        if st.button("← Retake Survey", use_container_width=True):
            st.session_state.page = "survey"
            trigger_scroll()
            st.rerun()
    with c3:
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.page = "about"
            trigger_scroll()
            st.rerun()

    st.markdown(
        '<div class="footer-note">⚠️ Research prototype only — not a clinical diagnostic tool. '
        'Results do not replace professional mental health advice.</div>',
        unsafe_allow_html=True
    )