"""
04_feature_engineering_shared.py
Purpose:
  - Derive engineered features from Nusratt TRAINING split only
  - Apply same logic to Nusratt test and Souvik
  - Use only transferable features (inputs available in both datasets)
  - No mixing of direct and proxy variables inside the same feature
  - 5–7 compact engineered features maximum
  - Save shared feature datasets

SHARED ENGINEERED FEATURES (6 total):
  1. behavioral_intensity_index   — usage + conflicts (both direct on Nusratt; usage direct + purposeless_use proxy on Souvik)
     NOTE: conflicts ↔ purposeless_use is a proxy match. This feature uses ONE proxy variable only.
  2. psychological_distress_index — mental_health_score proxy (depression_freq + bothered_by_worries on Souvik)
     Pure proxy construction, consistent across both sides.
  3. sleep_disturbance_proxy      — sleep_issue_score (Souvik Q20) / rescaled sleep_hours inverse (Nusratt)
     Both treated as sleep disturbance proxies.
  4. age_norm                     — direct match, normalised
  5. is_student                   — direct proxy flag
  6. usage_hours_sq               — squared usage (non-linear interaction) — direct input

NOT INCLUDED (would mix direct+proxy or unavailable in Nusratt):
  - social_comparison_index (Souvik Q15 has no Nusratt match)
  - distraction_concentration_index (Souvik Q12/Q14 have no Nusratt equivalent)
  - emotional_vulnerability_index (Q19 not in Nusratt)
"""

import pandas as pd
import numpy as np
import os, pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

os.makedirs("outputs", exist_ok=True)
os.makedirs("models",  exist_ok=True)

# ── Load cleaned data ─────────────────────────────────────────────────────────
n = pd.read_csv("outputs/nusratt_clean.csv")
s = pd.read_csv("outputs/souvik_clean.csv")

# ── Train/test split (stratified) — MUST happen BEFORE any fitting ───────────
from sklearn.model_selection import train_test_split

X_all = n.copy()
y_clf = n["addiction_label"]
y_reg = n["addicted_score"]

n_train, n_test = train_test_split(
    n, test_size=0.2, random_state=42, stratify=y_clf
)
n_train = n_train.reset_index(drop=True)
n_test  = n_test.reset_index(drop=True)

print(f"Nusratt train: {n_train.shape}, test: {n_test.shape}")
print("Train label dist:\n", n_train["addiction_label"].value_counts())


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING FUNCTIONS
# All parameters (clip bounds, scale ranges) are derived from n_train only.
# ══════════════════════════════════════════════════════════════════════════════

def make_nusratt_features(df, params=None, fit=False):
    """
    Build shared features for Nusratt rows.
    params: dict of fitted statistics (filled when fit=True)
    Returns: (feature_df, params)
    """
    d = df.copy()

    # 1. behavioral_intensity_index
    #    Direct: avg_daily_usage_hours (0-24); conflicts (0-5 int)
    #    Both normalised to [0,1] then averaged.
    if fit:
        params["usage_max"]     = d["avg_daily_usage_hours"].quantile(0.99)
        params["conflicts_max"] = d["conflicts"].max()

    usage_norm     = (d["avg_daily_usage_hours"].clip(0, params["usage_max"])
                      / params["usage_max"])
    conflicts_norm = d["conflicts"].clip(0, params["conflicts_max"]) / params["conflicts_max"]
    d["behavioral_intensity_index"] = (usage_norm + conflicts_norm) / 2

    # 2. psychological_distress_index
    #    Nusratt: mental_health_score (1-10), higher = better mental health.
    #    We invert so that higher index = more distress (consistent with Souvik side).
    if fit:
        params["mh_min"] = d["mental_health_score"].min()
        params["mh_max"] = d["mental_health_score"].max()
    mh_range = params["mh_max"] - params["mh_min"] or 1
    d["psychological_distress_index"] = (
        1 - (d["mental_health_score"] - params["mh_min"]) / mh_range
    )

    # 3. sleep_disturbance_proxy
    #    Nusratt: sleep_hours. Fewer hours → more disturbance.
    #    Map to [0,1]: disturbance = 1 - normalised_sleep
    if fit:
        params["sleep_min"] = d["sleep_hours"].min()
        params["sleep_max"] = d["sleep_hours"].max()
    sleep_range = params["sleep_max"] - params["sleep_min"] or 1
    d["sleep_disturbance_proxy"] = (
        1 - (d["sleep_hours"].clip(params["sleep_min"], params["sleep_max"])
             - params["sleep_min"]) / sleep_range
    )

    # 4. age_norm (direct, continuous)
    if fit:
        params["age_min"] = d["age"].min()
        params["age_max"] = d["age"].max()
    age_range = params["age_max"] - params["age_min"] or 1
    d["age_norm"] = (d["age"].clip(params["age_min"], params["age_max"])
                     - params["age_min"]) / age_range

    # 5. is_student (proxy flag)
    #    Nusratt: academic_level ∈ {High School, Undergraduate, Graduate}
    student_levels = {"High School", "Undergraduate", "Graduate"}
    d["is_student"] = d["academic_level"].isin(student_levels).astype(int)

    # 6. usage_hours_sq (non-linear usage term, direct input)
    if fit:
        params["usage_sq_max"] = (d["avg_daily_usage_hours"].clip(0, params["usage_max"]) ** 2).quantile(0.99)
    d["usage_hours_sq"] = (
        d["avg_daily_usage_hours"].clip(0, params["usage_max"]) ** 2
    ).clip(0, params["usage_sq_max"]) / params["usage_sq_max"]

    feature_cols = [
        "behavioral_intensity_index",
        "psychological_distress_index",
        "sleep_disturbance_proxy",
        "age_norm",
        "is_student",
        "usage_hours_sq",
    ]
    return d[feature_cols], params


def make_souvik_features(df, params):
    """
    Build shared features for Souvik rows using FITTED params from Nusratt train.
    """
    d = df.copy()

    # 1. behavioral_intensity_index
    #    Souvik proxy: purposeless_use (1-5 Likert) in place of conflicts
    usage_norm        = (d["avg_daily_usage_hours"].clip(0, params["usage_max"])
                         / params["usage_max"])
    purposeless_norm  = (d["purposeless_use"].clip(1, 5) - 1) / 4  # scale to [0,1]
    d["behavioral_intensity_index"] = (usage_norm + purposeless_norm) / 2

    # 2. psychological_distress_index
    #    Souvik: (depression_freq + bothered_by_worries) / 2, scaled to 1-10 → then invert
    mh_proxy = ((d["depression_freq"].clip(1,5) + d["bothered_by_worries"].clip(1,5)) / 2)
    # Scale 1-5 Likert to 1-10
    mh_proxy_10 = (mh_proxy - 1) / 4 * 9 + 1
    mh_range = params["mh_max"] - params["mh_min"] or 1
    d["psychological_distress_index"] = (
        1 - (mh_proxy_10.clip(params["mh_min"], params["mh_max"])
             - params["mh_min"]) / mh_range
    )

    # 3. sleep_disturbance_proxy
    #    Souvik: sleep_issue_score (1-5). Higher = more sleep disturbance.
    #    Map to [0,1] directly (already a disturbance measure, no inversion needed).
    d["sleep_disturbance_proxy"] = (d["sleep_issue_score"].clip(1,5) - 1) / 4

    # 4. age_norm
    age_range = params["age_max"] - params["age_min"] or 1
    d["age_norm"] = (d["age"].clip(params["age_min"], params["age_max"])
                     - params["age_min"]) / age_range

    # 5. is_student
    student_occupations = {
        "University Student", "School Student"
    }
    d["is_student"] = d["occupation_status"].str.strip().str.title().isin(
        student_occupations
    ).astype(int)

    # 6. usage_hours_sq
    d["usage_hours_sq"] = (
        d["avg_daily_usage_hours"].clip(0, params["usage_max"]) ** 2
    ).clip(0, params["usage_sq_max"]) / params["usage_sq_max"]

    feature_cols = [
        "behavioral_intensity_index",
        "psychological_distress_index",
        "sleep_disturbance_proxy",
        "age_norm",
        "is_student",
        "usage_hours_sq",
    ]
    return d[feature_cols]


# ── Fit on training split ─────────────────────────────────────────────────────
params = {}
X_train_feats, params = make_nusratt_features(n_train, params, fit=True)
X_test_feats,  _      = make_nusratt_features(n_test,  params, fit=False)
X_souvik_feats        = make_souvik_features(s, params)

# ── Impute residual NaNs (median from train) ──────────────────────────────────
imputer = SimpleImputer(strategy="median")
X_train_feats = pd.DataFrame(
    imputer.fit_transform(X_train_feats),
    columns=X_train_feats.columns
)
X_test_feats = pd.DataFrame(
    imputer.transform(X_test_feats),
    columns=X_test_feats.columns
)
X_souvik_feats = pd.DataFrame(
    imputer.transform(X_souvik_feats),
    columns=X_souvik_feats.columns
)

# ── Attach labels ─────────────────────────────────────────────────────────────
X_train_feats["addiction_label"] = n_train["addiction_label"].values
X_train_feats["addicted_score"]  = n_train["addicted_score"].values
X_test_feats["addiction_label"]  = n_test["addiction_label"].values
X_test_feats["addicted_score"]   = n_test["addicted_score"].values

# ── Save ──────────────────────────────────────────────────────────────────────
X_train_feats.to_csv("outputs/features_train.csv", index=False)
X_test_feats.to_csv("outputs/features_test.csv",   index=False)
X_souvik_feats.to_csv("outputs/features_souvik.csv", index=False)

pickle.dump(params,   open("models/fe_params.pkl",  "wb"))
pickle.dump(imputer,  open("models/imputer.pkl",    "wb"))

# Save engineered feature list
feat_names = [c for c in X_train_feats.columns if c not in ("addiction_label","addicted_score")]
with open("outputs/retained_engineered_features.txt","w") as f:
    f.write("RETAINED ENGINEERED FEATURES\n")
    f.write("="*40 + "\n")
    for fn in feat_names:
        f.write(f"  {fn}\n")

print("Feature engineering complete. Saved features_train/test/souvik.csv")
print("Features:", feat_names)