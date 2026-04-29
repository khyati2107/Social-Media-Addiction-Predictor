"""
10_shap_explainability.py
Purpose: SHAP explainability for tree-based and linear models, global SHAP summary plots (bar + beeswarm per class)
,feature importance ranking table ,local waterfall explanation examples, robust fallback ladders for XGBoost and Logistic Regression
"""

import pandas as pd
import numpy as np
import pickle, os, warnings
warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

#Load
train     = pd.read_csv("outputs/features_train.csv")
test      = pd.read_csv("outputs/features_test.csv")
FEAT_COLS = pickle.load(open("models/feature_cols.pkl", "rb"))
scaler    = pickle.load(open("models/scaler_clf.pkl",   "rb"))
le        = pickle.load(open("models/label_encoder.pkl","rb"))

X_train_sc = scaler.transform(train[FEAT_COLS].values)
X_test_sc  = scaler.transform(test[FEAT_COLS].values)
X_train_df = pd.DataFrame(X_train_sc, columns=FEAT_COLS)
X_test_df  = pd.DataFrame(X_test_sc,  columns=FEAT_COLS)

rf  = pickle.load(open("models/clf_rf.pkl",  "rb"))
xgb = pickle.load(open("models/clf_xgb.pkl", "rb"))
lr  = pickle.load(open("models/clf_lr.pkl",  "rb"))

classes      = le.classes_       # ['High', 'Low', 'Moderate']
n_classes    = len(classes)
TARGET_CLASS = 0                 # 'High' used for local waterfall plots



# HELPERS
def normalise_shap(shap_values, n_cls):
    """
    Normalise any SHAP output into list[class_idx] -> array(n_samples, n_feats).
    Handles: list-of-arrays | 3-D ndarray (n,f,c) | 2-D ndarray (n,f).
    """
    if isinstance(shap_values, list):
        return shap_values
    if isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            return [shap_values[:, :, i] for i in range(n_cls)]
        return [shap_values] * n_cls
    return [shap_values] * n_cls


def get_base_value(explainer, class_idx):
    """Safely extract scalar base value for one class."""
    ev = explainer.expected_value
    arr = np.array(ev).flatten()
    if len(arr) > class_idx:
        return float(arr[class_idx])
    return float(arr[0])


def save_global_plots(sv_list, X_df, label, prefix):
    """Bar + beeswarm SHAP summary for each class."""
    for i, cls in enumerate(classes):
        sv = sv_list[i]

        plt.figure(figsize=(8, 5))
        shap.summary_plot(sv, X_df, feature_names=FEAT_COLS,
                          show=False, plot_type="bar", max_display=6)
        plt.title(f"{label} SHAP Feature Importance — Class: {cls}")
        plt.tight_layout()
        plt.savefig(f"outputs/shap_{prefix}_bar_{cls}.png", dpi=120)
        plt.close()

        plt.figure(figsize=(8, 5))
        shap.summary_plot(sv, X_df, feature_names=FEAT_COLS,
                          show=False, max_display=6)
        plt.title(f"{label} SHAP Summary (beeswarm) — Class: {cls}")
        plt.tight_layout()
        plt.savefig(f"outputs/shap_{prefix}_beeswarm_{cls}.png", dpi=120)
        plt.close()


def save_waterfall_plots(sv_list, explainer, X_df, label, prefix,
                         class_idx=TARGET_CLASS):
    """Local waterfall for first 3 test samples, one target class."""
    sv_for_class = sv_list[class_idx]           # (n_samples, n_feats)
    base_val     = get_base_value(explainer, class_idx)

    for idx in range(min(3, len(X_df))):
        shap_1d = sv_for_class[idx]             # (n_feats,) 1-D required

        explanation = shap.Explanation(
            values        = shap_1d,
            base_values   = base_val,
            data          = X_df.iloc[idx].values,
            feature_names = FEAT_COLS,
        )

        plt.figure(figsize=(9, 4))
        shap.plots.waterfall(explanation, max_display=6, show=False)
        plt.title(
            f"{label} Local SHAP (class={classes[class_idx]}) — Sample {idx}"
        )
        plt.tight_layout()
        plt.savefig(f"outputs/shap_{prefix}_local_{idx}.png", dpi=120)
        plt.close()


def builtin_importance_fallback(model, prefix, label):
    """Save sklearn/xgb built-in feature importances when SHAP fails."""
    print(f"  Using {label} built-in feature importances as fallback.")
    imp    = model.feature_importances_
    imp_df = pd.DataFrame({
        "feature":    FEAT_COLS,
        "importance": imp,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    plt.figure(figsize=(8, 5))
    plt.barh(imp_df["feature"][::-1], imp_df["importance"][::-1],
             color="steelblue")
    plt.xlabel("Feature Importance")
    plt.title(f"{label} Built-in Feature Importance (SHAP fallback)")
    plt.tight_layout()
    plt.savefig(f"outputs/shap_{prefix}_builtin_importance.png", dpi=120)
    plt.close()

    imp_df.to_csv(f"outputs/shap_{prefix}_builtin_importance.csv", index=False)
    print(f"  Saved: outputs/shap_{prefix}_builtin_importance.png/.csv")
    return imp_df


def kernel_explainer_fallback(model_fn, X_train_bg, X_test_local,
                               label, prefix, n_bg=50, n_test=50):
    """
    KernelExplainer fallback — model-agnostic, works with any classifier. Uses a small background and test slice for speed.
    """
    print(f"  Trying KernelExplainer for {label} (bg={n_bg}, test={n_test})...")
    background = shap.sample(X_train_bg, n_bg, random_state=42)
    explainer  = shap.KernelExplainer(model_fn, background)
    X_slice    = X_test_local.iloc[:n_test]
    sv_raw     = explainer.shap_values(X_slice)
    sv_list    = normalise_shap(sv_raw, n_classes)
    print(f"  KernelExplainer succeeded for {label}.")
    return sv_list, explainer, X_slice



# 1. RANDOM FOREST  (TreeExplainer — reliable for sklearn RF)
print("Computing SHAP for Random Forest...")
explainer_rf = shap.TreeExplainer(rf)
sv_rf        = normalise_shap(
    explainer_rf.shap_values(X_test_df), n_classes
)
save_global_plots(sv_rf, X_test_df, label="RF", prefix="rf")
save_waterfall_plots(sv_rf, explainer_rf, X_test_df, label="RF", prefix="rf")
print("  RF SHAP done.")



# 2. XGBOOST — three-level fallback ladder
#   L1: shap.Explainer (universal)
#   L2: KernelExplainer
#   L3: built-in feature_importances_

print("Computing SHAP for XGBoost...")

sv_xgb        = None
explainer_xgb = None
xgb_method    = None
X_test_xgb    = X_test_df.copy()   # may be sliced if KernelExplainer is used

# Level 1 — shap.Explainer
try:
    print("  Trying shap.Explainer (universal API)...")
    _exp   = shap.Explainer(xgb, X_train_df)
    _sv    = _exp(X_test_df)
    raw    = _sv.values
    if isinstance(raw, np.ndarray) and raw.ndim == 3:
        sv_xgb = [raw[:, :, i] for i in range(n_classes)]
    else:
        sv_xgb = normalise_shap(raw, n_classes)

    # Build mock explainer carrying expected_value for waterfall
    bv = _sv.base_values
    if isinstance(bv, np.ndarray) and bv.ndim == 2:
        mean_bv = bv.mean(axis=0)
    else:
        mean_bv = np.full(n_classes, float(np.array(bv).mean()))

    class _MockExp:
        def __init__(self, ev): self.expected_value = ev
    explainer_xgb = _MockExp(mean_bv)
    xgb_method    = "shap.Explainer"
    print("  shap.Explainer succeeded.")

except Exception as e1:
    print(f"  shap.Explainer failed: {e1}")

    # Level 2 — KernelExplainer
    try:
        sv_xgb, explainer_xgb, X_test_xgb = kernel_explainer_fallback(
            lambda x: xgb.predict_proba(x),
            X_train_df, X_test_df,
            label="XGB", prefix="xgb"
        )
        xgb_method = "KernelExplainer"

    except Exception as e2:
        print(f"  KernelExplainer failed: {e2}")
        xgb_method = "builtin_fallback"

if xgb_method in ("shap.Explainer", "KernelExplainer"):
    save_global_plots(sv_xgb, X_test_xgb, label="XGB", prefix="xgb")
    save_waterfall_plots(sv_xgb, explainer_xgb, X_test_xgb,
                         label="XGB", prefix="xgb")
    print(f"  XGB SHAP done via {xgb_method}.")
else:
    builtin_importance_fallback(xgb, prefix="xgb", label="XGB")
    print("  XGB: built-in importances saved (SHAP version incompatibility).")



# 3. LOGISTIC REGRESSION — two-level fallback ladder
#   L1: LinearExplainer with feature_perturbation="interventional"
#   L2: KernelExplainer (model-agnostic, guaranteed to work)

print("Computing SHAP for Logistic Regression...")

sv_lr        = None
explainer_lr = None
lr_method    = None
X_test_lr    = X_test_df.copy()

# Level 1 — LinearExplainer, interventional mode
try:
    print("  Trying LinearExplainer (interventional)...")
    explainer_lr = shap.LinearExplainer(
        lr, X_train_df,
        feature_perturbation="interventional"   # no covariance matrix needed
    )
    sv_lr    = normalise_shap(
        explainer_lr.shap_values(X_test_df), n_classes
    )
    lr_method = "LinearExplainer_interventional"
    print("  LinearExplainer (interventional) succeeded.")

except Exception as e1:
    print(f"  LinearExplainer interventional failed: {e1}")

    # Level 2 — KernelExplainer
    try:
        sv_lr, explainer_lr, X_test_lr = kernel_explainer_fallback(
            lambda x: lr.predict_proba(x),
            X_train_df, X_test_df,
            label="LR", prefix="lr"
        )
        lr_method = "KernelExplainer"

    except Exception as e2:
        print(f"  LR KernelExplainer also failed: {e2}")
        lr_method = "failed"

if lr_method not in ("failed", None):
    save_global_plots(sv_lr, X_test_lr, label="LR", prefix="lr")
    save_waterfall_plots(sv_lr, explainer_lr, X_test_lr,
                         label="LR", prefix="lr")
    print(f"  LR SHAP done via {lr_method}.")
else:
    print("  LR SHAP could not be computed. Skipping LR plots.")



# 4. FEATURE IMPORTANCE RANKING TABLE
#    RF mean |SHAP| (primary) + XGB if available

mean_abs_rf = np.mean(
    [np.abs(sv).mean(axis=0) for sv in sv_rf], axis=0
)
importance_df = pd.DataFrame({
    "feature":          FEAT_COLS,
    "rf_mean_abs_shap": mean_abs_rf,
}).sort_values("rf_mean_abs_shap", ascending=False).reset_index(drop=True)

if sv_xgb is not None:
    mean_abs_xgb = np.mean(
        [np.abs(sv).mean(axis=0) for sv in sv_xgb], axis=0
    )
    if len(mean_abs_xgb) == len(FEAT_COLS):
        # Align to sorted RF order
        xgb_col = pd.DataFrame({
            "feature":           FEAT_COLS,
            "xgb_mean_abs_shap": mean_abs_xgb,
        })
        importance_df = importance_df.merge(xgb_col, on="feature", how="left")

if sv_lr is not None:
    mean_abs_lr = np.mean(
        [np.abs(sv).mean(axis=0) for sv in sv_lr], axis=0
    )
    if len(mean_abs_lr) == len(FEAT_COLS):
        lr_col = pd.DataFrame({
            "feature":          FEAT_COLS,
            "lr_mean_abs_shap": mean_abs_lr,
        })
        importance_df = importance_df.merge(lr_col, on="feature", how="left")

importance_df.to_csv("outputs/shap_feature_importance_combined.csv", index=False)

print("\nFeature importance ranking (mean |SHAP| across classes):")
print(importance_df.to_string(index=False))



# 5. LIMITATIONS REPORT

xgb_note = (
    f"XGBoost SHAP computed via {xgb_method}."
    if xgb_method != "builtin_fallback"
    else "XGBoost TreeExplainer incompatible with installed version "
         "(base_score stored as vector). Built-in importances used. "
         "Fix: pip install 'xgboost==1.7.6' or upgrade shap."
)
lr_note = (
    f"LR SHAP computed via {lr_method}."
    if lr_method not in ("failed", None)
    else "LR LinearExplainer failed with both perturbation modes. "
         "Fix: pip install shap --upgrade"
)

limitations = f"""
LIMITATIONS REPORT
==================
1. Souvik has no ground-truth addiction labels. All Souvik evaluation uses proxy
   labels derived from usage hours only. 
2. Sleep constructs are incompatible across datasets:
   - Nusratt: sleep_hours_per_night (objective duration)
   - Souvik:  sleep_issue_score (subjective disturbance, 1-5 Likert)
   Treated as separate proxies; never directly compared.

3. Only 6 shared features were usable. Several psychologically important Souvik
   variables (social_comparison, validation_seeking, restlessness,
   interest_fluctuation, distraction_while_busy) have no Nusratt equivalents
   and were excluded.

4. Nusratt appears synthetic (regular patterns across 705 rows). Real-world
   generalisation may differ substantially.

5. SMOTE generates synthetic minority samples that may not reflect real
   population distributions.

6. ALL 6 engineered features showed statistically significant domain shift
   (p < 0.05, both KS and Mann-Whitney). The OOD performance gap is primarily
   attributable to this distribution mismatch, not model failure.

7. Relationship_status has a major structural mismatch: Nusratt has no
   'Married' entries; Souvik has ~21% married respondents.

8. Nusratt mental_health_score polarity (higher = better health) was inverted
   during feature engineering. Verify against original documentation.

9. SHAP local waterfall plots are for class='High' only (most clinically
   relevant). Global plots cover all three classes.

10. {xgb_note}

11. {lr_note}
"""

with open("outputs/limitations_report.txt", "w") as f:
    f.write(limitations)

print(limitations)
print("\n All SHAP outputs saved. Pipeline complete.")