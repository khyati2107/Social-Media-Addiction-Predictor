"""
06_evaluate_classification.py
Purpose: Evaluate classifiers on:
  1. Nusratt test set (in-distribution)
  2. Souvik (out-of-distribution / external validation)
Clearly separated. Performance gap explicitly reported.
Souvik proxy labels created separately, never used as ground truth.
"""

import pandas as pd
import numpy as np
import pickle, os, warnings
warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    roc_auc_score
)

# ── Load ──────────────────────────────────────────────────────────────────────
test  = pd.read_csv("outputs/features_test.csv")
souv  = pd.read_csv("outputs/features_souvik.csv")
FEAT_COLS = pickle.load(open("models/feature_cols.pkl","rb"))
scaler    = pickle.load(open("models/scaler_clf.pkl","rb"))
le        = pickle.load(open("models/label_encoder.pkl","rb"))

X_test  = scaler.transform(test[FEAT_COLS].values)
y_test  = le.transform(test["addiction_label"].values)
X_souv  = scaler.transform(souv[FEAT_COLS].values)

model_names = ["lr","rf","xgb","svm"]
models = {n: pickle.load(open(f"models/clf_{n}.pkl","rb")) for n in model_names}
classes = le.classes_

# ── Evaluation helper ─────────────────────────────────────────────────────────
def evaluate(model, X, y_true, dataset_label, model_label):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X) if hasattr(model,"predict_proba") else None

    acc    = accuracy_score(y_true, y_pred)
    mac_f1 = f1_score(y_true, y_pred, average="macro")
    prec   = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec    = recall_score(y_true, y_pred, average="macro", zero_division=0)
    auc    = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro") if y_prob is not None and len(np.unique(y_true)) > 1 else float("nan")

    print(f"\n[{dataset_label}] {model_label}")
    print(f"  Acc={acc:.3f}  MacF1={mac_f1:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  AUC={auc:.3f}")
    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))

    # Confusion matrix plot
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes,
                cmap="Blues", ax=ax)
    ax.set_title(f"{model_label} — {dataset_label}")
    ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    plt.tight_layout()
    fname = f"outputs/cm_{model_label}_{dataset_label.replace(' ','_')}.png"
    plt.savefig(fname, dpi=120); plt.close()

    return {"model": model_label, "dataset": dataset_label,
            "accuracy": acc, "macro_f1": mac_f1,
            "precision": prec, "recall": rec, "roc_auc": auc}

# ── Run evaluation ────────────────────────────────────────────────────────────
results = []
for mn, m in models.items():
    results.append(evaluate(m, X_test, y_test,  "Nusratt_Test", mn.upper()))

# ── Souvik proxy labels (reference only, NOT ground truth) ───────────────────
# Proxy: bucket avg_daily_usage_hours + behavioral_intensity_index
raw_s = pd.read_csv("outputs/souvik_clean.csv")
def souvik_proxy_label(row):
    score = (row["avg_daily_usage_hours"] / 6.0) * 10   # rough 0-10 scale
    if score <= 4:   return "Low"
    elif score <= 6: return "Moderate"
    else:            return "High"

raw_s["proxy_addiction_label"] = raw_s.apply(souvik_proxy_label, axis=1)
raw_s["proxy_addiction_label"].to_csv("outputs/souvik_proxy_labels.csv", index=False)
print("\n[NOTE] Souvik proxy labels saved. NOT used as ground truth.")
print("Proxy label distribution:\n", raw_s["proxy_addiction_label"].value_counts())

# Evaluate each model on Souvik with proxy labels (clearly marked)
y_souv_proxy = le.transform(raw_s["proxy_addiction_label"].values)
for mn, m in models.items():
    results.append(evaluate(m, X_souv, y_souv_proxy,
                            "Souvik_OOD_ProxyOnly", mn.upper()))

# ── Performance gap report ────────────────────────────────────────────────────
df_res = pd.DataFrame(results)
df_res.to_csv("outputs/classification_metrics.csv", index=False)

print("\n" + "="*60)
print("PERFORMANCE GAP REPORT (In-dist vs OOD)")
print("="*60)
for mn in [m.upper() for m in model_names]:
    id_row  = df_res[(df_res.model==mn) & (df_res.dataset=="Nusratt_Test")].iloc[0]
    ood_row = df_res[(df_res.model==mn) & (df_res.dataset=="Souvik_OOD_ProxyOnly")].iloc[0]
    gap_acc = id_row.accuracy  - ood_row.accuracy
    gap_f1  = id_row.macro_f1 - ood_row.macro_f1
    print(f"\n{mn}:")
    print(f"  Nusratt Acc={id_row.accuracy:.3f}  MacF1={id_row.macro_f1:.3f}")
    print(f"  Souvik  Acc={ood_row.accuracy:.3f}  MacF1={ood_row.macro_f1:.3f}")
    print(f"  GAP     Acc={gap_acc:+.3f}  MacF1={gap_f1:+.3f}")

with open("outputs/performance_gap_report.txt","w") as f:
    f.write("PERFORMANCE GAP REPORT\n")
    f.write("NOTE: Souvik evaluation uses PROXY labels (not ground truth).\n")
    f.write("OOD drop is an expected consequence of dataset shift.\n\n")
    for mn in [m.upper() for m in model_names]:
        id_row  = df_res[(df_res.model==mn) & (df_res.dataset=="Nusratt_Test")].iloc[0]
        ood_row = df_res[(df_res.model==mn) & (df_res.dataset=="Souvik_OOD_ProxyOnly")].iloc[0]
        f.write(f"{mn}  ID_Acc={id_row.accuracy:.3f}  OOD_Acc={ood_row.accuracy:.3f}  "
                f"Gap={id_row.accuracy-ood_row.accuracy:+.3f}\n")

print("\nMetrics and plots saved to outputs/")