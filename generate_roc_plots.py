"""
generate_roc_plots.py
Generates ROC curve plots (one-vs-rest, per class) for all 4 classifiers
on the Nusratt test set. Saves individual model plots + a combined comparison.
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import os

os.makedirs("outputs", exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
test      = pd.read_csv("outputs/features_test.csv")
FEAT_COLS = pickle.load(open("models/feature_cols.pkl", "rb"))
scaler    = pickle.load(open("models/scaler_clf.pkl", "rb"))
le        = pickle.load(open("models/label_encoder.pkl", "rb"))

X_test = scaler.transform(test[FEAT_COLS].values)
y_test = le.transform(test["addiction_label"].values)
classes = le.classes_   # ['High', 'Low', 'Moderate']
n_classes = len(classes)

# Binarize for OvR ROC
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

model_names = {"LR": "lr", "RF": "rf", "XGB": "xgb", "SVM": "svm"}
colors_per_class = {"High": "#E63946", "Low": "#2A9D8F", "Moderate": "#E9C46A"}
model_colors = {"LR": "#4361EE", "RF": "#F77F00", "XGB": "#7B2D8B", "SVM": "#2EC4B6"}

models = {label: pickle.load(open(f"models/clf_{fname}.pkl", "rb"))
          for label, fname in model_names.items()}

# ── 1. Per-model ROC (one plot per model, all 3 classes) ─────────────────────
for model_label, model in models.items():
    y_prob = model.predict_proba(X_test)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("white")

    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2,
                color=colors_per_class[cls],
                label=f"Class: {cls}  (AUC = {roc_auc:.3f})")

    # Macro average
    all_fpr = np.unique(np.concatenate([
        roc_curve(y_test_bin[:, i], y_prob[:, i])[0] for i in range(n_classes)
    ]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)
    ax.plot(all_fpr, mean_tpr, lw=2.5, linestyle="--",
            color="#333333", label=f"Macro Avg  (AUC = {macro_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k:", lw=1, alpha=0.5, label="Random Classifier")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curves — {model_label} (Nusratt Test Set)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"outputs/roc_{model_label}_nusratt.png", dpi=150)
    plt.close()
    print(f"Saved: outputs/roc_{model_label}_nusratt.png  (Macro AUC={macro_auc:.3f})")

# ── 2. Combined: Macro ROC for all 4 models on one chart ─────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
ax.set_facecolor("#F8F9FA")
fig.patch.set_facecolor("white")

for model_label, model in models.items():
    y_prob = model.predict_proba(X_test)
    all_fpr = np.unique(np.concatenate([
        roc_curve(y_test_bin[:, i], y_prob[:, i])[0] for i in range(n_classes)
    ]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)
    ax.plot(all_fpr, mean_tpr, lw=2.5,
            color=model_colors[model_label],
            label=f"{model_label}  (AUC = {macro_auc:.3f})")

ax.plot([0, 1], [0, 1], "k:", lw=1, alpha=0.5, label="Random Classifier")
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.05])
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("Macro-Averaged ROC Curves — All Models\n(Nusratt Test Set, 3-Class OvR)", fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/roc_all_models_combined.png", dpi=150)
plt.close()
print("Saved: outputs/roc_all_models_combined.png")

# ── 3. 2x2 grid: per-model, all classes (paper-ready figure) ─────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("ROC Curves by Model — Nusratt Test Set (One-vs-Rest)", fontsize=14, fontweight="bold")

for ax, (model_label, model) in zip(axes.flatten(), models.items()):
    y_prob = model.predict_proba(X_test)
    ax.set_facecolor("#F8F9FA")

    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2,
                color=colors_per_class[cls],
                label=f"{cls}  (AUC={roc_auc:.3f})")

    all_fpr = np.unique(np.concatenate([
        roc_curve(y_test_bin[:, i], y_prob[:, i])[0] for i in range(n_classes)
    ]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)
    ax.plot(all_fpr, mean_tpr, lw=2.5, linestyle="--",
            color="#333333", label=f"Macro  (AUC={macro_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k:", lw=1, alpha=0.4)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_title(f"{model_label}", fontsize=13, fontweight="bold")
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/roc_2x2_grid.png", dpi=150)
plt.close()
print("Saved: outputs/roc_2x2_grid.png")

print("\nAll ROC plots generated successfully.")