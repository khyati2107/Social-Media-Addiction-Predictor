"""
09_domain_shift_analysis.py
Purpose: Explicit domain shift analysis between Nusratt train and Souvik.
- Mann-Whitney U and KS tests for numeric features
- Categorical distribution comparisons
- Domain shift report
- Distribution plots
"""

import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ks_2samp, mannwhitneyu

# ── Load feature matrices ─────────────────────────────────────────────────────
train = pd.read_csv("outputs/features_train.csv")
souv  = pd.read_csv("outputs/features_souvik.csv")

FEAT_COLS = [c for c in train.columns if c not in ("addiction_label","addicted_score")]

X_train = train[FEAT_COLS]
X_souv  = souv[FEAT_COLS]

# ── Numeric statistical tests ─────────────────────────────────────────────────
print("\nDOMAIN SHIFT ANALYSIS — Numeric Features")
print("="*60)
shift_results = []
for col in FEAT_COLS:
    a = X_train[col].dropna().values
    b = X_souv[col].dropna().values
    ks_stat, ks_p  = ks_2samp(a, b)
    mw_stat, mw_p  = mannwhitneyu(a, b, alternative="two-sided")
    significant = (ks_p < 0.05) or (mw_p < 0.05)
    flag = "** SHIFT **" if significant else "ok"
    print(f"  {col:40s}  KS_p={ks_p:.4f}  MW_p={mw_p:.4f}  {flag}")
    shift_results.append({
        "feature": col,
        "train_mean": np.mean(a), "souvik_mean": np.mean(b),
        "train_std": np.std(a),   "souvik_std": np.std(b),
        "ks_stat": ks_stat, "ks_p": ks_p,
        "mw_stat": mw_stat, "mw_p": mw_p,
        "significant_shift": significant
    })

df_shift = pd.DataFrame(shift_results)
df_shift.to_csv("outputs/domain_shift_stats.csv", index=False)

# ── Distribution plots ────────────────────────────────────────────────────────
n_feats = len(FEAT_COLS)
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for i, col in enumerate(FEAT_COLS):
    ax = axes[i]
    ax.hist(X_train[col].dropna(), bins=25, alpha=0.6, label="Nusratt Train", color="steelblue", density=True)
    ax.hist(X_souv[col].dropna(),  bins=25, alpha=0.6, label="Souvik OOD",    color="orange",    density=True)
    ax.set_title(col, fontsize=9)
    ax.legend(fontsize=7)
for j in range(i+1, len(axes)):
    axes[j].axis("off")
plt.suptitle("Feature Distributions: Nusratt Train vs Souvik", fontsize=12)
plt.tight_layout()
plt.savefig("outputs/domain_shift_distributions.png", dpi=120)
plt.close()

# ── Categorical comparison from RAW cleaned datasets ─────────────────────────
n_raw = pd.read_csv("outputs/nusratt_clean.csv")
s_raw = pd.read_csv("outputs/souvik_clean.csv")

cat_report = []
for col, scol in [("gender","gender"), ("relationship_status","relationship_status")]:
    n_dist = n_raw[col].value_counts(normalize=True).rename("nusratt_pct")
    s_dist = s_raw[scol].value_counts(normalize=True).rename("souvik_pct")
    merged = pd.concat([n_dist, s_dist], axis=1).fillna(0)
    merged["abs_diff"] = (merged["nusratt_pct"] - merged["souvik_pct"]).abs()
    print(f"\nCategorical shift — {col}:\n{merged.to_string()}")
    merged["feature"] = col
    cat_report.append(merged)

cat_df = pd.concat(cat_report)
cat_df.to_csv("outputs/domain_shift_categorical.csv")

# ── Domain shift summary report ───────────────────────────────────────────────
shifted = df_shift[df_shift["significant_shift"]]
not_shifted = df_shift[~df_shift["significant_shift"]]

report = f"""
DOMAIN SHIFT REPORT
===================
Nusratt train N={len(X_train)}  |  Souvik N={len(X_souv)}

SIGNIFICANTLY SHIFTED FEATURES (KS or MW p < 0.05):
{shifted[['feature','train_mean','souvik_mean','ks_p','mw_p']].to_string(index=False) if len(shifted)>0 else '  None'}

STABLE FEATURES (no significant shift):
{not_shifted[['feature','train_mean','souvik_mean','ks_p','mw_p']].to_string(index=False) if len(not_shifted)>0 else '  None'}

INTERPRETATION:
- Shifted features indicate the Souvik population differs from Nusratt on those constructs.
- This is the primary source of OOD performance drop in classification evaluation.
- Features derived from proxy variables (psychological_distress_index, sleep_disturbance_proxy)
  are particularly susceptible to shift because they are constructed from different instruments.
- Shifted features do NOT invalidate the model; they explain the performance gap.
"""

print(report)
with open("outputs/domain_shift_report.txt","w") as f:
    f.write(report)

print("Domain shift analysis complete. Outputs saved.")