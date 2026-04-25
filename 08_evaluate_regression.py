"""
08_evaluate_regression.py
Purpose: Evaluate regression models on Nusratt test set.
Regression is Nusratt-only (no ground-truth score for Souvik).
"""

import pandas as pd
import numpy as np
import pickle, os, warnings
warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Load ──────────────────────────────────────────────────────────────────────
test = pd.read_csv("outputs/features_test.csv")
FEAT_COLS  = pickle.load(open("models/feature_cols.pkl","rb"))
scaler_reg = pickle.load(open("models/scaler_reg.pkl","rb"))

X_test = scaler_reg.transform(test[FEAT_COLS].values)
y_test = test["addicted_score"].values

models_reg = {
    "Ridge":   pickle.load(open("models/reg_ridge.pkl","rb")),
    "RF_Reg":  pickle.load(open("models/reg_rf.pkl","rb")),
    "XGB_Reg": pickle.load(open("models/reg_xgb.pkl","rb")),
}

results = []
for name, m in models_reg.items():
    y_pred = m.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    print(f"{name}: MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")
    results.append({"model":name,"MAE":mae,"RMSE":rmse,"R2":r2})

    # Scatter plot
    fig, ax = plt.subplots(figsize=(5,4))
    ax.scatter(y_test, y_pred, alpha=0.5, s=15)
    lims = [min(y_test.min(), y_pred.min())-0.5, max(y_test.max(), y_pred.max())+0.5]
    ax.plot(lims, lims, "r--", lw=1)
    ax.set_xlabel("True Addicted Score"); ax.set_ylabel("Predicted")
    ax.set_title(f"{name} — Nusratt Test")
    plt.tight_layout()
    plt.savefig(f"outputs/reg_scatter_{name}.png", dpi=120); plt.close()

df_reg = pd.DataFrame(results)
df_reg.to_csv("outputs/regression_metrics.csv", index=False)
print("\nRegression metrics saved.")
print(df_reg.to_string(index=False))