"""
07_train_regression.py
Purpose: Optional Nusratt-only regression branch.
Predict raw addicted_score (1-10 continuous).
Models: Ridge, Random Forest Regressor, XGBoost Regressor
Fit only on Nusratt training split.
"""

import pandas as pd
import numpy as np
import pickle, os, warnings
warnings.filterwarnings("ignore")
os.makedirs("models",  exist_ok=True)
os.makedirs("outputs", exist_ok=True)

from sklearn.linear_model    import Ridge
from sklearn.ensemble        import RandomForestRegressor
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import KFold, cross_validate
from xgboost                 import XGBRegressor

# ── Load ──────────────────────────────────────────────────────────────────────
train = pd.read_csv("outputs/features_train.csv")
FEAT_COLS = pickle.load(open("models/feature_cols.pkl","rb"))

X_train = train[FEAT_COLS].values
y_train = train["addicted_score"].values

scaler_reg = StandardScaler()
X_train_sc = scaler_reg.fit_transform(X_train)

# ── Cross-validation ──────────────────────────────────────────────────────────
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = {"r2":"r2", "neg_mae":"neg_mean_absolute_error",
           "neg_rmse":"neg_root_mean_squared_error"}

def cv_report_reg(model, X, y, name):
    res = cross_validate(model, X, y, cv=cv, scoring=scoring)
    print(f"\n[CV-Reg] {name}")
    print(f"  R²  : {res['test_r2'].mean():.3f} ± {res['test_r2'].std():.3f}")
    print(f"  MAE : {-res['test_neg_mae'].mean():.3f}")
    print(f"  RMSE: {-res['test_neg_rmse'].mean():.3f}")
    return res

# Model 1: Ridge
ridge = Ridge(alpha=1.0)
cv_report_reg(ridge, X_train_sc, y_train, "Ridge")
ridge.fit(X_train_sc, y_train)

# Model 2: Random Forest Regressor
rf_reg = RandomForestRegressor(
    n_estimators=300, max_features="sqrt",
    min_samples_leaf=5, random_state=42, n_jobs=-1
)
cv_report_reg(rf_reg, X_train_sc, y_train, "RandomForestRegressor")
rf_reg.fit(X_train_sc, y_train)

# Model 3: XGBoost Regressor
xgb_reg = XGBRegressor(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, n_jobs=-1
)
cv_report_reg(xgb_reg, X_train_sc, y_train, "XGBoostRegressor")
xgb_reg.fit(X_train_sc, y_train)

# ── Save ──────────────────────────────────────────────────────────────────────
pickle.dump(ridge,     open("models/reg_ridge.pkl","wb"))
pickle.dump(rf_reg,    open("models/reg_rf.pkl","wb"))
pickle.dump(xgb_reg,   open("models/reg_xgb.pkl","wb"))
pickle.dump(scaler_reg,open("models/scaler_reg.pkl","wb"))

print("\nRegression models saved.")