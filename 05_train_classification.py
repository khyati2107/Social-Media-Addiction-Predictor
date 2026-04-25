"""
05_train_classification.py
Purpose: Train multi-class addiction classifiers on Nusratt training split.
Models: Logistic Regression, Random Forest, XGBoost
- SMOTE only on training split
- Cross-validation on training split
- Save trained models
"""

import pandas as pd
import numpy as np
import pickle, os, warnings
warnings.filterwarnings("ignore")
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

from sklearn.linear_model   import LogisticRegression
from sklearn.ensemble       import RandomForestClassifier
from sklearn.svm            import SVC
from sklearn.preprocessing  import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics         import make_scorer, f1_score
from imblearn.over_sampling  import SMOTE
from xgboost                 import XGBClassifier

# ── Load ──────────────────────────────────────────────────────────────────────
train = pd.read_csv("outputs/features_train.csv")
FEAT_COLS = [c for c in train.columns if c not in ("addiction_label","addicted_score")]

X_train = train[FEAT_COLS].values
y_raw   = train["addiction_label"].values

le = LabelEncoder()
y_train = le.fit_transform(y_raw)   # Low=0, Moderate=1, High=2 (alphabetical)
print("Class mapping:", dict(zip(le.classes_, le.transform(le.classes_))))
print("Class distribution:", dict(zip(*np.unique(y_train, return_counts=True))))

# ── Scale (fit on train, apply later in evaluation) ───────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)

# ── SMOTE on training split only ─────────────────────────────────────────────
sm = SMOTE(random_state=42, k_neighbors=3)
X_train_sm, y_train_sm = sm.fit_resample(X_train_sc, y_train)
print(f"After SMOTE: {X_train_sm.shape}, dist={dict(zip(*np.unique(y_train_sm, return_counts=True)))}")

# ── Cross-validation helper ───────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    "accuracy": "accuracy",
    "macro_f1": make_scorer(f1_score, average="macro"),
}

def cv_report(model, X, y, name):
    res = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    print(f"\n[CV] {name}")
    print(f"  Acc  : {res['test_accuracy'].mean():.3f} ± {res['test_accuracy'].std():.3f}")
    print(f"  MacF1: {res['test_macro_f1'].mean():.3f} ± {res['test_macro_f1'].std():.3f}")
    return res

# ── Model 1: Logistic Regression (L2 regularised) ────────────────────────────
lr = LogisticRegression(
    C=0.5, max_iter=1000, class_weight="balanced",
    multi_class="multinomial", solver="lbfgs", random_state=42
)
cv_report(lr, X_train_sm, y_train_sm, "LogisticRegression")
lr.fit(X_train_sm, y_train_sm)

# ── Model 2: Random Forest (feature subsampling) ──────────────────────────────
rf = RandomForestClassifier(
    n_estimators=300, max_features="sqrt",      # subsampling
    min_samples_leaf=5, class_weight="balanced",
    random_state=42, n_jobs=-1
)
cv_report(rf, X_train_sm, y_train_sm, "RandomForest")
rf.fit(X_train_sm, y_train_sm)

# ── Model 3: XGBoost ─────────────────────────────────────────────────────────
xgb = XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,    # feature subsampling
    reg_alpha=0.1, reg_lambda=1.0,          # regularisation
    use_label_encoder=False,
    eval_metric="mlogloss",
    random_state=42, n_jobs=-1
)
cv_report(xgb, X_train_sm, y_train_sm, "XGBoost")
xgb.fit(X_train_sm, y_train_sm)

# ── Model 4: SVM ──────────────────────────────────────────────────────────────
svm = SVC(C=1.0, kernel="rbf", class_weight="balanced",
          probability=True, random_state=42)
cv_report(svm, X_train_sm, y_train_sm, "SVM")
svm.fit(X_train_sm, y_train_sm)

# ── Save models, scaler, encoder ─────────────────────────────────────────────
models = {"lr": lr, "rf": rf, "xgb": xgb, "svm": svm}
for name, m in models.items():
    pickle.dump(m, open(f"models/clf_{name}.pkl","wb"))

pickle.dump(scaler, open("models/scaler_clf.pkl","wb"))
pickle.dump(le,     open("models/label_encoder.pkl","wb"))
pickle.dump(FEAT_COLS, open("models/feature_cols.pkl","wb"))

print("\nAll classification models saved.")