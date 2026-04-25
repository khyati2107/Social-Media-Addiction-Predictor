# 📱 Social Media Addiction Predictor

A behavioral-psychological machine learning system that analyzes user habits and mental health indicators to classify social media addiction risk as **Low / Moderate / High**, with explainable AI (SHAP) and a deployed Streamlit web application.

> Associated IEEE paper submitted as part of coursework at BITS Pilani Dubai Campus.

---

## 👥 Authors

| Name | Department | Institution |
|------|-----------|-------------|
| Khyati Prashant Jetly | Computer Science | BITS Pilani Dubai |
| Srinjita Roy Chowdhury | Computer Science | BITS Pilani Dubai |
| Soorya Kiran Kakkarayil | Computer Science | BITS Pilani Dubai |

🏫 **Birla Institute of Technology and Science, Pilani — Dubai Campus**

---

## Project Structure

```text
.
├── 01_data_audit.py                  # Audit raw datasets (shape, dtypes, missing values)
├── 02_clean_datasets.py              # Clean & standardise both datasets independently
├── 03_column_mapping.py              # Build Nusratt ↔ Souvik feature mapping table
├── 04_feature_engineering_shared.py  # Engineer 6 transferable features (train-only fit)
├── 05_train_classification.py        # Train LR, RF, XGBoost, SVM classifiers + SMOTE
├── 06_evaluate_classification.py     # Evaluate in-distribution + OOD performance gap
├── 07_train_regression.py            # Train Ridge, RF, XGBoost regressors (Nusratt only)
├── 08_evaluate_regression.py         # Evaluate regression on Nusratt test set
├── 09_domain_shift_analysis.py       # KS + Mann-Whitney tests, distribution plots
├── 10_shap_explainability.py         # SHAP feature importance & explainability plots
├── generate_roc_plots.py             # ROC curves (per-model + combined 2x2 grid)
├── app.py                            # Streamlit web application
├── run_all.py                        # One-command full pipeline runner
├── requirements.txt
├── Nusratt.csv                       # Dataset 1 (labeled, training source)
├── Souvik.csv                        # Dataset 2 (unlabeled, OOD validation)
├── models/                           # Saved model pickle files (auto-generated)
└── outputs/                          # Metrics, plots, reports (auto-generated)
```
## 📊 Datasets

| Dataset | Source | Records | Labels |
|---------|--------|---------|--------|
| **Nusratt** | [Kaggle — Student Social Media Addiction Analysis](https://www.kaggle.com/datasets/zahranusratt/student-social-media-addiction-analysis-dataset) | 705 | ✅ `addicted_score` (1–10) |
| **Souvik** | [Kaggle — Social Media & Mental Health](https://www.kaggle.com/datasets/souvikahmed071/social-media-and-mental-health) | 481 | ❌ No ground-truth label (validation only) |

Nusratt is used for training and in-distribution testing. Souvik is used exclusively as an out-of-distribution (OOD) external validation set. Souvik has no ground-truth addiction labels — proxy labels are generated for reference only and never used in model training.

---

## ⚙️ Engineered Features

Six transferable features are derived from both datasets using parameters **fitted on the Nusratt training split only**, preventing data leakage:

| Feature | Description |
|---------|-------------|
| `behavioral_intensity_index` | Normalized usage hours + conflict/purposeless-use proxy |
| `psychological_distress_index` | Inverted mental health score (Nusratt) / depression + worry Likert mean (Souvik) |
| `sleep_disturbance_proxy` | Inverted sleep hours (Nusratt) / sleep issue score (Souvik) |
| `age_norm` | Min-max normalized age |
| `is_student` | Binary flag from academic level / occupation status |
| `usage_hours_sq` | Squared usage hours (non-linear interaction term) |

---

## 🤖 Models

### Classification (Low / Moderate / High)
- Logistic Regression (L2, balanced class weights)
- Random Forest (300 estimators, balanced)
- XGBoost (depth 4, L1+L2 regularisation)
- SVM (RBF kernel, balanced)

All classifiers use **SMOTE** (training split only) to address class imbalance, with **5-fold stratified cross-validation**.

### Regression (Nusratt only — predicts raw `addicted_score` 1–10)
- Ridge Regression
- Random Forest Regressor
- XGBoost Regressor

### Explainability
- **SHAP** (SHapley Additive exPlanations) for global feature importance and individual prediction explanations

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/Social-Media-Addiction-Predictor.git
cd Social-Media-Addiction-Predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline
```bash
python run_all.py
```

This executes all scripts in order — data audit → cleaning → feature engineering → training → evaluation → domain shift analysis → SHAP explainability. All outputs are saved to `outputs/` and `models/`.

### 4. Launch the Streamlit app
```bash
streamlit run app.py
```

> **Note:** Run the full pipeline at least once before launching the app so that model files are present in `models/`.

---

## 📈 Outputs

After running the pipeline, the `outputs/` directory will contain:

- `audit_nusratt.csv`, `audit_souvik.csv` — data quality reports
- `column_mapping.csv`, `proxy_mapping_report.txt` — feature mapping documentation
- `features_train.csv`, `features_test.csv`, `features_souvik.csv` — engineered feature matrices
- `classification_metrics.csv` — accuracy, macro-F1, precision, recall, AUC for all models
- `performance_gap_report.txt` — in-distribution vs OOD performance gap analysis
- `regression_metrics.csv` — MAE, RMSE, R² for regression models
- `domain_shift_stats.csv`, `domain_shift_report.txt` — statistical shift analysis (KS + Mann-Whitney)
- `domain_shift_distributions.png` — feature distribution comparison plots
- `cm_*.png` — confusion matrices for all models × datasets
- `roc_*.png`, `roc_2x2_grid.png` — ROC curves (per-model + combined)
- `reg_scatter_*.png` — regression scatter plots

---

## 🔍 Domain Shift Analysis

Because the two datasets use different survey instruments, an explicit domain shift analysis is conducted using **Kolmogorov-Smirnov** and **Mann-Whitney U** tests on all engineered features. Features flagged as significantly shifted (p < 0.05) are documented and used to explain the expected OOD performance drop — this drop does not invalidate the model, but contextualises its generalisation behaviour.

---

## ⚠️ Limitations & Ethical Notes

- Souvik has **no ground-truth addiction labels**. OOD evaluation uses proxy labels derived from usage hours only — treat those numbers as indicative, not definitive.
- The `sleep_issue_score` (Souvik Q20) measures **sleep disturbance**, not sleep duration. It is used only as a proxy and never converted to hours.
- `avg_daily_usage_hours` in Souvik is binned text — midpoint approximation introduces ±0.5h noise.
- Addiction classification thresholds (≤4 = Low, ≤6 = Moderate, >6 = High) are operationally defined from the Nusratt dataset and may not generalise to clinical definitions.
- This tool is intended for **research purposes only** and should not be used for clinical diagnosis.

---

## 📦 Requirements
