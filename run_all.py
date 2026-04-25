"""
run_all.py
Runs the full pipeline in order. Execute: python run_all.py
"""
import subprocess, sys

scripts = [
    "01_data_audit.py",
    "02_clean_datasets.py",
    "03_column_mapping.py",
    "04_feature_engineering_shared.py",
    "05_train_classification.py",
    "06_evaluate_classification.py",
    "07_train_regression.py",
    "08_evaluate_regression.py",
    "09_domain_shift_analysis.py",
    "10_shap_explainability.py",
]

for script in scripts:
    print(f"\n{'='*60}")
    print(f"RUNNING: {script}")
    print('='*60)
    result = subprocess.run([sys.executable, script], capture_output=False)
    if result.returncode != 0:
        print(f"ERROR in {script}. Stopping.")
        sys.exit(1)

print("\n✓ Full pipeline completed successfully.")