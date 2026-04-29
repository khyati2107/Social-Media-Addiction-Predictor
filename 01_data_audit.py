"""
01_data_audit.py Purpose: Audit raw datasets — shape, dtypes, missing values, value distributions.
"""

import pandas as pd
import os

os.makedirs("outputs", exist_ok=True)

# Load 
nusratt = pd.read_csv("Nusratt.csv")
souvik  = pd.read_csv("Souvik.csv")

def audit(df, name):
    print(f"\n{'='*60}")
    print(f"DATASET: {name}   shape={df.shape}")
    print(f"{'='*60}")
    info = pd.DataFrame({
        "dtype":   df.dtypes,
        "missing": df.isna().sum(),
        "pct_missing": (df.isna().mean() * 100).round(2),
        "nunique": df.nunique(),
        "sample":  df.iloc[0] if len(df) > 0 else None
    })
    print(info.to_string())
    print("\n-- Numeric summary --")
    print(df.describe(include="number").T.to_string())

audit(nusratt, "Nusratt")
audit(souvik,  "Souvik")

# Save audit summaries
nusratt_info = pd.DataFrame({
    "dtype":   nusratt.dtypes,
    "missing": nusratt.isna().sum(),
    "pct_missing": (nusratt.isna().mean() * 100).round(2),
    "nunique": nusratt.nunique()
})
souvik_info = pd.DataFrame({
    "dtype":   souvik.dtypes,
    "missing": souvik.isna().sum(),
    "pct_missing": (souvik.isna().mean() * 100).round(2),
    "nunique": souvik.nunique()
})
nusratt_info.to_csv("outputs/audit_nusratt.csv")
souvik_info.to_csv("outputs/audit_souvik.csv")
print("\nAudit CSVs saved to outputs/")