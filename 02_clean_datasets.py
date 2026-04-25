"""
02_clean_datasets.py
Purpose: Clean both datasets independently.
- Standardise column names to lowercase_snake_case
- Type-cast, strip whitespace, recode categoricals
- Map Souvik's ordinal text responses to numeric scales
- Drop non-SM-users from Souvik (Q6 == 'No')
- Save cleaned CSVs
"""

import pandas as pd
import re
import os

os.makedirs("outputs", exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def to_snake(s):
    s = re.sub(r'[^\w\s]', '', str(s))
    s = re.sub(r'\s+', '_', s.strip().lower())
    return s

# ── NUSRATT ───────────────────────────────────────────────────────────────────
n = pd.read_csv("Nusratt.csv")
n.columns = [to_snake(c) for c in n.columns]

# Rename for clarity
n = n.rename(columns={
    "avg_daily_usage_hours":        "avg_daily_usage_hours",
    "affects_academic_performance": "affects_academic",
    "sleep_hours_per_night":        "sleep_hours",
    "mental_health_score":          "mental_health_score",
    "conflicts_over_social_media":  "conflicts",
    "addicted_score":               "addicted_score",
    "relationship_status":          "relationship_status",
    "most_used_platform":           "most_used_platform",
    "academic_level":               "academic_level",
})

# Drop student_id — not a feature
n = n.drop(columns=["student_id"], errors="ignore")

# Enforce numeric
for col in ["age","avg_daily_usage_hours","sleep_hours","mental_health_score","conflicts","addicted_score"]:
    n[col] = pd.to_numeric(n[col], errors="coerce")

# Standardise categoricals
n["gender"] = n["gender"].str.strip().str.title()
n["relationship_status"] = n["relationship_status"].str.strip().str.title()
n["affects_academic"] = n["affects_academic"].str.strip().str.title()
n["academic_level"] = n["academic_level"].str.strip().str.title()

# Classification label
def label_addiction(score):
    if score <= 4:  return "Low"
    elif score <= 6: return "Moderate"
    else:           return "High"

n["addiction_label"] = n["addicted_score"].apply(label_addiction)

# Drop rows with missing target
n = n.dropna(subset=["addicted_score"])
n = n.reset_index(drop=True)

print(f"Nusratt cleaned: {n.shape}")
print(n["addiction_label"].value_counts())
n.to_csv("outputs/nusratt_clean.csv", index=False)


# ── SOUVIK ────────────────────────────────────────────────────────────────────
s = pd.read_csv("Souvik.csv")

# Rename columns to short snake_case
col_map = {
    "Timestamp":                                                                       "timestamp",
    "1. What is your age?":                                                            "age",
    "2. Gender":                                                                       "gender",
    "3. Relationship Status":                                                          "relationship_status",
    "4. Occupation Status":                                                            "occupation_status",
    "5. What type of organizations are you affiliated with?":                          "org_type",
    "6. Do you use social media?":                                                     "uses_sm",
    "7. What social media platforms do you commonly use?":                             "platforms_used",
    "8. What is the average time you spend on social media every day?":                "avg_daily_usage_text",
    "9. How often do you find yourself using Social media without a specific purpose?": "purposeless_use",
    "10. How often do you get distracted by Social media when you are busy doing something?": "distraction_busy",
    "11. Do you feel restless if you haven't used Social media in a while?":           "restlessness",
    "12. On a scale of 1 to 5, how easily distracted are you?":                       "easily_distracted",
    "13. On a scale of 1 to 5, how much are you bothered by worries?":                "bothered_by_worries",
    "14. Do you find it difficult to concentrate on things?":                         "concentration_difficulty",
    "15. On a scale of 1-5, how often do you compare yourself to other successful people through the use of social media?": "social_comparison",
    "16. Following the previous question, how do you feel about these comparisons, generally speaking?": "comparison_feeling_text",
    "17. How often do you look to seek validation from features of social media?":     "validation_seeking",
    "18. How often do you feel depressed or down?":                                   "depression_freq",
    "19. On a scale of 1 to 5, how frequently does your interest in daily activities fluctuate?": "interest_fluctuation",
    "20. On a scale of 1 to 5, how often do you face issues regarding sleep?":        "sleep_issue_score",
}
s = s.rename(columns=col_map)

# Keep only SM users (Q6 == 'Yes')
s = s[s["uses_sm"].str.strip().str.title() == "Yes"].copy()

# Map text usage to numeric hours (midpoint approximation)
usage_map = {
    "Less than an Hour":     0.5,
    "Between 1 and 2 hours": 1.5,
    "Between 2 and 3 hours": 2.5,
    "Between 3 and 4 hours": 3.5,
    "Between 4 and 5 hours": 4.5,
    "More than 5 hours":     6.0,
}
s["avg_daily_usage_hours"] = s["avg_daily_usage_text"].map(usage_map)

# Enforce numeric Likert columns
likert_cols = [
    "purposeless_use","distraction_busy","restlessness",
    "easily_distracted","bothered_by_worries","concentration_difficulty",
    "social_comparison","validation_seeking","depression_freq",
    "interest_fluctuation","sleep_issue_score"
]
for col in likert_cols:
    s[col] = pd.to_numeric(s[col], errors="coerce")

s["age"] = pd.to_numeric(s["age"], errors="coerce")

# Standardise gender — collapse Nonbinary / others → "Other"
def norm_gender(g):
    g = str(g).strip().title()
    if g in ("Male","Female"): return g
    return "Other"
s["gender"] = s["gender"].apply(norm_gender)

# Standardise relationship_status
def norm_rel(r):
    r = str(r).strip().title()
    mapping = {
        "In A Relationship": "In Relationship",
        "In A Relation":     "In Relationship",
    }
    return mapping.get(r, r)
s["relationship_status"] = s["relationship_status"].apply(norm_rel)

# Drop rows with missing usage hours (can't impute intent)
s = s.dropna(subset=["avg_daily_usage_hours"])
s = s.reset_index(drop=True)

dropped_report = {
    "souvik_non_sm_users_dropped": int((pd.read_csv("Souvik.csv")["6. Do you use social media?"].str.strip().str.title() == "No").sum()),
    "souvik_missing_usage_dropped": int(s["avg_daily_usage_hours"].isna().sum()),
}
print(f"Souvik cleaned: {s.shape}")
print("Dropped:", dropped_report)

s.to_csv("outputs/souvik_clean.csv", index=False)

import json
with open("outputs/dropped_report.json","w") as f:
    json.dump(dropped_report, f, indent=2)

print("Cleaned datasets saved.")