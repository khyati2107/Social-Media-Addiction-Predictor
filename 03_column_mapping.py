"""
03_column_mapping.py
Purpose: Build the explicit Nusratt ↔ Souvik feature mapping table.
Categories: direct_match or proxy_match or no_match
Output: outputs/column_mapping.csv, outputs/proxy_mapping_report.txt
"""

import pandas as pd
import os

os.makedirs("outputs", exist_ok=True)

mapping = [
    # Nusratt column           Souvik column(s)             match_type  notes
    ("age",                    "age",                        "direct",
     "Both numeric age"),
    ("gender",                 "gender",                     "direct",
     "Both Male/Female; Souvik has 'Other' category"),
    ("relationship_status",    "relationship_status",        "direct",
     "Both have Single/In Relationship/Married/Complicated"),
    ("avg_daily_usage_hours",  "avg_daily_usage_hours",      "direct",
     "Nusratt: continuous float; Souvik: text binned → midpoint float"),
    ("conflicts",              "purposeless_use",            "proxy",
     "Conflicts over SM (0-5 int) vs purposeless use frequency (1-5 Likert). "
     "Both tap compulsive/problematic engagement; not identical constructs."),
    ("mental_health_score",    "depression_freq + bothered_by_worries",  "proxy",
     "Nusratt: composite MH score (1-10); Souvik: two Likert items. "
     "Proxy: average of Q13+Q18 scaled to 1-10 (see feature engineering)."),
    ("sleep_hours",            "sleep_issue_score",          "proxy",
     "Nusratt: actual sleep hours; Souvik Q20: sleep disturbance index (1-5). "
     "DO NOT treat as direct equivalent. Used only as sleep_disturbance proxy."),
    ("most_used_platform",     "platforms_used",             "proxy",
     "Nusratt: single platform; Souvik: multi-select list. "
     "Shared feature: platform_diversity_score (count of platforms listed)."),
    ("academic_level",         "occupation_status",          "proxy",
     "Nusratt: academic level; Souvik: occupation. "
     "Mapped to is_student flag (both have student categories)."),
    ("affects_academic",       None,                         "no_match",
     "No equivalent question in Souvik."),
    ("country",                None,                         "no_match",
     "No country/location field in Souvik."),
    ("addicted_score",         None,                         "no_match",
     "Nusratt target variable; Souvik has no ground-truth addiction label."),
    # Souvik-only columns (no Nusratt equivalent)
    (None, "easily_distracted",         "no_match",
     "Souvik Q12: distraction scale. Partially captured in distraction_concentration_index."),
    (None, "concentration_difficulty",  "no_match",
     "Souvik Q14: concentration difficulty. Same note as above."),
    (None, "social_comparison",         "no_match",
     "Souvik Q15: social comparison frequency. Used in social_comparison_index."),
    (None, "validation_seeking",        "no_match",
     "Souvik Q17: validation seeking. No direct Nusratt equivalent."),
    (None, "interest_fluctuation",      "no_match",
     "Souvik Q19: interest fluctuation. No direct Nusratt equivalent."),
    (None, "restlessness",              "no_match",
     "Souvik Q11: restlessness. No direct Nusratt equivalent."),
    (None, "distraction_busy",          "no_match",
     "Souvik Q10: distraction while busy. No direct Nusratt equivalent."),
    (None, "org_type",                  "no_match",
     "Souvik: org affiliation. No Nusratt equivalent."),
]

df_map = pd.DataFrame(mapping, columns=[
    "nusratt_column","souvik_column","match_type","notes"
])
df_map.to_csv("outputs/column_mapping.csv", index=False)

#Shared features to engineer (both sides must have inputs, no mixing)
shared_direct = df_map[df_map["match_type"] == "direct"]["nusratt_column"].dropna().tolist()
shared_proxy  = df_map[df_map["match_type"] == "proxy"]["nusratt_column"].dropna().tolist()

report = f"""
PROXY MAPPING REPORT
====================
Direct matches (usable as-is after harmonisation):
{chr(10).join('  - ' + c for c in shared_direct)}

Proxy matches (engineered or approximated):
{chr(10).join('  - ' + c for c in shared_proxy)}

"""

with open("outputs/proxy_mapping_report.txt","w") as f:
    f.write(report)

print(report)
print("Mapping table saved: outputs/column_mapping.csv")