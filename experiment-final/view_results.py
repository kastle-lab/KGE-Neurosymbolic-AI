# view_results.py

import pandas as pd

basepath = "./500people"   

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 180)
pd.set_option("display.max_rows", 100)

summary_all = pd.read_csv(f"{basepath}/prediction_error_summary_all_people.csv")
summary_missing = pd.read_csv(f"{basepath}/prediction_error_summary_missing_only.csv")
all_errors = pd.read_csv(f"{basepath}/prediction_errors_all_runs.csv")

print("\nSUMMARY — ALL PEOPLE")
print(summary_all.round(4).to_string(index=False))

print("\nSUMMARY — MISSING ONLY")
print(summary_missing.round(4).to_string(index=False))

print("\nWORST 20 MISSING-ONLY ERRORS")
missing = all_errors[
    (all_errors["exists_in_run"] == False) &
    (all_errors["absolute_error"].notna())
]

worst = missing.sort_values("absolute_error", ascending=False).head(20)
print(worst.round(4).to_string(index=False))

print("\nBEST 20 MISSING-ONLY ERRORS")
best = missing.sort_values("absolute_error", ascending=True).head(20)
print(best.round(4).to_string(index=False))