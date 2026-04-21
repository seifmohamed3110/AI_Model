"""
training/validate_data.py

Phase 1 helper:
- Prints dataset distribution (field + grade)
- Checks manual review progress from data/manual_review_checklist.csv
"""

import os
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "slim_resumes_with_strong.csv")
MANUAL_PATH = os.path.join(BASE_DIR, "data", "manual_review_checklist.csv")


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    print("=== Dataset Validation ===")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}\n")

    if "field" in df.columns:
        print("Field distribution:")
        print(df["field"].value_counts().to_string())
        print()

    if "grade" in df.columns:
        print("Grade distribution:")
        print(df["grade"].value_counts().sort_index().to_string())
        print()

    if os.path.exists(MANUAL_PATH):
        manual = pd.read_csv(MANUAL_PATH)
        done = int((manual["status"].astype(str).str.lower() == "done").sum())
        total = len(manual)
        print("Manual review progress:")
        print(f"Reviewed: {done}/{total}")
        if done < 20:
            print("WARNING: Target not met: need at least 20 manually reviewed resumes.")
        else:
            print("OK: Manual review minimum target met.")
    else:
        print("WARNING: manual_review_checklist.csv not found.")


if __name__ == "__main__":
    main()
