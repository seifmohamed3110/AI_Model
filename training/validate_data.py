"""
training/validate_data.py

Responsibilities:
- Validate slim_resumes_with_strong.csv
- Show dataset size, columns, missing values, field distribution, and grade distribution
- Check manual review progress from data/manual_review_checklist.csv
- Warn if class distribution or manual review progress looks unhealthy
"""

import os
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "slim_resumes_with_strong.csv")
MANUAL_PATH = os.path.join(BASE_DIR, "data", "manual_review_checklist.csv")


def print_distribution(series: pd.Series, title: str) -> None:
    counts = series.value_counts(dropna=False)
    percentages = (counts / len(series) * 100).round(2)

    print(title)
    for label, count in counts.items():
        pct = percentages[label]
        print(f"  {label}: {count} ({pct}%)")
    print()


def main() -> None:
    print("=== Dataset Validation ===")

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Dataset not found: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH, dtype=str, encoding="utf-8", on_bad_lines="skip")
    df.columns = df.columns.str.strip()

    print(f"Dataset: {os.path.basename(DATA_PATH)}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}\n")

    # ── Missing values check ────────────────────────────────────────────────
    important_cols = ["Resume_str", "field", "grade"]
    missing_summary = {}

    for col in important_cols:
        if col in df.columns:
            missing_summary[col] = int(df[col].isna().sum())
        else:
            missing_summary[col] = "MISSING COLUMN"

    print("Important column check:")
    for col, val in missing_summary.items():
        print(f"  {col}: {val}")
    print()

    # ── Basic cleaning view ─────────────────────────────────────────────────
    if "Resume_str" in df.columns:
        valid_resume_mask = df["Resume_str"].astype(str).str.strip().str.len() > 50
        print(f"Usable resumes (>50 chars): {int(valid_resume_mask.sum())}/{len(df)}")
        print()

    # ── Field distribution ──────────────────────────────────────────────────
    if "field" in df.columns:
        print_distribution(df["field"], "Field distribution:")

        field_counts = df["field"].value_counts()
        low_fields = field_counts[field_counts < 50]
        if not low_fields.empty:
            print("WARNING: Some field classes are very small:")
            for label, count in low_fields.items():
                print(f"  {label}: {count}")
            print()

    # ── Grade distribution ──────────────────────────────────────────────────
    if "grade" in df.columns:
        df["grade"] = pd.to_numeric(df["grade"], errors="coerce")
        print_distribution(df["grade"], "Grade distribution:")

        grade_counts = df["grade"].value_counts()
        low_grades = grade_counts[grade_counts < 50]
        if not low_grades.empty:
            print("WARNING: Some grade classes are very small:")
            for label, count in low_grades.items():
                print(f"  grade {label}: {count}")
            print()

    # ── Manual review progress ──────────────────────────────────────────────
    if os.path.exists(MANUAL_PATH):
        manual = pd.read_csv(MANUAL_PATH, dtype=str, encoding="utf-8", on_bad_lines="skip")
        manual.columns = manual.columns.str.strip()

        print("=== Manual Review Progress ===")
        required_manual_cols = ["resume_id", "career_label", "quality_label", "status"]
        missing_manual_cols = [c for c in required_manual_cols if c not in manual.columns]

        if missing_manual_cols:
            print(f"WARNING: manual_review_checklist.csv is missing columns: {missing_manual_cols}")
        else:
            status_series = manual["status"].astype(str).str.strip().str.lower()
            done_mask = status_series == "done"
            done = int(done_mask.sum())
            total = len(manual)

            print(f"Reviewed: {done}/{total}")

            if done > 0:
                reviewed = manual[done_mask].copy()

                career_filled = int(
                    reviewed["career_label"].astype(str).str.strip().replace({"": pd.NA}).notna().sum()
                )
                quality_filled = int(
                    reviewed["quality_label"].astype(str).str.strip().replace({"": pd.NA}).notna().sum()
                )

                print(f"Reviewed rows with career_label filled: {career_filled}/{done}")
                print(f"Reviewed rows with quality_label filled: {quality_filled}/{done}")

            if done < 20:
                print("WARNING: Target not met: need at least 20 manually reviewed resumes.")
            else:
                print("OK: Manual review minimum target met.")
        print()
    else:
        print("WARNING: manual_review_checklist.csv not found.\n")

    print("=== Validation Complete ===")


if __name__ == "__main__":
    main()