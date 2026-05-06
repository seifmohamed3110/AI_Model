"""
training/validate_data.py

Responsibilities:
- Validate labeled_resumes.csv (output of label_data.py)
- Optionally validate manual_gold_dataset.csv
- Show dataset size, columns, missing values, career label distribution,
  quality label distribution, and sub-score distributions
- Warn if class distribution or data integrity looks unhealthy

Column contract (matches manual_gold_dataset.csv):
    cv_text           — resume plain text
    career_label      — tech | business | creative | other
    quality_label     — 0 | 1 | 2
    ats_score         — 0 | 1 | 2   (optional, present in gold only)
    writing_score     — 0 | 1 | 2   (optional, present in gold only)
    achievement_score — 0 | 1 | 2   (optional, present in gold only)
    relevance_score   — 0 | 1 | 2   (optional, present in gold only)
    final_score       — 0-100       (optional, present in gold only)
"""

import os
import pandas as pd

BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH      = os.path.join(BASE_DIR, "data", "labeled_resumes.csv")
GOLD_PATH      = os.path.join(BASE_DIR, "data", "manual_gold_dataset.csv")

QUALITY_NAMES  = {0: "weak", 1: "average", 2: "strong"}
VALID_CAREERS  = {"tech", "business", "creative", "other"}
SUB_SCORE_COLS = ["ats_score", "writing_score", "achievement_score", "relevance_score"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def print_distribution(series: pd.Series, title: str) -> None:
    counts      = series.value_counts(dropna=False)
    percentages = (counts / len(series) * 100).round(2)
    print(title)
    for label, count in counts.items():
        pct = percentages[label]
        print(f"  {label}: {count} ({pct}%)")
    print()


def validate_file(path: str, label: str, encoding: str = "utf-8") -> None:
    print(f"=== {label} ===")

    if not os.path.exists(path):
        print(f"ERROR: File not found: {path}\n")
        return

    df = pd.read_csv(path, dtype=str, encoding=encoding, on_bad_lines="skip")
    df.columns = df.columns.str.strip()

    print(f"File:    {os.path.basename(path)}")
    print(f"Rows:    {len(df)}")
    print(f"Columns: {list(df.columns)}\n")

    # ── Required columns check ────────────────────────────────────────────────
    required_cols = ["cv_text", "career_label", "quality_label"]
    print("Required column check:")
    all_present = True
    for col in required_cols:
        if col in df.columns:
            missing = int(df[col].isna().sum())
            print(f"  {col}: {missing} missing")
        else:
            print(f"  {col}: MISSING COLUMN ⚠")
            all_present = False
    print()

    if not all_present:
        print("WARNING: Cannot continue validation — required columns missing.\n")
        return

    # ── Usable rows ───────────────────────────────────────────────────────────
    valid_mask = df["cv_text"].astype(str).str.strip().str.len() > 50
    print(f"Usable resumes (cv_text > 50 chars): {int(valid_mask.sum())}/{len(df)}\n")

    df = df[valid_mask].copy()

    # ── Career label distribution ─────────────────────────────────────────────
    df["career_label"] = df["career_label"].astype(str).str.strip().str.lower()
    print_distribution(df["career_label"], "Career label distribution:")

    unknown_careers = set(df["career_label"].unique()) - VALID_CAREERS
    if unknown_careers:
        print(f"WARNING: Unknown career labels found: {unknown_careers}\n")

    low_careers = df["career_label"].value_counts()
    low_careers = low_careers[low_careers < 10]
    if not low_careers.empty:
        print("WARNING: Some career labels have very few samples (< 10):")
        for lbl, cnt in low_careers.items():
            print(f"  {lbl}: {cnt}")
        print()

    # ── Quality label distribution ────────────────────────────────────────────
    df["quality_label"] = pd.to_numeric(df["quality_label"], errors="coerce")
    bad_quality = df["quality_label"].isna().sum()
    if bad_quality:
        print(f"WARNING: {bad_quality} rows have non-numeric quality_label.\n")
    df = df.dropna(subset=["quality_label"])
    df["quality_label"] = df["quality_label"].astype(int)

    quality_display = df["quality_label"].map(
        lambda x: f"{x} ({QUALITY_NAMES.get(x, '?')})"
    )
    print_distribution(quality_display, "Quality label distribution:")

    grade_counts = df["quality_label"].value_counts()
    low_grades   = grade_counts[grade_counts < 10]
    if not low_grades.empty:
        print("WARNING: Some quality labels have very few samples (< 10):")
        for lbl, cnt in low_grades.items():
            print(f"  grade {lbl} ({QUALITY_NAMES.get(lbl, '?')}): {cnt}")
        print()

    # ── Sub-score distributions (present in gold dataset only) ────────────────
    present_sub = [c for c in SUB_SCORE_COLS if c in df.columns]
    if present_sub:
        print("Sub-score distributions (0=low / 1=mid / 2=high):")
        for col in present_sub:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            counts = df[col].value_counts(dropna=False).sort_index()
            print(f"  {col}: {dict(counts)}")
        print()

    if "final_score" in df.columns:
        df["final_score"] = pd.to_numeric(df["final_score"], errors="coerce")
        print("final_score statistics:")
        print(df["final_score"].describe().round(2).to_string())
        print()

    # ── Text length sanity ────────────────────────────────────────────────────
    df["_text_len"] = df["cv_text"].astype(str).str.split().str.len()
    print("Word count statistics per resume:")
    print(df["_text_len"].describe().round(1).to_string())
    print()

    print(f"=== Validation Complete: {label} ===\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Validate auto-labeled training data
    validate_file(DATA_PATH, "Labeled Resumes (auto-scored)", encoding="utf-8")

    # Validate manual gold dataset if present
    if os.path.exists(GOLD_PATH):
        validate_file(GOLD_PATH, "Manual Gold Dataset", encoding="latin-1")

        # Cross-check: confirm both files share the same column schema
        df_main = pd.read_csv(DATA_PATH, nrows=1, dtype=str, encoding="utf-8")
        df_gold = pd.read_csv(GOLD_PATH, nrows=1, dtype=str, encoding="latin-1")
        df_main.columns = df_main.columns.str.strip()
        df_gold.columns = df_gold.columns.str.strip()

        shared_cols  = set(df_main.columns) & set(df_gold.columns)
        only_in_main = set(df_main.columns) - set(df_gold.columns)
        only_in_gold = set(df_gold.columns) - set(df_main.columns)

        print("=== Schema Comparison ===")
        print(f"Shared columns:         {sorted(shared_cols)}")
        if only_in_main:
            print(f"Only in labeled data:   {sorted(only_in_main)}")
        if only_in_gold:
            print(f"Only in gold dataset:   {sorted(only_in_gold)}")
        print()
    else:
        print(f"INFO: Manual gold dataset not found at: {GOLD_PATH}")
        print("      Skipping gold validation.\n")


if __name__ == "__main__":
    main()