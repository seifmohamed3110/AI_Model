"""
training/train_scorer.py

Responsibilities:
- Read data/labeled_resumes.csv (output of label_data.py)
  and optionally merge data/manual_gold_dataset.csv for gold supervision
- Extract features from each resume
- Train XGBoost classifier on quality_label  (0=weak / 1=average / 2=strong)
- Save models/scorer_model.pkl

Column contract (matches manual_gold_dataset.csv):
    cv_text           — resume plain text
    career_label      — field bucket (used by feature extractor)
    quality_label     — target: 0 | 1 | 2
    ats_score         — optional gold sub-score (0-2)
    writing_score     — optional gold sub-score (0-2)
    achievement_score — optional gold sub-score (0-2)
    relevance_score   — optional gold sub-score (0-2)
    final_score       — optional gold composite (0-100)
"""

import os
import sys
import pickle
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ── Setup paths ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from modules.features import extract_features   # noqa: E402  (path inserted above)

DATA_PATH  = os.path.join(BASE_DIR, "data", "labeled_resumes.csv")
GOLD_PATH  = os.path.join(BASE_DIR, "data", "manual_gold_dataset.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "scorer_model.pkl")
RESULTS_PATH = os.path.join(BASE_DIR, "SCORER_EVALUATION_RESULTS.md")

os.makedirs(MODEL_DIR, exist_ok=True)


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading dataset...")

df = pd.read_csv(DATA_PATH, dtype=str, encoding="utf-8", on_bad_lines="skip")
df.columns = df.columns.str.strip()

# Merge gold-label data (human-reviewed, higher signal)
if os.path.exists(GOLD_PATH):
    print(f"  Merging gold labels from: {os.path.basename(GOLD_PATH)}")
    gold = pd.read_csv(GOLD_PATH, dtype=str, encoding="latin-1", on_bad_lines="skip")
    gold.columns = gold.columns.str.strip()
    df = pd.concat([df, gold], ignore_index=True)
    print(f"  Total rows after merge: {len(df)}")

# ── Clean ─────────────────────────────────────────────────────────────────────
df = df.dropna(subset=["cv_text", "quality_label"])
df["cv_text"] = df["cv_text"].astype(str).str.strip()
df = df[df["cv_text"].str.len() > 50]

df["quality_label"] = pd.to_numeric(df["quality_label"], errors="coerce")
df = df.dropna(subset=["quality_label"])
df["quality_label"] = df["quality_label"].astype(int)

# career_label is used by extract_features for field-specific signals
if "career_label" not in df.columns:
    df["career_label"] = "other"
df["career_label"] = df["career_label"].fillna("other").astype(str).str.strip().str.lower()

print(f"Rows: {len(df)}")
print("\nQuality label distribution:")
print(df["quality_label"].value_counts().sort_index().rename(
    {0: "weak (0)", 1: "average (1)", 2: "strong (2)"}
).to_string())

# ── Extract features ──────────────────────────────────────────────────────────
print("\nExtracting features...")

feature_rows = [
    extract_features(row["cv_text"], career_label=row["career_label"])
    for _, row in df.iterrows()
]

X = pd.DataFrame(feature_rows)
y = df["quality_label"].values

print(f"Feature shape: {X.shape}")

# ── Train/Test split ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# ── Handle class imbalance ────────────────────────────────────────────────────
counter = Counter(y_train)
total   = sum(counter.values())

sample_weights = np.array([
    total / (len(counter) * counter[label])
    for label in y_train
])

# ── Train model ───────────────────────────────────────────────────────────────
print("\nTraining scorer model...")

model = XGBClassifier(
    n_estimators=250,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=1,
)

model.fit(X_train, y_train, sample_weight=sample_weights)

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\nEvaluating scorer model...")

y_pred = model.predict(X_test)

accuracy        = accuracy_score(y_test, y_pred)
macro_f1        = f1_score(y_test, y_pred, average="macro")
macro_precision = precision_score(y_test, y_pred, average="macro")
macro_recall    = recall_score(y_test, y_pred, average="macro")

class_names = ["weak", "average", "strong"]
labels      = [0, 1, 2]

per_class_f1        = f1_score(y_test, y_pred, average=None, labels=labels)
per_class_precision = precision_score(y_test, y_pred, average=None, labels=labels)
per_class_recall    = recall_score(y_test, y_pred, average=None, labels=labels)

print("\n==============================")
print(f"Accuracy:        {accuracy:.4f}")
print(f"Macro F1:        {macro_f1:.4f}")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall:    {macro_recall:.4f}")
print("==============================\n")

print("Per-class metrics:")
for i, name in enumerate(class_names):
    print(f"  {name:<10}  precision={per_class_precision[i]:.3f}  "
          f"recall={per_class_recall[i]:.3f}  f1={per_class_f1[i]:.3f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ── Save evaluation report ────────────────────────────────────────────────────
with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    f.write("# Scorer Model Evaluation\n\n")
    f.write(f"Accuracy:        {accuracy:.4f}\n")
    f.write(f"Macro F1:        {macro_f1:.4f}\n")
    f.write(f"Macro Precision: {macro_precision:.4f}\n")
    f.write(f"Macro Recall:    {macro_recall:.4f}\n\n")
    f.write("## Per-class metrics\n")
    for i, name in enumerate(class_names):
        f.write(f"  {name:<10}  precision={per_class_precision[i]:.3f}  "
                f"recall={per_class_recall[i]:.3f}  f1={per_class_f1[i]:.3f}\n")
    f.write("\n## Classification Report\n")
    f.write(classification_report(y_test, y_pred, target_names=class_names))
    f.write("\n## Confusion Matrix\n")
    f.write(str(confusion_matrix(y_test, y_pred)))

print(f"\nSaved evaluation results: {RESULTS_PATH}")

# ── Save model ────────────────────────────────────────────────────────────────
payload = {
    "model":         model,
    "feature_names": list(X.columns),
}

with open(MODEL_PATH, "wb") as f:
    pickle.dump(payload, f)

print(f"\nModel saved to: {MODEL_PATH}")
print(f"Features used:  {len(X.columns)}")