"""
training/train_scorer.py

Responsibilities:
- Read data/slim_resumes_with_strong.csv
- Extract features from each resume
- Train XGBoost classifier on grade labels
- Save models/scorer_model.pkl
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import Counter

# ── Setup paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from modules.features import extract_features

DATA_PATH  = os.path.join(BASE_DIR, "data", "slim_resumes_with_strong.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "scorer_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading dataset...")

df = pd.read_csv(DATA_PATH, dtype=str, encoding="utf-8", on_bad_lines="skip")
df.columns = df.columns.str.strip()

df = df.dropna(subset=["Resume_str", "grade"])
df["Resume_str"] = df["Resume_str"].astype(str).str.strip()
df = df[df["Resume_str"].str.len() > 50]

# Ensure grade is numeric
df["grade"] = df["grade"].astype(int)

print(f"Rows: {len(df)}")
print("\nGrade distribution:")
print(df["grade"].value_counts().sort_index())

# ── Extract features ─────────────────────────────────────────────────────────
print("\nExtracting features...")

feature_rows = []
for text in df["Resume_str"]:
    feature_rows.append(extract_features(text))

X = pd.DataFrame(feature_rows)
y = df["grade"].values

print(f"Feature shape: {X.shape}")

# ── Train/Test split ─────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ── Handle class imbalance ───────────────────────────────────────────────────
counter = Counter(y_train)
total = sum(counter.values())

sample_weights = np.array([
    total / (len(counter) * counter[label])
    for label in y_train
])

# ── Train model ──────────────────────────────────────────────────────────────
print("\nTraining model...")

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

# ── Evaluate ────────────────────────────────────────────────────────────────
print("\nEvaluating model...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print(f"Accuracy: {accuracy:.4f}")
print("==============================\n")

print("Classification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["weak", "average", "strong"]
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ── Save model ──────────────────────────────────────────────────────────────
payload = {
    "model": model,
    "feature_names": list(X.columns),
}

with open(MODEL_PATH, "wb") as f:
    pickle.dump(payload, f)

print(f"\nModel saved to: {MODEL_PATH}")
print(f"Features used: {len(X.columns)}")