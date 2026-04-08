"""
training/train_scorer.py

Responsibilities:
- Read data/slim_resumes.csv
- Extract hand-crafted features from each resume
- Train XGBoost classifier on features + grade labels
- Save models/scorer_model.pkl

Run once from the project root:
    python training/train_scorer.py
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Add project root to path so we can import modules
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from modules.features import extract_features

DATA_PATH  = os.path.join(BASE_DIR, "data",   "slim_resumes.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "scorer_model.pkl")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH, dtype=str, encoding="utf-8", on_bad_lines="skip")
df.columns = df.columns.str.strip()
df = df.dropna(subset=["Resume_str", "grade"])
df["Resume_str"] = df["Resume_str"].astype(str).str.strip()
df = df[df["Resume_str"].str.len() > 50]
df["grade"] = df["grade"].astype(int)

print(f"  Rows loaded: {len(df)}")
print(f"  Grade distribution:\n{df['grade'].value_counts().sort_index().to_string()}\n")

# ── Extract features ──────────────────────────────────────────────────────────
print("Extracting features...")
feature_rows = []
for text in df["Resume_str"]:
    feature_rows.append(extract_features(text))

X = pd.DataFrame(feature_rows)
y = df["grade"].values

print(f"  Feature matrix shape: {X.shape}")
print(f"  Features: {list(X.columns)}\n")

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ── Train XGBoost ─────────────────────────────────────────────────────────────
print("Training XGBoost scorer...")
# Compute class weights to handle imbalance
from collections import Counter
counter = Counter(y_train)
total = sum(counter.values())
sample_weights = np.array([
    total / (len(counter) * counter[label])
    for label in y_train
])

model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=1,
)
model.fit(X_train, y_train, sample_weight=sample_weights)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
print("\nClassification report on held-out 20%:")
print(classification_report(y_test, y_pred,
      target_names=["weak", "average", "strong"]))

# ── Save model and feature names ──────────────────────────────────────────────
payload = {
    "model":         model,
    "feature_names": list(X.columns),
}
with open(MODEL_PATH, "wb") as f:
    pickle.dump(payload, f)

print(f"✓ Saved: {MODEL_PATH}")
print(f"  Feature count: {len(X.columns)}")