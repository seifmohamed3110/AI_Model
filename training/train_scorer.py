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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

# Add project root to path so we can import modules
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from modules.features import extract_features

DATA_PATH  = os.path.join(BASE_DIR, "data", "slim_resumes_with_strong.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "scorer_model.pkl")
RESULTS_PATH = os.path.join(BASE_DIR, "EVALUATION_RESULTS.md")
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

# Overall accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n{'='*60}")
print(f"OVERALL ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"{'='*60}")

# Precision, Recall, F1 per class
precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
grade_labels = ["weak", "average", "strong"]

print("\nPer-Class Metrics:")
print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
print("-" * 55)
for i, cls in enumerate(grade_labels):
    print(f"{cls:<15} {precision[i]:>10.4f} {recall[i]:>10.4f} {f1[i]:>10.4f} {support[i]:>10}")

# Weighted averages
print("\nWeighted Averages:")
print(f"  Precision: {np.average(precision, weights=support):.4f}")
print(f"  Recall:    {np.average(recall, weights=support):.4f}")
print(f"  F1-Score:  {np.average(f1, weights=support):.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:\n{cm}")

# Sample predictions
print("\nSample Predictions (first 10):")
print(f"{'Actual':<15} {'Predicted':<15} {'Feature Preview':<50}")
print("-" * 80)
for i in range(min(10, len(y_test))):
    actual = grade_labels[y_test[i]]
    predicted = grade_labels[y_pred[i]]
    preview = f"features={X_test.iloc[i].shape[0]}"
    print(f"{actual:<15} {predicted:<15} {preview:<50}")

# Append to evaluation results
eval_results = []
eval_results.append("\n---\n\n")
eval_results.append("# Evaluation Results — Quality Scorer\n\n")
eval_results.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
eval_results.append(f"**Dataset:** slim_resumes_with_strong.csv ({len(df)} resumes)\n")
eval_results.append(f"**Algorithm:** XGBoost Classifier\n")
eval_results.append(f"**Features:** {X.shape[1]} hand-crafted features\n")
eval_results.append(f"**Train/Test Split:** 80/20 (stratified)\n")
eval_results.append(f"**Random Seed:** 42\n\n")

eval_results.append("## Overall Metrics\n\n")
eval_results.append(f"| Metric | Value |\n")
eval_results.append(f"|--------|-------|\n")
eval_results.append(f"| **Accuracy** | **{accuracy:.4f}** ({accuracy*100:.2f}%) |\n")
eval_results.append(f"| Total Test Samples | {len(y_test)} |\n\n")

eval_results.append("## Per-Class Metrics\n\n")
eval_results.append("| Class | Precision | Recall | F1-Score | Support |\n")
eval_results.append("|-------|-----------|--------|----------|--------|\n")
for i, cls in enumerate(grade_labels):
    eval_results.append(f"| {cls} | {precision[i]:.4f} | {recall[i]:.4f} | {f1[i]:.4f} | {support[i]} |\n")

eval_results.append("\n## Confusion Matrix\n\n")
eval_results.append("```\n")
eval_results.append(f"Predicted →\n{cm}\n")
eval_results.append("```\n\n")

eval_results.append("## Sample Predictions\n\n")
eval_results.append("| Actual | Predicted | Feature Snapshot |\n")
eval_results.append("|--------|-----------|------------------|\n")
for i in range(min(10, len(y_test))):
    actual = grade_labels[y_test[i]]
    predicted = grade_labels[y_pred[i]]
    wc = int(X_test.iloc[i].get("word_count", 0))
    qc = int(X_test.iloc[i].get("quantification_count", 0))
    bc = int(X_test.iloc[i].get("bullet_count", 0))
    eval_results.append(f"| {actual} | {predicted} | words={wc}, metrics={qc}, bullets={bc} |\n")
eval_results.append("\n")

eval_results.append("## Interpretation\n\n")
if accuracy >= 0.75:
    eval_results.append(f"✅ **Model meets target accuracy (≥75%):** {accuracy:.2%}\n")
else:
    eval_results.append(f"⚠️ **Model below target accuracy (≥75%):** {accuracy:.2%}\n")

# Append to existing file
if os.path.exists(RESULTS_PATH):
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        existing = f.read()
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(existing + "".join(eval_results))
else:
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("".join(eval_results))
print(f"\n[OK] Updated evaluation results: {RESULTS_PATH}")

# ── Save model and feature names ──────────────────────────────────────────────
payload = {
    "model":         model,
    "feature_names": list(X.columns),
}
with open(MODEL_PATH, "wb") as f:
    pickle.dump(payload, f)

print(f"[OK] Saved: {MODEL_PATH}")
print(f"  Feature count: {len(X.columns)}")