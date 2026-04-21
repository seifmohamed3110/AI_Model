"""
training/train_career.py

Responsibilities:
- Read data/slim_resumes.csv
- Train TF-IDF + Logistic Regression career field classifier
- Save models/career_model.pkl
- Save models/career_classes.pkl

Run once from the project root:
    python training/train_career.py
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(BASE_DIR, "data", "slim_resumes_with_strong.csv")
MODEL_DIR    = os.path.join(BASE_DIR, "models")
RESULTS_PATH = os.path.join(BASE_DIR, "EVALUATION_RESULTS.md")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH, dtype=str, encoding="utf-8", on_bad_lines="skip")
df.columns = df.columns.str.strip()
df = df.dropna(subset=["field", "Resume_str"])
df["Resume_str"] = df["Resume_str"].astype(str).str.strip()
df = df[df["Resume_str"].str.len() > 50]

print(f"  Rows after cleaning: {len(df)}")
print(f"  Field distribution:\n{df['field'].value_counts().to_string()}\n")

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df["Resume_str"], df["field"],
    test_size=0.20,
    random_state=42,
    stratify=df["field"]
)

# ── Build pipeline ────────────────────────────────────────────────────────────
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=5_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        strip_accents="unicode",
        analyzer="word",
    )),
    ("clf", LogisticRegression(
        max_iter=500,
        C=5.0,
        solver="lbfgs",
        random_state=42,
    )),
])

# ── Train ─────────────────────────────────────────────────────────────────────
print("Training career classifier...")
pipeline.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = pipeline.predict(X_test)

# Overall accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n{'='*60}")
print(f"OVERALL ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"{'='*60}")

# Precision, Recall, F1 per class
precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
classes = pipeline.classes_

print("\nPer-Class Metrics:")
print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
print("-" * 55)
for i, cls in enumerate(classes):
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
print(f"{'Actual':<15} {'Predicted':<15} {'Text Preview':<50}")
print("-" * 80)
for i in range(min(10, len(y_test))):
    actual = y_test.iloc[i]
    predicted = y_pred[i]
    preview = X_test.iloc[i][:50].replace('\n', ' ')
    print(f"{actual:<15} {predicted:<15} {preview:<50}")

# Save evaluation results
eval_results = []
eval_results.append("# Evaluation Results — Career Field Classifier\n")
eval_results.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
eval_results.append(f"**Dataset:** slim_resumes_with_strong.csv ({len(df)} resumes)\n")
eval_results.append(f"**Algorithm:** TF-IDF + Logistic Regression\n")
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
for i, cls in enumerate(classes):
    eval_results.append(f"| {cls} | {precision[i]:.4f} | {recall[i]:.4f} | {f1[i]:.4f} | {support[i]} |\n")

eval_results.append("\n## Confusion Matrix\n\n")
eval_results.append("```\n")
eval_results.append(f"Predicted →\n{cm}\n")
eval_results.append("```\n\n")

eval_results.append("## Sample Predictions\n\n")
eval_results.append("| Actual | Predicted | Resume Preview |\n")
eval_results.append("|--------|-----------|----------------|\n")
for i in range(min(10, len(y_test))):
    actual = y_test.iloc[i]
    predicted = y_pred[i]
    preview = X_test.iloc[i][:80].replace("\n", " ").replace("|", " ")
    eval_results.append(f"| {actual} | {predicted} | {preview}... |\n")
eval_results.append("\n")

eval_results.append("## Interpretation\n\n")
if accuracy >= 0.75:
    eval_results.append(f"✅ **Model meets target accuracy (≥75%):** {accuracy:.2%}\n")
else:
    eval_results.append(f"⚠️ **Model below target accuracy (≥75%):** {accuracy:.2%}\n")

# Save to file
with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    f.write("".join(eval_results))
print(f"\n[OK] Saved evaluation results: {RESULTS_PATH}")

# ── Save ──────────────────────────────────────────────────────────────────────
model_path   = os.path.join(MODEL_DIR, "career_model.pkl")
classes_path = os.path.join(MODEL_DIR, "career_classes.pkl")

with open(model_path, "wb") as f:
    pickle.dump(pipeline, f)

with open(classes_path, "wb") as f:
    pickle.dump(list(pipeline.classes_), f)

print(f"\n[OK] Saved: {model_path}")
print(f"[OK] Saved: {classes_path}")
print(f"  Classes: {list(pipeline.classes_)}")