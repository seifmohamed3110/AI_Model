"""
training/train_career.py

Responsibilities:
- Read data/slim_resumes_with_strong.csv
- Train TF-IDF + Logistic Regression career field classifier
- Save models/career_model.pkl
- Save models/career_classes.pkl
"""

import os
import pickle
import pandas as pd
import numpy as np
import re

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(BASE_DIR, "data", "slim_resumes_with_strong.csv")
MODEL_DIR    = os.path.join(BASE_DIR, "models")
RESULTS_PATH = os.path.join(BASE_DIR, "EVALUATION_RESULTS.md")

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Simple text cleaning ─────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)  # remove URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # remove symbols
    text = re.sub(r"\s+", " ", text)  # normalize spaces
    return text.strip()

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data...")

df = pd.read_csv(DATA_PATH, dtype=str, encoding="utf-8", on_bad_lines="skip")
df.columns = df.columns.str.strip()

df = df.dropna(subset=["field", "Resume_str"])
df["Resume_str"] = df["Resume_str"].astype(str).str.strip()
df = df[df["Resume_str"].str.len() > 50]

# Clean text
df["clean_text"] = df["Resume_str"].apply(clean_text)

print(f"Rows after cleaning: {len(df)}")
print("\nField distribution:")
print(df["field"].value_counts().to_string())

# ── Train/Test split ─────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["field"],
    test_size=0.20,
    random_state=42,
    stratify=df["field"]
)

# ── Build pipeline ───────────────────────────────────────────────────────────
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=7000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=3,
        max_df=0.9,
        stop_words="english",
    )),
    ("clf", LogisticRegression(
        max_iter=600,
        C=4.0,
        solver="lbfgs",
        random_state=42,
    )),
])

# ── Train ─────────────────────────────────────────────────────────────────────
print("\nTraining career classifier...")
pipeline.fit(X_train, y_train)

# ── Evaluate ─────────────────────────────────────────────────────────────────
y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print(f"Accuracy: {accuracy:.4f}")
print("==============================\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ── Save evaluation report ───────────────────────────────────────────────────
with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    f.write("Career Model Evaluation\n")
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred))
    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred)))

print(f"\nSaved evaluation results: {RESULTS_PATH}")

# ── Save model ───────────────────────────────────────────────────────────────
model_path   = os.path.join(MODEL_DIR, "career_model.pkl")
classes_path = os.path.join(MODEL_DIR, "career_classes.pkl")

with open(model_path, "wb") as f:
    pickle.dump(pipeline, f)

with open(classes_path, "wb") as f:
    pickle.dump(list(pipeline.classes_), f)

print(f"\nSaved: {model_path}")
print(f"Saved: {classes_path}")
print(f"Classes: {list(pipeline.classes_)}")