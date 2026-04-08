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
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(BASE_DIR, "data", "slim_resumes.csv")
MODEL_DIR    = os.path.join(BASE_DIR, "models")
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
print("\nClassification report on held-out 20%:")
print(classification_report(y_test, y_pred))

# ── Save ──────────────────────────────────────────────────────────────────────
model_path   = os.path.join(MODEL_DIR, "career_model.pkl")
classes_path = os.path.join(MODEL_DIR, "career_classes.pkl")

with open(model_path, "wb") as f:
    pickle.dump(pipeline, f)

with open(classes_path, "wb") as f:
    pickle.dump(list(pipeline.classes_), f)

print(f"\n✓ Saved: {model_path}")
print(f"✓ Saved: {classes_path}")
print(f"  Classes: {list(pipeline.classes_)}")