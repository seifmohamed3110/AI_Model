# Evaluation Results — Career Field Classifier
**Generated:** 2026-04-21 19:54:23
**Dataset:** slim_resumes_with_strong.csv (3413 resumes)
**Algorithm:** TF-IDF + Logistic Regression
**Train/Test Split:** 80/20 (stratified)
**Random Seed:** 42

## Overall Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | **0.8448** (84.48%) |
| Total Test Samples | 683 |

## Per-Class Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| business | 0.8140 | 0.9170 | 0.8624 | 229 |
| creative | 0.8193 | 0.6126 | 0.7010 | 111 |
| marketing | 1.0000 | 1.0000 | 1.0000 | 56 |
| other | 0.8168 | 0.8250 | 0.8209 | 200 |
| tech | 0.9286 | 0.8966 | 0.9123 | 87 |

## Confusion Matrix

```
Predicted →
[[210   8   0   8   3]
 [ 21  68   0  22   0]
 [  0   0  56   0   0]
 [ 25   7   0 165   3]
 [  2   0   0   7  78]]
```

## Sample Predictions

| Actual | Predicted | Resume Preview |
|--------|-----------|----------------|
| creative | creative | GRAPHIC DESIGNER       Summary     Driven Graphic Artist adept at managing heavy... |
| tech | tech | ENGINEERING MANAGER         Core Qualifications          Executive Decision Make... |
| business | business | FINANCE MANAGER       Summary    Outgoing Sales Manager offering superb customer... |
| other | other | QUALITY ASSURANCE ADVOCATE           Summary    I have recently completed five y... |
| other | other | WS         BARTENDER (ON CALL)       Summary     Hardworking and reliable Fitnes... |
| business | business | MANAGER, INDUSTRY ANALYST RELATIONS       Summary     Creative communications pr... |
| other | business | DIRECTOR OF NATIONAL SALES- US. HEALTHCARE           Executive Profile     SALES... |
| other | other | HISTORY TEACHER         Experience      History Teacher  ,     08/2006   to   Cu... |
| other | other | RESEARCH MOLECULAR/RESEARCH  MICROBIOLOGIST/RESEARCH ECOLOGIST (RESEARCH ASSOCIA... |
| business | business | CONSULTANT       Summary     As a proud Microsoft employee, I'm driven by Custom... |

## Interpretation

✅ **Model meets target accuracy (≥75%):** 84.48%

---

# Evaluation Results — Quality Scorer

**Generated:** 2026-04-21 19:55:04
**Dataset:** slim_resumes_with_strong.csv (3413 resumes)
**Algorithm:** XGBoost Classifier
**Features:** 27 hand-crafted features
**Train/Test Split:** 80/20 (stratified)
**Random Seed:** 42

## Overall Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | **0.6984** (69.84%) |
| Total Test Samples | 683 |

## Per-Class Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| weak | 0.3436 | 0.4912 | 0.4043 | 114 |
| average | 0.7225 | 0.7123 | 0.7174 | 351 |
| strong | 0.9828 | 0.7844 | 0.8724 | 218 |

## Confusion Matrix

```
Predicted →
[[ 56  58   0]
 [ 98 250   3]
 [  9  38 171]]
```

## Sample Predictions

| Actual | Predicted | Feature Snapshot |
|--------|-----------|------------------|
| average | average | words=70, metrics=0, bullets=6 |
| average | average | words=108, metrics=3, bullets=6 |
| average | weak | words=60, metrics=0, bullets=0 |
| weak | average | words=67, metrics=0, bullets=0 |
| average | weak | words=74, metrics=0, bullets=0 |
| weak | weak | words=57, metrics=0, bullets=0 |
| average | average | words=65, metrics=0, bullets=0 |
| strong | strong | words=269, metrics=8, bullets=0 |
| average | average | words=61, metrics=0, bullets=0 |
| average | average | words=52, metrics=0, bullets=0 |

## Interpretation

⚠️ **Model below target accuracy (≥75%):** 69.84%
