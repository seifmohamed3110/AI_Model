# Dataset Documentation

## Overview

This folder contains the resume datasets used for training the CV Analyzer's machine learning models.

## Files

| File | Description | Rows |
|------|-------------|------|
| `slim_resumes_with_strong.csv` | **Main training dataset** - balanced with strong resumes | 12,810 |
| `slim_resumes.csv` | Original dataset | 6,416 |
| `labeled_resumes.csv` | Intermediate labeled data | - |
| `raw_resumes.csv` | Raw input data (pre-processing) | - |
| `manual_review_checklist.csv` | Phase-1 manual quality review tracker (20 target) | 20 |

## Dataset Schema: `slim_resumes_with_strong.csv`

### Columns

| Column | Type | Description |
|--------|------|-------------|
| `Resume_str` | Text | Raw resume text content |
| `Category` | Text | Job category/field (HR, Engineering, etc.) |
| `field` | Text | Broad career field: `tech`, `marketing`, `creative`, `business`, `other` |
| `score` | Float | Numerical quality score (0-100 scale) |
| `grade` | Integer | Quality label: `0`=weak, `1`=average, `2`=strong |

### Label Rubric

#### Career Field (`field`)
- **tech**: Software developers, engineers, data scientists, IT professionals
- **marketing**: Marketing specialists, digital marketers, brand managers
- **creative**: Designers, artists, video editors, UI/UX professionals
- **business**: HR, finance, operations, product management, administration
- **other**: Roles that don't fit the above categories

#### Quality Grade (`grade`)
| Grade | Label | Criteria |
|-------|-------|----------|
| 0 | weak | Missing core sections, no quantified achievements, generic language, formatting issues |
| 1 | average | Has core sections, some metrics, decent formatting, standard language |
| 2 | strong | Complete sections, quantified achievements, clean formatting, strong action verbs |

### Current Distribution

**Grade Distribution:**
| Grade | Count | Percentage |
|-------|-------|------------|
| 0 (weak) | 568 | 4.4% |
| 1 (average) | 1,754 | 13.7% |
| 2 (strong) | 1,091 | 8.5% |

**Field Distribution:**
| Field | Count |
|-------|-------|
| business | 1,144 |
| other | 998 |
| creative | 553 |
| tech | 438 |
| marketing | 280 |

## Data Processing Workflow

```
raw_resumes.csv
    ↓
label_data.py (keyword-based pre-labeling)
    ↓
labeled_resumes.csv
    ↓
manual review + filtering
    ↓
slim_resumes.csv → slim_resumes_with_strong.csv (final)
```

## Usage

### For Training Career Classifier
```python
import pandas as pd
df = pd.read_csv('data/slim_resumes_with_strong.csv')
X = df['Resume_str']  # Input text
y = df['field']       # Target: career field
```

### For Training Quality Scorer
```python
import pandas as pd
df = pd.read_csv('data/slim_resumes_with_strong.csv')
X = extract_features(df['Resume_str'])  # Custom features
y = df['grade']      # Target: 0=weak, 1=average, 2=strong
```

## Notes

- **Scoring Method**: Current grades are generated using rule-based scoring (keyword density, section presence, formatting heuristics)
- **Limitation**: Rule-based labels may not capture nuanced human judgment of resume quality
- **Manual Review Workflow**: Fill `manual_review_checklist.csv` with reviewer-approved labels (`status=done`) for at least 20 resumes
