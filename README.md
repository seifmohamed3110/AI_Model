# CV Analyzer - Professional Resume Analysis System

An intelligent Flask-based web application that analyzes resumes using machine learning to provide personalized career field detection, quality scoring, and actionable improvement suggestions.

## Features

- **Career Field Detection**: Automatically classifies resumes into tech, marketing, creative, business, or other fields
- **Quality Scoring**: Evaluates resume strength (weak/average/strong) using trained ML models
- **ATS Analysis**: Checks for Applicant Tracking System compatibility issues
- **Writing Analysis**: Identifies grammar, style, and formatting improvements
- **Personalized Suggestions**: Provides field-specific, ranked recommendations

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Models (First Time Only)

```bash
# Train career field classifier
python training/train_career.py

# Train quality scorer
python training/train_scorer.py
```

### 3. Run Flask App

```bash
python app.py
```

Open http://localhost:5000 in your browser.

## Project Structure

```
resume_analyzer/
├── app.py                      # Flask application + API endpoints
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── data/                       # Datasets
│   ├── slim_resumes_with_strong.csv   # Main training data (12,810 resumes)
│   ├── slim_resumes.csv               # Original dataset
│   ├── labeled_resumes.csv            # Intermediate labeled data
│   ├── raw_resumes.csv                # Raw input data
│   └── README.md                      # Dataset documentation
│
├── models/                     # Trained ML models
│   ├── career_model.pkl        # Career field classifier
│   ├── career_classes.pkl      # Career class labels
│   └── scorer_model.pkl        # Quality scorer
│
├── modules/                    # Core logic modules
│   ├── career.py               # Career field detection
│   ├── scorer.py               # Quality scoring
│   ├── features.py             # Feature extraction
│   ├── extractor.py            # PDF/DOCX text extraction
│   ├── ats.py                  # ATS compatibility checks
│   ├── writing.py              # Writing quality analysis
│   └── suggestions.py          # Suggestion generation
│
├── training/                   # Training scripts
│   ├── label_data.py           # Data labeling utilities
│   ├── train_career.py         # Train career classifier
│   └── train_scorer.py         # Train quality scorer
│
└── templates/                  # HTML templates
    └── index.html              # Frontend UI
```

## Training Workflow

### Step 1: Prepare Data (Optional)

If you need to re-label data:

```bash
python training/label_data.py
python training/validate_data.py
```

### Step 2: Train Career Classifier

```bash
python training/train_career.py
```

**Input:** `data/slim_resumes_with_strong.csv`  
**Output:** `models/career_model.pkl`, `models/career_classes.pkl`  
**Algorithm:** TF-IDF + Logistic Regression  
**Classes:** tech, marketing, creative, business, other

### Step 3: Train Quality Scorer

```bash
python training/train_scorer.py
```

**Input:** `data/slim_resumes_with_strong.csv`  
**Output:** `models/scorer_model.pkl`  
**Algorithm:** XGBoost Classifier  
**Classes:** weak (0), average (1), strong (2)

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Train/Test Split | 80/20 |
| Random Seed | 42 |
| Class Weighting | Enabled (for scorer) |

## API Reference

### `POST /analyze`

Upload a resume for analysis.

**Request:**
- `file`: PDF or DOCX file

**Response:**
```json
{
  "score": 83.7,
  "grade": "strong",
  "detected_field": "tech",
  "critical_issues": [],
  "quick_wins": [],
  "field_specific_improvements": [],
  "ats_issues": [],
  "writing_issues": [],
  "improvements": [],
  "summary": "..."
}
```

### `GET /`

Returns the main HTML page.

### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true
}
```

## Module Details

### modules/extractor.py
Extracts text from PDF (using PyMuPDF) and DOCX (using python-docx) files.

### modules/features.py
Extracts hand-crafted features:
- Section presence (education, experience, skills, etc.)
- Contact information completeness
- Quantification metrics (numbers, percentages)
- Action verb usage
- Length metrics

### modules/career.py
Detects career field using:
1. Keyword matching (fallback)
2. Trained ML model (primary)

### modules/scorer.py
Scores resume quality using trained XGBoost model on extracted features.

### modules/ats.py
Checks ATS compatibility:
- Missing sections
- Formatting issues
- Keyword optimization

### modules/writing.py
Analyzes writing quality:
- Passive voice detection
- Weak verb identification
- Bullet point structure

### modules/suggestions.py
Generates ranked, field-specific improvement suggestions.

## Evaluation Results

See `EVALUATION_RESULTS.md` for detailed model performance metrics.

## Requirements

- Python 3.8+
- See `requirements.txt` for full dependency list

## License

This project is part of a graduation project at SAMS 4th Year.

## Contact

For questions or issues, please contact the development team.
