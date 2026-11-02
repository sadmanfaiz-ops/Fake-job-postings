# CSE424 Project: Multi-Modal Pattern Recognition for Fake Job Postings

## Overview
This project detects fraudulent job postings using **multi-modal pattern recognition**.
It compares three model families:

1. **Logistic Regression** (`logistic_regression`) with TF-IDF text + categorical + numeric fusion.
2. **Random Forest** (`random_forest`) with reduced text features (TF-IDF + SVD) + numeric features.
3. **BERT** (`bert`) using sentence-transformer embeddings (`all-MiniLM-L6-v2`) + Logistic Regression head.

The system automatically picks the best model by **F1-score** and saves all artifacts.

## Dataset
- File: `fake_job_postings.csv`
- Target label: `fraudulent` (0 = real, 1 = fake)

## Features Used
### Text modality
- `title`, `company_profile`, `description`, `requirements`, `benefits`
- `department`, `function`, `industry`, `location`

### Metadata modality
- Categorical: `employment_type`, `required_experience`, `required_education`, `industry`, `function`, `department`
- Numeric: `telecommuting`, `has_company_logo`, `has_questions`
- Engineered:
  - Salary features parsed from `salary_range`: `salary_lower`, `salary_upper`, `salary_mid`, `salary_available`
  - Length features: `title_len`, `description_len`, `requirements_len`, `benefits_len`, `company_profile_len`, `word_count_total`

## Installation
```bash
pip install -r requirements.txt
```

## Train + Evaluate
```bash
python "CSE424 Project.py" --data "fake_job_postings.csv" --output-dir "outputs"
```

Optional:
```bash
python "CSE424 Project.py" --data "fake_job_postings.csv" --output-dir "outputs" --test-size 0.2 --random-state 42
```

Optional (skip BERT for faster run):
```bash
python "CSE424 Project.py" --data "fake_job_postings.csv" --output-dir "outputs" --skip-bert
```

## Inference (Optional)
Prepare a JSON file containing one job object or a list of job objects, then run:

```bash
python "CSE424 Project.py" --predict-json sample_input.json --model-path outputs/best_model.joblib
```

## Output Artifacts
Inside `outputs/`:
- `model_comparison.csv` → accuracy, precision, recall, f1, roc_auc for all models
- `metrics_summary.json` → full report and per-model confusion matrices
- `best_model.joblib` → best-performing model
- `best_model_info.json` → summary metadata

## Notes
- Class imbalance is handled by `class_weight="balanced"`.
- Evaluation uses stratified train/test split.
- Best model is selected by F1-score, suitable for imbalanced fraud detection.
- BERT requires `sentence-transformers` and downloads the selected pretrained model on first run.
