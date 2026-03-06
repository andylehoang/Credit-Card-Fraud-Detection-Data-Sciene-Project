# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Credit card fraud detection project using the Kaggle "kartik2112/fraud-detection" simulated dataset. The project classifies transactions as fraudulent or legitimate using a Random Forest classifier (scikit-learn). The notebook and documentation are partly in German.

## Environment Setup

```bash
conda env create -f environment.yml
conda activate fraud-detection
```

Key dependencies: Python 3.12, pandas 2.2+, scikit-learn 1.5+, plotly 5.20+, seaborn 0.13+.

## Running the Notebook

```bash
jupyter notebook notebook.ipynb
```

The notebook uses the `fraud-detection` conda environment kernel (`python3`).

## Data

- `fraudTrain.csv` (~1.3M rows, 351 MB) and `fraudTest.csv` (~556K rows, 150 MB) are local CSV files in the project root
- Target column: `is_fraud` (binary: 0=legitimate, 1=fraud)
- 23 columns including transaction amount (`amt`), merchant `category`, geospatial coordinates (`lat`/`long`, `merch_lat`/`merch_long`), temporal (`trans_date_trans_time`, `unix_time`), and demographic features
- Highly imbalanced dataset — fraud cases are a small minority; use `class_weight="balanced_subsample"` and evaluate with PR-AUC, not accuracy

## Architecture & ML Pipeline

1. **Data loading**: `pd.read_csv()` — the CSV files are already present locally
2. **EDA**: matplotlib, seaborn, plotly for visualization
3. **Preprocessing**: `StandardScaler` for numeric features, `LabelEncoder` for low-cardinality categoricals (`category`, `gender`), frequency encoding for high-cardinality (`merchant`, `job`)
4. **Imbalance handling**: `class_weight="balanced_subsample"` (SMOTE was tested and removed — it decreased PR-AUC)
5. **Modeling**: Single optimized Random Forest with `max_depth=20`, `min_samples_leaf=50`, `max_samples=0.5`
6. **Tuning**: `RandomizedSearchCV` on 20% subsample, retrained on full data
7. **Evaluation**: `classification_report`, `confusion_matrix`, `roc_auc_score`, `average_precision_score`
8. **Model persistence**: `joblib` for saving/loading models

## Current State

The notebook has a complete ML pipeline (~26 cells):
1. **EDA & preprocessing** (Steps 1-6): data profiling, imbalance analysis, drift analysis, fraud patterns, leakage checks, feature engineering with StandardScaler and frequency encoding
2. **Classical ML** (Steps 7-11): Optimized RandomForest baseline with class_weight, threshold calibration (F1-optimal and recall-85% SLA), RandomizedSearchCV hyperparameter tuning on subsample, validation checks, model persistence
