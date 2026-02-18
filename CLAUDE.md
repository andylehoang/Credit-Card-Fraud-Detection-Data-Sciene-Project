# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Credit card fraud detection project using the Kaggle "kartik2112/fraud-detection" simulated dataset. The project classifies transactions as fraudulent or legitimate using both classical ML (scikit-learn) and deep learning (TensorFlow/Keras). The notebook and documentation are partly in German.

## Environment Setup

```bash
conda env create -f environment.yml
conda activate fraud-detection
```

Key dependencies: Python 3.12, pandas 2.2, scikit-learn 1.6, imbalanced-learn 0.12, TensorFlow 2.20.0 (pip-installed), plotly 5.24, seaborn 0.13.

## Running the Notebook

```bash
jupyter notebook notebook.ipynb
```

The notebook uses the `fraud-detection` conda environment kernel (`python3`).

## Data

- `fraudTrain.csv` (~1.3M rows, 351 MB) and `fraudTest.csv` (~556K rows, 150 MB) are local CSV files in the project root
- Target column: `is_fraud` (binary: 0=legitimate, 1=fraud)
- 23 columns including transaction amount (`amt`), merchant `category`, geospatial coordinates (`lat`/`long`, `merch_lat`/`merch_long`), temporal (`trans_date_trans_time`, `unix_time`), and demographic features
- Highly imbalanced dataset — fraud cases are a small minority; use SMOTE and evaluate with ROC-AUC, not accuracy

## Architecture & ML Pipeline

The intended pipeline (from imports in notebook.ipynb):

1. **Data loading**: `pd.read_csv()` — the CSV files are already present locally
2. **EDA**: matplotlib, seaborn, plotly for visualization
3. **Preprocessing**: `StandardScaler` for feature scaling, encoding for categoricals (`category`, `gender`, `merchant`, `job`)
4. **Imbalance handling**: `SMOTE` from imbalanced-learn
5. **Modeling**: scikit-learn classifiers + TensorFlow/Keras neural network
6. **Evaluation**: `classification_report`, `confusion_matrix`, `roc_auc_score`
7. **Model persistence**: `joblib` for saving/loading models

## Current State

The project is in early scaffolding stage. Only 2 notebook cells exist: imports (working) and a broken Kaggle download cell (empty `file_path`). No preprocessing, modeling, or evaluation code has been written yet.
