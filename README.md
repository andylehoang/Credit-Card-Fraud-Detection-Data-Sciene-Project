# Credit-Card-Fraud-Detection-Data-Sciene-Project
Projekt fuer die Klassifizierung von Kreditkartenbetrug aus dem bereitgestellten Datensatz.

## Summarized Code Review by Logical Block (`notebook.ipynb`)

1. **Environment and dependency setup**  
What it does: Loads all required libraries, silences warnings, and prints versions.  
Review: Good reproducibility signal and clear library grouping.  
Risk/Improvement: `warnings.filterwarnings('ignore')` can hide important issues during development.

2. **Raw data loading and initial validation**  
What it does: Loads train/test CSV files and reports shape, fraud rate, columns, and nulls.  
Review: Strong first checkpoint for schema and class-imbalance visibility.  
Risk/Improvement: Paths are hardcoded; add config/CLI/env-based paths for portability.

3. **Geospatial helper (`haversine_km`)**  
What it does: Computes vectorized transaction distance between cardholder and merchant.  
Review: Efficient, correct feature for fraud context.  
Risk/Improvement: No explicit guard for invalid/missing coordinates.

4. **Feature engineering pipeline (`engineer_features`)**  
What it does: Adds time features, age, distance, and drops raw identity/location fields.  
Review: Clean feature construction and good copy-based immutability pattern.  
Risk/Improvement: Fixed reference date for age can become stale if dataset period changes.

5. **Feature engineering execution and schema check**  
What it does: Applies engineering to train/test and prints resulting schema details.  
Review: Good consistency check after transformation.  
Risk/Improvement: Could add strict train-vs-test column equality assertion.

6. **Categorical encoding (train fit, test fallback)**  
What it does: Fits `LabelEncoder` on train categories and maps unseen test categories to `-1`.  
Review: Correct anti-leakage approach and practical handling of unseen labels.  
Risk/Improvement: Integer label encoding can imply false ordinal meaning for tree-linear hybrids.

7. **Feature/target split**  
What it does: Defines feature columns and separates `X`/`y` for both splits.  
Review: Clear and standard separation step.  
Risk/Improvement: Add assertion that target exists and is binary before split.

8. **Scaling and NumPy export**  
What it does: Fits `StandardScaler` on train only, transforms both sets, converts to NumPy arrays.  
Review: Correct preprocessing order for leakage prevention.  
Risk/Improvement: Not all planned models need scaling; consider model-specific preprocessing pipelines.

9. **Final dataset audit and retention note**  
What it does: Prints final array stats and keeps unscaled DataFrames for EDA.  
Review: Useful final sanity check and practical dual-format retention.  
Risk/Improvement: Add saved artifacts (scaler/encoders) and a small validation report for reproducible inference.
