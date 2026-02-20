# Credit-Card-Fraud-Detection-Data-Sciene-Project
Projekt fuer die Klassifizierung von Kreditkartenbetrug aus dem bereitgestellten Datensatz.

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

- `Data/fraudTrain.csv` (~1.3M rows, 351 MB) and `Data/fraudTest.csv` (~556K rows, 150 MB) are local CSV files in `Data/`
- Target column: `is_fraud` (binary: 0=legitimate, 1=fraud)
- 23 columns including transaction amount (`amt`), merchant `category`, geospatial coordinates (`lat`/`long`, `merch_lat`/`merch_long`), temporal (`trans_date_trans_time`, `unix_time`), and demographic features
- Highly imbalanced: fraud cases are ~0.5% of all transactions

---

## Notebook Steps (Implemented)

### Step 1 — Data Profiling
Loads train/test CSVs and runs structural health checks: shape, schema equality, null counts, fraud rate per split, and temporal holdout validation (all train timestamps must precede all test timestamps).

### Step 2 — Class Imbalance Analysis
Quantifies fraud rarity: counts, rates, and imbalance ratio per split. Documents the modeling implication: accuracy is not a useful metric; use precision-recall metrics instead.

### Step 3 — Drift Analysis (Train vs Test)
Detects distribution changes between training and deployment periods using PSI (Population Stability Index) for numeric features and share-shift tables for categoricals.

| PSI | Risk |
|---|---|
| < 0.10 | Low |
| 0.10 – 0.25 | Medium |
| ≥ 0.25 | High |

### Step 4 — Fraud Pattern Mining
Computes fraud lift tables by `category`, `hour`, `amt_band`, and `category × hour` interaction hotspots. Lift > 1 means a group is riskier than the global baseline. High-lift + high-support groups become candidate features.

### Step 5 — Leakage and Proxy-Risk Analysis

**What is leakage?**
Leakage means a feature contains information that would not be available at the moment of making a real prediction. For example, if a field only has a value *after* fraud is confirmed, using it in training produces inflated accuracy that disappears in production.

Each column is classified with a risk level and a recommended action:

| Risk | Action | Meaning | Example columns |
|---|---|---|---|
| High | `drop_now` | Direct identifier — unique per transaction, useless as signal, only causes leakage | `cc_num`, `trans_num`, `first`, `last`, `street` |
| High | `engineer_then_drop_raw` | Raw form encodes sequence/time artifacts; extract safe signals first | `trans_date_trans_time`, `unix_time` |
| Medium | `keep_with_monitor` | Personally-linked or location proxies; allowed but track for bias/drift | `dob`, `zip`, `city`, `state`, `job` |
| Low | `keep` | No major leakage or proxy risk | `amt`, `category`, `merchant` |

**Output of this step:** a `risk_df` DataFrame with one row per feature listing `risk_level`, `reason`, and `recommendation`. Also prints identifier overlap counts between train and test (a zero overlap on `trans_num` confirms no data bleed).

---

### Step 6 — Feature Strategy Decision

This step translates the risk assessment from Step 5 into a **concrete, named policy** that all downstream cells share. Rather than each cell making its own drop/keep decisions, one authoritative set of constants is defined here.

**Key variables defined (used by Steps 7 and 8):**

| Variable | Type | Purpose |
|---|---|---|
| `DROP_NOW_COLS` | list | Columns to remove immediately, never used |
| `DERIVED_FEATURE_MAP` | dict | Maps each raw column to the new features it generates |
| `KEEP_RAW_COLS` | list | Columns that enter the model directly after encoding/scaling |
| `CATEGORICAL_FEATURES` | list | Subset of keep cols that need label-encoding |
| `NUMERIC_FEATURES_AFTER_ENGINEERING` | list | Final numeric feature names after all derivations |
| `RAW_DROP_AFTER_ENGINEERING` | list | All raw columns to drop once derived features are ready |
| `AGE_REFERENCE_DATE` | Timestamp | Fixed date used to compute customer age consistently |

**Decision summary:**

```
Raw column              Action                  Result
───────────────────────────────────────────────────────────────────
cc_num, first, last,
street, trans_num,   →  drop_now             →  removed entirely
zip, city, state

trans_date_trans_time →  engineer_then_drop   →  hour, day_of_week, month
dob                  →  engineer_then_drop   →  age
lat/long,
merch_lat/long       →  engineer_then_drop   →  distance_km
unix_time            →  drop (no signal)     →  removed

merchant             →  keep_raw             →  label-encoded integer
category             →  keep_raw             →  label-encoded integer
gender               →  keep_raw             →  label-encoded integer
job                  →  keep_raw             →  label-encoded integer
amt                  →  keep_raw             →  numeric, scaled
city_pop             →  keep_raw             →  numeric, scaled
```

**Output of this step:** a `feature_strategy_df` table printed for audit, plus the policy constants used downstream.

---

### Step 7 — Feature Engineering

**Function:** `engineer_features(df)` — takes a raw DataFrame and returns a transformed copy. Applies identically to both train and test to guarantee the same schema.

#### Sub-step A: Derive new features

**Time features** (from `trans_date_trans_time`):

| New feature | What it captures | Fraud relevance |
|---|---|---|
| `hour` | Hour of day (0–23) | Fraud peaks at night (Step 4 hotspots confirmed this) |
| `day_of_week` | Day of week (0=Mon, 6=Sun) | Weekend patterns differ from weekdays |
| `month` | Month of year (1–12) | Seasonal fraud shifts (holiday periods) |

**Age** (from `dob`):
```
age = (AGE_REFERENCE_DATE − date_of_birth).days / 365.25
```
Uses a fixed reference date (`2020-06-21`) so age is consistent across all rows regardless of when the notebook is run.

**Geographic distance** (from `lat`, `long`, `merch_lat`, `merch_long`):
```
distance_km = Haversine(cardholder_location, merchant_location)
```
The Haversine formula gives the straight-line distance on the Earth's surface in km. A cardholder buying in a city far from their home address is a known fraud signal.

#### Sub-step B: Drop raw columns

After all derived features exist, `RAW_DROP_AFTER_ENGINEERING` removes every source column so no raw form survives into modeling.

**Schema before `engineer_features()`:** 23 columns (raw CSV minus index)
**Schema after `engineer_features()`:** 11 columns

```
Before → After

trans_date_trans_time  →  hour, day_of_week, month  (3 new, raw dropped)
dob                    →  age                        (1 new, raw dropped)
lat, long,
merch_lat, merch_long  →  distance_km               (1 new, 4 raws dropped)
cc_num, first, last,
street, city, state,
zip, trans_num,
unix_time              →  (dropped, no replacement)

merchant               →  merchant                   (unchanged)
category               →  category                   (unchanged)
amt                    →  amt                        (unchanged)
gender                 →  gender                     (unchanged)
city_pop               →  city_pop                   (unchanged)
job                    →  job                        (unchanged)
is_fraud               →  is_fraud                   (target, unchanged)
```

---

### Step 7b — Encoding and Scaling

This cell converts the engineered DataFrame into model-ready NumPy arrays.

#### Categorical encoding with `LabelEncoder`

The four categorical columns (`merchant`, `category`, `gender`, `job`) contain strings. Models require numbers.
`LabelEncoder` assigns each unique string a fixed integer based on the training set:

```
category strings  →  integer codes
─────────────────────────────────
"grocery_pos"     →  3
"shopping_net"    →  9
"gas_transport"   →  2
...
```

**Critical rule — fit only on train:**
The encoder learns the mapping from training data only. If a test transaction has a category not seen in training, it gets mapped to `-1` (a safe unknown signal) instead of raising an error.

**Output printed:** for each column, the number of unique train classes and how many test values were unseen.

#### Scaling with `StandardScaler`

Each feature is rescaled to have mean ≈ 0 and standard deviation ≈ 1:

```
scaled_value = (raw_value − train_mean) / train_std
```

**Critical rule — fit only on train:**
`train_mean` and `train_std` are computed from training data only, then applied to test. Using test statistics would let test data influence preprocessing — a form of leakage.

**Why scaling is needed:**
`amt` ranges from ~$1 to ~$28,000. `hour` ranges from 0 to 23. Without scaling, Logistic Regression treats a 1-unit change in `amt` and a 1-unit change in `hour` as equally important — which they are not. Scaling puts all features on the same magnitude so the optimizer treats them fairly.

> Note: Random Forest is scale-invariant (trees split by threshold, not magnitude), so scaling does not affect its results — but it does not harm it either.

#### Final output arrays

| Variable | Shape | dtype | Contents |
|---|---|---|---|
| `X_train_np` | (1,296,675 × 11) | float64 | Scaled training features |
| `X_test_np` | (555,719 × 11) | float64 | Scaled test features |
| `y_train_np` | (1,296,675,) | int32 | Fraud labels (0 or 1) |
| `y_test_np` | (555,719,) | int32 | Fraud labels (0 or 1) |

These four arrays are the only inputs to all models in Step 8.

### Step 8 — Baseline Modeling (Precision-First)

Trains two class-weighted baseline models and compares them:

| Model | Type | Imbalance handling |
|---|---|---|
| `log_reg_balanced` | Logistic Regression | `class_weight="balanced"` |
| `rf_balanced` | Random Forest (300 trees) | `class_weight="balanced_subsample"` |

#### Model Output Reference

**Per-transaction scores (`y_score_test`):**
A 1D NumPy array of length ~555K. Each value is P(fraud) ∈ [0, 1]. Values near 1 are high-confidence fraud predictions. This array feeds Step 9's threshold tuning.

**Comparison table (`baseline_results_df`):**

| Column | What it measures |
|---|---|
| `roc_auc` | Ranking quality across all thresholds (0.5 = random) |
| `pr_auc` | Area under the precision-recall curve — **primary selection metric** |
| `precision_at_0_5` | Of flagged transactions, fraction that are real fraud |
| `recall_at_0_5` | Of all real frauds, fraction the model caught |
| `f1_at_0_5` | Harmonic mean of precision and recall at threshold 0.5 |
| `alert_count_at_0_5` | Number of transactions flagged — operational workload |
| `alert_rate_at_0_5` | Fraction of all test transactions flagged |

**Selection rule:** the model with the highest `pr_auc` is forwarded to Step 9 as `best_model`.

---

## Current State

Steps 1–8 are implemented and documented in `notebook.ipynb`. Steps 9 and 10 are planned (see below).

---

## Step 9 and Step 10 (Team Planning Notes)

### Step 9: Threshold Tuning (Precision-First)
**Goal**
Choose a decision threshold on model probability scores that meets business precision targets while keeping useful recall.

**Why this step exists**
Fraud models output probabilities. Operations need a hard decision rule (`fraud` vs `not fraud`), and that rule is the threshold.

**Inputs**
1. Best baseline model selected in Step 8.
2. Test-set fraud scores (`predict_proba` output).
3. Precision-first objective (high precision, controlled alert volume).

**Core method**
1. Sweep thresholds (for example `0.01` to `0.99`).
2. Compute precision, recall, F1, and alert rate for each threshold.
3. Select candidate thresholds that satisfy precision targets (example: `95%`, `97%`, `99%`).
4. Compare recall loss and alert volume tradeoff across candidates.

**Deliverables**
1. Threshold comparison table.
2. Recommended operating threshold.
3. Business interpretation: expected alert volume and expected fraud catch rate.

### Step 10: Final Recommendation
**Goal**
Finalize the model + threshold package that should move forward to implementation.

**Why this step exists**
Team needs one clear, decision-ready recommendation rather than multiple partially evaluated options.

**Inputs**
1. Step 6 feature strategy decisions.
2. Step 8 model comparison results.
3. Step 9 threshold-tuning results.

**Decision rule**
1. Prioritize precision-first requirement.
2. Among models meeting precision expectations, choose the one with best recall/PR-AUC and acceptable alert volume.
3. Confirm no leakage-risk violations from Step 5.

**Deliverables**
1. Selected model name and version.
2. Selected threshold and justification.
3. Final metric snapshot (PR-AUC, precision, recall, alert rate).
4. Implementation handoff notes (feature list, preprocessing, threshold policy).

### Team Discussion Checklist
1. What minimum precision must be guaranteed in production?
2. What alert volume per day/week is operationally acceptable?
3. Is lower recall acceptable to maintain precision target?
4. Do we require different thresholds by transaction segment later?

---

## Code Review Notes

| Block | Risk / Improvement |
|---|---|
| `warnings.filterwarnings('ignore')` | Can hide important issues during development; remove for debugging |
| Hardcoded CSV paths in Step 1 | Add config/env-based paths for portability |
| `haversine_km` | No guard for missing/invalid coordinates |
| Fixed `AGE_REFERENCE_DATE` | Can become stale if dataset period changes |
| `LabelEncoder` on nominals | Implies false ordinal meaning for Logistic Regression; consider target-encoding `merchant` |
| `StandardScaler` applied to label-encoded ints | Harmless but misleading for tree models (which are scale-invariant) |
| Scaler/encoders not persisted | Add `joblib.dump()` calls for reproducible inference |
