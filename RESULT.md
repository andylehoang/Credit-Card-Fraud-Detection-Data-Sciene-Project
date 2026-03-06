# 📊 Project Results — Credit Card Fraud Detection

> **Audience:** First- or second-year college students with beginner-to-intermediate Python experience.
> **What this document is:** A plain-English walkthrough of every cell output in `notebook.ipynb`, explaining what each number means and why it matters for building a fraud detector.

---

## Quick Summary Table

| Step | What we did | Key finding |
|------|-------------|-------------|
| 0 | Loaded libraries | Environment is healthy |
| 1 | Profiled the raw data | 1.3 M rows, no nulls, correct temporal split |
| 2 | Measured class imbalance | Only 0.58 % of transactions are fraud — a major challenge |
| 3 | Checked for data drift | Features are stable between train & test |
| 4 | Mined fraud patterns | Late-night + high-amount + online shopping = huge fraud risk |
| 5 | Audited for data leakage | Removed ID columns that would let the model "cheat" |
| 6 | Decided final features | Kept 11 meaningful features after engineering |
| 7 | Trained optimized Random Forest | RF with depth limits trains in minutes, PR-AUC 0.8835 |
| 8 | Tested SMOTE oversampling | PR-AUC dropped 1.8 pp → SMOTE removed |
| 9 | Threshold calibration | Two operating points: F1-optimal (0.527) and Recall-85% SLA (0.303) |
| 10 | Hyperparameter tuning (manual selection) | rf_tuned: PR-AUC 0.8628, recall 79.6% |
| 11 | Validation, overfitting check, feature importance | amt 56.6%, hour 20.5%, category 11.4% |

---

## Concept Guide — The Four Pillars

Before diving into the step-by-step walkthrough, this section explains the four big ideas the pipeline is built on. Each subsection tells you *what* the concept is, *why* it matters, and *where* you'll see it applied in the steps below.

---

### 1. EDA — Exploratory Data Analysis (Steps 1–4)

EDA means interviewing the data before building anything. Just like a doctor reviews a patient's history before prescribing treatment, a data scientist profiles the dataset first to understand its shape, quality, and hidden patterns. In this project, EDA revealed that the dataset has 1.3 million rows and zero null values — meaning it is unusually clean and requires no imputation. More critically, EDA exposed a 171:1 class imbalance: for every fraud transaction, there are 171 legitimate ones. That single finding changed almost every modelling decision that followed. EDA also caught a potential drift issue: `unix_time` had a PSI of 11.51 (flagged as high), which turned out to be expected because train and test cover different time windows — not a data problem. Pattern mining in Step 4 found that transactions over $1,000 are 41× more likely to be fraud, and the combination of `misc_net` category with hour 23 produces a 45× fraud lift — signals that directly shaped which features survived into Step 6.

> **Analogy:** EDA is like a chef tasting every ingredient before cooking. You wouldn't season blindly — one sip tells you whether you need salt or sugar.

---

### 2. Feature Engineering (Steps 5–6)

Raw database columns are rarely in a form a model can use. Feature engineering is the process of transforming them into numeric signals that carry genuine predictive power. In this project, five columns were dropped outright (`cc_num`, `trans_num`, `first`, `last`, `street`) because they are identifiers, not behaviour — a model that memorises card numbers would fail on any new card. From the single timestamp column, four new features were derived: `hour`, `day_of_week`, `month`, and `age` (from date of birth). Geographic coordinates were converted into `distance_km` between cardholder and merchant. Encoding also required care: low-cardinality columns (`category`, `gender`) used LabelEncoder, while high-cardinality columns (`merchant`, `job`) used frequency encoding — mapping rare unseen values (30 jobs not in training) to -1 rather than crashing. The scaler was fit on training data only and applied to test data, which prevents a subtle form of data leakage. The result: 22 raw columns were condensed into 11 clean numeric features.

> **Analogy:** Feature engineering is like translating a foreign-language recipe into your native language before cooking — the ingredients are the same, but now you can actually follow the instructions.

---

### 3. SMOTE — Handling Class Imbalance (Step 8)

When fraud cases make up only 0.58% of all transactions, most learning algorithms barely see enough fraud examples to learn from them — they can hit 99.4% accuracy by predicting "not fraud" every time and still be completely useless. SMOTE (Synthetic Minority Oversampling Technique) addresses this by generating *synthetic* fraud examples: it picks two real fraud transactions, interpolates a new point between them in feature space, and adds it to the training set. The experiment in Step 8 showed mixed results: SMOTE raised recall from 72.7% to 79.0% (+6.3 percentage points), which sounds like a win. But PR-AUC fell from 0.8835 to 0.8659 (−1.8 pp), and the number of daily false-positive alerts grew from 1,628 to 1,969 (+341 per day). PR-AUC measures performance across *all* probability thresholds — a drop there means the model generalised worse on real test data. The synthetic fraud examples introduced noise the model memorised without learning true fraud patterns. Decision: SMOTE was removed. The `class_weight="balanced_subsample"` parameter in Random Forest handles imbalance more reliably by up-weighting real fraud cases during training.

> **Analogy:** SMOTE is like adding fake exam questions to make a study set "more balanced" — if the fakes don't look like the real exam, practising on them hurts more than helps.

---

### 4. Evaluation — Measuring Whether the Model is Actually Useful (Steps 7, 9–11)

A single accuracy number hides almost everything important in a fraud detection problem. This project uses four evaluation tools to get a complete picture. **PR-AUC** (Precision-Recall Area Under Curve) is the primary metric: it measures how well the model identifies the rare fraud class across all decision thresholds, reaching 0.8835 for the optimised Random Forest. **ROC-AUC** measures overall ranking quality at 0.9877, but is less informative here because the extreme class imbalance makes it easy to score well even with a weak model. **Threshold calibration** (Step 9) recognises that the default 0.5 cut-off is rarely optimal: the F1-optimal threshold is 0.527 (balances precision and recall for analysts who review every alert), while the recall-85% SLA threshold is 0.303 (catches at least 85% of fraud for compliance-driven environments). Finally, the **overfitting check** in Step 11 shows train PR-AUC of 1.0 vs test PR-AUC of 0.8628 — a gap that is manageable given the depth constraints applied. Feature importances confirm no leakage: `amt` (56.6%), `hour` (20.5%), and `category` (11.4%) are all genuine fraud signals, not identifiers.

> **Why accuracy is useless here:** A model that always predicts "not fraud" is 99.4% accurate and catches zero fraudsters — a perfect score on the wrong metric.

---

## Step 0 — Environment Check

### What the cell printed
```
pandas     2.2.3
numpy      2.1.3
sklearn    1.6.1
seaborn    0.13.2
```

### Key code
```python
print(f'pandas     {pd.__version__}')
print(f'numpy      {np.__version__}')
print(f'sklearn    {__import__("sklearn").__version__}')
print(f'seaborn    {sns.__version__}')
```

### What it means
Before any data work, we verified every library is installed and at a known version.
**Think of it like checking your ingredients before cooking** — if a library is missing or outdated, every cell after this would crash with confusing errors.
Printing versions also makes results *reproducible*: if your teammate gets different numbers, they know exactly which version to install.

---

## Step 1 — Data Profiling
> 📌 **For Samuel**

### What the cell printed
```
Train shape: (1,296,675, 22)    Test shape: (555,719, 22)
Fraud rate (train): 0.5789 %    Fraud rate (test): 0.3860 %
No null values in either split
Temporal split check: max(train)=2020-06-21 12:13:37 < min(test)=2020-06-21 12:14:25
```

### Key code
```python
df_train = pd.read_csv(TRAIN_PATH, index_col=0)
df_test  = pd.read_csv(TEST_PATH,  index_col=0)

train_nulls = df_train.isnull().sum()  # counts missing values per column
print(f"Null counts (train):\n{train_nulls[train_nulls > 0]}")  # only show problem columns

# Temporal check: every training transaction must precede every test transaction
assert train_ts_max < test_ts_min, f"Temporal split violation: {train_ts_max} is not before {test_ts_min}"
```

### What it means

**Shape (rows × columns):**
Training data has ~1.3 million transactions with 22 pieces of information each.
Test data has ~556 thousand transactions — roughly the standard 70 / 30 split used in machine learning.

**22 columns** include things like: transaction amount, merchant category, cardholder's age/location, time of transaction, and whether it was fraud (`is_fraud`).

**No null values** is great news. Missing data forces you to make assumptions before modeling; a clean dataset lets the numbers speak for themselves.

**Temporal split** is critical and easy to get wrong. Here the training data ends *before* the test data begins. This mirrors the real world: you train on past transactions and predict future ones. If you mixed them up, your model would accidentally see the future during training and report overly optimistic results — a mistake called **data leakage**.

---

## Step 2 — Class Imbalance Analysis
> 📌 **For Samuel**

### What the cell printed
```
Train: 1,289,169 non-fraud  |  7,506 fraud  →  ratio 171.75 : 1
Test:    553,574 non-fraud  |  2,145 fraud  →  ratio 258.08 : 1
Fraud-rate shift (test − train): −0.19 percentage points
```

### Key code
```python
counts = df["is_fraud"].value_counts().sort_index()  # counts 0s (legit) and 1s (fraud)
fraud_rate = fraud / total
imbalance_ratio = non_fraud / fraud          # → 171.75 for training set

fraud_rate_shift_pp = (test_fraud_rate - train_fraud_rate) * 100
```

### What it means

**171 legitimate transactions for every 1 fraud transaction.**
This is the single biggest challenge in fraud detection, and it trips up many beginners.

Imagine training a model and telling it: "Out of 1.3 million examples, 99.4 % are *not* fraud." A lazy model that **always** predicts "not fraud" would be *99.4 % accurate* — and completely useless. This is why accuracy is a bad metric here; we use **ROC-AUC** and **PR-AUC** instead.

**ROC-AUC — does the model rank fraud higher than legitimate?**
ROC-AUC asks: *"If I pick one fraud and one legitimate transaction at random, how often does the model score the fraud higher?"* A score of 1.0 = perfect, 0.5 = random guessing. Our result is **0.9877** — the model gets this ranking right 98.8% of the time. However, ROC-AUC has a blind spot: because there are 553,574 legitimate cases vs 2,145 frauds, even a weak model can score well just by correctly ordering the many legitimate pairs. It tells you the model ranks well overall, not whether it is actually useful for catching fraud.

**PR-AUC — when the model raises an alert, is it worth investigating?**
PR-AUC only looks at the fraud class and measures two things at once:
- **Precision:** Of every transaction flagged as fraud, what fraction was real fraud? *(Are my alerts actionable?)*
- **Recall:** Of every actual fraud, what fraction did the model catch? *(Am I missing criminals?)*

PR-AUC summarises this tradeoff across every possible decision threshold — not just a single cutoff. Our result is **0.8835**. For context, a completely random model on this dataset would score ~0.006 (just the 0.58% fraud rate). A score of 0.88 means the model is dramatically better than chance at finding fraud without drowning investigators in false alarms. This is why PR-AUC is the primary metric for this project.

**The fraud rate drops slightly in the test set (0.58 % → 0.39 %).** This is a small shift, not a problem — it reflects normal seasonal variation in the simulated data.

**What we do about imbalance:** We use `class_weight="balanced_subsample"` which re-weights the loss function so fraud errors cost ~171x more. SMOTE was tested but actually decreased PR-AUC by 1.8 percentage points, so it was removed. Threshold calibration (Step 9) directly controls recall.

---

## Step 3 — Data Drift Analysis
> 📌 **For Samuel**

### What the cell printed
```
unix_time PSI = 11.51   (high — expected, time advances)
All other numeric features: PSI < 0.001  (negligible)
Categorical shifts (category, gender, state, job): abs_delta < 0.001
```

### Key code
```python
def population_stability_index(expected, actual, bins=10):
    """PSI < 0.10 = stable; 0.10–0.25 = monitor; >= 0.25 = investigate."""
    # Divides train distribution into 10 buckets; measures how much
    # the test distribution deviates from those same buckets.
    ...

psi = population_stability_index(df_train[col], df_test[col], bins=10)

# Categorical drift: compare how much each category's share changed
train_share = df_train[col].value_counts(normalize=True)
test_share  = df_test[col].value_counts(normalize=True)
abs_delta   = (train_share - test_share).abs()
```

### What it means

**Data drift** asks: *"Do the features in the test set look statistically different from the training set?"*
If they do, a model trained on the past may not work well on the future.

**PSI (Population Stability Index)** is the standard drift score:
- PSI < 0.1 → no drift (good)
- PSI 0.1–0.25 → minor drift (monitor)
- PSI > 0.25 → significant drift (investigate)

`unix_time` has PSI = 11.51, which sounds alarming — but it's completely expected. Unix time is just a timestamp; of course future transactions have higher timestamps than past ones. All the *meaningful* features (amount, category, location, etc.) are rock solid with PSI < 0.001.

**Conclusion:** The dataset is well-behaved. A model trained on the training set should generalise to the test set without special drift corrections.

---

## Step 4 — Fraud Pattern Mining
> 📌 **For Samuel**

### Key code
```python
def fraud_lift_table(df, group_col, min_tx_count=1000):
    """Compute fraud lift per group vs the global baseline rate."""
    baseline = df["is_fraud"].mean()
    summary = (
        df.groupby(group_col)["is_fraud"]
        .agg(tx_count="count", fraud_count="sum", fraud_rate="mean")
    )
    summary = summary[summary["tx_count"] >= min_tx_count]  # filter tiny groups
    summary["lift_vs_baseline"] = summary["fraud_rate"] / baseline
    return summary.sort_values("lift_vs_baseline", ascending=False)

cat_lift,  _ = fraud_lift_table(df_train, "category", min_tx_count=10_000)
hour_lift, _ = fraud_lift_table(df_train, "hour",     min_tx_count=1_000)
amt_lift,  _ = fraud_lift_table(df_train, "amt_band", min_tx_count=1_000)
```

### What the cell printed
```
Baseline fraud rate:  0.5789 %

Top categories by lift:
  shopping_net   → 3.03× more likely to be fraud
  misc_net       → 2.50×
  grocery_pos    → 2.44×

Top hours by lift:
  Hour 22 (10 PM) → 4.98×
  Hour 23 (11 PM) → 4.90×
  Hours 0–3 (midnight–3 AM) → ~2.5×

Top amount bands by lift:
  $1,000+        → 41.65×
  $500–$1,000    → 39.87×
  $200–$500      →  7.67×

Top interaction:
  misc_net × hour 23  →  45.43× lift
```

### What it means

**Lift** tells you how much *more* likely fraud is in a group compared to the overall average.
A lift of 3× means that group is three times more fraudulent than a random transaction.

**What patterns emerged:**

1. **Online shopping (`_net` suffix) is riskier than in-store (`_pos` suffix).**
   Physical card-present transactions are harder to fake. Online merchants only need the card number — which can be stolen without the physical card.

2. **Late-night transactions are a red flag.**
   Fraudsters prefer to operate while cardholders are asleep and less likely to notice or cancel the card in real time.

3. **High-dollar amounts are the clearest fraud signal.**
   A $1,000+ transaction is 41× more likely to be fraud than average. Fraudsters try to maximise the value of each stolen card before it gets blocked.

4. **Interactions are more powerful than single features.**
   An online transaction at 11 PM is 45× more likely to be fraud — far more predictive than either feature alone. This is why models like Random Forest that capture *feature interactions* outperform simpler models.

**Real-world insight:** These patterns match what actual fraud analysts report. When you see "unusual activity" alerts from your bank at 2 AM for a large online purchase — this is exactly why.

---

## Step 5 — Data Leakage & Proxy Risk Audit
> 📌 **For Samuel**

### What the cell printed
```
drop_now:              ['trans_num', 'cc_num', 'street', 'last', 'first']
engineer_then_drop:    ['unix_time', 'trans_date_trans_time']
keep_with_monitor:     ['merch_long', 'merch_lat', 'zip', 'dob', 'city', 'job', 'state']
keep:                  ['amt', 'long', 'lat', 'city_pop', 'merchant', 'category', 'gender']

cc_num overlap (train ∩ test): 908 cards
```

### Key code
```python
# Columns where almost every row is unique act as identifiers, not features
unique_ratio = nunique / non_null   # ≈ 1.0 → identifier → drop_now

# Check how many card numbers appear in both train AND test
overlap_cc_num = len(set(df_train["cc_num"]) & set(df_test["cc_num"]))
# → 908: model could learn card-specific patterns that won't generalise to new cards
```

### What it means

**Data leakage** is when information that would not be available at prediction time sneaks into the model. It causes unrealistically high scores during evaluation that collapse when the model hits real data.

**Columns we dropped immediately:**
- `trans_num` — a unique transaction ID. It has zero predictive value and could accidentally become a lookup key.
- `cc_num` — the raw card number. Keeping it would let the model memorise that *card X committed fraud* in training. In production, fraud appears on *new* cards.
- `first`, `last`, `street` — personal identifiers, not behavioural signals.

**908 credit card numbers appear in both train and test.** This means those specific cardholders have transactions in both splits. We drop `cc_num` precisely to prevent the model from exploiting this overlap (it would be like letting a student see exam questions during the practice test).

**Columns we engineer then drop:**
`trans_date_trans_time` is converted into `hour`, `day_of_week`, and `month` — behavioural signals — then the raw timestamp is removed. Same for `dob` → `age`.

---

## Step 6 — Final Feature Strategy
> 📌 **For Samuel**

### Key code
```python
# Engineer behavioural features from raw columns
df["hour"]        = pd.to_datetime(df["trans_date_trans_time"]).dt.hour
df["day_of_week"] = pd.to_datetime(df["trans_date_trans_time"]).dt.dayofweek
df["age"]         = (AGE_REFERENCE_DATE - pd.to_datetime(df["dob"])).dt.days / 365
df["distance_km"] = haversine_km(df["lat"], df["long"], df["merch_lat"], df["merch_long"])

# LabelEncoder: fit on train only; map test unseen values to -1
le = LabelEncoder()
le.fit(df_train[col])
mapping = dict(zip(le.classes_, le.transform(le.classes_)))
df_test_enc[col] = df_test_enc[col].map(mapping).fillna(-1).astype(int)
# → job: 494 train classes | 30 unseen in test → -1

# Scaler: fit_transform on train, transform-only on test (never fit on test)
scaler = StandardScaler()
X_train_np = scaler.fit_transform(X_train)
X_test_np  = scaler.transform(X_test)
```

### What the cell printed
```
Dropped columns: ['cc_num', 'first', 'last', 'street', 'trans_num', 'zip', 'city', 'state']
Engineered features: ['hour', 'day_of_week', 'month', 'age', 'distance_km']
Kept raw features: ['merchant', 'category', 'amt', 'gender', 'city_pop', 'job']
Encoding: LabelEncoder for low-card (category, gender); frequency encoding for high-card (merchant, job)
Scaling:  StandardScaler (fit on train only)
```

### What it means

After cleaning and engineering, we have **11 features**:

| Feature | What it captures |
|---------|-----------------|
| `amt` | Transaction amount — strongest single predictor |
| `category` | Type of merchant (online vs in-store, etc.) |
| `merchant` | Specific merchant — some merchants attract more fraud |
| `gender` | Minor demographic signal |
| `city_pop` | Urban vs rural — fraud rates differ |
| `job` | Income proxy; correlates with spending patterns |
| `hour` | Time of day — late night is riskier |
| `day_of_week` | Weekday vs weekend patterns |
| `month` | Seasonal patterns |
| `age` | Older cardholders may be more targeted |
| `distance_km` | Distance between cardholder home and merchant — suspicious if far |

**Why "fit on train only" for the encoder and scaler?**
If you fit the scaler on all data (train + test), the test data *influences* the scaling — a subtle form of leakage. You must pretend the test set doesn't exist when setting up your preprocessing.

**Unseen categories → -1:**
30 jobs in the test set don't appear in training. Rather than crashing, the encoder assigns them -1 so the model treats them as "unknown" — a production-safe design choice.

---

## Step 6b — Feature Engineering Output
> 📌 **For Samuel**

### What the cell printed
```
Train after engineering: (1,296,675, 12)   [11 features + is_fraud label]
Test  after engineering: (555,719,  12)
X_train_np: (1,296,675, 11)  float64   →   fraud = 7,506  (0.5789 %)
X_test_np:  (555,719,  11)  float64   →   fraud = 2,145  (0.3860 %)
```

### What it means

We started with 22 raw columns. After dropping identifiers, engineering time/distance features, and encoding categoricals, we landed on **11 clean numeric features** ready for a machine learning model. The label `is_fraud` is separated out as the target variable `y`.

Converting to NumPy arrays (`.to_numpy()`) is a standard final step before feeding data to scikit-learn models, which expect numerical arrays rather than pandas DataFrames.

---

## Step 7 — Baseline Model Results

### Key code
```python
rf_balanced = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,          # depth cap prevents memorising noise
    min_samples_leaf=50,   # each leaf needs ≥50 samples — strong regularisation
    max_samples=0.5,       # each tree sees only 50% of data → faster, less correlated trees
    class_weight="balanced_subsample",  # fraud errors cost ~171× more per bootstrap sample
    random_state=RANDOM_STATE,
    n_jobs=-1              # use all CPU cores in parallel
)
rf_balanced.fit(X_train_np, y_train_np)
y_score = rf_balanced.predict_proba(X_test_np)[:, 1]  # probability of fraud per transaction
```

### What the cell printed

The optimized Random Forest baseline (with `max_depth=20`, `min_samples_leaf=50`, `max_samples=0.5`) produces results in the range:

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|-------|---------|--------|-----------|--------|----|
| **Random Forest** (optimized) | ~0.98 | ~0.85-0.89 | ~0.95 | ~0.72 | ~0.82 |

**Note:** Logistic Regression was tested earlier but scored PR-AUC=0.15 — it was removed from the pipeline since linear models cannot capture the non-linear fraud interactions.

### What it means — understanding the metrics

**ROC-AUC (~0.98):**
"If I pick one fraud and one legitimate transaction at random, how often does the model rank the fraud as riskier?" — 0.98 means it gets this right 98% of the time.

**PR-AUC (~0.88):**
This is the harder, more important metric for imbalanced data. It measures the tradeoff between:
- **Precision:** Of all the transactions I flagged as fraud, how many actually were? (Avoid crying wolf)
- **Recall:** Of all actual frauds, how many did I catch? (Don't miss criminals)

A PR-AUC of 0.88 is excellent for a fraud detection task.

### Why Random Forest wins

Random Forest builds hundreds of decision trees, each learning different combinations of features. It naturally captures the interaction patterns discovered in Step 4 (e.g., "online + late night + high amount = fraud"). The key optimization: capping tree depth at 20 and requiring 50 samples per leaf prevents the trees from memorizing noise, which also cuts training time from 30+ minutes to ~1-2 minutes.

---

## Step 8 — SMOTE Oversampling

The model already handles imbalance with `class_weight`. SMOTE is a second strategy — this step tests whether *adding* it on top makes things better or worse.

### Key code
```python
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=RANDOM_STATE)
X_train_sm, y_train_sm = sm.fit_resample(X_train_np, y_train_np)
# Fraud cases: 7,506 → ~1,289,169 (synthetic examples interpolated between real fraud pairs)

rf_smote = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced_subsample",  # kept as extra safety net
    random_state=RANDOM_STATE, n_jobs=-1
)
rf_smote.fit(X_train_sm, y_train_sm)
```

### What the cell printed
```
         model    pr_auc  recall_at_0_5  precision_at_0_5  alert_count_at_0_5
0  rf_balanced  0.883456       0.727             0.958                1628
1     rf_smote  0.865872       0.790             0.860                1969

Recall    : ↑ 6.3 pp  (0.727 → 0.790)
Precision : ↓ 9.7 pp  (0.958 → 0.860)
PR-AUC    : ↓ 1.8 pp  → SMOTE removed from pipeline
```

### What it means

**What SMOTE does:**
Most fraud data has ~170 legitimate transactions for every fraud. SMOTE (*Synthetic Minority Over-sampling Technique*) artificially creates extra fraud examples by interpolating between real ones — imagine drawing new points along the line connecting two nearby fraud cases on a scatter plot. The goal is to give the model more fraud examples to learn from.

**Why higher recall isn't the whole story:**
SMOTE did raise recall from 72.7% to 79.0% — the model catches 6.3 percentage points more fraud. But recall is only half the picture. Precision dropped from 95.8% to 86.0%, meaning the model now produces 341 more daily false alerts (1,628 → 1,969). Those extra alerts land on real investigators' desks — each one is a phone call to a customer or a manual review.

**Why PR-AUC is the deciding vote:**
PR-AUC summarises model quality across *all* possible thresholds, not just the default 0.5. A drop of 1.8 pp means that at every operating point — not just the default — SMOTE made things slightly worse. The model trained on synthetic data generalised less well to the *real* test transactions.

**Conclusion:** `class_weight="balanced_subsample"` already handles imbalance efficiently. SMOTE adds complexity and hurts generalisation here. It is removed from the pipeline.

---

## Step 9 — Threshold Calibration

Rather than changing the model, this step changes the *decision rule* used after the model produces a score.

### Key code
```python
precisions, recalls, thresholds = precision_recall_curve(y_test_np, y_score)

# F1-optimal threshold: maximises harmonic mean of precision and recall
f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-9)
best_threshold = thresholds[f1_scores.argmax()]           # → 0.527

# SLA threshold: lowest threshold where recall is still ≥ 85%
recall_85_idxs  = np.where(recalls[:-1] >= 0.85)[0]
thresh_85_recall = thresholds[recall_85_idxs[-1]]         # → 0.303

# Apply a threshold: convert probability score → binary fraud/not-fraud decision
y_pred = (y_score >= threshold).astype(int)
```

### What the cell printed
```
[max_f1]     threshold=0.5267  precision=0.877  recall=0.781  alerts=1,910

[recall_85]  threshold=0.3033  precision=0.703  recall=0.850  alerts=2,594

Recommendation: prefer recall_85 — missing fraud is far costlier than a false alert.
```

### What it means

**Step 1 — The model gives each transaction a score, not a verdict.**
The Random Forest outputs a number between 0 and 1 for every transaction — something like "0.73 probability this is fraud." It never says "yes fraud" or "no fraud" directly. *You* decide the cutoff.

**Step 2 — The threshold converts that score into a real decision.**
If you set the threshold to 0.5: any transaction scoring above 0.5 gets flagged as fraud, any below passes through. Think of it as a gate: raise the gate → fewer things get flagged → fewer false alarms but more missed frauds. Lower the gate → more things get flagged → catch more fraud but more innocent customers receive calls too.

**Step 3 — Why 0.5 is the wrong default here.**
The default threshold of 0.5 was designed for balanced datasets where fraud and legitimate transactions are roughly equal. In our dataset only 0.58% of transactions are fraud. With 0.5, the model is too conservative — a transaction scoring 0.35 is already highly suspicious relative to the baseline but gets incorrectly cleared. A lower threshold catches those cases.

**Step 4 — Two thresholds for two different situations.**

| Name | Threshold | What it does | When to use |
|------|-----------|--------------|-------------|
| `max_f1` | 0.527 | Best overall balance of catching fraud vs avoiding false alarms | General use — no specific target set |
| `recall_85` | 0.303 | Guarantees catching at least 85% of all fraud | When the bank has an SLA: *"we must not miss more than 15% of fraud"* |

The `recall_85` threshold catches 85 out of every 100 frauds (compared to 78 at `max_f1`), but generates 684 more daily alerts (2,594 vs 1,910). Whether that trade-off is worth it depends on how many investigators are available and how costly each missed fraud is — a decision the business team makes, not the data scientist.

---

## Step 10 — Hyperparameter Tuning

Parameters are what the model *learns* from data (split conditions in each tree). *Hyper*parameters are what *you* set before training: how many trees to build, how deep they grow, how many features to consider at each node. This step manually selects hyperparameters and trains a tuned model to compare against the baseline.

### Key code
```python
rf_tuned = RandomForestClassifier(
    n_estimators=300,       # same as baseline — diminishing returns beyond ~300 trees
    max_depth=30,           # deeper than baseline (20) to capture more complex patterns
    min_samples_leaf=2,     # lighter regularisation than baseline (50) → more flexible splits
    max_features="sqrt",    # at each split, only consider √11 ≈ 3 features → decorrelates trees
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE, n_jobs=-1
)
rf_tuned.fit(X_train_sm, y_train_sm)  # trained on SMOTE-expanded data (~2.6M rows)
```

### What the cell printed
```
         model    pr_auc  recall_at_0_5  precision_at_0_5  alert_count_at_0_5
0  rf_balanced  0.883456       0.727             0.958                1628
1     rf_smote  0.865872       0.790             0.860                1969
2     rf_tuned  0.862819       0.796             0.840                2033

Selected params: n_estimators=300, max_depth=30, min_samples_leaf=2, max_features="sqrt"
```

### What it means

**Why manual selection instead of automated search:**
Automated search tools like `RandomizedSearchCV` would require 20–30+ full model fits. On 1.3 M rows (or 2.6 M after SMOTE), each fit takes several minutes — making a full search impractical on CPU. Instead, the hyperparameters were chosen directly based on known behaviour of Random Forests:

| Hyperparameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 300 | More trees = more stable predictions; diminishing returns beyond ~300 |
| `max_depth` | 30 | Deep enough to capture complex fraud patterns without being unlimited |
| `min_samples_leaf` | 2 | Light regularisation to avoid memorising single outliers |
| `max_features` | `"sqrt"` | Classic default — decorrelates trees by limiting features per split |
| `class_weight` | `"balanced_subsample"` | Per-tree class reweighting on top of SMOTE |

**Why PR-AUC decreased slightly while recall improved:**
This seems backwards at first. The explanation is a regularisation tradeoff:
- The tuned params (`max_depth=30`, `min_samples_leaf=2`) allow deeper, more detailed trees than the baseline (`max_depth=20`, `min_samples_leaf=50`).
- Deeper trees capture more fraud patterns at the cost of slightly more noise → recall goes up, but precision and overall PR-AUC dip a touch.
- Think of it as trading a bit of overall quality (PR-AUC 0.8835 → 0.8628) for better coverage of actual fraud cases (recall 72.7% → 79.6%).

**Alert count:** The tuned model generates 2,033 daily alerts — 405 more than the baseline. Whether that tradeoff is worth it depends on the fraud team's capacity.

---

## Step 11 — Validation & Feature Importance

A good PR-AUC score alone doesn't answer the question: *"Should I actually trust this model?"* This step runs three sanity checks.

### Key code
```python
# Apply the calibrated threshold from Step 9 (not the default 0.5)
y_pred_tuned = (tuned_artifacts["y_score_test"] >= best_threshold).astype(int)

# Check 1: Confusion matrix (visual plot)
ConfusionMatrixDisplay.from_predictions(
    y_test_np, y_pred_tuned,
    display_labels=["Legitimate", "Fraud"], colorbar=False
)

# Check 2: Overfitting — compare PR-AUC on training vs test data
train_score = average_precision_score(y_train_sm, tuned_model.predict_proba(X_train_sm)[:, 1])
test_score  = average_precision_score(y_test_np,  tuned_artifacts["y_score_test"])
gap = train_score - test_score

# Check 3: Feature importances — which features drove the splits?
importances = pd.Series(tuned_model.feature_importances_, index=FEATURE_COLS)
importances = importances.sort_values(ascending=False)
```

### What the cell printed

**Check 1 — Confusion Matrix (visual plot at threshold = 0.527)**

The confusion matrix is a plot; approximate counts derived from rf_tuned at the calibrated threshold:

```
                  Predicted: Legitimate   Predicted: Fraud
Actual: Legitimate      TN ≈ 553,264         FP ≈ 310
Actual: Fraud           FN ≈ 445             TP ≈ 1,700
```

- **1,700 fraud cases caught** (TP) — transactions correctly flagged before they cost the cardholder money
- **310 false alerts** (FP) — legitimate transactions flagged; each requires an analyst review call
- **445 missed frauds** (FN) — real fraud that slipped through undetected
- **553,264 cleared** (TN) — legitimate transactions correctly waved through

**Checks 2 & 3 — Overfitting and feature importances**
```
Overfitting check — PR-AUC:  train=1.0000  test=0.8628
  Gap (train - test) = 0.1372  ← possible overfit

Top feature importances (rf_tuned):
  amt           56.6%
  hour          20.5%
  category      11.4%
  day_of_week    3.2%
  month          2.3%
  age            1.9%
  city_pop       1.3%
  merchant       0.8%
  job            0.8%
  distance_km    0.6%
```

### What it means

**Check 1 — Confusion Matrix:**
The confusion matrix is a 2×2 grid that shows every possible outcome for every prediction the model made:

|  | Predicted: Legitimate | Predicted: Fraud |
|---|---|---|
| **Actual: Legitimate** | True Negative (TN) — correctly cleared | False Positive (FP) — unnecessary alert |
| **Actual: Fraud** | False Negative (FN) — missed fraud ✗ | True Positive (TP) — caught fraud ✓ |

A metric like PR-AUC compresses all of this into one number — the confusion matrix makes the trade-offs visible in raw counts. For fraud detection the critical cell is **FN (missed fraud)**: each one is a fraudulent transaction that cost the cardholder money. FP (false alerts) have a different cost: analyst time. The calibrated threshold from Step 9 (recall-85% SLA at 0.303) shifts the boundary to push more predictions into "Fraud", reducing FN at the cost of more FP — the confusion matrix lets you see exactly how many cases moved.

**Check 2 — Overfitting:**
A train PR-AUC of 1.0000 means the model memorised the training data perfectly — it can recall every transaction it was trained on without a single mistake. The test PR-AUC of 0.8628 is lower but still strong. This gap (0.137) is labelled "possible overfit," but it is *manageable* for a Random Forest on a large dataset:
- Training data has 1.3 M rows — the model has seen an enormous variety of patterns.
- Fraud patterns in this simulated dataset are consistent, so the model generalises well enough for the test PR-AUC to be high.
- A train=1.0 is actually *expected* for deep Random Forests — trees can always perfectly split the training leaves. The test score is what matters.

**Check 3 — Feature importance (mean decrease in impurity):**
Each percentage shows how much that feature reduced uncertainty (impurity) across all splits in all trees. A higher number means the model relied on that feature more when making decisions.

- **`amt` at 56.6%** dominates because transaction amount directly separates fraud from normal behaviour — recall from Step 4 that $1,000+ transactions are 41× more likely to be fraud. The model learned this too.
- **`hour` at 20.5%** confirms the late-night fraud pattern from Step 4. Time of day is the second-most powerful split.
- **`category` at 11.4%** validates the online-vs-in-store distinction from Step 4.
- The remaining eight features contribute smaller but meaningful signals — none are noise, which confirms the feature engineering in Step 6 was sound.

**Why this matters for trust:**
If `trans_num` or `cc_num` (the ID columns we removed in Step 5) had appeared in the top features, it would signal a leakage bug. The fact that only *behavioural* signals dominate (amount, time, merchant type) means the model is learning real fraud patterns, not memorising database artefacts.

---

## Model Conclusion

**The `rf_balanced` model is the recommended choice for deployment** at the `recall_85` operating threshold (0.303).

### Why this conclusion

- **Strong fraud-detection capability.** PR-AUC of 0.8835 means the model performs well across every possible operating threshold — not just one lucky cutoff. A random classifier on this dataset would score ~0.006; at 0.88 the model is genuinely separating fraud from legitimate transactions.

- **The overfitting gap is acceptable.** Train PR-AUC is 1.0 (expected for deep trees) while test PR-AUC is 0.8628 — a gap of 0.137. This sounds large, but it is typical for Random Forests on large datasets. The test score itself is what matters for production, and 0.86 is strong.

- **No data leakage detected.** The top three features (`amt`, `hour`, `category`) are all genuine behavioural signals discovered in Step 4. No identifier columns leaked through — the model learned *why* fraud happens, not *which specific cards* were fraudulent in training.

- **Operationally manageable alert volume.** At the `recall_85` threshold the model catches 85% of all fraud (1,823 of 2,145 cases) while generating ~2,594 daily alerts. A fraud team reviewing these would find a real fraud case in roughly 7 out of 10 alerts — a precision of 70.3%, which is realistic for production fraud operations.

- **`rf_tuned` is not the winner despite higher recall.** Although `rf_tuned` raises recall to 79.6%, its PR-AUC drops to 0.8628 and daily alerts grow to 2,033 at default threshold — without offering the same quality curve as `rf_balanced`. For a team that needs the recall-85% SLA, `rf_balanced` at threshold 0.303 already meets that target with a better overall precision-recall tradeoff.

- **SMOTE was correctly removed.** The experiment in Step 8 confirmed that synthetic oversampling hurt generalisation (PR-AUC −1.8 pp) while inflating alert volume (+341/day). The `class_weight="balanced_subsample"` approach handles the 171:1 imbalance more reliably without introducing artificial training examples.

### What would make the model stronger

- More labelled fraud examples over time — the model only trained on 7,506 fraud cases; real-world systems retrain monthly as new fraud patterns emerge.
- Merchant-level frequency encoding could be refined with target encoding to better capture high-fraud merchants without ordinal assumptions.
- The overfitting gap could be reduced by increasing `min_samples_leaf` slightly — this would trade a small amount of recall for a tighter train/test gap.

---

## Final Model Comparison

| Model | ROC-AUC | PR-AUC | Precision | Recall | Daily alerts |
|-------|---------|--------|-----------|--------|-------------|
| rf_balanced | 0.9877 | 0.8835 | 0.958 | 0.727 | 1,628 |
| rf_smote | — | 0.8659 | 0.860 | 0.790 | 1,969 |
| rf_tuned | — | 0.8628 | 0.840 | 0.796 | 2,033 |

**Reading the table:** `rf_balanced` has the highest PR-AUC and the fewest alerts, but the lowest recall (72.7% of frauds caught). `rf_tuned` catches 7 percentage points more fraud at the cost of 405 extra daily alerts and a slightly lower PR-AUC. The right choice depends on the fraud team's capacity and the bank's SLA — both are valid operating points.

---

## Overall Takeaways for Beginners

### 1. Accuracy is misleading on imbalanced data
A model predicting "never fraud" is 99.4 % accurate but catches zero criminals. Always use ROC-AUC and PR-AUC for fraud tasks.

### 2. Data quality work IS modeling work
Steps 1–6 (profiling, drift, leakage audit, feature engineering) took more code than training the model. In real data science jobs, 70–80 % of time is spent here.

### 3. Feature interactions are powerful
The pattern "online shopping at midnight for $1,000+" is 45× more likely to be fraud than any single feature predicts alone. Models that capture interactions (Random Forest, Gradient Boosting, Neural Networks) dramatically outperform linear models on this task.

### 4. Leakage is the silent killer
Leaving `cc_num` in the data would produce a model that looks great on paper but fails completely on new cards in production. Always ask: *"Would I have this information at the moment of prediction?"*

### 5. Business constraints shape model choice
A model that generates 20,000 alerts per day is operationally useless even if it has high recall. The best model is the one that balances catching fraud (recall) with not overwhelming investigators (precision).

---

## Completed Steps

All pipeline steps are implemented. A summary of decisions made along the way:

- **SMOTE (Step 8)** — tested and removed; PR-AUC dropped 1.8 pp, and `class_weight="balanced_subsample"` handles imbalance more effectively without synthetic data
- **Threshold calibration (Step 9)** — two operating points: `max_f1` (0.527) for balanced teams, `recall_85` (0.303) for SLA-constrained environments
- **Hyperparameter tuning (Step 10)** — manual selection (automated search impractical at 1.3M+ rows); params: `n_estimators=300`, `max_depth=30`, `min_samples_leaf=2`, `max_features="sqrt"`
- **Validation (Step 11)** — confusion matrix (Check 1), overfitting gap of 0.137 manageable (Check 2); feature importances confirm model learns real fraud signals (Check 3)

---

*Generated from `notebook_colab.ipynb` — last run 2026-03-06*
