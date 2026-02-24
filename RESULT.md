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
| 7 | Trained two baseline models | Random Forest crushes Logistic Regression on this task |

---

## Step 0 — Environment Check

### What the cell printed
```
pandas     2.2.3
numpy      2.1.3
sklearn    1.6.1
tensorflow 2.20.0
seaborn    0.13.2
```

### What it means
Before any data work, we verified every library is installed and at a known version.
**Think of it like checking your ingredients before cooking** — if a library is missing or outdated, every cell after this would crash with confusing errors.
Printing versions also makes results *reproducible*: if your teammate gets different numbers, they know exactly which version to install.

---

## Step 1 — Data Profiling

### What the cell printed
```
Train shape: (1,296,675, 22)    Test shape: (555,719, 22)
Fraud rate (train): 0.5789 %    Fraud rate (test): 0.3860 %
No null values in either split
Temporal split check: max(train) < min(test)  ✓
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

### What the cell printed
```
Train: 1,289,169 non-fraud  |  7,506 fraud  →  ratio 171.75 : 1
Test:    553,574 non-fraud  |  2,145 fraud  →  ratio 258.08 : 1
Fraud-rate shift (test − train): −0.19 percentage points
```

### What it means

**171 legitimate transactions for every 1 fraud transaction.**
This is the single biggest challenge in fraud detection, and it trips up many beginners.

Imagine training a model and telling it: "Out of 1.3 million examples, 99.4 % are *not* fraud." A lazy model that **always** predicts "not fraud" would be *99.4 % accurate* — and completely useless. This is why accuracy is a bad metric here; we use **ROC-AUC** and **Precision-Recall AUC** instead (see Step 7).

**The fraud rate drops slightly in the test set (0.58 % → 0.39 %).** This is a small shift, not a problem — it reflects normal seasonal variation in the simulated data.

**What we do about imbalance:** Later in the pipeline, we use a technique called **SMOTE** (Synthetic Minority Over-sampling Technique), which generates synthetic fraud examples so the model sees enough fraud cases to learn the pattern.

---

## Step 3 — Data Drift Analysis

### What the cell printed
```
unix_time PSI = 11.51   (high — expected, time advances)
All other numeric features: PSI < 0.001  (negligible)
Categorical shifts (category, gender, state, job): abs_delta < 0.001
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

### What the cell printed
```
drop_now:              ['trans_num', 'cc_num', 'street', 'last', 'first']
engineer_then_drop:    ['unix_time', 'trans_date_trans_time']
keep_with_monitor:     ['merch_long', 'merch_lat', 'zip', 'dob', 'city', 'job', 'state']
keep:                  ['amt', 'long', 'lat', 'city_pop', 'merchant', 'category', 'gender']

cc_num overlap (train ∩ test): 908 cards
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

### What the cell printed
```
Dropped columns: ['cc_num', 'first', 'last', 'street', 'trans_num', 'zip', 'city', 'state']
Engineered features: ['hour', 'day_of_week', 'month', 'age', 'distance_km']
Kept raw features: ['merchant', 'category', 'amt', 'gender', 'city_pop', 'job']
Encoding: LabelEncoder (fit on train only; unseen → -1)
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

### What the cell printed

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Alerts | Alert Rate |
|-------|---------|--------|-----------|--------|----|--------|-----------|
| **Random Forest** (balanced) | **0.9877** | **0.8835** | **0.9576** | **0.7268** | **0.8264** | 1,628 | 0.29 % |
| Logistic Regression (balanced) | 0.8513 | 0.1452 | 0.0758 | 0.7413 | 0.1376 | 20,965 | 3.77 % |

**Selected candidate for tuning: Random Forest**

### What it means — understanding the metrics

**ROC-AUC (0.99 for RF):**
"If I pick one fraud and one legitimate transaction at random, how often does the model rank the fraud as riskier?" — 0.99 means it gets this right 99 % of the time. A score of 0.5 would be random guessing.

**PR-AUC (0.88 for RF vs 0.15 for LR):**
This is the harder, more important metric for imbalanced data. It measures the tradeoff between:
- **Precision:** Of all the transactions I flagged as fraud, how many actually were? (Avoid crying wolf)
- **Recall:** Of all actual frauds, how many did I catch? (Don't miss criminals)

A PR-AUC of 0.88 is excellent. Logistic Regression's 0.15 is barely better than random on this imbalanced dataset.

**Alert counts — the business reality:**
- Random Forest fires **1,628 alerts** on 555 K test transactions — a 0.29 % alert rate. Bank investigators can feasibly review this.
- Logistic Regression fires **20,965 alerts** — investigators would be overwhelmed, and each alert costs money.

### Why Random Forest wins

Random Forest builds hundreds of decision trees, each learning different combinations of features. It naturally captures the interaction patterns discovered in Step 4 (e.g., "online + late night + high amount = fraud"). Logistic Regression is a linear model — it can only combine features additively and struggles with these nonlinear, high-lift interactions.

**Visual summary of the gap:**

```
PR-AUC (higher = better at finding fraud precisely)

Random Forest   ██████████████████████████████████████████  0.88
Logistic Reg.   ███████                                     0.15
```

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

## What Comes Next (Planned Steps)

- **SMOTE oversampling** on the training set to balance classes before retraining
- **Hyperparameter tuning** of the Random Forest (number of trees, depth, class weights)
- **Gradient Boosting / XGBoost** as a second strong baseline
- **Neural Network** (TensorFlow/Keras) for comparison
- **Threshold calibration** — moving the decision boundary to optimise the precision/recall tradeoff for a specific business requirement

---

*Generated from `notebook.ipynb` — last run 2026-02-24*
