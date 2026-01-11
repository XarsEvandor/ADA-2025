# 🎯 ADA MCQ EXAM - COMPREHENSIVE STUDY NOTES
## Verified Against 2025 Mock MCQs, Past Exams & Course Materials

---

# 📊 PART 1: REGRESSION & STATISTICS

## 1.1 Linear Regression (OLS)

**Model:** `y = β₀ + β₁X₁ + β₂X₂ + ... + ε`

**Key Concepts:**
- **R²** = fraction of variance explained (0 to 1)
  - R² = 1 - (variance of residuals / variance of y)
  - Higher R² = model explains more variance
- **Residuals** = actual - predicted
  - Mean of residuals = 0 (always!)
- **p-value** = probability of seeing coefficient if true value were 0
- **Coefficient interpretation:** "1-unit increase in X associated with β change in Y"

**Formula Syntax:**
- `C(variable)` = categorical (creates dummy variables)
- `a * b` = `a + b + a:b` (main effects + interaction)
- `a : b` = interaction term only

## 1.2 Logistic Regression ⭐ VERY HIGH PRIORITY

**Purpose:** Predicts probability of binary outcome

**The Chain:** `Probability p ↔ Odds p/(1-p) ↔ Log-odds log[p/(1-p)]`

**Model:** `log[p/(1-p)] = β₀ + β₁X₁ + β₂X₂`

### Coefficient Interpretation (EXAM TESTED - BE PRECISE!)

```
β = change in LOG-ODDS per unit increase in X
To convert to ODDS multiplier: e^β

EXAMPLE FROM 2025 MOCK EXAM:
logit(p) = -3 + 0.02×Age + 0.5×Smoker

✅ "Aging by 1 year increases ODDS by ~2%" (e^0.02 ≈ 1.02)
❌ "Odds for smokers is 3× non-smoker" (FALSE: e^0.5 ≈ 1.65)
❌ "Log-odds increase 50% for smokers" (FALSE: increases BY 0.5, not 50%)
❌ "Log-odds is non-linear function" (FALSE: it's linear!)
```

**Quick approximation:** When β is small, e^β ≈ 1 + β

**Loss function:** Cross-entropy (NOT squared error)

**Connection:** Logistic regression = neural network with no hidden layers + cross-entropy loss ✅

## 1.3 Log Transformations

| Transformation | Interpretation |
|----------------|----------------|
| log(y) ~ X | Multiplicative model; +1 in X → +β% in outcome |
| y ~ log(X) | 1% increase in X → β/100 change in y |

**Log-log plot** (FROM MOCK EXAM):
- Used for power law distributions
- **Linearizes** the relationship (slope = exponent)
- ❌ NOT for normalization
- ❌ NOT for removing outliers
- ❌ NOT to make it bell-shaped

## 1.4 Hypothesis Testing ⭐ HIGH PRIORITY

**P-value:** Probability of seeing data this extreme IF null hypothesis is true

**Common Misconceptions (from lecture):**

| Statement | TRUE/FALSE |
|-----------|------------|
| p=0.05 means null has 5% chance of being true | ❌ FALSE |
| p≥0.05 means no difference between groups | ❌ FALSE |
| Statistically significant = clinically important | ❌ FALSE |
| Studies with p on opposite sides of 0.05 are conflicting | ❌ FALSE |
| p=0.05 means 5% Type I error rate if rejecting | ✅ TRUE |

**Multiple hypothesis testing:** Use α/n (Bonferroni correction)

## 1.5 Confidence Intervals

- 95% CI = range where true value lies with 95% confidence
- CI contains 0 → effect NOT significant at 5% level
- Overlapping CIs → difference may not be significant

---

# 🔬 PART 2: OBSERVATIONAL STUDIES & CAUSALITY

## 2.1 Experiments vs Observational Studies ⭐ HIGH PRIORITY

| Randomized Experiment | Observational Study |
|----------------------|---------------------|
| Researcher controls treatment | Subjects self-select |
| Random assignment balances confounders | Confounders problematic |
| Gold standard for causality | Correlation ≠ Causation |
| Expensive, ethical issues | Cheaper, more feasible |

**Hierarchy:** Randomized experiment > Natural experiment > Matched observational study

## 2.2 Confounders ⭐ TESTED VERBATIM IN MOCK EXAM

**Definition:** Variable that affects BOTH:
1. ✅ The treatment assignment AND
2. ✅ The outcome

**NOT a confounder:**
- ❌ Variable that only affects treatment (not outcome)
- ❌ Variable that only affects outcome (not treatment)
- ❌ Variable measured AFTER treatment

**Memory trick:** "Common Cause" - must cause BOTH things

**Example:** "Motivation to quit smoking" confounds cessation studies (affects who tries to quit AND health outcomes)

## 2.3 Propensity Score Matching ⭐ TESTED IN MOCK EXAM

**Definition:** Probability of receiving treatment given observed covariates

**Two subjects with EQUAL propensity scores have:**
- ✅ Similar distribution of OBSERVED covariates
- ❌ NOT identical covariates
- ❌ NOT identical outcomes  
- ❌ NOT identical treatment effects

**Critical Limitation:** Cannot control for UNOBSERVED confounders

**Process:**
1. Estimate propensity scores (logistic regression)
2. Match treated/control with similar scores (e.g., bipartite matching)
3. Compare outcomes between matched pairs
4. Verify balance was achieved

## 2.4 Sensitivity Analysis

**Purpose:** Quantify how wrong the naive model can be without changing conclusions

**Γ (Gamma):**
- Γ = 1: naive model is true (no hidden bias)
- Γ = 2: identical-looking subjects could have 2× different treatment odds
- Higher Γ needed to invalidate → more robust conclusion

**Example:** Smoking-cancer link requires Γ > 6 to invalidate → very robust

---

# 🤖 PART 3: MACHINE LEARNING - SUPERVISED

## 3.1 Supervised vs Unsupervised

| Supervised | Unsupervised |
|-----------|--------------|
| Has labeled data (X, y) | Only data X |
| Classification (discrete y) | Clustering |
| Regression (continuous y) | Dimensionality reduction |

## 3.2 Bias-Variance Tradeoff ⭐ VERY HIGH PRIORITY

**Fundamental equation:** `Error² = Bias² + Variance`

| High Bias (Underfitting) | High Variance (Overfitting) |
|--------------------------|----------------------------|
| Model too simple | Model too complex |
| Can't capture patterns | Memorizes training data |
| Training error HIGH | Training error LOW |
| Train ≈ Test (both bad) | Train << Test (gap!) |
| More data WON'T help | More data WILL help |

**Complexity Controls:**

| Parameter | ↑ Complexity (↓Bias, ↑Variance) | ↓ Complexity (↑Bias, ↓Variance) |
|-----------|--------------------------------|--------------------------------|
| k in k-NN | Smaller k | Larger k |
| Tree depth | Deeper | Shallower |
| # Features | More | Fewer |
| Regularization | Less (lower λ) | More (higher λ) |

**Solutions for overfitting:**
- Regularization (Ridge, Lasso)
- Feature selection
- Cross-validation for model selection
- More training data
- Ensemble methods

## 3.3 k-Nearest Neighbors (k-NN)

**Properties:**
- No training time (lazy learner) - stores all data
- Can handle complex, non-linear decision boundaries
- Doesn't need retraining when new data added
- Small k → overfitting; Large k → underfitting

**Choose k:** Cross-validation

## 3.4 Decision Trees

**Construction:** Greedy top-down divide-and-conquer
- Split on most "discriminative" attribute
- **Information gain** (ID3, C4.5) or **Gini impurity** (CART)

**Stopping:** All samples same class OR no attributes left

**Overfitting:** Deep trees overfit → use **pruning**
- Pre-pruning: stop early (min samples per leaf)
- Post-pruning: build full tree, then remove nodes if validation accuracy doesn't decrease

**Bias-Variance:** Deeper tree = lower bias, higher variance

## 3.5 Random Forests ⭐ HIGH PRIORITY

**Ensemble of decision trees using:**
1. **Bootstrap sampling** (different data per tree)
2. **Random feature selection** at each node (typically √p features)

**Why it works:** Diversity between trees reduces variance

| Random Forest | Boosted Trees (e.g., XGBoost) |
|---------------|-------------------------------|
| Parallel training | Sequential training |
| Deeper trees (10-30) | Shallower trees (4-8) |
| Reduces VARIANCE | Reduces BIAS |
| Less prone to overfitting | Can overfit if not tuned |

## 3.6 Cross-Validation ⭐ HIGH PRIORITY (EXAM TRAPS!)

**Purpose:** Model selection, hyperparameter tuning

**k-fold CV:**
1. Split data into k parts
2. Train on k-1, validate on 1
3. Repeat k times, average results

**TRUE statements:**
- ✅ Useful when not enough data for train/val/test split
- ✅ Used to select hyperparameters

**FALSE statements (EXAM TRAPS):**
- ❌ Guaranteed to prevent overfitting
- ❌ Reporting CV performance used for tuning = unbiased estimate (IT'S BIASED!)

**Leave-one-out:** k = n (computationally expensive)

---

# 📏 PART 4: CLASSIFICATION METRICS ⭐ VERY HIGH PRIORITY

## 4.1 Confusion Matrix

```
                 ACTUAL
              Pos     Neg
PREDICTED  ┌───────┬───────┐
    Pos    │  TP   │  FP   │ ← Precision = TP/(TP+FP)
           ├───────┼───────┤
    Neg    │  FN   │  TN   │
           └───────┴───────┘
              ↑
        Recall = TP/(TP+FN)
```

## 4.2 Key Metrics - MEMORIZE

| Metric | Formula | Question Answered |
|--------|---------|-------------------|
| **Precision** | TP/(TP+FP) | Of predicted positives, how many correct? |
| **Recall** (Sensitivity, TPR) | TP/(TP+FN) | Of actual positives, how many found? |
| **Accuracy** | (TP+TN)/Total | What fraction correct overall? |
| **F1 Score** | 2PR/(P+R) | Harmonic mean of P and R |
| **FPR** | FP/(FP+TN) | Of actual negatives, how many wrongly flagged? |
| **Specificity** | TN/(TN+FP) | Of actual negatives, how many correctly identified? |

## 4.3 When to Use What ⭐ EXAM FAVORITE

### When Accuracy FAILS (FROM MOCK EXAM):

```
Example: Rare disease (1% prevalence), 1000 patients
Predictions: 50 positive, 950 negative
TP=5, FP=45, FN=5, TN=945

Accuracy = 950/1000 = 95% 👍 looks great!
Precision = 5/50 = 10% 😱 terrible!
Recall = 5/10 = 50%

Correct interpretation: "Accuracy is high but precision is low; 
risky if false alarms are costly"
```

**Accuracy is BAD when:**
- ✅ Classes are imbalanced
- ✅ Costs of FP ≠ FN
- ✅ Detecting rare events (fraud, disease)

**Use instead:**
- F1 Score (imbalanced classes)
- Precision (FP costly - e.g., spam filter)
- Recall (FN costly - e.g., disease screening, cancer detection)

## 4.4 ROC Curve ⭐ HIGH PRIORITY

- **Y-axis:** True Positive Rate (Recall)
- **X-axis:** False Positive Rate
- Created by varying classification threshold

**AUC (Area Under Curve):**
- AUC = 0.5 → random classifier (diagonal line)
- AUC = 1.0 → perfect classifier
- Higher = Better

---

# 📝 PART 5: TEXT & NLP

## 5.1 Text Preprocessing Pipeline

1. **Tokenization:** Split text into words
2. **Stopword removal:** Remove common words (the, is, at)
3. **Lemmatization:** Map to normalized lexicon entries (better → good)
4. **Stemming:** Chop off suffixes (running → run) - loses POS info!
5. **Case folding:** Convert to lowercase

**From Quiz:**
- Lemmatization maps to normalized lexicon entries ✅
- Stemming reduces sparsity but loses part-of-speech info ✅
- Tokenization harder for Chinese, German (no explicit delimiters) ✅

## 5.2 Bag of Words

**Definition:** Multiset of words (keeps count, ignores order)

**Example:** "what you see is what you get" → {what:2, you:2, see:1, is:1, get:1}

**Properties:**
- High dimensional (vocabulary size)
- Very sparse
- Loses word order and syntax

## 5.3 TF-IDF ⭐ VERY HIGH PRIORITY

**TF-IDF(d,w) = TF(d,w) × IDF(w)**

- **TF:** How often word appears in document
- **IDF:** log(total docs / docs containing word)
  - Rare words → high IDF
  - Common words → low IDF (down-weighted)

**Matrix structure:**
- Rows = documents
- Columns = words

### Cosine Similarity (EXAM TESTED)

**From 2025 Mock Exam:**
```
Doc 1: "The weather today is sunny"
Doc 2: "It rained yesterday"  
Doc 3: "I am going shopping today"

After preprocessing:
- Docs 1 & 2 share NO words → cosine = 0 ✅
- Docs 2 & 3 share NO words → cosine = 0 ✅
- Docs 1 & 3 share "today" → cosine ≠ 0 ✅
```

**Key insight:** No shared words = cosine similarity = 0

**Sparsity problem:** Most document pairs share no words → cosine = 0

## 5.4 Topic Modeling

### LSA (Latent Semantic Analysis)

- Matrix factorization: TF-IDF ≈ A × B
- Reduces dimensionality (solves sparsity)
- Topics = vectors (NOT probability distributions!)

### LDA (Latent Dirichlet Allocation) ⭐ KNOW THE DIFFERENCE

- **Probabilistic** model
- Topic = probability distribution over words
- Document = probability distribution over topics
- **Unsupervised** (K = number of topics is input)
- Uses maximum likelihood

**EXAM TRAP:** "LSA's document representation is a probability distribution" → FALSE! (That's LDA)

## 5.5 Word Embeddings (Word2Vec)

- ✅ Lower dimensional than bag of words
- ✅ Similar words have similar vectors
- ✅ Can compute sentence vectors (average word vectors)
- ❌ Dimensions do NOT have clear interpretation

---

# 🔗 PART 6: NETWORKS ⭐ HIGH PRIORITY

## 6.1 Graph Basics

- **Directed vs Undirected**
- **Degree:** Number of edges to node
  - Directed: in-degree + out-degree
- **Bipartite:** Two groups, edges only between groups
  - Sum of degrees in partition 1 = sum in partition 2 ✅
  - Can be directed ✅
  - Projection CAN be complete graph ✅

## 6.2 Centrality Measures ⭐ VERY HIGH PRIORITY

| Measure | What it Measures | Key Insight |
|---------|------------------|-------------|
| **Degree** | # connections | Simple, local |
| **Betweenness** | Shortest paths through node | Bridge/broker detection |
| **Closeness** | Avg distance to all nodes | How "central" |
| **PageRank** | Random walk steady-state | INCOMING edges only! |

### PageRank ⭐ EXAM FAVORITE

**"Luggage randomly traveling between airports - where does it end up?"**
**Answer: PageRank** (steady-state probability of random walk)

**Key facts:**
- Based ONLY on incoming edges
- Removing outgoing edges → PageRank UNCHANGED ✅
- Node with only outgoing edges → LOW PageRank
- Inverting all edges → PageRank CHANGES

### Betweenness Centrality ⭐ CALCULATION TESTED

**From Mock Exam:** Graph with A-B-C and D connected to B
```
All shortest paths between {A,C,D} go through B
Nodes A, C, D are endpoints → betweenness = 0

Answer: [A=0, B=2/3, C=0, D=0]
```

**Key insight:** 
- Endpoints of paths have betweenness = 0
- Can have HIGH betweenness with LOW degree (bridges)

## 6.3 Clustering Coefficient

**Local:** How connected are a node's neighbors?
- 1 = neighbors form complete clique
- 0 = no edges between neighbors

**Global (Transitivity):** Ratio of triangles to connected triples

## 6.4 Community Detection

- **Girvan-Newman:** Remove edges with highest betweenness iteratively
- **Louvain:** Bottom-up, merge communities that improve modularity

---

# ⚡ PART 7: SPARK & BIG DATA ⭐ HIGH PRIORITY

## 7.1 Transformations vs Actions (TESTED!)

| TRANSFORMATIONS (Lazy) | ACTIONS (Execute!) |
|------------------------|-------------------|
| map() | collect() |
| filter() | count() |
| flatMap() | **reduce()** ← TRAP! |
| distinct() | take(n) |
| groupByKey() | first() |
| reduceByKey() | saveAsTextFile() |

**Memory trick:** Actions return values to driver or save to storage

**Lazy evaluation:** Transformations build DAG; Actions execute it

## 7.2 persist() / cache() ⭐ TESTED IN MOCK EXAM

**Problem:**
```python
rdd2 = rdd1.map(f1)
list1 = rdd2.filter(f2).collect()  # map(f1) runs
list2 = rdd2.filter(f3).collect()  # map(f1) runs AGAIN!
```

**Solution:**
```python
rdd2 = rdd1.map(f1)
rdd2.persist()  # ← Add this after the transformation!
list1 = rdd2.filter(f2).collect()
list2 = rdd2.filter(f3).collect()
```

**Exam answer:** "Add rdd2.persist() after the map" ✅

## 7.3 Accumulators ⭐ TESTED IN MOCK EXAM

**Why normal variables fail:**
```python
count = 0
def increment(x):
    count += 1  # ❌ Variable COPIED to workers, not updated!
rdd.foreach(increment)
print(count)  # Still 0!
```

**Solution:**
```python
count = sc.accumulator(0)
def increment(x):
    count.add(1)  # ✅ Updates aggregated back to driver
rdd.foreach(increment)
print(count.value)  # Correct!
```

---

# 🔄 PART 8: CLUSTERING (Unsupervised)

## 8.1 K-Means ⭐ HIGH PRIORITY

**Algorithm:**
1. Initialize K centroids (randomly or K-means++)
2. Assign each point to nearest centroid
3. Recompute centroids as mean of assigned points
4. Repeat until convergence

**Limitations:**
- ❌ Only finds **convex (spherical) clusters**
- ❌ Does NOT output optimal K
- ❌ Does NOT guarantee global optimum (local minimum)
- ❌ Sensitive to initialization

**K-means++:** Better initialization - spread out initial centroids

## 8.2 Choosing K

**Elbow method:** Plot SSE vs K, find the "elbow"

**Silhouette score:**
```
s(i) = (b(i) - a(i)) / max(a(i), b(i))

a(i) = avg distance to points in OWN cluster
b(i) = avg distance to points in NEAREST OTHER cluster

Range: [-1, 1]
Higher = better (well-separated, cohesive clusters)
```

## 8.3 DBSCAN

- **Density-based** clustering
- Can find **non-convex** clusters (unlike K-means)
- Automatically detects **noise/outliers**
- Parameters: ε (radius), MinPts (minimum points)

## 8.4 Hierarchical Clustering

- **Agglomerative (bottom-up):** Start with each point as cluster, merge
- **Divisive (top-down):** Start with one cluster, split
- Produces dendrogram
- Linkage: single, complete, average, Ward's

---

# 📉 PART 9: DIMENSIONALITY REDUCTION

## 9.1 PCA (Principal Component Analysis)

- Finds directions of **maximum variance**
- Principal components are **orthogonal**
- First PC = most variance
- Linear transformation
- Used for visualization, denoising, feature extraction

## 9.2 t-SNE

- Good for **visualization** (2D/3D)
- **Non-linear**
- Preserves **local structure** better than PCA
- Computationally expensive
- Hyperparameter: perplexity

---

# 🚨 TOP 15 EXAM TRAPS (Verified from Mock MCQs)

| # | The Trap | Truth |
|---|----------|-------|
| 1 | "Accuracy is good for imbalanced classes" | ❌ Use F1/Precision/Recall |
| 2 | "Cross-validation prevents overfitting" | ❌ Helps SELECT models only |
| 3 | "CV performance for tuning = unbiased" | ❌ It's BIASED |
| 4 | "PageRank uses outgoing edges" | ❌ Only INCOMING |
| 5 | "Propensity matching handles unobserved confounders" | ❌ Only OBSERVED |
| 6 | "LSA gives probability distributions" | ❌ That's LDA |
| 7 | "Word2vec dimensions are interpretable" | ❌ NOT interpretable |
| 8 | "K-means finds any cluster shape" | ❌ Only convex |
| 9 | "Log-log normalizes data" | ❌ LINEARIZES power laws |
| 10 | "reduce() is a transformation" | ❌ It's an ACTION |
| 11 | "p=0.05 means 5% chance null is true" | ❌ Misinterpretation |
| 12 | "High accuracy = good model" | ❌ Check class balance |
| 13 | "More features always help" | ❌ Overfitting risk |
| 14 | "Smoker odds 3x" from β=0.5 | ❌ e^0.5 ≈ 1.65 |
| 15 | "Equal propensity = identical covariates" | ❌ Similar DISTRIBUTION |

---

# 🧮 ESSENTIAL FORMULAS

```
CLASSIFICATION:
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 × (P × R) / (P + R)
Accuracy = (TP + TN) / Total

CLUSTERING:
Silhouette = (b - a) / max(a, b)

SIMILARITY:
Jaccard(A,B) = |A ∩ B| / |A ∪ B|
Cosine = (A·B) / (||A|| × ||B||)

REGRESSION:
R² = 1 - (variance of residuals / variance of y)

LOGISTIC:
Log-odds = log(p / (1-p))
Odds multiplier from β: e^β
When β small: e^β ≈ 1 + β

TEXT:
IDF(word) = log(total docs / docs containing word)
TF-IDF = TF × IDF
```

---

# 🧠 MEMORY AIDS

**PRECISION vs RECALL:**
- **P**recision = **P**ositive predictive value ("of my predictions...")
- **R**ecall = **R**etrieve all relevant ("of actual positives...")

**Bias-Variance:**
- **S**imple model = High **B**ias (Systematic error)
- **C**omplex model = High **V**ariance (Changes with data)

**Confounders:** "Common Cause" - affects BOTH treatment AND outcome

**PageRank:** "Popularity contest" - based on who links TO you

**K-means:** "Spheres only" - can't find weird shapes

**LSA vs LDA:** "LSA = Linear algebra, LDA = Distributions"

**Spark Actions:** "Things that give you an ANSWER" (collect, count, reduce)

---

# 🎯 EXAM STRATEGY

1. **Read ALL options** before answering
2. **Absolute words** ("always", "never", "guarantees") → usually FALSE
3. **Causation claims** in observational studies → usually FALSE
4. **"High accuracy = good"** → check for imbalanced classes
5. **When stuck:** eliminate obviously wrong answers
6. **PageRank:** "incoming edges only"
7. **Spark:** transformation vs action?
8. **Logistic regression:** odds vs log-odds vs probability

---

*Sources: ADA 2025 Mock MCQs (62.5% score analyzed), Lectures 1-11, Past Exams 2019-2023, Course Exercises, Quiz Solutions*

*Last updated: Study session for exam*