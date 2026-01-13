# Data Interpretation Decision Tree for Applied Data Analysis (ADA)

> **Purpose**: A systematic, step-by-step guide for interpreting any data analysis task within the ADA course scope. Follow these questions in order to determine the appropriate approach.

---

## PHASE 0: BEFORE YOU TOUCH THE DATA

### Q0.1: What is my analysis goal?
- **Descriptive**: Summarize/understand the data → Go to Phase 1
- **Predictive**: Build a model to predict outcomes → Go to Phase 3 (Supervised Learning)
- **Causal**: Determine if X *causes* Y → Go to Phase 4 (Causal Analysis)
- **Exploratory**: Discover patterns/structure → Go to Phase 5 (Unsupervised Learning)

### Q0.2: What type of study generated this data?
| Study Type | Key Feature | Causal Claims? |
|------------|-------------|----------------|
| **Randomized Controlled Trial (RCT)** | Random assignment to treatment/control | ✅ Yes (gold standard) |
| **Observational Study** | No randomization; groups differ naturally | ⚠️ Only with careful analysis (matching, propensity scores) |
| **Survey/Cross-sectional** | Snapshot at one time point | ❌ No (correlation only) |

**Source**: Lecture 3 (Observational Studies), Lecture 6

---

## PHASE 1: DATA UNDERSTANDING & CLEANING

### Q1.1: What are my variable types?

| Variable Type | Examples | Appropriate Statistics | Appropriate Visualizations |
|---------------|----------|------------------------|---------------------------|
| **Continuous** | Age, income, temperature | Mean, std, median, IQR | Histogram, boxplot, KDE |
| **Categorical (Nominal)** | Gender, country, color | Mode, frequencies | Bar plot, pie chart |
| **Categorical (Ordinal)** | Education level, ratings | Median, percentiles | Bar plot (ordered) |
| **Binary** | Yes/No, 0/1 | Proportion, count | Bar plot |

**Source**: Lecture 11, Describing_data_solution.ipynb

### Q1.2: Do I have missing data?
```
IF missing data exists:
  → Ask: Is it Missing Completely at Random (MCAR), Missing at Random (MAR), 
         or Not Missing at Random (NMAR)?
  → Options: Drop rows, impute (mean/median/mode), or flag as separate category
  → ALWAYS report how you handled it
```
**Source**: Lecture 1

### Q1.3: Do I have outliers?
- **Detection methods**: Boxplot (points beyond whiskers), Z-scores (|z| > 3), IQR method
- **Decision**: Are they errors or genuine extreme values?
  - If errors → Remove or correct
  - If genuine → Consider robust statistics (median instead of mean) or transformation

### Q1.4: What is my sample size?
| Sample Size | Statistical Test Considerations |
|-------------|--------------------------------|
| Small (n < 30) | Non-parametric tests preferred; CLT may not apply |
| Medium (30 ≤ n < 100) | Parametric tests generally OK if approximately normal |
| Large (n ≥ 100) | Central Limit Theorem applies; parametric methods robust |

**Source**: Lecture 11

---

## PHASE 2: DESCRIBING & COMPARING DATA

### Q2.1: Am I describing ONE variable?

**For continuous variables:**
- **Central tendency**: Mean (sensitive to outliers) vs. Median (robust)
- **Spread**: Standard deviation (with mean) vs. IQR (with median)
- **Shape**: Skewness (use histogram/KDE to visualize)
- **Always report**: n, measure of center, measure of spread, and visualize!

**For categorical variables:**
- Report frequencies/proportions
- Visualize with bar plots

**Source**: Lecture 11, useful_functions.pdf

### Q2.2: Am I comparing TWO or more groups?

| Comparison Type | Appropriate Test | When to Use |
|-----------------|------------------|-------------|
| 2 groups, continuous outcome | t-test (parametric) | Normal data, equal variance |
| 2 groups, continuous outcome | Mann-Whitney U (non-parametric) | Non-normal or ordinal data |
| 2+ groups, continuous outcome | ANOVA / Kruskal-Wallis | Comparing >2 group means |
| 2 categorical variables | Chi-square test | Testing independence |

**Key Question**: Are my samples independent or paired?
- **Independent**: Different subjects in each group
- **Paired**: Same subjects measured twice (use paired t-test)

### Q2.3: Am I examining relationships between variables?

| Variable Types | Method | Measures |
|----------------|--------|----------|
| Both continuous | Scatter plot + Correlation | Pearson r (linear), Spearman ρ (monotonic) |
| One continuous, one categorical | Boxplots by group | Group means comparison |
| Both categorical | Contingency table | Chi-square, Cramér's V |

**⚠️ CRITICAL WARNING**: Correlation ≠ Causation!

**Source**: Lecture 11 ("Correlation != causation - even trickier with >2 variables")

### Q2.4: Do I need to quantify uncertainty?

**Two approaches (from course):**

1. **Hypothesis Testing**:
   - H₀ (null hypothesis): No effect/no difference
   - Calculate test statistic → p-value
   - If p < α (typically 0.05): reject H₀
   - **Caution**: p = 0.05 means 1 in 20 chance of false positive!

2. **Confidence Intervals** (PREFERRED per course):
   - 95% CI: Range where true parameter likely lies
   - Interpretation: "If we repeated this experiment many times, 95% of CIs would contain the true value"
   - **Parametric**: Assume distribution (typically Normal)
   - **Non-parametric (Bootstrap)**: No distributional assumptions

**Source**: Lecture 11 ("Confidence intervals preferred!")

### Q2.5: Am I doing multiple comparisons?

```
IF testing multiple hypotheses simultaneously:
  → Apply correction (Bonferroni: α_adjusted = α / number_of_tests)
  → Without correction: high false positive rate!
```

**Source**: Lecture 11 ("Careful when performing multiple tests - apply correction")

---

## PHASE 3: SUPERVISED LEARNING (PREDICTION)

### Q3.1: What is my outcome variable type?

| Outcome Type | Task | Models to Consider |
|--------------|------|-------------------|
| Continuous | Regression | Linear Regression, Ridge, Lasso |
| Binary | Classification | Logistic Regression, Decision Trees, Random Forest, kNN |
| Multi-class | Classification | Same as binary (extended) |

**Source**: Lecture 4, Lecture 5

### Q3.2: Linear Regression Interpretation

**Model**: y = β₀ + β₁X₁ + β₂X₂ + ... + ε

| Term | Interpretation |
|------|---------------|
| β₀ (intercept) | Expected y when all X = 0 |
| β₁ (coefficient) | Change in y for 1-unit increase in X₁, *holding other variables constant* |
| R² | Fraction of variance explained (0 to 1) |
| p-value for βᵢ | Probability of seeing this coefficient if true βᵢ = 0 |

**Assumptions to check**:
1. **Validity**: Outcome reflects phenomenon; model includes relevant predictors
2. **Linearity**: Linear in predictors (can use transformations: log, polynomial)
3. **Independence of errors**: No autocorrelation
4. **Homoscedasticity**: Equal variance of errors
5. **Normality of errors**: Less important for large samples

**Source**: Lecture 2, Regression_analysis_solution.ipynb

### Q3.3: Logistic Regression Interpretation

**Model**: logit(p) = log(p/(1-p)) = β₀ + β₁X₁ + ...

| Term | Interpretation |
|------|---------------|
| βᵢ | Change in log-odds for 1-unit increase in Xᵢ |
| exp(βᵢ) | **Odds ratio**: multiplier to odds for 1-unit increase in Xᵢ |
| exp(βᵢ) ≈ 1 + βᵢ | For small βᵢ: approximate percentage change in odds |

**Example from Mock MCQ**: If logit(p) = -3 + 0.02×Age + 0.5×Smoker
- Aging 1 year: odds multiply by exp(0.02) ≈ 1.02 → **~2% increase**
- Being smoker: odds multiply by exp(0.5) ≈ 1.65 → ~65% increase

**Source**: Lecture 4, ADA_Mock_MCQs

### Q3.4: How do I evaluate my model?

**For Regression:**
- MSE (Mean Squared Error): Average squared prediction error
- R²: Variance explained

**For Classification:**

| Metric | Formula | When to Use |
|--------|---------|-------------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Balanced classes, equal error costs |
| **Precision** | TP/(TP+FP) | Cost of false positives is high |
| **Recall (Sensitivity)** | TP/(TP+FN) | Cost of false negatives is high |
| **F1 Score** | 2×(Prec×Rec)/(Prec+Rec) | Balance precision/recall |
| **AUC-ROC** | Area under ROC curve | Overall discriminative ability |

**⚠️ When NOT to use accuracy**:
- Imbalanced classes (e.g., 99% negative, 1% positive)
- Different costs for FP vs FN (e.g., fraud detection, medical screening)

**Source**: Lecture 5, ADA_Mock_MCQs, SupervisedLearning_solution.ipynb

### Q3.5: How do I prevent overfitting?

**The Bias-Variance Tradeoff:**
- **High bias** (underfitting): Model too simple, misses patterns
- **High variance** (overfitting): Model too complex, fits noise

| Model Complexity | Bias | Variance | Training Error | Validation Error |
|------------------|------|----------|----------------|------------------|
| Too simple | High | Low | High | High |
| Just right | Medium | Medium | Medium | Medium (≈ training) |
| Too complex | Low | High | Low | High (>> training) |

**Solutions:**
1. **Cross-validation**: Split data into k folds, rotate validation set
   - Use for model selection and hyperparameter tuning
   - Report performance on held-out test set
2. **Regularization**: 
   - Ridge (L2): Penalizes large coefficients
   - Lasso (L1): Can zero out coefficients (feature selection)
3. **Pruning** (for trees): Remove branches that don't improve validation
4. **More data**: Helps high-variance models

**Source**: Lecture 4, Lecture 5

### Q3.6: Decision Tree for Model Selection

```
Start with baseline model (e.g., predicting mean/mode)
   ↓
Try simple models first (linear/logistic regression)
   ↓
IF simple model performs poorly:
   → Consider more flexible models (trees, random forests)
   ↓
Use cross-validation to compare models
   ↓
Select model with best validation performance
   ↓
Report final performance on TEST set (never used before!)
```

**Source**: Lecture 5 ("Data collection → Model selection → Model assessment")

---

## PHASE 4: CAUSAL ANALYSIS (OBSERVATIONAL DATA)

### Q4.1: Can I make causal claims?

**Decision tree:**
```
Was treatment randomly assigned?
├── YES (RCT) → Causal claims supported
└── NO (Observational) → CAUTION! Confounders may exist
    ↓
    Apply matching/propensity score methods
    ↓
    Still cannot eliminate unobserved confounders
```

### Q4.2: What is a confounder?

A **confounder** is a variable that:
1. Affects the probability of receiving treatment, AND
2. Affects the outcome

**Example**: Comparing hospital mortality rates without accounting for patient severity (sicker patients go to specialized hospitals → hospitals appear worse)

**NOT a confounder**:
- Variable that only affects treatment (not outcome)
- Variable that only affects outcome (not treatment)
- Variable measured AFTER treatment (mediator)

**Source**: Lecture 3, ADA_Mock_MCQs

### Q4.3: How do I use Propensity Score Matching?

**Steps:**
1. **Estimate propensity scores**: Use logistic regression
   - Outcome: Treatment (0/1)
   - Predictors: All pre-treatment covariates
   - Propensity score = P(treatment | covariates)

2. **Match treated to control**: 
   - Each treated unit paired with control unit with similar propensity score
   - Methods: Nearest neighbor, optimal matching (Hungarian algorithm)

3. **Check balance**: 
   - After matching, covariate distributions should be similar in both groups
   - If not balanced → matching failed

4. **Compare outcomes**: 
   - Now valid to compare treated vs. control outcomes

**Key insight from Mock MCQ**: "Two subjects with equal propensity scores will have *similar distribution of observed covariates*" (not identical covariates, not identical outcomes)

**⚠️ Important**: No train/test split for propensity scores! Goal is balance, not prediction.

**Source**: Lecture 3, Observational_studies_solution.ipynb

### Q4.4: Interpreting results from observational studies

**Always acknowledge limitations:**
1. Can only control for *observed* confounders
2. Unobserved confounders may still bias results
3. Use sensitivity analysis: "How strong would an unobserved confounder need to be to explain away our findings?"

**Source**: Lecture 3 (Smoking and lung cancer example)

---

## PHASE 5: UNSUPERVISED LEARNING

### Q5.1: What is my unsupervised goal?

| Goal | Methods | Output |
|------|---------|--------|
| **Clustering**: Group similar observations | K-means, DBSCAN, Hierarchical | Discrete cluster labels |
| **Dimensionality Reduction**: Simplify high-D data | PCA, t-SNE | Continuous lower-D representation |

**Source**: Lecture 6

### Q5.2: K-Means Clustering

**How it works:**
1. Initialize k centroids randomly
2. Assign each point to nearest centroid
3. Update centroids to mean of assigned points
4. Repeat until convergence

**Choosing k:**
- **Silhouette score**: Maximize (higher = better separation)
  - s(i) = (b(i) - a(i)) / max(a(i), b(i))
  - a(i) = avg distance to own cluster
  - b(i) = avg distance to nearest other cluster
- **Elbow method**: Plot SSE vs. k, find "elbow"

**Limitations of K-means:**
- Only finds **convex** (spherical) clusters
- Sensitive to initialization
- Must specify k in advance

**Source**: Lecture 6, Unsupervised_Learning_solutions.ipynb

### Q5.3: When to use DBSCAN instead?

Use DBSCAN when:
- Clusters have non-convex shapes (e.g., crescents)
- You don't know the number of clusters
- There are outliers (DBSCAN marks them as noise)

**Source**: Unsupervised_Learning_solutions.ipynb

### Q5.4: Dimensionality Reduction

**PCA (Principal Component Analysis):**
- Finds directions of maximum variance
- Linear transformation
- Good for: visualization, preprocessing, noise reduction

**t-SNE:**
- Non-linear; preserves local structure
- Better for visualization of clusters
- Not deterministic (depends on random initialization)

**Source**: Lecture 10, Unsupervised_Learning_solutions.ipynb

---

## PHASE 6: INTERPRETATION CHECKLIST

### Before reporting results, verify:

- [ ] **Sample size**: Is n large enough for my analysis?
- [ ] **Assumptions checked**: Distribution, independence, homoscedasticity?
- [ ] **Uncertainty quantified**: CI or p-value reported?
- [ ] **Effect size considered**: Statistical significance ≠ practical importance
- [ ] **Multiple comparisons**: Correction applied if needed?
- [ ] **Causation claims**: Only if RCT or proper observational study methods?
- [ ] **Limitations acknowledged**: What could bias these results?

---

## QUICK REFERENCE: KEY FORMULAS

### Statistics
| Concept | Formula |
|---------|---------|
| Mean | x̄ = Σxᵢ/n |
| Variance | s² = Σ(xᵢ - x̄)²/(n-1) |
| Pearson correlation | r = Σ(xᵢ - x̄)(yᵢ - ȳ) / √[Σ(xᵢ - x̄)² × Σ(yᵢ - ȳ)²] |
| Odds | odds = p / (1-p) |
| Log odds | logit(p) = log(p / (1-p)) |
| Probability from odds | p = odds / (1 + odds) |

### Classification Metrics
| Metric | Formula |
|--------|---------|
| Accuracy | (TP + TN) / N |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1 | 2 × (Precision × Recall) / (Precision + Recall) |

### Regression
| Concept | Interpretation |
|---------|---------------|
| Coefficient βᵢ in linear reg | Δy for 1-unit Δxᵢ |
| Coefficient βᵢ in log reg | Δlog-odds for 1-unit Δxᵢ |
| exp(βᵢ) in log reg | Odds ratio |
| R² | Proportion of variance explained |

---

## COMMON EXAM TRAPS (from Mock MCQs and Course Materials)

1. **Correlation vs. Causation**: Never claim causation without proper study design
2. **p-value misinterpretation**: p=0.03 does NOT mean "3% chance null is true"
3. **Accuracy on imbalanced data**: 95% accuracy meaningless if 95% is one class
4. **Propensity score matching**: Creates balance in *observed* covariates only
5. **Cross-validation for propensity scores**: NOT needed (goal isn't prediction)
6. **K-means on non-convex clusters**: Will fail; use DBSCAN
7. **Forgetting interaction terms**: y ~ X₁ + X₂ misses X₁×X₂ effects
8. **Confusing coefficient interpretation**: In multiple regression, coefficients are "holding other variables constant"
9. **Logistic regression coefficients**: Remember exp(β) gives odds ratio, not probability
10. **Mutual information vs. Pearson**: MI captures non-linear relationships; Pearson only linear

---

## VISUALIZATION SELECTION GUIDE

| What to show | Variable types | Recommended plot |
|--------------|---------------|------------------|
| Distribution of 1 variable | Continuous | Histogram, KDE, Boxplot |
| Distribution of 1 variable | Categorical | Bar plot |
| Relationship of 2 variables | Both continuous | Scatter plot |
| Relationship of 2 variables | 1 cont., 1 cat. | Boxplots by group |
| Relationship of 2 variables | Both categorical | Heatmap, stacked bar |
| Many variables | Mixed | Pairplot, scatter matrix |
| High-dimensional | Any | PCA/t-SNE reduced to 2D |

**Source**: Lecture 10, Becoming_a_DataVizard_solution.ipynb

---

*Sources: ADA Lectures 1-11, Exercise notebooks (Regression, Supervised Learning, Unsupervised Learning, Observational Studies, Describing Data, Data Visualization), Mock MCQs, useful_functions.pdf*