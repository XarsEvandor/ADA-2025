```
TASK: Create a standalone Python script called `exam_blitz.py` that brute-force pre-computes every possible analysis on exam data.

GOAL: Reduce the exam to a lookup task. Instead of coding under pressure, I run this script at exam start and spend remaining time matching questions to pre-computed outputs.

INPUT: 
- Path to a directory containing exam data files (.csv, .jsonl, .graphml, .json, .parquet, .txt)
- Optional: Path to exam .ipynb file for context extraction

OUTPUT: A comprehensive report (printed + saved as HTML/markdown) containing every plausible analysis, organized by category.

---

CONTEXT EXTRACTION (when exam .ipynb provided):
- Parse exam notebook to extract:
  - Data file paths/names mentioned (e.g., "exam1.jsonl", "exam2.graphml")
  - Variable names used in code cells (e.g., "df", "G", "X", "y")  
  - Column names referenced (e.g., "speaker", "length", "season")
  - Function/method calls hinted at (e.g., "PageRank", "confusion_matrix")
- Use extracted names in all output labels and code snippets for direct copy-paste
- Prioritize analyses involving explicitly mentioned columns/methods
- If no exam notebook provided, use generic names and run all analyses

---

ANALYSES TO PRE-COMPUTE:

1. DATA OVERVIEW (per dataset)
   - Shape, dtypes, memory usage
   - .describe() for all numeric columns
   - .value_counts() for all categorical columns (top 20)
   - Missing values count and percentage
   - Sample rows (head + tail)
   - Unique value counts per column
   - Column name list (for quick reference)

2. DISTRIBUTIONS & CORRELATIONS
   - Histogram for every numeric column
   - Correlation matrix heatmap for all numeric pairs
   - Pairwise scatterplots (sample if too many)
   
3. REGRESSION (auto-detect plausible combinations)
   - OLS regression for every numeric DV ~ categorical IV (with Treatment encoding)
   - OLS regression for every numeric DV ~ numeric IV
   - Multi-variable regressions for plausible combinations
   - Print full summary tables
   - Flag significant coefficients (p < 0.05)
   - If exam mentions specific formula syntax, run that exactly

4. TEXT ANALYSIS (if text columns detected)
   - Token counts, vocabulary size
   - TF-IDF matrix (top 1000 terms)
   - Word frequency distributions per category (e.g., per speaker)
   - Sample TF-IDF values for inspection
   - Tokenized samples

5. NETWORK ANALYSIS (if .graphml or edge-like data detected)
   - Node/edge counts
   - Degree distribution (in/out if directed)
   - All centrality metrics: degree, PageRank, betweenness, closeness, eigenvector
   - Top 10 nodes by each metric
   - Connected components count
   - Density, diameter (if feasible)
   - Multi-edge counts if MultiGraph
   - Subgraph analyses if node attributes allow filtering

6. CLASSIFICATION (if plausible target column detected, e.g., <10 unique values)
   - Train/test split (70/30, random_state=42)
   - Decision tree classifier (random_state=42)
   - Logistic regression
   - Random forest
   - Accuracy score for each
   - Normalized confusion matrix + heatmap
   - Random baseline comparison
   - Classification report (precision, recall, F1)

7. GROUPBY AGGREGATIONS
   - For each categorical column: group by it and compute mean/sum/count of all numeric columns
   - Cross-tabulations for categorical pairs
   - Pivot tables for plausible combinations

8. TIME SERIES (if datetime-like column detected)
   - Parse and sort by time
   - Aggregations by time period (day, week, month, season, year)
   - Trend plots
   - Rolling averages

9. ID PARSING (if ID columns with structure detected, e.g., "s01_e05_c03_u012")
   - Extract components into separate columns
   - Aggregations by extracted components
   - This is common in ADA exams — prioritize

10. STANDARD PLOTS (saved as labeled images in report)
    - Bar charts for all categorical value counts (top N)
    - Box plots for numeric by categorical
    - Heatmaps for confusion matrices and correlations
    - Network visualization (spring layout, sized by PageRank)
    - Line plots for time-based data
    - Scatter with regression line for numeric pairs

---

REPORT FORMAT:
- HTML file with collapsible sections
- Each section contains:
  - Analysis title (searchable)
  - Code snippet used (copy-paste ready, using exam variable names)
  - Output (table, plot, or summary)
  - Key observations auto-highlighted (e.g., "p < 0.05", "top node: X")
- Table of contents at top with anchor links
- Search-friendly: all text, no images-only sections

---

REQUIREMENTS:
- Study ALL provided past exams to identify what analyses were asked — ensure full coverage
- Auto-detect column types and data relationships
- Handle messy data gracefully (skip on error, log warning, continue)
- Label everything clearly: "Regression: length ~ C(season, Treatment(reference='s01'))"
- Code snippets must use exam-legal packages only: pandas, numpy, statsmodels, sklearn, networkx, matplotlib, seaborn
- Print progress during execution so I know it's working
- Target runtime: under 3 minutes on typical exam data

---

CLI INTERFACE:
python exam_blitz.py /path/to/exam/data/
python exam_blitz.py /path/to/exam/data/ --exam exam.ipynb

- First argument: directory containing data files
- Optional --exam flag: path to exam notebook for context extraction
- Save report to `exam_blitz_report.html` in current directory
- Also print summary to terminal

---

SCRIPT STRUCTURE:
- Modular: separate functions per analysis type
- Context extractor module for .ipynb parsing
- Data loader that handles all file types
- Analysis orchestrator that runs all modules
- Report generator that compiles HTML output
- Main function with CLI parsing

---

CONSTRAINTS:
- No network calls — fully offline
- Script can use any packages for its own logic
- Output code snippets must only use exam-allowed packages
- Fail gracefully: if one analysis errors, log it and continue to next
- Prioritize analyses that appeared most frequently in past exams
- Must handle: pandas DataFrames, networkx graphs, sklearn matrices, statsmodels results

---

Analyze all provided course materials thoroughly to ensure no common analysis type is missed. The script's value is in its comprehensiveness — it should answer 80%+ of possible questions before I even read them.

MAKE SURE TO ANALYSE ALL EXAMS AND EXCERCISE NOTEBOOKS PROVIDED

Output the complete, runnable script.
```