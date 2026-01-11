#!/usr/bin/env python3
"""
EXAM BLITZ v5.0 - COMPLETE ADA Exam Pre-Computation Script
============================================================
Based on v4.2, enhanced with ALL course material coverage.

NEW IN v5.0:
- Propensity Score Matching (Observational Studies - Lecture 6)
- k-NN Classifier (Lecture 4, SupervisedLearning exercise)
- Random Forest Classifier (Lecture 5, SupervisedLearning exercise)
- ROC Curve / AUC Score (AppliedML exercise)
- Bootstrap Confidence Intervals (Describing_data exercise)
- Hypothesis Testing (t-test, chi-squared) (Lecture 11)
- PCA / t-SNE Visualization (Unsupervised_Learning exercise)
- Silhouette Score for Clustering
- Enhanced copy-pastable code for ALL sections

COMPLETE EXAM COVERAGE:
- 2019: YouTube (LogisticRegressionCV, RidgeCV, binary target from median, groupby)
- 2020: Wikispeedia (SGDClassifier, Pipeline+GridSearchCV, TF-IDF, class_weight, connectivity)
- 2021: Faculty Hiring (cumsum plots, quintiles, LogisticRegression C=10, Spearman, PageRank-score correlation)
- 2022: Wikipedia RfA (ECDF, triangles, structural balance, signed graphs)
- 2023: Friends TV (ID parsing, Treatment encoding, confusion matrix normalized)

COURSE MATERIAL COVERAGE:
- Lecture 4: k-NN, choosing k, cross-validation
- Lecture 5: Classification pipeline, confusion matrix, precision/recall/F1, decision trees, random forests
- Lecture 6: Observational studies, propensity scores, matching
- Lecture 7: Text processing, TF-IDF, document classification
- Lecture 8: Topic detection, LSA, SVD, word2vec
- Lecture 10: Clustering, dimensionality reduction
- Lecture 11: Hypothesis testing, confidence intervals, bootstrap

Usage:
    python exam_blitz_v5_complete.py /path/to/exam/data/
    python exam_blitz_v5_complete.py /path/to/exam/data/ --exam exam.ipynb -o report.html
"""

import os
import sys
import re
import json
import gzip
import warnings
import argparse
import base64
from pathlib import Path
from datetime import datetime
from collections import Counter
from io import BytesIO

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import (LogisticRegression, LogisticRegressionCV, 
                                   RidgeCV, SGDClassifier, LinearRegression)
from sklearn.metrics import (confusion_matrix, classification_report, 
                             accuracy_score, balanced_accuracy_score,
                             mean_absolute_error, mean_squared_error,
                             roc_curve, auc, roc_auc_score, silhouette_score,
                             precision_score, recall_score, f1_score)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Stats imports
import statsmodels.formula.api as smf
import statsmodels.api as sm
import scipy.stats as stats

# Network imports
import networkx as nx

# Constants - EXAM STANDARDS
RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.3  # Exam 2023 standard
ALT_TEST_SIZE = 0.4      # Exam 2021 standard
FIGURE_DPI = 100
MAX_CATEGORIES = 20
TOP_N = 20


# =============================================================================
# EXAM CONTEXT EXTRACTOR
# =============================================================================

class ExamContextExtractor:
    """Extract and analyze context from exam notebook to prioritize analyses."""
    
    def __init__(self):
        self.data_files = set()
        self.method_hints = set()
        self.column_names = set()
        self.formulas = []
        self.questions = []
        self.analysis_keywords = {}
        
    def parse_notebook(self, notebook_path):
        """Parse Jupyter notebook to extract exam context."""
        print(f"📖 Parsing exam notebook: {notebook_path}")
        
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = json.load(f)
        except Exception as e:
            print(f"  ⚠️ Could not parse notebook: {e}")
            return
            
        for cell in nb.get('cells', []):
            cell_type = cell.get('cell_type', '')
            source = ''.join(cell.get('source', []))
            
            if cell_type == 'markdown':
                self._extract_questions(source)
            
            # Extract data file references
            file_patterns = [
                r'["\']([^"\']+\.(?:jsonl|csv|tsv|graphml|json|parquet|txt|gz))["\']',
                r'read_(?:csv|json|parquet)\(["\']([^"\']+)["\']',
                r'gzip\.open\(["\']([^"\']+)["\']',
            ]
            for pattern in file_patterns:
                matches = re.findall(pattern, source)
                for m in matches:
                    fname = Path(m).name
                    self.data_files.add(fname)
            
            # Extract method hints - ENHANCED for v5
            method_patterns = [
                (r'PageRank|pagerank', 'pagerank'),
                (r'confusion_matrix', 'confusion_matrix'),
                (r'DecisionTree', 'decisiontree'),
                (r'LogisticRegression(?!CV)', 'logisticregression'),
                (r'LogisticRegressionCV', 'logisticregressioncv'),
                (r'SGDClassifier', 'sgdclassifier'),
                (r'RidgeCV|Ridge\(', 'ridge'),
                (r'train_test_split', 'train_test_split'),
                (r'TfidfVectorizer|TF-IDF|tfidf', 'tfidf'),
                (r'Treatment\(', 'treatment'),
                (r'MultiDiGraph', 'multidigraph'),
                (r'DiGraph', 'digraph'),
                (r'centrality', 'centrality'),
                (r'in_degree|out_degree', 'degree'),
                (r'cumsum', 'cumsum'),
                (r'qcut|quintile', 'quintile'),
                (r'spearman', 'spearman'),
                (r'ecdfplot|ECDF', 'ecdf'),
                (r'enumerate_all_cliques', 'cliques'),
                (r'structural.?balance', 'structural_balance'),
                (r'triangle', 'triangle'),
                (r'connected_component', 'connected_component'),
                (r'groupby', 'groupby'),
                (r'Pipeline', 'pipeline'),
                (r'GridSearchCV', 'gridsearch'),
                (r'class_weight', 'class_weight'),
                (r'balanced_accuracy', 'balanced_accuracy'),
                # NEW in v5
                (r'KNeighbors|knn|k-nn', 'knn'),
                (r'RandomForest', 'randomforest'),
                (r'roc_curve|ROC|AUC', 'roc_auc'),
                (r'bootstrap', 'bootstrap'),
                (r'ttest|t-test|t_test', 'ttest'),
                (r'chi2|chi-squared|chi_squared', 'chi_squared'),
                (r'propensity', 'propensity'),
                (r'max_weight_matching', 'matching'),
                (r'PCA', 'pca'),
                (r'TSNE|t-SNE', 'tsne'),
                (r'silhouette', 'silhouette'),
            ]
            for pattern, hint in method_patterns:
                if re.search(pattern, source, re.IGNORECASE):
                    self.method_hints.add(hint)
            
            # Extract column names from bracket notation
            col_patterns = [
                r'\[["\'](\w+)["\']\]',
                r'\.(\w+)\s*[=<>!]',
                r'groupby\(["\'](\w+)["\']',
                r'groupby\(\[["\'](\w+)',
            ]
            for pattern in col_patterns:
                self.column_names.update(re.findall(pattern, source))
            
            # Extract regression formulas
            formula_pattern = r'formula\s*=\s*["\']([^"\']+)["\']'
            self.formulas.extend(re.findall(formula_pattern, source))
        
        self._extract_keywords()
        
        print(f"  📁 Data files found: {self.data_files}")
        print(f"  📊 Column hints: {list(self.column_names)[:15]}...")
        print(f"  🔧 Method hints: {self.method_hints}")
        print(f"  📐 Formulas: {self.formulas}")
        print(f"  ❓ Questions found: {len(self.questions)}")
    
    def _extract_questions(self, source):
        """Extract question text from markdown."""
        question_patterns = [
            r'(?:^|\n)\s*(?:\d+\.|(?:\*\*)?(?:Task|Question)\s*\d+)',
            r'/Discuss:/',
            r'/True or false:/',
            r'(?:^|\n)\s*[A-Z]\.\s+',
        ]
        for pattern in question_patterns:
            if re.search(pattern, source):
                self.questions.append(source[:500])
                break
    
    def _extract_keywords(self):
        """Extract analysis keywords from questions."""
        all_text = ' '.join(self.questions).lower()
        
        keyword_map = {
            'pagerank': ['pagerank', 'page rank', 'importance'],
            'degree': ['degree', 'in-degree', 'out-degree', 'indegree', 'outdegree'],
            'centrality': ['centrality', 'betweenness', 'closeness'],
            'regression': ['regression', 'coefficient', 'r-squared', 'r²', 'ols'],
            'classification': ['classifier', 'classification', 'predict', 'accuracy'],
            'tfidf': ['tf-idf', 'tfidf', 'term frequency'],
            'clustering': ['cluster', 'k-means', 'kmeans'],
            'network': ['graph', 'network', 'node', 'edge'],
            'correlation': ['correlation', 'correlate', 'relationship'],
            'significance': ['significant', 'p-value', 'hypothesis'],
            'balance': ['balance', 'structural balance', 'triangle'],
            'quintile': ['quintile', 'quartile', 'percentile'],
            'cumulative': ['cumulative', 'cumsum', 'fraction'],
            # NEW in v5
            'propensity': ['propensity', 'matching', 'treatment', 'control', 'observational'],
            'bootstrap': ['bootstrap', 'confidence interval', 'resampling'],
            'roc': ['roc', 'auc', 'receiver operating'],
        }
        
        for key, terms in keyword_map.items():
            count = sum(all_text.count(term) for term in terms)
            if count > 0:
                self.analysis_keywords[key] = count
    
    def get_priority_analyses(self):
        """Return which analyses should be prioritized based on exam content."""
        priorities = {}
        
        # Map method hints to analysis types
        hint_to_analysis = {
            'pagerank': 'network_centrality',
            'centrality': 'network_centrality',
            'degree': 'network_degree',
            'multidigraph': 'network_analysis',
            'digraph': 'network_analysis',
            'treatment': 'regression_categorical',
            'logisticregression': 'classification_logreg',
            'logisticregressioncv': 'classification_logreg_cv',
            'decisiontree': 'classification_tree',
            'sgdclassifier': 'classification_sgd',
            'ridge': 'regression_ridge',
            'tfidf': 'text_analysis',
            'confusion_matrix': 'classification_evaluation',
            'train_test_split': 'ml_evaluation',
            'cumsum': 'cumulative_analysis',
            'quintile': 'quintile_analysis',
            'spearman': 'correlation_analysis',
            'ecdf': 'distribution_analysis',
            'cliques': 'triangle_analysis',
            'triangle': 'triangle_analysis',
            'structural_balance': 'structural_balance',
            'connected_component': 'connectivity_analysis',
            'pipeline': 'ml_pipeline',
            'gridsearch': 'hyperparameter_tuning',
            'groupby': 'aggregation_analysis',
            # NEW in v5
            'knn': 'classification_knn',
            'randomforest': 'classification_rf',
            'roc_auc': 'roc_analysis',
            'bootstrap': 'bootstrap_analysis',
            'ttest': 'hypothesis_testing',
            'propensity': 'propensity_matching',
            'matching': 'propensity_matching',
            'pca': 'dimensionality_reduction',
            'tsne': 'dimensionality_reduction',
            'silhouette': 'clustering_evaluation',
        }
        
        for hint in self.method_hints:
            if hint in hint_to_analysis:
                priorities[hint_to_analysis[hint]] = 'HIGH'
        
        # Add keyword-based priorities
        for keyword, count in self.analysis_keywords.items():
            if count >= 2:
                priorities[keyword] = 'HIGH'
        
        return priorities


# =============================================================================
# REPORT BUILDER
# =============================================================================

class ReportBuilder:
    """Builds comprehensive HTML report with embedded base64 images."""
    
    def __init__(self):
        self.sections = []
        self.toc = []
        self.figure_counter = 0
        self.exam_context = None
        
    def set_exam_context(self, context: ExamContextExtractor):
        """Set exam context for report customization."""
        self.exam_context = context
        
    def add_section(self, title, content, code=None, level=1, key_findings=None):
        section_id = f"section-{len(self.sections)}"
        self.toc.append((level, title, section_id))
        
        # Check if this is a priority section based on exam context
        priority_marker = ""
        if self.exam_context:
            priorities = self.exam_context.get_priority_analyses()
            title_lower = title.lower()
            for analysis, priority in priorities.items():
                if analysis.replace('_', ' ') in title_lower or analysis in title_lower:
                    priority_marker = '<span class="priority-badge">⭐ HIGH PRIORITY</span>'
                    break
        
        html = f"""
        <div class="section level-{level}" id="{section_id}">
            <h{level+1} class="section-title" onclick="toggleSection('{section_id}-content')">
                {title} {priority_marker}
            </h{level+1}>
            <div class="section-content" id="{section_id}-content">
        """
        
        if key_findings:
            html += f"""
                <div class="key-findings">
                    <strong>🔑 Key Findings:</strong>
                    <ul>{"".join(f"<li>{f}</li>" for f in key_findings)}</ul>
                </div>
            """
        
        if code:
            html += f"""
                <details class="code-block" open>
                    <summary>📝 Copy-Paste Code</summary>
                    <pre><code>{self._escape_html(code)}</code></pre>
                </details>
            """
            
        html += f"""
                <div class="content">{content}</div>
            </div>
        </div>
        """
        self.sections.append(html)
        
    def add_dataframe(self, df, title="", max_rows=30):
        if df is None or (hasattr(df, 'empty') and df.empty):
            return "<p><em>No data available</em></p>"
        
        if hasattr(df, 'to_html'):
            html = df.head(max_rows).to_html(classes='dataframe', escape=True)
        else:
            html = f"<pre>{str(df)[:2000]}</pre>"
            
        return f"""
        <div class="dataframe-container">
            {f'<h5>{title}</h5>' if title else ''}
            {html}
            {f'<p class="more-rows">... and {len(df) - max_rows} more rows</p>' if len(df) > max_rows else ''}
        </div>
        """
        
    def save_figure(self, fig, name):
        self.figure_counter += 1
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close(fig)
        return f'<img src="data:image/png;base64,{img_base64}" alt="{name}" class="figure">'
        
    def _escape_html(self, text):
        return (str(text).replace('&', '&amp;')
                   .replace('<', '&lt;')
                   .replace('>', '&gt;'))
    
    def build(self, title="Exam Blitz Report"):
        toc_html = "<ul class='toc-list'>"
        for level, section_title, section_id in self.toc:
            indent = "&nbsp;" * ((level - 1) * 4)
            toc_html += f'<li class="toc-level-{level}">{indent}<a href="#{section_id}">{section_title}</a></li>'
        toc_html += "</ul>"
        
        return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
        <style>
            :root {{
                --bg-primary: #0d1117;
                --bg-secondary: #161b22;
                --bg-tertiary: #1c2128;
                --bg-elevated: #21262d;
                --border: #30363d;
                --border-hover: #484f58;
                --text-primary: #e6edf3;
                --text-secondary: #8b949e;
                --text-muted: #6e7681;
                --accent-blue: #58a6ff;
                --accent-purple: #bc8cff;
                --accent-green: #3fb950;
                --accent-yellow: #d29922;
                --accent-red: #f85149;
                --accent-orange: #ff8c42;
                --code-bg: #161b22;
                --shadow: rgba(0, 0, 0, 0.5);
            }}
            
            * {{ box-sizing: border-box; margin: 0; padding: 0; }}
            html {{ scroll-behavior: smooth; scroll-padding-top: 80px; }}
            
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif;
                line-height: 1.6;
                background: var(--bg-primary);
                color: var(--text-primary);
                font-size: 14px;
            }}
            
            .header {{
                position: sticky; top: 0; z-index: 100;
                background: var(--bg-secondary);
                border-bottom: 1px solid var(--border);
                padding: 15px 0;
                box-shadow: 0 2px 8px var(--shadow);
            }}
            
            .header-content {{
                max-width: 1400px; margin: 0 auto; padding: 0 30px;
                display: flex; align-items: center; justify-content: space-between;
                flex-wrap: wrap; gap: 15px;
            }}
            
            .header h1 {{ font-size: 24px; font-weight: 600; color: var(--text-primary); margin: 0; }}
            .header-meta {{ color: var(--text-secondary); font-size: 12px; margin-top: 5px; }}
            
            .container {{ max-width: 1400px; margin: 0 auto; padding: 20px 30px; }}
            
            .search-box {{ margin-bottom: 20px; }}
            .search-box input {{
                width: 100%; padding: 12px 20px; font-size: 15px;
                background: var(--bg-secondary); border: 1px solid var(--border);
                border-radius: 8px; color: var(--text-primary); outline: none;
            }}
            .search-box input:focus {{ border-color: var(--accent-blue); }}
            
            .toc {{
                background: var(--bg-secondary); border: 1px solid var(--border);
                border-radius: 8px; padding: 20px; margin-bottom: 30px;
            }}
            .toc h3 {{ color: var(--text-primary); font-size: 16px; margin: 0 0 15px 0; }}
            .toc-list {{ list-style-type: none; padding: 0; margin: 0; }}
            .toc-list li {{ margin: 6px 0; }}
            .toc a {{ color: var(--accent-blue); text-decoration: none; }}
            .toc a:hover {{ color: var(--accent-purple); text-decoration: underline; }}
            
            .section {{
                background: var(--bg-secondary); border: 1px solid var(--border);
                border-radius: 8px; padding: 24px; margin-bottom: 20px;
            }}
            .section.level-2 {{ margin-left: 20px; background: var(--bg-tertiary); }}
            .section.level-3 {{ margin-left: 40px; background: var(--bg-elevated); }}
            
            .section-title {{ cursor: pointer; display: flex; align-items: center; gap: 10px; margin-bottom: 15px; }}
            h2, h3, h4, h5 {{ color: var(--text-primary); font-weight: 600; }}
            h4 {{ color: var(--accent-blue); font-size: 16px; margin: 20px 0 12px 0; }}
            h5 {{ color: var(--accent-purple); font-size: 14px; margin: 15px 0 10px 0; }}
            
            .dataframe {{ width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 12px; }}
            .dataframe th, .dataframe td {{
                padding: 8px 12px; text-align: left;
                border: 1px solid var(--border);
            }}
            .dataframe th {{ background: var(--bg-tertiary); font-weight: 600; }}
            .dataframe tr:nth-child(even) {{ background: var(--bg-tertiary); }}
            
            .code-block {{ margin: 15px 0; }}
            .code-block summary {{
                cursor: pointer; padding: 10px 15px;
                background: var(--bg-tertiary); border-radius: 6px;
                font-weight: 500; color: var(--accent-green);
            }}
            .code-block pre {{
                background: var(--code-bg); padding: 15px;
                border-radius: 0 0 6px 6px; overflow-x: auto;
                border: 1px solid var(--border); border-top: none;
            }}
            .code-block code {{ color: var(--text-primary); font-family: 'Fira Code', monospace; font-size: 13px; }}
            
            .figure {{ max-width: 100%; height: auto; border-radius: 8px; margin: 15px 0; }}
            
            .key-findings {{
                background: rgba(63, 185, 80, 0.1); border-left: 4px solid var(--accent-green);
                padding: 15px; margin: 15px 0; border-radius: 0 6px 6px 0;
            }}
            .warning {{
                background: rgba(210, 153, 34, 0.1); border-left: 4px solid var(--accent-yellow);
                padding: 15px; margin: 15px 0; border-radius: 0 6px 6px 0;
            }}
            .critical {{
                background: rgba(248, 81, 73, 0.1); border-left: 4px solid var(--accent-red);
                padding: 15px; margin: 15px 0; border-radius: 0 6px 6px 0;
            }}
            .info {{
                background: rgba(88, 166, 255, 0.1); border-left: 4px solid var(--accent-blue);
                padding: 15px; margin: 15px 0; border-radius: 0 6px 6px 0;
            }}
            
            .priority-badge {{
                background: var(--accent-yellow); color: black;
                padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600;
            }}
            
            .footer {{
                text-align: center; padding: 30px;
                color: var(--text-muted); font-size: 12px;
                border-top: 1px solid var(--border); margin-top: 40px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <div class="header-content">
                <div>
                    <h1>🎯 {title}</h1>
                    <div class="header-meta">
                        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
                        Coverage: ALL ADA Course Material + Exams 2019-2023
                    </div>
                </div>
            </div>
        </div>
        
        <div class="container">
            <div class="search-box">
                <input type="text" id="searchInput" 
                    placeholder="🔍 Search (PageRank, confusion matrix, propensity, bootstrap, ROC, etc.)..." 
                    onkeyup="searchReport()">
            </div>
            
            <div class="toc">
                <h3>📑 Table of Contents</h3>
                {toc_html}
            </div>
            
            <div id="sections-container">
                {"".join(self.sections)}
            </div>
            
            <div class="footer">
                <strong>Exam Blitz v5.0 (Complete)</strong> | 
                Full ADA Course Coverage | {self.figure_counter} figures
            </div>
        </div>
        
        <script>
            function searchReport() {{
                const query = document.getElementById('searchInput').value.toLowerCase().trim();
                const sections = document.querySelectorAll('.section');
                if (query === '') {{ sections.forEach(s => s.style.display = 'block'); return; }}
                sections.forEach(s => {{ s.style.display = s.textContent.toLowerCase().includes(query) ? 'block' : 'none'; }});
            }}
            function toggleSection(id) {{
                const el = document.getElementById(id);
                if (el) el.style.display = el.style.display === 'none' ? 'block' : 'none';
            }}
        </script>
    </body>
    </html>
    """


# =============================================================================
# DATA LOADING
# =============================================================================

class DataLoader:
    """Load all file formats used across exams."""
    
    @staticmethod
    def load_file(filepath):
        filepath = Path(filepath)
        name = filepath.stem
        if '.csv' in name or '.tsv' in name:
            name = name.replace('.csv', '').replace('.tsv', '')
        suffix = ''.join(filepath.suffixes).lower()
        
        try:
            if '.csv.gz' in suffix or (suffix == '.gz' and '.csv' in str(filepath).lower()):
                return name, pd.read_csv(filepath, compression='gzip')
            elif '.tsv.gz' in suffix or (suffix == '.gz' and '.tsv' in str(filepath).lower()):
                return name, pd.read_csv(filepath, compression='gzip', sep='\t')
            elif suffix == '.csv':
                return name, pd.read_csv(filepath)
            elif suffix == '.tsv':
                return name, pd.read_csv(filepath, sep='\t')
            elif suffix == '.jsonl':
                return name, pd.read_json(filepath, lines=True)
            elif suffix == '.json':
                return name, pd.read_json(filepath)
            elif suffix == '.parquet':
                return name, pd.read_parquet(filepath)
        except Exception as e:
            print(f"  ⚠️ Could not load {filepath}: {e}")
        return None, None
    
    @staticmethod
    def find_data_files(directory):
        extensions = ['*.csv', '*.csv.gz', '*.tsv', '*.tsv.gz', '*.jsonl', '*.json', '*.parquet']
        files = []
        for ext in extensions:
            files.extend(Path(directory).rglob(ext))
        return sorted(set(files))


# =============================================================================
# ID PARSING (Exam 2023)
# =============================================================================

class IDParser:
    """Parse structured IDs like s01_e05_c03_u012."""
    
    def __init__(self, report: ReportBuilder):
        self.report = report
    
    def detect_and_parse(self, df, name):
        """Detect and parse structured ID columns."""
        parsed_cols = []
        
        for col in df.columns:
            if df[col].dtype != 'object':
                continue
            
            sample = df[col].dropna().head(100).astype(str)
            pattern = r'^[a-z]\d+_[a-z]\d+(?:_[a-z]\d+)*$'
            match_ratio = sample.str.match(pattern, case=False).mean()
            
            if match_ratio > 0.8:
                parsed_cols.append(col)
                df = self._parse_structured_id(df, col)
        
        if parsed_cols:
            content = f"<p>Detected structured IDs in: {', '.join(parsed_cols)}</p>"
            
            code = """# Parse structured IDs (Exam 2023 - Friends)
# Pattern: s01_e05_c03_u012 -> season, episode, conversation, utterance

def parse_id(id_str):
    parts = id_str.split('_')
    result = {}
    for part in parts:
        prefix = part[0]
        number = int(part[1:])
        if prefix == 's': result['season'] = number
        elif prefix == 'e': result['episode'] = number
        elif prefix == 'c': result['conversation'] = number
        elif prefix == 'u': result['utterance'] = number
    return result

# Apply to dataframe
parsed = df['id'].apply(parse_id).apply(pd.Series)
df = pd.concat([df, parsed], axis=1)"""
            content += f'<details class="code-block"><summary>📝 ID Parsing Code</summary><pre><code>{code}</code></pre></details>'
            
            self.report.add_section(f"🔍 ID Parsing: {name}", content, level=2)
        
        return df, parsed_cols
    
    def _parse_structured_id(self, df, col):
        """Parse structured ID column into components."""
        df = df.copy()
        
        def parse_single(id_str):
            if pd.isna(id_str):
                return {}
            parts = str(id_str).split('_')
            result = {}
            for part in parts:
                if len(part) > 1 and part[0].isalpha() and part[1:].isdigit():
                    prefix = part[0].lower()
                    number = int(part[1:])
                    result[f'{col}_{prefix}'] = number
            return result
        
        parsed = df[col].apply(parse_single).apply(pd.Series)
        for c in parsed.columns:
            if c not in df.columns:
                df[c] = parsed[c]
        
        return df


# =============================================================================
# DATA OVERVIEW ANALYZER
# =============================================================================

class DataOverviewAnalyzer:
    """Generate comprehensive data overview."""
    
    def __init__(self, report: ReportBuilder):
        self.report = report
    
    def analyze(self, df, name):
        print(f"  📊 Generating overview for {name}...")
        
        content = f"""
        <h4>📊 Basic Info</h4>
        <table class="dataframe">
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Rows</td><td>{len(df):,}</td></tr>
            <tr><td>Columns</td><td>{len(df.columns)}</td></tr>
            <tr><td>Memory</td><td>{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB</td></tr>
        </table>
        """
        
        # Column types
        dtype_counts = df.dtypes.astype(str).value_counts()
        content += "<h4>Column Types</h4>"
        content += self.report.add_dataframe(dtype_counts.reset_index().rename(
            columns={'index': 'Type', 0: 'Count'}))
        
        # Sample data
        content += "<h4>Sample Data (first 5 rows)</h4>"
        content += self.report.add_dataframe(df.head(5))
        
        # Numeric summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            content += "<h4>Numeric Summary</h4>"
            content += self.report.add_dataframe(df[numeric_cols].describe().round(3))
        
        # Missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            missing_df = missing[missing > 0].reset_index()
            missing_df.columns = ['Column', 'Missing']
            missing_df['Percent'] = (missing_df['Missing'] / len(df) * 100).round(1)
            content += "<h4>Missing Values</h4>"
            content += self.report.add_dataframe(missing_df)
        
        code = """# Data Overview
import pandas as pd

# Basic info
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Dtypes:\\n{df.dtypes}")

# Missing values
print(f"\\nMissing values:\\n{df.isnull().sum()}")

# Numeric summary
print(f"\\nNumeric summary:\\n{df.describe()}")

# Sample
print(f"\\nSample:\\n{df.head()}")"""
        content += f'<details class="code-block"><summary>📝 Overview Code</summary><pre><code>{code}</code></pre></details>'
        
        self.report.add_section(f"📋 Data Overview: {name}", content, level=1)


# =============================================================================
# HYPOTHESIS TESTING ANALYZER (NEW in v5)
# =============================================================================

class HypothesisTestingAnalyzer:
    """Hypothesis testing: t-test, chi-squared, etc. (Lecture 11)"""
    
    def __init__(self, report: ReportBuilder):
        self.report = report
    
    def analyze(self, df, name):
        print(f"  📊 Running hypothesis tests for {name}...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [c for c in df.select_dtypes(include=['object', 'category']).columns
                          if 2 <= df[c].nunique() <= 10]
        
        if len(numeric_cols) < 1 or len(categorical_cols) < 1:
            return
        
        content = ""
        findings = []
        
        # T-tests for numeric columns grouped by categorical
        for cat_col in categorical_cols[:2]:
            for num_col in numeric_cols[:3]:
                result = self._ttest_by_group(df, num_col, cat_col)
                if result:
                    content += result['content']
                    if result.get('significant'):
                        findings.append(result['finding'])
        
        # Chi-squared test for categorical associations
        if len(categorical_cols) >= 2:
            result = self._chi_squared_test(df, categorical_cols[0], categorical_cols[1])
            if result:
                content += result['content']
        
        if content:
            self.report.add_section(f"📊 Hypothesis Testing: {name}", content, level=2, 
                                   key_findings=findings[:5] if findings else None)
    
    def _ttest_by_group(self, df, num_col, cat_col):
        """Perform t-test comparing groups."""
        try:
            groups = df[cat_col].dropna().unique()
            if len(groups) != 2:
                return None
            
            group1 = df[df[cat_col] == groups[0]][num_col].dropna()
            group2 = df[df[cat_col] == groups[1]][num_col].dropna()
            
            if len(group1) < 5 or len(group2) < 5:
                return None
            
            t_stat, p_value = stats.ttest_ind(group1, group2)
            
            content = f"<h5>T-Test: {num_col} by {cat_col}</h5>"
            content += f"""
            <table class="dataframe">
                <tr><th>Group</th><th>Mean</th><th>Std</th><th>N</th></tr>
                <tr><td>{groups[0]}</td><td>{group1.mean():.4f}</td><td>{group1.std():.4f}</td><td>{len(group1)}</td></tr>
                <tr><td>{groups[1]}</td><td>{group2.mean():.4f}</td><td>{group2.std():.4f}</td><td>{len(group2)}</td></tr>
            </table>
            <p><strong>t-statistic:</strong> {t_stat:.4f}</p>
            <p><strong>p-value:</strong> {p_value:.6f} {'✅ Significant (p < 0.05)' if p_value < 0.05 else '❌ Not significant'}</p>
            """
            
            code = f"""# T-Test (Lecture 11, Describing_data exercise)
from scipy import stats

# Compare {num_col} between {cat_col} groups
group1 = df[df['{cat_col}'] == '{groups[0]}']['{num_col}'].dropna()
group2 = df[df['{cat_col}'] == '{groups[1]}']['{num_col}'].dropna()

# Two-sample t-test (independent samples)
t_stat, p_value = stats.ttest_ind(group1, group2)

print(f"Group 1 mean: {{group1.mean():.4f}}")
print(f"Group 2 mean: {{group2.mean():.4f}}")
print(f"t-statistic: {{t_stat:.4f}}")
print(f"p-value: {{p_value:.6f}}")

# Interpretation
if p_value < 0.05:
    print("Reject null hypothesis - significant difference")
else:
    print("Cannot reject null hypothesis - no significant difference")"""
            content += f'<details class="code-block"><summary>📝 T-Test Code</summary><pre><code>{code}</code></pre></details>'
            
            return {
                'content': content,
                'significant': p_value < 0.05,
                'finding': f"{num_col} differs by {cat_col} (p={p_value:.4f})" if p_value < 0.05 else None
            }
        except:
            return None
    
    def _chi_squared_test(self, df, col1, col2):
        """Chi-squared test for independence."""
        try:
            contingency = pd.crosstab(df[col1], df[col2])
            if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                return None
            
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
            
            content = f"<h5>Chi-Squared Test: {col1} vs {col2}</h5>"
            content += f"""
            <p><strong>Chi-squared statistic:</strong> {chi2:.4f}</p>
            <p><strong>Degrees of freedom:</strong> {dof}</p>
            <p><strong>p-value:</strong> {p_value:.6f} {'✅ Significant' if p_value < 0.05 else '❌ Not significant'}</p>
            """
            content += self.report.add_dataframe(contingency, "Contingency Table")
            
            code = f"""# Chi-Squared Test for Independence (Lecture 5, 11)
from scipy import stats
import pandas as pd

# Create contingency table
contingency = pd.crosstab(df['{col1}'], df['{col2}'])
print("Contingency Table:")
print(contingency)

# Chi-squared test
chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

print(f"\\nChi-squared: {{chi2:.4f}}")
print(f"Degrees of freedom: {{dof}}")
print(f"p-value: {{p_value:.6f}}")

# Interpretation
if p_value < 0.05:
    print("Variables are NOT independent (reject H0)")
else:
    print("Variables may be independent (cannot reject H0)")"""
            content += f'<details class="code-block"><summary>📝 Chi-Squared Code</summary><pre><code>{code}</code></pre></details>'
            
            return {'content': content}
        except:
            return None


# =============================================================================
# BOOTSTRAP CONFIDENCE INTERVAL ANALYZER (NEW in v5)
# =============================================================================

class BootstrapAnalyzer:
    """Bootstrap confidence intervals (Describing_data exercise, Lecture 11)"""
    
    def __init__(self, report: ReportBuilder):
        self.report = report
    
    def analyze(self, df, name):
        print(f"  📊 Computing bootstrap CIs for {name}...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 1:
            return
        
        content = "<h4>Bootstrap 95% Confidence Intervals</h4>"
        content += """
        <div class="info">
            <strong>Bootstrap Method:</strong> Sample with replacement, compute statistic, repeat 1000+ times.
            95% CI = 2.5th and 97.5th percentiles of bootstrap distribution.
        </div>
        """
        
        results = []
        for col in numeric_cols[:5]:
            data = df[col].dropna().values
            if len(data) < 10:
                continue
            
            ci_low, ci_high = self._bootstrap_ci(data)
            results.append({
                'Column': col,
                'Mean': np.mean(data),
                'CI_Lower': ci_low,
                'CI_Upper': ci_high,
                'CI_Width': ci_high - ci_low
            })
        
        if results:
            results_df = pd.DataFrame(results).round(4)
            content += self.report.add_dataframe(results_df)
            
            # Visualization
            fig, ax = plt.subplots(figsize=(10, max(4, len(results) * 0.5)))
            y_pos = range(len(results))
            means = [r['Mean'] for r in results]
            errors = [[r['Mean'] - r['CI_Lower'] for r in results],
                     [r['CI_Upper'] - r['Mean'] for r in results]]
            
            ax.barh(y_pos, means, xerr=errors, capsize=5, color='steelblue', alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([r['Column'] for r in results])
            ax.set_xlabel('Mean with 95% CI')
            ax.set_title('Bootstrap Confidence Intervals')
            ax.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            content += self.report.save_figure(fig, f"{name}_bootstrap_ci")
        
        code = """# Bootstrap Confidence Intervals (Describing_data exercise)
import numpy as np

def bootstrap_confidence_interval(data, iterations=1000, confidence=0.95):
    '''
    Compute bootstrap confidence interval for the mean.
    
    Parameters:
    - data: array of values
    - iterations: number of bootstrap samples (default 1000)
    - confidence: confidence level (default 0.95)
    
    Returns:
    - (lower_bound, upper_bound) tuple
    '''
    means = np.zeros(iterations)
    n = len(data)
    
    for i in range(iterations):
        # Sample WITH REPLACEMENT (key for bootstrap!)
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        means[i] = np.mean(bootstrap_sample)
    
    # Percentiles for confidence interval
    alpha = (1 - confidence) / 2
    lower_bound = np.percentile(means, alpha * 100)
    upper_bound = np.percentile(means, (1 - alpha) * 100)
    
    return lower_bound, upper_bound

# Example usage
data = df['column'].dropna().values
ci_lower, ci_upper = bootstrap_confidence_interval(data, iterations=1000)
print(f"Mean: {np.mean(data):.4f}")
print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

# IMPORTANT EXAM NOTE:
# For 95% CI with 10000 iterations:
# - Lower bound = 250th value (2.5th percentile)
# - Upper bound = 9750th value (97.5th percentile)"""
        content += f'<details class="code-block"><summary>📝 Bootstrap Code</summary><pre><code>{code}</code></pre></details>'
        
        self.report.add_section(f"📊 Bootstrap Analysis: {name}", content, level=2)
    
    def _bootstrap_ci(self, data, iterations=1000, confidence=0.95):
        """Compute bootstrap confidence interval."""
        means = np.zeros(iterations)
        n = len(data)
        
        for i in range(iterations):
            bootstrap_sample = np.random.choice(data, size=n, replace=True)
            means[i] = np.mean(bootstrap_sample)
        
        alpha = (1 - confidence) / 2
        return np.percentile(means, alpha * 100), np.percentile(means, (1 - alpha) * 100)


# =============================================================================
# DISTRIBUTION ANALYZER (ECDF, Cumsum - Exams 2021, 2022)
# =============================================================================

class DistributionAnalyzer:
    """Distribution analysis: ECDF, cumsum, histograms (Exams 2021, 2022)."""
    
    def __init__(self, report: ReportBuilder):
        self.report = report
    
    def analyze(self, df, name):
        print(f"  📊 Running distribution analysis for {name}...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 1:
            return
        
        content = ""
        
        # ECDF plots (Exam 2022)
        content += self._ecdf_analysis(df, numeric_cols[:4], name)
        
        # Cumulative sum plots (Exam 2021)
        content += self._cumsum_analysis(df, numeric_cols[:4], name)
        
        # Distribution overview
        content += self._distribution_overview(df, numeric_cols[:6], name)
        
        if content:
            self.report.add_section(f"📊 Distribution Analysis: {name}", content, level=2)
    
    def _ecdf_analysis(self, df, numeric_cols, name):
        """ECDF plots (Exam 2022 - Wikipedia RfA)."""
        content = "<h4>ECDF Plots (Exam 2022)</h4>"
        
        try:
            n_cols = min(4, len(numeric_cols))
            fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
            if n_cols == 1:
                axes = [axes]
            
            for idx, col in enumerate(numeric_cols[:n_cols]):
                data = df[col].dropna().values
                if len(data) < 2:
                    continue
                
                # Compute ECDF
                sorted_data = np.sort(data)
                ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                
                axes[idx].plot(sorted_data, ecdf, linewidth=2)
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('ECDF')
                axes[idx].set_title(f'ECDF of {col}')
                axes[idx].grid(True, alpha=0.3)
                
                # Add median line
                median_val = np.median(data)
                axes[idx].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label=f'Median={median_val:.2f}')
                axes[idx].axvline(x=median_val, color='r', linestyle='--', alpha=0.5)
                axes[idx].legend(fontsize=8)
            
            plt.tight_layout()
            content += self.report.save_figure(fig, f"{name}_ecdf")
            
            code = """# ECDF Plot (Exam 2022)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Method 1: Manual ECDF computation
data = df['column'].dropna().values
sorted_data = np.sort(data)
ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

plt.plot(sorted_data, ecdf)
plt.xlabel('Value')
plt.ylabel('ECDF')
plt.title('Empirical Cumulative Distribution Function')

# Method 2: Using seaborn (preferred)
sns.ecdfplot(data=df, x='column')

# Method 3: Statsmodels ECDF
from statsmodels.distributions.empirical_distribution import ECDF
ecdf_func = ECDF(df['column'].dropna())
x = np.linspace(df['column'].min(), df['column'].max(), 100)
plt.plot(x, ecdf_func(x))

# ECDF Interpretation:
# - Y-axis: Proportion of data points <= x
# - Steeper slope = more data concentrated there
# - Jump at x = how many observations have exactly that value"""
            content += f'<details class="code-block"><summary>📝 ECDF Code</summary><pre><code>{code}</code></pre></details>'
            
        except Exception as e:
            content += f"<p>Error: {e}</p>"
        
        return content
    
    def _cumsum_analysis(self, df, numeric_cols, name):
        """Cumulative sum plots (Exam 2021 - Faculty Hiring)."""
        content = "<h4>Cumulative Sum Plots (Exam 2021)</h4>"
        
        try:
            # Look for columns that might be good for cumsum analysis
            # Typically used with sorted data by some ranking
            
            n_cols = min(3, len(numeric_cols))
            fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
            if n_cols == 1:
                axes = [axes]
            
            for idx, col in enumerate(numeric_cols[:n_cols]):
                data = df[col].dropna()
                if len(data) < 2:
                    continue
                
                # Sort and compute cumulative sum
                sorted_data = data.sort_values(ascending=False).values
                cumsum = np.cumsum(sorted_data)
                cumsum_frac = cumsum / cumsum[-1]  # Normalize to [0, 1]
                
                x = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                
                axes[idx].plot(x, cumsum_frac, linewidth=2)
                axes[idx].set_xlabel('Fraction of items (sorted)')
                axes[idx].set_ylabel(f'Cumulative fraction of {col}')
                axes[idx].set_title(f'Cumsum: {col}')
                axes[idx].grid(True, alpha=0.3)
                
                # Add reference line for uniform distribution
                axes[idx].plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Uniform')
                axes[idx].legend()
            
            plt.tight_layout()
            content += self.report.save_figure(fig, f"{name}_cumsum")
            
            code = """# Cumulative Sum Plot (Exam 2021)
import numpy as np
import matplotlib.pyplot as plt

# Sort data by some ranking (e.g., score, PageRank)
df_sorted = df.sort_values('score', ascending=False)

# Compute cumulative sum (e.g., of hires, publications)
cumsum = df_sorted['hires'].cumsum()

# Normalize to get cumulative fraction
cumsum_frac = cumsum / cumsum.iloc[-1]

# X-axis: fraction of institutions
x = np.arange(1, len(cumsum) + 1) / len(cumsum)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x, cumsum_frac, label='Actual')
plt.plot([0, 1], [0, 1], 'r--', label='Uniform (baseline)')
plt.xlabel('Fraction of institutions (by score)')
plt.ylabel('Cumulative fraction of hires')
plt.title('Cumulative Distribution of Hires')
plt.legend()
plt.grid(True, alpha=0.3)

# Interpretation:
# - If line is above diagonal: top-ranked have MORE than their share
# - If line is below diagonal: top-ranked have LESS than their share
# - Straight diagonal = uniform distribution

# Example from Exam 2021:
# "Top 10% of institutions produce what % of faculty?"
# Answer: cumsum_frac[int(0.1 * len(df))]"""
            content += f'<details class="code-block"><summary>📝 Cumsum Code</summary><pre><code>{code}</code></pre></details>'
            
        except Exception as e:
            content += f"<p>Error: {e}</p>"
        
        return content
    
    def _distribution_overview(self, df, numeric_cols, name):
        """Distribution overview with histograms and box plots."""
        content = "<h4>Distribution Overview</h4>"
        
        try:
            n_cols = min(4, len(numeric_cols))
            
            fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))
            
            for idx, col in enumerate(numeric_cols[:n_cols]):
                data = df[col].dropna()
                if len(data) < 2:
                    continue
                
                # Histogram
                axes[0, idx].hist(data, bins=30, alpha=0.7, color='steelblue', edgecolor='white')
                axes[0, idx].set_xlabel(col)
                axes[0, idx].set_ylabel('Count')
                axes[0, idx].set_title(f'Histogram: {col}')
                axes[0, idx].axvline(data.mean(), color='red', linestyle='--', label=f'Mean={data.mean():.2f}')
                axes[0, idx].axvline(data.median(), color='green', linestyle='--', label=f'Median={data.median():.2f}')
                axes[0, idx].legend(fontsize=8)
                
                # Box plot
                axes[1, idx].boxplot(data, vert=True)
                axes[1, idx].set_ylabel(col)
                axes[1, idx].set_title(f'Boxplot: {col}')
                axes[1, idx].grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            content += self.report.save_figure(fig, f"{name}_distributions")
            
            # Statistics table
            stats_data = []
            for col in numeric_cols[:n_cols]:
                data = df[col].dropna()
                stats_data.append({
                    'Column': col,
                    'Mean': data.mean(),
                    'Median': data.median(),
                    'Std': data.std(),
                    'Min': data.min(),
                    'Max': data.max(),
                    'Skew': data.skew()
                })
            
            stats_df = pd.DataFrame(stats_data).round(4)
            content += self.report.add_dataframe(stats_df, "Distribution Statistics")
            
            code = """# Distribution Analysis
import matplotlib.pyplot as plt
import seaborn as sns

# Histogram with KDE
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df['column'], kde=True, ax=ax)
ax.axvline(df['column'].mean(), color='red', linestyle='--', label='Mean')
ax.axvline(df['column'].median(), color='green', linestyle='--', label='Median')
ax.legend()

# Box plot
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, y='column')

# Multiple columns comparison
plt.figure(figsize=(12, 6))
df[['col1', 'col2', 'col3']].boxplot()

# Violin plot (combines boxplot and KDE)
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='category', y='value')

# Summary statistics
print(df['column'].describe())
print(f"Skewness: {df['column'].skew():.4f}")
print(f"Kurtosis: {df['column'].kurtosis():.4f}")"""
            content += f'<details class="code-block"><summary>📝 Distribution Code</summary><pre><code>{code}</code></pre></details>'
            
        except Exception as e:
            content += f"<p>Error: {e}</p>"
        
        return content


# =============================================================================
# CORRELATION ANALYZER
# =============================================================================

class CorrelationAnalyzer:
    """Correlation analysis including Spearman (Exam 2021)."""
    
    def __init__(self, report: ReportBuilder):
        self.report = report
    
    def analyze(self, df, name):
        print(f"  📈 Computing correlations for {name}...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            return
        
        use_cols = numeric_cols[:10]
        df_numeric = df[use_cols].dropna()
        
        if len(df_numeric) < 10:
            return
        
        content = ""
        findings = []
        
        # Pearson correlation
        pearson_corr = df_numeric.corr(method='pearson')
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        sns.heatmap(pearson_corr, annot=len(use_cols) <= 8, fmt='.2f', 
                   cmap='RdBu_r', center=0, ax=axes[0], vmin=-1, vmax=1)
        axes[0].set_title('Pearson Correlation')
        
        # Spearman correlation (Exam 2021)
        spearman_corr = df_numeric.corr(method='spearman')
        sns.heatmap(spearman_corr, annot=len(use_cols) <= 8, fmt='.2f',
                   cmap='RdBu_r', center=0, ax=axes[1], vmin=-1, vmax=1)
        axes[1].set_title('Spearman Rank Correlation')
        
        plt.tight_layout()
        content += self.report.save_figure(fig, f"{name}_correlations")
        
        # Find strongest correlations
        corr_pairs = []
        for i, col1 in enumerate(use_cols):
            for j, col2 in enumerate(use_cols):
                if i < j:
                    p_corr = pearson_corr.loc[col1, col2]
                    s_corr = spearman_corr.loc[col1, col2]
                    corr_pairs.append({
                        'Var1': col1, 'Var2': col2,
                        'Pearson': p_corr, 'Spearman': s_corr
                    })
        
        if corr_pairs:
            corr_df = pd.DataFrame(corr_pairs)
            corr_df['Abs_Spearman'] = corr_df['Spearman'].abs()
            top_corr = corr_df.nlargest(10, 'Abs_Spearman')[['Var1', 'Var2', 'Pearson', 'Spearman']].round(3)
            content += "<h5>Strongest Correlations</h5>"
            content += self.report.add_dataframe(top_corr)
            
            if len(top_corr) > 0:
                top = top_corr.iloc[0]
                findings.append(f"Strongest: {top['Var1']} ~ {top['Var2']} (ρ={top['Spearman']:.3f})")
        
        code = """# Correlation Analysis (Exam 2021 - Spearman)
import pandas as pd
from scipy import stats

# Pearson correlation (linear relationship)
pearson_corr = df[numeric_cols].corr(method='pearson')

# Spearman rank correlation (monotonic relationship)
# IMPORTANT: Exam 2021 specifically asks for Spearman!
spearman_corr = df[numeric_cols].corr(method='spearman')

# For specific pair with p-value:
corr, p_value = stats.spearmanr(df['col1'], df['col2'])
print(f"Spearman correlation: {corr:.4f}")
print(f"p-value: {p_value:.6f}")

# Pearson with p-value:
corr, p_value = stats.pearsonr(df['col1'], df['col2'])
print(f"Pearson correlation: {corr:.4f}")
print(f"p-value: {p_value:.6f}")

# Plot heatmap
import seaborn as sns
sns.heatmap(spearman_corr, annot=True, cmap='RdBu_r', center=0, vmin=-1, vmax=1)"""
        content += f'<details class="code-block"><summary>📝 Correlation Code</summary><pre><code>{code}</code></pre></details>'
        
        self.report.add_section(f"📈 Correlation Analysis: {name}", content, level=2,
                               key_findings=findings if findings else None)


# =============================================================================
# GROUPBY ANALYZER
# =============================================================================

class GroupbyAnalyzer:
    """GroupBy aggregation analysis."""
    
    def __init__(self, report: ReportBuilder):
        self.report = report
    
    def analyze(self, df, name):
        print(f"  📊 Running groupby analysis for {name}...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [c for c in df.select_dtypes(include=['object', 'category']).columns
                          if 2 <= df[c].nunique() <= MAX_CATEGORIES]
        
        if not numeric_cols or not categorical_cols:
            return
        
        content = ""
        
        for cat_col in categorical_cols[:2]:
            for num_col in numeric_cols[:2]:
                grouped = df.groupby(cat_col)[num_col].agg(['mean', 'std', 'count', 'median'])
                grouped = grouped.round(3).reset_index()
                
                content += f"<h5>{num_col} by {cat_col}</h5>"
                content += self.report.add_dataframe(grouped.head(15))
                
                # Bar plot
                if len(grouped) <= 15:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    grouped_sorted = grouped.sort_values('mean', ascending=False)
                    ax.bar(range(len(grouped_sorted)), grouped_sorted['mean'], 
                          yerr=grouped_sorted['std'], capsize=3, alpha=0.7)
                    ax.set_xticks(range(len(grouped_sorted)))
                    ax.set_xticklabels(grouped_sorted[cat_col], rotation=45, ha='right')
                    ax.set_ylabel(f'Mean {num_col}')
                    ax.set_title(f'{num_col} by {cat_col}')
                    ax.grid(True, alpha=0.3, axis='y')
                    plt.tight_layout()
                    content += self.report.save_figure(fig, f"{name}_{num_col}_by_{cat_col}")
        
        code = """# GroupBy Analysis
import pandas as pd

# Basic groupby with multiple aggregations
grouped = df.groupby('category_col')['numeric_col'].agg(['mean', 'std', 'count', 'median'])

# Multiple columns
grouped = df.groupby('category_col').agg({
    'col1': ['mean', 'std'],
    'col2': ['sum', 'count']
})

# Custom aggregations
grouped = df.groupby('category_col').agg(
    mean_val=('numeric_col', 'mean'),
    total=('numeric_col', 'sum'),
    n=('numeric_col', 'count')
)

# Plot with error bars
import matplotlib.pyplot as plt
grouped = df.groupby('category_col')['numeric_col'].agg(['mean', 'std']).reset_index()
plt.bar(grouped['category_col'], grouped['mean'], yerr=grouped['std'], capsize=5)
plt.xticks(rotation=45)"""
        content += f'<details class="code-block"><summary>📝 GroupBy Code</summary><pre><code>{code}</code></pre></details>'
        
        if content:
            self.report.add_section(f"📊 GroupBy Analysis: {name}", content, level=2)


# =============================================================================
# REGRESSION ANALYZER
# =============================================================================

class RegressionAnalyzer:
    """OLS, Ridge, Treatment encoding regression analysis."""
    
    def __init__(self, report: ReportBuilder):
        self.report = report
    
    def analyze(self, df, name):
        print(f"  📈 Running regression analyses for {name}...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [c for c in df.select_dtypes(include=['object', 'category']).columns
                          if 2 <= df[c].nunique() <= 15]
        
        content = ""
        findings = []
        
        # OLS with Treatment encoding (Exam 2023)
        if categorical_cols and numeric_cols:
            result = self._ols_with_treatment(df, numeric_cols[0], categorical_cols[0], name)
            if result:
                content += result['content']
                findings.extend(result.get('findings', []))
        
        # Ridge regression (Exam 2019)
        if len(numeric_cols) >= 3:
            content += self._ridge_regression(df, numeric_cols, name)
        
        # Numeric OLS
        if len(numeric_cols) >= 2:
            result = self._ols_numeric(df, numeric_cols, name)
            if result:
                content += result
        
        if content:
            self.report.add_section(f"📈 Regression Analysis: {name}", content, level=2,
                                   key_findings=findings[:5] if findings else None)
    
    def _ols_with_treatment(self, df, num_col, cat_col, name):
        """OLS with Treatment (categorical) encoding (Exam 2023)."""
        try:
            df_clean = df[[num_col, cat_col]].dropna()
            if len(df_clean) < 20:
                return None
            
            # Escape special characters in category names
            df_clean = df_clean.copy()
            df_clean[cat_col] = df_clean[cat_col].astype(str).str.replace(r'[^\w\s]', '_', regex=True)
            
            # Get reference category (most frequent)
            ref = df_clean[cat_col].value_counts().index[0]
            
            formula = f'{num_col} ~ C({cat_col}, Treatment(reference="{ref}"))'
            
            try:
                model = smf.ols(formula=formula, data=df_clean).fit()
            except:
                formula = f'{num_col} ~ C({cat_col})'
                model = smf.ols(formula=formula, data=df_clean).fit()
            
            content = f"<h5>OLS with Treatment Encoding: {num_col} ~ {cat_col}</h5>"
            content += f"<p><strong>Reference category:</strong> {ref}</p>"
            content += f"<p><strong>R²:</strong> {model.rsquared:.4f}</p>"
            
            summary_df = pd.DataFrame({
                'coefficient': model.params,
                'std_err': model.bse,
                'p_value': model.pvalues,
                'significant': ['✅' if p < 0.05 else '' for p in model.pvalues]
            }).round(4)
            content += self.report.add_dataframe(summary_df)
            
            content += """
            <div class="info">
                <strong>Interpretation:</strong>
                <ul>
                    <li>Intercept = mean when category = reference</li>
                    <li>Other coefficients = change vs reference</li>
                    <li>p < 0.05 = statistically significant</li>
                </ul>
            </div>
            """
            
            code = f"""# OLS with Treatment Encoding (Exam 2023)
import statsmodels.formula.api as smf

# Treatment encoding: one category is reference, others compared to it
# Reference = "{ref}" (most frequent category)

formula = '{num_col} ~ C({cat_col}, Treatment(reference="{ref}"))'
model = smf.ols(formula=formula, data=df).fit()
print(model.summary())

# Key interpretation:
# - Intercept = mean {num_col} when {cat_col} = "{ref}"
# - C({cat_col})[T.X] = how much category X differs from reference

# Alternative: simple categorical encoding
formula_simple = '{num_col} ~ C({cat_col})'
model_simple = smf.ols(formula=formula_simple, data=df).fit()"""
            content += f'<details class="code-block"><summary>📝 OLS Treatment Code</summary><pre><code>{code}</code></pre></details>'
            
            return {'content': content, 'findings': [f"R²={model.rsquared:.3f} for {num_col}~{cat_col}"]}
        except Exception as e:
            return None
    
    def _ridge_regression(self, df, numeric_cols, name):
        """Ridge regression with CV (Exam 2019)."""
        content = "<h5>Ridge Regression (RidgeCV)</h5>"
        
        try:
            df_clean = df[numeric_cols].dropna()
            if len(df_clean) < 20:
                return content + "<p>Not enough data</p>"
            
            y_col = numeric_cols[0]
            X_cols = numeric_cols[1:min(6, len(numeric_cols))]
            
            X = df_clean[X_cols].values
            y = df_clean[y_col].values
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=DEFAULT_TEST_SIZE, random_state=RANDOM_STATE
            )
            
            alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
            ridge = RidgeCV(alphas=alphas)
            ridge.fit(X_train, y_train)
            
            train_score = ridge.score(X_train, y_train)
            test_score = ridge.score(X_test, y_test)
            y_pred = ridge.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            
            content += f"""
            <table class="dataframe">
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Target</td><td>{y_col}</td></tr>
                <tr><td>Features</td><td>{', '.join(X_cols)}</td></tr>
                <tr><td>Best alpha</td><td>{ridge.alpha_}</td></tr>
                <tr><td>Train R²</td><td>{train_score:.4f}</td></tr>
                <tr><td>Test R²</td><td>{test_score:.4f}</td></tr>
                <tr><td>Test MAE</td><td>{mae:.4f}</td></tr>
            </table>
            """
            
            coef_df = pd.DataFrame({'feature': X_cols, 'coefficient': ridge.coef_}).round(4)
            content += self.report.add_dataframe(coef_df, "Coefficients")
            
            code = f"""# RidgeCV (Exam 2019)
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

X = df[{X_cols}].values
y = df['{y_col}'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# RidgeCV automatically selects best alpha via cross-validation
alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
ridge = RidgeCV(alphas=alphas)
ridge.fit(X_train, y_train)

print(f"Best alpha: {{ridge.alpha_}}")
print(f"Train R²: {{ridge.score(X_train, y_train):.4f}}")
print(f"Test R²: {{ridge.score(X_test, y_test):.4f}}")
print(f"Coefficients: {{dict(zip({X_cols}, ridge.coef_))}}")"""
            content += f'<details class="code-block"><summary>📝 RidgeCV Code</summary><pre><code>{code}</code></pre></details>'
            
        except Exception as e:
            content += f"<p>Error: {e}</p>"
        
        return content
    
    def _ols_numeric(self, df, numeric_cols, name):
        """OLS with numeric predictors."""
        try:
            df_clean = df[numeric_cols[:5]].dropna()
            if len(df_clean) < 20:
                return None
            
            y_col = numeric_cols[0]
            x_cols = numeric_cols[1:min(4, len(numeric_cols))]
            
            formula = f"{y_col} ~ " + " + ".join(x_cols)
            model = smf.ols(formula=formula, data=df_clean).fit()
            
            content = f"<h5>Numeric OLS: {formula}</h5>"
            content += f"<p><strong>R²:</strong> {model.rsquared:.4f}</p>"
            
            summary_df = pd.DataFrame({
                'coefficient': model.params,
                'p_value': model.pvalues,
                'significant': ['✅' if p < 0.05 else '' for p in model.pvalues]
            }).round(4)
            content += self.report.add_dataframe(summary_df)
            
            return content
        except:
            return None


# =============================================================================
# CLASSIFICATION ANALYZER (ENHANCED in v5)
# =============================================================================

class ClassificationAnalyzer:
    """Complete classification: DT, LogReg, k-NN, Random Forest, ROC/AUC."""
    
    def __init__(self, report: ReportBuilder):
        self.report = report
    
    def analyze(self, df, name):
        print(f"  🎯 Running classification analyses for {name}...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [c for c in df.select_dtypes(include=['object', 'category']).columns
                          if 2 <= df[c].nunique() <= MAX_CATEGORIES]
        
        # Create binary targets from numeric columns (Exam 2019)
        df = df.copy()
        for num_col in numeric_cols[:2]:
            if df[num_col].nunique() > 10:
                median_val = df[num_col].median()
                binary_col = f"{num_col}_above_median"
                df[binary_col] = (df[num_col] > median_val).map({True: 'above', False: 'below'})
                categorical_cols.append(binary_col)
        
        if len(numeric_cols) < 1 or len(categorical_cols) < 1:
            return
        
        content = ""
        findings = []
        
        for target_col in categorical_cols[:2]:
            feature_cols = [c for c in numeric_cols if c != target_col and '_above_median' not in target_col]
            if len(feature_cols) < 1:
                continue
            
            # Decision Tree (Exam 2023)
            result = self._decision_tree(df, feature_cols, target_col, name)
            if result:
                content += result['content']
                findings.extend(result.get('findings', []))
            
            # Binary classification only
            if df[target_col].nunique() == 2:
                # k-NN (NEW in v5)
                result = self._knn_classifier(df, feature_cols, target_col, name)
                if result:
                    content += result['content']
                
                # Random Forest (NEW in v5)
                result = self._random_forest(df, feature_cols, target_col, name)
                if result:
                    content += result['content']
                
                # Logistic Regression with ROC/AUC (ENHANCED in v5)
                result = self._logistic_with_roc(df, feature_cols, target_col, name)
                if result:
                    content += result['content']
                
                # LogisticRegressionCV (Exam 2019)
                result = self._logistic_regression_cv(df, feature_cols, target_col, name)
                if result:
                    content += result['content']
        
        if content:
            self.report.add_section(f"🎯 Classification: {name}", content, level=2, 
                                   key_findings=findings[:5])
    
    def _decision_tree(self, df, feature_cols, target_col, name, test_size=0.3):
        """DecisionTree classifier (Exam 2023)."""
        try:
            use_cols = feature_cols[:10]
            df_clean = df[use_cols + [target_col]].dropna()
            
            if len(df_clean) < 10:
                return None
            
            X = df_clean[use_cols].values
            y, labels = pd.factorize(df_clean[target_col])
            n_classes = len(labels)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=RANDOM_STATE
            )
            
            dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
            dt.fit(X_train, y_train)
            y_pred = dt.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            random_baseline = 1 / n_classes
            
            content = f"<h5>DecisionTree: {target_col}</h5>"
            content += f"""
            <table class="dataframe">
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Accuracy</td><td>{acc:.4f}</td></tr>
                <tr><td>Balanced Accuracy</td><td>{bal_acc:.4f}</td></tr>
                <tr><td>Random Baseline</td><td>{random_baseline:.4f}</td></tr>
            </table>
            """
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            cm_norm = cm / cm.sum()
            
            content += """
            <div class="critical">
                <strong>⚠️ EXAM STANDARD:</strong> Confusion matrix normalized so <strong>ALL cells sum to 1</strong>!
            </div>
            """
            
            if n_classes <= 10:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                sns.heatmap(cm, annot=True, fmt='d', ax=axes[0], cmap='Blues',
                           xticklabels=labels, yticklabels=labels)
                axes[0].set_title('Raw Counts')
                axes[0].set_xlabel('Predicted')
                axes[0].set_ylabel('Actual')
                
                sns.heatmap(cm_norm, annot=True, fmt='.3f', ax=axes[1], cmap='Blues',
                           xticklabels=labels, yticklabels=labels)
                axes[1].set_title('Normalized (ALL sum to 1)')
                axes[1].set_xlabel('Predicted')
                axes[1].set_ylabel('Actual')
                
                plt.tight_layout()
                content += self.report.save_figure(fig, f"cm_dt_{name}")
            
            code = f"""# DecisionTree (Exam 2023)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size={test_size}, random_state=42
)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
balanced_acc = balanced_accuracy_score(y_test, y_pred)
random_baseline = 1 / n_classes  # NOT most frequent class!

# CRITICAL: Normalize so ALL cells sum to 1
cm = confusion_matrix(y_test, y_pred)
cm_normalized = cm / cm.sum()  # NOT cm / cm.sum(axis=1)!

print(f"Accuracy: {{accuracy:.4f}}")
print(f"Balanced Accuracy: {{balanced_acc:.4f}}")

# Feature importance
for feat, imp in sorted(zip(feature_names, dt.feature_importances_), 
                        key=lambda x: -x[1]):
    print(f"  {{feat}}: {{imp:.4f}}")"""
            content += f'<details class="code-block"><summary>📝 DecisionTree Code</summary><pre><code>{code}</code></pre></details>'
            
            return {'content': content, 'findings': [f"DT acc={acc:.3f}, bal_acc={bal_acc:.3f}"]}
        except:
            return None
    
    def _knn_classifier(self, df, feature_cols, target_col, name):
        """k-NN Classifier (Lecture 4, SupervisedLearning exercise) - NEW in v5."""
        try:
            use_cols = feature_cols[:10]
            df_clean = df[use_cols + [target_col]].dropna()
            
            if len(df_clean) < 20:
                return None
            
            X = df_clean[use_cols].values
            y, labels = pd.factorize(df_clean[target_col])
            
            # Scale features for k-NN
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=DEFAULT_TEST_SIZE, random_state=RANDOM_STATE
            )
            
            # Test different k values
            k_values = [1, 3, 5, 7, 9, 11, 15]
            results = []
            
            for k in k_values:
                if k >= len(X_train):
                    continue
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(X_train, y_train)
                train_acc = knn.score(X_train, y_train)
                test_acc = knn.score(X_test, y_test)
                results.append({'k': k, 'train_acc': train_acc, 'test_acc': test_acc})
            
            if not results:
                return None
            
            results_df = pd.DataFrame(results).round(4)
            best_k = results_df.loc[results_df['test_acc'].idxmax(), 'k']
            
            content = f"<h5>k-NN Classifier: {target_col}</h5>"
            content += self.report.add_dataframe(results_df)
            content += f"<p><strong>Best k:</strong> {int(best_k)}</p>"
            
            # Plot k vs accuracy
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot([r['k'] for r in results], [r['train_acc'] for r in results], 'b-o', label='Train')
            ax.plot([r['k'] for r in results], [r['test_acc'] for r in results], 'r-o', label='Test')
            ax.set_xlabel('k (number of neighbors)')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'k-NN: Effect of k on Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            content += self.report.save_figure(fig, f"knn_{name}")
            
            code = """# k-NN Classifier (Lecture 4, SupervisedLearning exercise)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# IMPORTANT: Scale features for k-NN (distance-based)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# Test different k values to find optimal
k_values = [1, 3, 5, 7, 9, 11, 15]
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    print(f"k={k}: Train acc={knn.score(X_train, y_train):.3f}, "
          f"Test acc={knn.score(X_test, y_test):.3f}")

# Final model with best k
best_k = 5  # Choose based on test accuracy
knn_final = KNeighborsClassifier(n_neighbors=best_k)
knn_final.fit(X_train, y_train)

# k-NN properties (EXAM QUESTIONS):
# - No training time (lazy learner)
# - Handles complex decision boundaries
# - No retraining needed for new data
# - BUT: slow prediction, sensitive to k choice"""
            content += f'<details class="code-block"><summary>📝 k-NN Code</summary><pre><code>{code}</code></pre></details>'
            
            return {'content': content}
        except:
            return None
    
    def _random_forest(self, df, feature_cols, target_col, name):
        """Random Forest Classifier (Lecture 5, SupervisedLearning) - NEW in v5."""
        try:
            use_cols = feature_cols[:10]
            df_clean = df[use_cols + [target_col]].dropna()
            
            if len(df_clean) < 20:
                return None
            
            X = df_clean[use_cols].values
            y, labels = pd.factorize(df_clean[target_col])
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=DEFAULT_TEST_SIZE, random_state=RANDOM_STATE
            )
            
            # Test different n_estimators
            n_trees = [1, 5, 10, 20, 50]
            results = []
            
            for n in n_trees:
                rf = RandomForestClassifier(n_estimators=n, max_depth=3, random_state=RANDOM_STATE)
                rf.fit(X_train, y_train)
                
                # Cross-validation scores
                cv_precision = cross_val_score(rf, X_train, y_train, cv=5, scoring='precision_macro').mean()
                cv_recall = cross_val_score(rf, X_train, y_train, cv=5, scoring='recall_macro').mean()
                test_acc = rf.score(X_test, y_test)
                
                results.append({
                    'n_trees': n, 
                    'test_acc': test_acc,
                    'cv_precision': cv_precision,
                    'cv_recall': cv_recall
                })
            
            results_df = pd.DataFrame(results).round(4)
            
            content = f"<h5>Random Forest: {target_col}</h5>"
            content += self.report.add_dataframe(results_df)
            
            # Feature importance from best model
            rf_best = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=RANDOM_STATE)
            rf_best.fit(X_train, y_train)
            
            importance_df = pd.DataFrame({
                'feature': use_cols,
                'importance': rf_best.feature_importances_
            }).sort_values('importance', ascending=False).round(4)
            content += self.report.add_dataframe(importance_df.head(10), "Feature Importance")
            
            # Plot
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            axes[0].plot([r['n_trees'] for r in results], [r['cv_precision'] for r in results], 'b-o', label='Precision')
            axes[0].plot([r['n_trees'] for r in results], [r['cv_recall'] for r in results], 'r-o', label='Recall')
            axes[0].set_xlabel('Number of Trees')
            axes[0].set_ylabel('Score')
            axes[0].set_title('Precision/Recall vs Number of Trees')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            axes[1].barh(importance_df['feature'].head(8), importance_df['importance'].head(8))
            axes[1].set_xlabel('Importance')
            axes[1].set_title('Feature Importance')
            
            plt.tight_layout()
            content += self.report.save_figure(fig, f"rf_{name}")
            
            code = """# Random Forest (Lecture 5, SupervisedLearning exercise)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Random Forest with cross-validation for precision/recall
# EXAM PATTERN: Vary n_estimators, plot precision/recall

n_trees_range = range(1, 21)
precision_scores = []
recall_scores = []

for n in n_trees_range:
    rf = RandomForestClassifier(n_estimators=n, max_depth=3, random_state=0)
    
    # 10-fold cross-validation
    precision = cross_val_score(rf, X, y, cv=10, scoring='precision').mean()
    recall = cross_val_score(rf, X, y, cv=10, scoring='recall').mean()
    
    precision_scores.append(precision)
    recall_scores.append(recall)

# Plot precision/recall curves
import matplotlib.pyplot as plt
plt.plot(n_trees_range, precision_scores, label='Precision')
plt.plot(n_trees_range, recall_scores, label='Recall')
plt.xlabel('Number of Trees')
plt.ylabel('Score')
plt.legend()

# Feature importance
rf = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=0)
rf.fit(X_train, y_train)
for feat, imp in sorted(zip(feature_names, rf.feature_importances_), 
                        key=lambda x: -x[1]):
    print(f"  {feat}: {imp:.4f}")"""
            content += f'<details class="code-block"><summary>📝 Random Forest Code</summary><pre><code>{code}</code></pre></details>'
            
            return {'content': content}
        except:
            return None
    
    def _logistic_with_roc(self, df, feature_cols, target_col, name):
        """Logistic Regression with ROC/AUC (AppliedML exercise) - NEW in v5."""
        try:
            use_cols = feature_cols[:10]
            df_clean = df[use_cols + [target_col]].dropna()
            
            if len(df_clean) < 20 or df_clean[target_col].nunique() != 2:
                return None
            
            X = df_clean[use_cols].values
            y, labels = pd.factorize(df_clean[target_col])
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=DEFAULT_TEST_SIZE, random_state=RANDOM_STATE
            )
            
            # Logistic regression
            lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
            lr.fit(X_train, y_train)
            
            # Predictions
            y_pred = lr.predict(X_test)
            y_pred_proba = lr.predict_proba(X_test)[:, 1]
            
            # Metrics
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # ROC/AUC
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            auc_score = auc(fpr, tpr)
            
            content = f"<h5>Logistic Regression with ROC/AUC: {target_col}</h5>"
            content += f"""
            <table class="dataframe">
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Accuracy</td><td>{acc:.4f}</td></tr>
                <tr><td>Precision</td><td>{precision:.4f}</td></tr>
                <tr><td>Recall</td><td>{recall:.4f}</td></tr>
                <tr><td>F1 Score</td><td>{f1:.4f}</td></tr>
                <tr><td><strong>AUC</strong></td><td><strong>{auc_score:.4f}</strong></td></tr>
            </table>
            """
            
            # ROC curve plot
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # ROC curve
            axes[0].plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc_score:.3f})')
            axes[0].plot([0, 1], [0, 1], 'r--', label='Random')
            axes[0].set_xlabel('False Positive Rate')
            axes[0].set_ylabel('True Positive Rate')
            axes[0].set_title('ROC Curve')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Coefficients
            coef_df = pd.DataFrame({
                'feature': use_cols,
                'coefficient': lr.coef_[0]
            }).sort_values('coefficient', key=abs, ascending=False)
            
            axes[1].barh(coef_df['feature'].head(10), coef_df['coefficient'].head(10))
            axes[1].set_xlabel('Coefficient')
            axes[1].set_title('Feature Coefficients')
            axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            
            plt.tight_layout()
            content += self.report.save_figure(fig, f"logreg_roc_{name}")
            
            code = """# Logistic Regression with ROC/AUC (AppliedML exercise)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# Fit logistic regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

# Predictions and probabilities
y_pred = lr.predict(X_test)
y_pred_proba = lr.predict_proba(X_test)[:, 1]  # Probability of positive class

# Classification metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

# ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc_score = auc(fpr, tpr)
print(f"AUC: {auc_score:.4f}")

# Plot ROC curve
import matplotlib.pyplot as plt
plt.plot(fpr, tpr, label=f'ROC (AUC = {auc_score:.3f})')
plt.plot([0, 1], [0, 1], 'r--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

# Cross-validation ROC (from exercise)
from sklearn.model_selection import cross_val_predict
y_pred_cv = cross_val_predict(lr, X, y, cv=10, method='predict_proba')
fpr_cv, tpr_cv, _ = roc_curve(y, y_pred_cv[:, 1])
auc_cv = auc(fpr_cv, tpr_cv)"""
            content += f'<details class="code-block"><summary>📝 ROC/AUC Code</summary><pre><code>{code}</code></pre></details>'
            
            return {'content': content}
        except:
            return None
    
    def _logistic_regression_cv(self, df, feature_cols, target_col, name):
        """LogisticRegressionCV (Exam 2019)."""
        try:
            use_cols = feature_cols[:10]
            df_clean = df[use_cols + [target_col]].dropna()
            
            if len(df_clean) < 20 or df_clean[target_col].nunique() != 2:
                return None
            
            X = df_clean[use_cols].values
            y, labels = pd.factorize(df_clean[target_col])
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=DEFAULT_TEST_SIZE, random_state=RANDOM_STATE
            )
            
            Cs = [0.01, 0.1, 1, 10, 100]
            lr_cv = LogisticRegressionCV(Cs=Cs, cv=3, random_state=RANDOM_STATE, max_iter=200)
            lr_cv.fit(X_train, y_train)
            
            test_acc = lr_cv.score(X_test, y_test)
            
            content = f"<h5>LogisticRegressionCV: {target_col}</h5>"
            content += f"""
            <table class="dataframe">
                <tr><th>Parameter</th><th>Value</th></tr>
                <tr><td>Cs tested</td><td>{Cs}</td></tr>
                <tr><td>Best C</td><td>{lr_cv.C_[0]}</td></tr>
                <tr><td>Test Accuracy</td><td>{test_acc:.4f}</td></tr>
            </table>
            """
            
            code = """# LogisticRegressionCV (Exam 2019)
from sklearn.linear_model import LogisticRegressionCV

Cs = [0.01, 0.1, 1, 10, 100]
lr_cv = LogisticRegressionCV(Cs=Cs, cv=3, random_state=42, max_iter=200)
lr_cv.fit(X_train, y_train)

print(f"Best C: {lr_cv.C_}")
print(f"Test Accuracy: {lr_cv.score(X_test, y_test):.4f}")"""
            content += f'<details class="code-block"><summary>📝 LogRegCV Code</summary><pre><code>{code}</code></pre></details>'
            
            return {'content': content}
        except:
            return None


# =============================================================================
# CLUSTERING ANALYZER (ENHANCED in v5)
# =============================================================================

class ClusteringAnalyzer:
    """K-Means with Silhouette, PCA, t-SNE visualization."""
    
    def __init__(self, report: ReportBuilder):
        self.report = report
    
    def analyze(self, df, name):
        print(f"  🎯 Running clustering analysis for {name}...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            return
        
        use_cols = numeric_cols[:10]
        df_clean = df[use_cols].dropna()
        
        if len(df_clean) < 30:
            return
        
        X = df_clean.values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        content = ""
        
        # K-Means with elbow and silhouette
        content += self._kmeans_analysis(X_scaled, name, use_cols)
        
        # Dimensionality reduction visualization
        content += self._dimensionality_reduction(X_scaled, name)
        
        if content:
            self.report.add_section(f"🎯 Clustering & Dim. Reduction: {name}", content, level=2)
    
    def _kmeans_analysis(self, X_scaled, name, feature_names):
        """K-Means with elbow and silhouette score."""
        content = "<h4>K-Means Clustering</h4>"
        
        try:
            K_range = range(2, min(10, len(X_scaled) // 5))
            inertias = []
            silhouettes = []
            
            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                inertias.append(kmeans.inertia_)
                silhouettes.append(silhouette_score(X_scaled, labels))
            
            # Find best k by silhouette
            best_k_idx = np.argmax(silhouettes)
            best_k = list(K_range)[best_k_idx]
            
            results_df = pd.DataFrame({
                'k': list(K_range),
                'inertia': inertias,
                'silhouette': silhouettes
            }).round(4)
            content += self.report.add_dataframe(results_df)
            content += f"<p><strong>Best k (by silhouette):</strong> {best_k}</p>"
            
            # Plots
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            axes[0].plot(K_range, inertias, 'bo-')
            axes[0].set_xlabel('k')
            axes[0].set_ylabel('Inertia (SSE)')
            axes[0].set_title('Elbow Method')
            axes[0].grid(True, alpha=0.3)
            
            axes[1].plot(K_range, silhouettes, 'go-')
            axes[1].axvline(x=best_k, color='r', linestyle='--', label=f'Best k={best_k}')
            axes[1].set_xlabel('k')
            axes[1].set_ylabel('Silhouette Score')
            axes[1].set_title('Silhouette Analysis')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            content += self.report.save_figure(fig, f"{name}_kmeans")
            
            code = """# K-Means Clustering with Silhouette (Unsupervised_Learning exercise)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal k
inertias = []
silhouettes = []

for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))
    
    print(f"k={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouettes[-1]:.4f}")

# Best k by silhouette
best_k = silhouettes.index(max(silhouettes)) + 2
print(f"\\nBest k: {best_k}")

# Final clustering
kmeans_final = KMeans(n_clusters=best_k, random_state=42)
labels = kmeans_final.fit_predict(X_scaled)
cluster_centers = kmeans_final.cluster_centers_

# Plot elbow
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(range(2, 10), inertias, 'bo-')
ax1.set_xlabel('k')
ax1.set_ylabel('Inertia')
ax1.set_title('Elbow Method')

ax2.plot(range(2, 10), silhouettes, 'go-')
ax2.set_xlabel('k')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Analysis')"""
            content += f'<details class="code-block"><summary>📝 K-Means Code</summary><pre><code>{code}</code></pre></details>'
            
        except Exception as e:
            content += f"<p>Error: {e}</p>"
        
        return content
    
    def _dimensionality_reduction(self, X_scaled, name):
        """PCA and t-SNE visualization (Unsupervised_Learning exercise) - NEW in v5."""
        content = "<h4>Dimensionality Reduction Visualization</h4>"
        
        try:
            # Limit samples for t-SNE (computationally expensive)
            n_samples = min(1000, len(X_scaled))
            X_sample = X_scaled[:n_samples]
            
            # PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_sample)
            
            # t-SNE
            tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=min(30, n_samples // 4))
            X_tsne = tsne.fit_transform(X_sample)
            
            # Cluster for coloring
            kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE)
            labels = kmeans.fit_predict(X_sample)
            
            # Plot
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6, s=20)
            axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
            axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
            axes[0].set_title('PCA Projection')
            plt.colorbar(scatter1, ax=axes[0], label='Cluster')
            
            scatter2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis', alpha=0.6, s=20)
            axes[1].set_xlabel('t-SNE 1')
            axes[1].set_ylabel('t-SNE 2')
            axes[1].set_title('t-SNE Projection')
            plt.colorbar(scatter2, ax=axes[1], label='Cluster')
            
            plt.tight_layout()
            content += self.report.save_figure(fig, f"{name}_dimred")
            
            # PCA explained variance
            content += f"""
            <p><strong>PCA Explained Variance:</strong></p>
            <ul>
                <li>PC1: {pca.explained_variance_ratio_[0]:.2%}</li>
                <li>PC2: {pca.explained_variance_ratio_[1]:.2%}</li>
                <li>Total: {sum(pca.explained_variance_ratio_):.2%}</li>
            </ul>
            """
            
            code = """# PCA and t-SNE (Unsupervised_Learning exercise)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Scale features first
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA - linear dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"PCA explained variance: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")

# t-SNE - non-linear dimensionality reduction
# Good for visualization, preserves local structure
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

# Plot both
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)
ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax1.set_title('PCA')

ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis', alpha=0.6)
ax2.set_xlabel('t-SNE 1')
ax2.set_ylabel('t-SNE 2')
ax2.set_title('t-SNE')

# KEY DIFFERENCES (EXAM):
# PCA: Linear, fast, preserves global structure, variance-based
# t-SNE: Non-linear, slow, preserves local structure, probability-based"""
            content += f'<details class="code-block"><summary>📝 PCA/t-SNE Code</summary><pre><code>{code}</code></pre></details>'
            
        except Exception as e:
            content += f"<p>Error: {e}</p>"
        
        return content


# =============================================================================
# PROPENSITY SCORE MATCHING (NEW in v5 - Lecture 6)
# =============================================================================

class PropensityMatchingAnalyzer:
    """Propensity Score Matching for Observational Studies (Lecture 6, Exercise)."""
    
    def __init__(self, report: ReportBuilder):
        self.report = report
    
    def analyze(self, df, name):
        print(f"  ⚖️ Checking for propensity score matching in {name}...")
        
        # Look for treatment/control indicators
        binary_cols = [c for c in df.columns 
                      if df[c].nunique() == 2 and df[c].dtype in ['int64', 'float64', 'object']]
        
        # Check for typical treatment column names
        treatment_cols = [c for c in df.columns 
                        if any(t in c.lower() for t in ['treat', 'treatment', 'control', 'group', 'intervention'])]
        
        candidate_cols = list(set(binary_cols + treatment_cols))
        
        if not candidate_cols:
            return
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in candidate_cols]
        
        if len(numeric_cols) < 2:
            return
        
        content = ""
        
        for treat_col in candidate_cols[:1]:  # Only analyze first candidate
            result = self._propensity_analysis(df, treat_col, numeric_cols, name)
            if result:
                content += result
        
        if content:
            self.report.add_section(f"⚖️ Propensity Score Analysis: {name}", content, level=2)
    
    def _propensity_analysis(self, df, treat_col, covariate_cols, name):
        """Run propensity score analysis."""
        try:
            df_clean = df[[treat_col] + covariate_cols].dropna()
            if len(df_clean) < 50:
                return None
            
            # Ensure binary treatment
            treat_values = df_clean[treat_col].unique()
            if len(treat_values) != 2:
                return None
            
            # Map to 0/1
            df_clean = df_clean.copy()
            df_clean['treat_binary'] = pd.factorize(df_clean[treat_col])[0]
            
            content = f"<h4>Propensity Score Matching: {treat_col}</h4>"
            
            content += """
            <div class="info">
                <strong>Observational Study Workflow (Lecture 6):</strong>
                <ol>
                    <li>Estimate propensity scores (probability of treatment)</li>
                    <li>Match treated with control based on propensity scores</li>
                    <li>Check balance after matching</li>
                    <li>Compare outcomes between matched groups</li>
                </ol>
            </div>
            """
            
            # Fit propensity score model
            use_covariates = covariate_cols[:6]
            formula = 'treat_binary ~ ' + ' + '.join(use_covariates)
            
            try:
                model = smf.logit(formula=formula, data=df_clean).fit(disp=0)
                df_clean['propensity_score'] = model.predict()
            except:
                # Fallback to sklearn
                X = df_clean[use_covariates].values
                y = df_clean['treat_binary'].values
                lr = LogisticRegression(max_iter=1000)
                lr.fit(X, y)
                df_clean['propensity_score'] = lr.predict_proba(X)[:, 1]
            
            # Distribution of propensity scores
            treated = df_clean[df_clean['treat_binary'] == 1]['propensity_score']
            control = df_clean[df_clean['treat_binary'] == 0]['propensity_score']
            
            content += f"""
            <h5>Propensity Score Distribution</h5>
            <table class="dataframe">
                <tr><th>Group</th><th>N</th><th>Mean PS</th><th>Std PS</th></tr>
                <tr><td>Treated</td><td>{len(treated)}</td><td>{treated.mean():.4f}</td><td>{treated.std():.4f}</td></tr>
                <tr><td>Control</td><td>{len(control)}</td><td>{control.mean():.4f}</td><td>{control.std():.4f}</td></tr>
            </table>
            """
            
            # Plot propensity score distributions
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(treated, bins=30, alpha=0.6, label='Treated', density=True)
            ax.hist(control, bins=30, alpha=0.6, label='Control', density=True)
            ax.set_xlabel('Propensity Score')
            ax.set_ylabel('Density')
            ax.set_title('Propensity Score Distribution by Group')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            content += self.report.save_figure(fig, f"{name}_propensity")
            
            # Covariate balance before matching
            content += "<h5>Covariate Balance (Before Matching)</h5>"
            balance_data = []
            for cov in use_covariates:
                treated_mean = df_clean[df_clean['treat_binary'] == 1][cov].mean()
                control_mean = df_clean[df_clean['treat_binary'] == 0][cov].mean()
                std_diff = (treated_mean - control_mean) / df_clean[cov].std()
                balance_data.append({
                    'Covariate': cov,
                    'Treated Mean': treated_mean,
                    'Control Mean': control_mean,
                    'Std Diff': std_diff
                })
            
            balance_df = pd.DataFrame(balance_data).round(4)
            content += self.report.add_dataframe(balance_df)
            
            code = """# Propensity Score Matching (Lecture 6, Observational Studies Exercise)
import pandas as pd
import numpy as np
import networkx as nx
import statsmodels.formula.api as smf

# Step 1: Estimate propensity scores using logistic regression
# Propensity score = P(treatment | covariates)

# Standardize continuous features first (optional but recommended)
for col in ['age', 'educ', 're74', 're75']:
    df[col] = (df[col] - df[col].mean()) / df[col].std()

# Fit logistic regression for propensity scores
formula = 'treat ~ age + educ + C(black) + C(hispan) + C(married) + C(nodegree) + re74 + re75'
model = smf.logit(formula=formula, data=df).fit()
df['propensity_score'] = model.predict()

print(model.summary())

# Step 2: Match treated with control using max weight matching
# Create bipartite graph for matching

def get_similarity(ps1, ps2):
    '''Similarity based on propensity score difference'''
    return 1 - np.abs(ps1 - ps2)

treated_df = df[df['treat'] == 1]
control_df = df[df['treat'] == 0]

G = nx.Graph()

# Add edges weighted by similarity
for t_idx, t_row in treated_df.iterrows():
    for c_idx, c_row in control_df.iterrows():
        similarity = get_similarity(t_row['propensity_score'], 
                                   c_row['propensity_score'])
        G.add_weighted_edges_from([(t_idx, c_idx, similarity)])

# Find maximum weight matching (Hungarian algorithm)
matching = nx.max_weight_matching(G, maxcardinality=True)

# Step 3: Extract matched pairs
matched_indices = []
for pair in matching:
    matched_indices.extend(pair)

matched_df = df.loc[matched_indices]

# Step 4: Check balance after matching
print("\\nBalance after matching:")
for col in covariates:
    treated_mean = matched_df[matched_df['treat'] == 1][col].mean()
    control_mean = matched_df[matched_df['treat'] == 0][col].mean()
    print(f"{col}: Treated={treated_mean:.3f}, Control={control_mean:.3f}")

# Step 5: Compare outcomes
treated_outcome = matched_df[matched_df['treat'] == 1]['re78'].mean()
control_outcome = matched_df[matched_df['treat'] == 0]['re78'].mean()
ate = treated_outcome - control_outcome  # Average Treatment Effect
print(f"\\nAverage Treatment Effect: {ate:.2f}")"""
            content += f'<details class="code-block"><summary>📝 Propensity Matching Code</summary><pre><code>{code}</code></pre></details>'
            
            return content
        except Exception as e:
            return None


# =============================================================================
# TEXT ANALYZER
# =============================================================================

class TextAnalyzer:
    """TF-IDF and text analysis with Pipeline+GridSearchCV (Exam 2020)."""
    
    def __init__(self, report: ReportBuilder):
        self.report = report
    
    def analyze(self, df, name):
        print(f"  📝 Running text analysis for {name}...")
        
        text_cols = []
        for col in df.columns:
            if df[col].dtype == 'object':
                sample = df[col].dropna().head(100)
                avg_len = sample.str.len().mean()
                if avg_len > 30:
                    text_cols.append(col)
        
        if not text_cols:
            return
        
        content = ""
        
        for col in text_cols[:2]:
            result = self._analyze_text_column(df, col, name)
            if result:
                content += result
        
        if content:
            self.report.add_section(f"📝 Text Analysis: {name}", content, level=2)
    
    def _analyze_text_column(self, df, col, name):
        try:
            texts = df[col].dropna().astype(str)
            if len(texts) < 5:
                return None
            
            content = f"<h5>TF-IDF: {col}</h5>"
            content += f"<p><strong>Documents:</strong> {len(texts):,}</p>"
            
            vectorizer = TfidfVectorizer(max_features=500, stop_words='english', min_df=2)
            tfidf_matrix = vectorizer.fit_transform(texts)
            vocab = vectorizer.get_feature_names_out()
            
            mean_tfidf = np.array(tfidf_matrix.mean(axis=0)).flatten()
            top_idx = mean_tfidf.argsort()[-20:][::-1]
            
            top_df = pd.DataFrame({
                'term': [vocab[i] for i in top_idx],
                'avg_tfidf': [mean_tfidf[i] for i in top_idx]
            }).round(4)
            content += self.report.add_dataframe(top_df, "Top Terms")
            
            code = """# TF-IDF with Pipeline + GridSearchCV (Exam 2020)
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Simple TF-IDF
vectorizer = TfidfVectorizer(max_features=150, stop_words='english')
X = vectorizer.fit_transform(documents).toarray()

# Pipeline with SGDClassifier (Exam 2020 pattern)
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(
        loss='log_loss',      # Logistic regression loss
        penalty='l2',
        alpha=1e-4,
        max_iter=5,
        tol=None,
        random_state=42,
        class_weight='balanced'  # Important for imbalanced classes!
    ))
])

# GridSearchCV for hyperparameter tuning
parameters = {
    'vect__ngram_range': [(1, 1), (1, 2)],
    'tfidf__use_idf': [True, False],
    'clf__alpha': [1e-4, 1e-3, 1e-2]
}

gs_clf = GridSearchCV(text_clf, parameters, cv=5, scoring='balanced_accuracy')
gs_clf.fit(X_train, y_train)

print(f"Best parameters: {gs_clf.best_params_}")
print(f"Best score: {gs_clf.best_score_:.4f}")
print(f"Test balanced accuracy: {balanced_accuracy_score(y_test, gs_clf.predict(X_test)):.4f}")"""
            content += f'<details class="code-block"><summary>📝 TF-IDF Pipeline Code</summary><pre><code>{code}</code></pre></details>'
            
            return content
        except:
            return None


# =============================================================================
# NETWORK BUILDER
# =============================================================================

class NetworkBuilder:
    """Build NetworkX graphs from dataframes."""
    
    def __init__(self, report: ReportBuilder):
        self.report = report
        self.graphs = {}
        self.node_attrs = {}
    
    def detect_and_build(self, dataframes):
        """Detect and build all possible graphs from dataframes."""
        
        node_dfs = {}
        
        for name, df in dataframes.items():
            cols_lower = {c.lower(): c for c in df.columns}
            
            # Check for edge list patterns
            if 'u' in cols_lower and 'v' in cols_lower:
                self._build_from_uv(name, df, cols_lower, node_dfs)
            elif 'src' in cols_lower and 'tgt' in cols_lower:
                self._build_from_src_tgt(name, df, cols_lower)
            elif 'source' in cols_lower and 'target' in cols_lower:
                self._build_from_source_target(name, df, cols_lower)
            elif 'speaker' in cols_lower and 'reply_to' in cols_lower:
                self._build_from_reply_to(name, df, cols_lower)
            
            # Check for node attribute patterns
            if any(k in cols_lower for k in ['u', 'node', 'id']) and len(df.columns) > 1:
                node_dfs[name] = df
        
        return self.graphs, self.node_attrs
    
    def _build_from_uv(self, name, df, cols_lower, node_dfs):
        """Build from U, V columns."""
        print(f"    → Building graph from U/V columns in {name}")
        
        u_col = cols_lower['u']
        v_col = cols_lower['v']
        
        edge_counts = df.groupby([u_col, v_col]).size()
        is_multi = edge_counts.max() > 1
        
        G = nx.MultiDiGraph() if is_multi else nx.DiGraph()
        
        edge_attrs = [c for c in df.columns if c.lower() not in ['u', 'v']]
        for _, row in df.iterrows():
            if pd.isna(row[u_col]) or pd.isna(row[v_col]):
                continue
            attrs = {col: row[col] for col in edge_attrs if pd.notna(row[col])}
            G.add_edge(row[u_col], row[v_col], **attrs)
        
        for node_name, node_df in node_dfs.items():
            node_cols_lower = {c.lower(): c for c in node_df.columns}
            node_col = None
            for key in ['u', 'node', 'id']:
                if key in node_cols_lower:
                    node_col = node_cols_lower[key]
                    break
            
            if node_col:
                attr_cols = [c for c in node_df.columns if c != node_col]
                for _, row in node_df.iterrows():
                    node = row[node_col]
                    if pd.notna(node) and node in G.nodes():
                        for col in attr_cols:
                            if pd.notna(row[col]):
                                G.nodes[node][col] = row[col]
                
                self.node_attrs[f"{name}_graph"] = node_df
        
        self.graphs[f"{name}_graph"] = G
    
    def _build_from_src_tgt(self, name, df, cols_lower):
        """Build from SRC, TGT, VOT columns (Exam 2022 - Signed graph)."""
        print(f"    → Building signed graph from SRC/TGT columns in {name}")
        
        src_col = cols_lower['src']
        tgt_col = cols_lower['tgt']
        vot_col = cols_lower.get('vot')
        
        G_undirected = nx.Graph()
        
        edge_votes = {}
        for _, row in df.iterrows():
            if pd.isna(row[src_col]) or pd.isna(row[tgt_col]):
                continue
            
            edge_key = tuple(sorted([row[src_col], row[tgt_col]]))
            
            if vot_col and pd.notna(row[vot_col]):
                if edge_key not in edge_votes:
                    edge_votes[edge_key] = []
                edge_votes[edge_key].append(row[vot_col])
        
        for (u, v), votes in edge_votes.items():
            agg_vote = sum(votes)
            final_vote = 1 if agg_vote > 0 else -1
            G_undirected.add_edge(u, v, VOT=final_vote)
        
        self.graphs[f"{name}_undirected"] = G_undirected
        
        G_directed = nx.DiGraph()
        for _, row in df.iterrows():
            if pd.isna(row[src_col]) or pd.isna(row[tgt_col]):
                continue
            attrs = {}
            if vot_col and pd.notna(row[vot_col]):
                attrs['VOT'] = row[vot_col]
            G_directed.add_edge(row[src_col], row[tgt_col], **attrs)
        self.graphs[f"{name}_directed"] = G_directed
    
    def _build_from_source_target(self, name, df, cols_lower):
        """Build from source, target columns."""
        print(f"    → Building graph from source/target columns in {name}")
        
        source_col = cols_lower['source']
        target_col = cols_lower['target']
        
        G = nx.DiGraph()
        edge_attrs = [c for c in df.columns if c.lower() not in ['source', 'target']]
        for _, row in df.iterrows():
            if pd.isna(row[source_col]) or pd.isna(row[target_col]):
                continue
            attrs = {col: row[col] for col in edge_attrs if pd.notna(row[col])}
            G.add_edge(row[source_col], row[target_col], **attrs)
        
        self.graphs[f"{name}_graph"] = G
    
    def _build_from_reply_to(self, name, df, cols_lower):
        """Build from reply_to column (Exam 2023 - Friends)."""
        print(f"    → Building reply network from {name}")
        
        speaker_col = cols_lower['speaker']
        reply_col = cols_lower['reply_to']
        id_col = None
        for key in ['id', 'utterance_id', 'msg_id']:
            if key in cols_lower:
                id_col = cols_lower[key]
                break
        
        G = nx.MultiDiGraph()
        
        if id_col:
            valid_rows = df[df[id_col].notna() & df[speaker_col].notna()]
            id_to_speaker = valid_rows.set_index(id_col)[speaker_col].to_dict()
            
            for _, row in df.iterrows():
                if pd.isna(row[speaker_col]) or pd.isna(row[reply_col]):
                    continue
                if row[reply_col] in id_to_speaker:
                    from_speaker = row[speaker_col]
                    to_speaker = id_to_speaker[row[reply_col]]
                    G.add_edge(from_speaker, to_speaker)
        
        self.graphs[f"{name}_reply_graph"] = G


# =============================================================================
# NETWORK ANALYZER
# =============================================================================

class NetworkAnalyzer:
    """Complete network analysis including triangles and structural balance."""
    
    def __init__(self, report: ReportBuilder):
        self.report = report
    
    def analyze(self, G, name, node_df=None):
        print(f"  🔍 Analyzing network: {name}")
        
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        
        if n_nodes == 0:
            return
        
        content = f"""
        <h4>📊 Basic Statistics</h4>
        <table class="dataframe">
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Nodes</td><td>{n_nodes:,}</td></tr>
            <tr><td>Edges</td><td>{n_edges:,}</td></tr>
            <tr><td>Type</td><td>{'Directed' if G.is_directed() else 'Undirected'}{', Multi' if G.is_multigraph() else ''}</td></tr>
            <tr><td>Density</td><td>{nx.density(G):.6f}</td></tr>
        </table>
        """
        
        findings = [f"{n_nodes:,} nodes, {n_edges:,} edges"]
        
        # Degree analysis
        content += self._analyze_degrees(G, name)
        
        # Centrality
        content += self._analyze_centrality(G, name, findings)
        
        # Connectivity
        content += self._analyze_connectivity(G, name)
        
        # Structural balance (for signed graphs)
        if not G.is_directed() and 'VOT' in nx.get_edge_attributes(G, 'VOT'):
            content += self._analyze_structural_balance(G, name, findings)
        
        self.report.add_section(f"🕸️ Network Analysis: {name}", content, level=2,
                               key_findings=findings[:5])
    
    def _analyze_degrees(self, G, name):
        """Degree distribution analysis."""
        content = "<h4>Degree Distribution</h4>"
        
        if G.is_directed():
            in_degrees = [d for _, d in G.in_degree()]
            out_degrees = [d for _, d in G.out_degree()]
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            axes[0].hist(in_degrees, bins=50, alpha=0.7, color='steelblue')
            axes[0].set_xlabel('In-Degree')
            axes[0].set_ylabel('Count')
            axes[0].set_title('In-Degree Distribution')
            axes[0].set_yscale('log')
            
            axes[1].hist(out_degrees, bins=50, alpha=0.7, color='coral')
            axes[1].set_xlabel('Out-Degree')
            axes[1].set_ylabel('Count')
            axes[1].set_title('Out-Degree Distribution')
            axes[1].set_yscale('log')
        else:
            degrees = [d for _, d in G.degree()]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(degrees, bins=50, alpha=0.7, color='steelblue')
            ax.set_xlabel('Degree')
            ax.set_ylabel('Count')
            ax.set_title('Degree Distribution')
            ax.set_yscale('log')
        
        plt.tight_layout()
        content += self.report.save_figure(fig, f"{name}_degree_dist")
        
        code = """# Degree Analysis
import networkx as nx

# Basic degree
degrees = dict(G.degree())

# For directed graphs
in_degrees = dict(G.in_degree())
out_degrees = dict(G.out_degree())

# Degree statistics
print(f"Mean degree: {np.mean(list(degrees.values())):.2f}")
print(f"Max degree: {max(degrees.values())}")

# Top nodes by degree
top_nodes = sorted(degrees.items(), key=lambda x: -x[1])[:10]
for node, deg in top_nodes:
    print(f"  {node}: {deg}")"""
        content += f'<details class="code-block"><summary>📝 Degree Code</summary><pre><code>{code}</code></pre></details>'
        
        return content
    
    def _analyze_centrality(self, G, name, findings):
        """Centrality measures including PageRank."""
        content = "<h4>Centrality Measures</h4>"
        
        try:
            # PageRank
            if G.is_directed():
                pagerank = nx.pagerank(G, alpha=0.85)
            else:
                pagerank = nx.pagerank(G.to_directed(), alpha=0.85)
            
            # Top nodes by PageRank
            top_pr = sorted(pagerank.items(), key=lambda x: -x[1])[:10]
            pr_df = pd.DataFrame(top_pr, columns=['Node', 'PageRank']).round(6)
            content += self.report.add_dataframe(pr_df, "Top 10 by PageRank")
            
            findings.append(f"Top PageRank: {top_pr[0][0]} ({top_pr[0][1]:.4f})")
            
            code = """# Centrality Analysis (Exam 2021)
import networkx as nx

# PageRank (most important for exams!)
pagerank = nx.pagerank(G, alpha=0.85)

# Top nodes by PageRank
top_nodes = sorted(pagerank.items(), key=lambda x: -x[1])[:10]
for node, pr in top_nodes:
    print(f"{node}: {pr:.6f}")

# Degree centrality
degree_cent = nx.degree_centrality(G)

# Betweenness centrality (expensive for large graphs)
betweenness = nx.betweenness_centrality(G)

# Eigenvector centrality
eigenvector = nx.eigenvector_centrality(G, max_iter=1000)

# Correlate PageRank with node attribute (Exam 2021 pattern)
from scipy import stats
scores = [G.nodes[n].get('score', 0) for n in G.nodes()]
pr_values = [pagerank[n] for n in G.nodes()]
corr, p_value = stats.spearmanr(scores, pr_values)
print(f"Spearman correlation (score vs PageRank): {corr:.4f}, p={p_value:.6f}")"""
            content += f'<details class="code-block"><summary>📝 Centrality Code</summary><pre><code>{code}</code></pre></details>'
            
        except Exception as e:
            content += f"<p>Error computing centrality: {e}</p>"
        
        return content
    
    def _analyze_connectivity(self, G, name):
        """Connectivity analysis."""
        content = "<h4>Connectivity</h4>"
        
        try:
            if G.is_directed():
                n_weak = nx.number_weakly_connected_components(G)
                n_strong = nx.number_strongly_connected_components(G)
                content += f"""
                <p><strong>Weakly connected components:</strong> {n_weak}</p>
                <p><strong>Strongly connected components:</strong> {n_strong}</p>
                """
            else:
                n_components = nx.number_connected_components(G)
                content += f"<p><strong>Connected components:</strong> {n_components}</p>"
                
                if n_components == 1:
                    content += "<p>Graph is <strong>connected</strong></p>"
            
            code = """# Connectivity Analysis (Exam 2020)
import networkx as nx

# For directed graphs
is_weakly = nx.is_weakly_connected(G)
is_strongly = nx.is_strongly_connected(G)
n_weak = nx.number_weakly_connected_components(G)
n_strong = nx.number_strongly_connected_components(G)

print(f"Weakly connected: {is_weakly}")
print(f"Strongly connected: {is_strongly}")
print(f"# Weak components: {n_weak}")
print(f"# Strong components: {n_strong}")

# Get largest component
if G.is_directed():
    largest_cc = max(nx.weakly_connected_components(G), key=len)
else:
    largest_cc = max(nx.connected_components(G), key=len)

G_largest = G.subgraph(largest_cc)
print(f"Largest component: {len(largest_cc)} nodes")"""
            content += f'<details class="code-block"><summary>📝 Connectivity Code</summary><pre><code>{code}</code></pre></details>'
            
        except Exception as e:
            content += f"<p>Error: {e}</p>"
        
        return content
    
    def _analyze_structural_balance(self, G, name, findings):
        """Structural balance analysis for signed graphs (Exam 2022)."""
        content = "<h4>Structural Balance (Signed Graph)</h4>"
        
        try:
            triangles = []
            for clique in nx.enumerate_all_cliques(G):
                if len(clique) == 3:
                    triangles.append(tuple(clique))
                elif len(clique) > 3:
                    break
            
            if not triangles:
                return content + "<p>No triangles found</p>"
            
            configs = {(1, 1, 1): 0, (1, 1, -1): 0, (1, -1, -1): 0, (-1, -1, -1): 0}
            
            for tri in triangles:
                signs = []
                for i in range(3):
                    u, v = tri[i], tri[(i + 1) % 3]
                    if G.has_edge(u, v):
                        vot = G[u][v].get('VOT', 1)
                        signs.append(1 if vot > 0 else -1)
                    else:
                        signs.append(1)
                signs = tuple(sorted(signs, reverse=True))
                if signs in configs:
                    configs[signs] += 1
            
            total = sum(configs.values())
            if total == 0:
                return content + "<p>No valid triangles</p>"
            
            strong_balance = (configs[(1, 1, 1)] + configs[(1, -1, -1)]) / total
            weak_balance = (configs[(1, 1, 1)] + configs[(1, -1, -1)] + configs[(-1, -1, -1)]) / total
            
            content += f"""
            <table class="dataframe">
                <tr><th>Configuration</th><th>Count</th><th>Fraction</th></tr>
                <tr><td>(+,+,+) Friend of friend is friend</td><td>{configs[(1,1,1)]}</td><td>{configs[(1,1,1)]/total:.3f}</td></tr>
                <tr><td>(+,+,-) Two friends, one enemy</td><td>{configs[(1,1,-1)]}</td><td>{configs[(1,1,-1)]/total:.3f}</td></tr>
                <tr><td>(+,-,-) Enemy of enemy is friend</td><td>{configs[(1,-1,-1)]}</td><td>{configs[(1,-1,-1)]/total:.3f}</td></tr>
                <tr><td>(-,-,-) All enemies</td><td>{configs[(-1,-1,-1)]}</td><td>{configs[(-1,-1,-1)]/total:.3f}</td></tr>
            </table>
            <p><strong>Total triangles:</strong> {total}</p>
            <p><strong>Strong balance:</strong> {strong_balance:.3f} (+ + + and + - -)</p>
            <p><strong>Weak balance:</strong> {weak_balance:.3f} (all except + + -)</p>
            """
            
            findings.append(f"Strong balance: {strong_balance:.3f}")
            
            code = """# Structural Balance (Exam 2022)
import networkx as nx

# Get all triangles
triangles = []
for clique in nx.enumerate_all_cliques(G):
    if len(clique) == 3:
        triangles.append(tuple(clique))
    elif len(clique) > 3:
        break

print(f"Found {len(triangles)} triangles")

# Count configurations
configs = {(1,1,1): 0, (1,1,-1): 0, (1,-1,-1): 0, (-1,-1,-1): 0}

for tri in triangles:
    signs = []
    for i in range(3):
        u, v = tri[i], tri[(i+1) % 3]
        # For simple Graph: G[u][v] returns edge data dict directly
        vot = G[u][v].get('VOT', 1)
        signs.append(1 if vot > 0 else -1)
    signs = tuple(sorted(signs, reverse=True))
    configs[signs] += 1

# Calculate balance
total = sum(configs.values())
strong_balance = (configs[(1,1,1)] + configs[(1,-1,-1)]) / total
weak_balance = (configs[(1,1,1)] + configs[(1,-1,-1)] + configs[(-1,-1,-1)]) / total

print(f"Strong balance: {strong_balance:.3f}")
print(f"Weak balance: {weak_balance:.3f}")

# Interpretation:
# (+,+,+): Friend of my friend is my friend - BALANCED
# (+,+,-): Two friends have common enemy - UNBALANCED (tension)
# (+,-,-): Enemy of my enemy is my friend - BALANCED
# (-,-,-): All enemies - BALANCED (weak) or UNBALANCED (strong)"""
            content += f'<details class="code-block"><summary>📝 Structural Balance Code</summary><pre><code>{code}</code></pre></details>'
            
        except Exception as e:
            content += f"<p>Error: {e}</p>"
        
        return content


# =============================================================================
# QUINTILE ANALYZER
# =============================================================================

class QuintileAnalyzer:
    """Quintile-based cross-tabulation analysis (Exam 2021)."""
    
    def __init__(self, report: ReportBuilder):
        self.report = report
    
    def analyze(self, G, node_df, name):
        """Create quintile cross-tabulation heatmap."""
        if node_df is None:
            return
        
        print(f"  📊 Running quintile analysis for {name}...")
        
        score_col = None
        node_col = None
        for c in node_df.columns:
            if c.lower() == 'score':
                score_col = c
            if c.lower() in ['u', 'node', 'id']:
                node_col = c
        
        if not score_col or not node_col:
            return
        
        content = "<h4>Quintile Cross-Tabulation (Exam 2021)</h4>"
        
        try:
            node_df = node_df.copy()
            
            try:
                node_df['quintile'] = pd.qcut(
                    node_df[score_col], q=5, 
                    labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                    duplicates='drop'
                )
            except:
                node_df['quintile'] = pd.cut(
                    node_df[score_col], bins=5,
                    labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
                )
            
            quintile_map = node_df.set_index(node_col)['quintile'].to_dict()
            unique_quintiles = sorted([q for q in node_df['quintile'].unique() if pd.notna(q)])
            n_quintiles = len(unique_quintiles)
            
            if n_quintiles < 2:
                return
            
            quintile_to_idx = {q: i for i, q in enumerate(unique_quintiles)}
            matrix = np.zeros((n_quintiles, n_quintiles))
            
            for u, v in G.edges():
                if u in quintile_map and v in quintile_map:
                    q_u = quintile_map[u]
                    q_v = quintile_map[v]
                    if pd.notna(q_u) and pd.notna(q_v):
                        matrix[quintile_to_idx[q_u], quintile_to_idx[q_v]] += 1
            
            if matrix.sum() == 0:
                return
            
            row_sums = matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            matrix_norm = matrix / row_sums
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            labels_list = [str(q) for q in unique_quintiles]
            
            sns.heatmap(matrix, annot=True, fmt='.0f', cmap='YlOrRd',
                       xticklabels=labels_list, yticklabels=labels_list, ax=axes[0])
            axes[0].set_xlabel('Target Quintile')
            axes[0].set_ylabel('Source Quintile')
            axes[0].set_title('Raw Counts')
            
            sns.heatmap(matrix_norm, annot=True, fmt='.2f', cmap='YlOrRd',
                       xticklabels=labels_list, yticklabels=labels_list, ax=axes[1])
            axes[1].set_xlabel('Target Quintile')
            axes[1].set_ylabel('Source Quintile')
            axes[1].set_title('Row-Normalized')
            
            plt.tight_layout()
            content += self.report.save_figure(fig, f"{name}_quintile")
            
            code = """# Quintile Analysis (Exam 2021)
import pandas as pd
import numpy as np

# Create quintiles based on score
# IMPORTANT: Use duplicates='drop' to handle ties
node_df['quintile'] = pd.qcut(
    node_df['score'], q=5, 
    labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
    duplicates='drop'
)

quintile_map = node_df.set_index('u')['quintile'].to_dict()

# Build cross-tabulation matrix
matrix = np.zeros((5, 5))
quintile_to_idx = {'Q1': 0, 'Q2': 1, 'Q3': 2, 'Q4': 3, 'Q5': 4}

for u, v in G.edges():
    if u in quintile_map and v in quintile_map:
        q_u, q_v = quintile_map[u], quintile_map[v]
        if pd.notna(q_u) and pd.notna(q_v):
            matrix[quintile_to_idx[q_u], quintile_to_idx[q_v]] += 1

# Row-wise normalization (fraction from each source)
matrix_norm = matrix / matrix.sum(axis=1, keepdims=True)

# Plot heatmap
import seaborn as sns
sns.heatmap(matrix_norm, annot=True, fmt='.2f',
           xticklabels=['Q1','Q2','Q3','Q4','Q5'],
           yticklabels=['Q1','Q2','Q3','Q4','Q5'])
plt.xlabel('Target Quintile')
plt.ylabel('Source Quintile')"""
            content += f'<details class="code-block"><summary>📝 Quintile Code</summary><pre><code>{code}</code></pre></details>'
            
            self.report.add_section(f"📊 Quintile Analysis: {name}", content, level=2)
            
        except Exception as e:
            print(f"    Error in quintile analysis: {e}")


# =============================================================================
# EXAM CHEAT SHEET GENERATOR
# =============================================================================

class ExamCheatSheet:
    """Generate a comprehensive cheat sheet with all exam-relevant code."""
    
    def __init__(self, report: ReportBuilder):
        self.report = report
    
    def generate(self):
        """Generate the complete cheat sheet."""
        print("📚 Generating Exam Cheat Sheet...")
        
        content = """
        <div class="info">
            <strong>📚 COMPLETE EXAM CHEAT SHEET</strong><br>
            All code snippets are copy-paste ready. Organized by topic for quick reference during exams.
        </div>
        """
        
        # 1. Data Loading
        content += self._data_loading_section()
        
        # 2. Pandas Operations
        content += self._pandas_section()
        
        # 3. Classification
        content += self._classification_section()
        
        # 4. Regression
        content += self._regression_section()
        
        # 5. Network Analysis
        content += self._network_section()
        
        # 6. Statistical Tests
        content += self._statistics_section()
        
        # 7. Text Analysis
        content += self._text_section()
        
        # 8. Clustering & Dimensionality Reduction
        content += self._clustering_section()
        
        # 9. Visualization
        content += self._visualization_section()
        
        # 10. Observational Studies
        content += self._observational_section()
        
        self.report.add_section("📚 EXAM CHEAT SHEET", content, level=1)
    
    def _data_loading_section(self):
        code = """# ============================================================
# DATA LOADING - All common formats
# ============================================================
import pandas as pd
import gzip
import json

# CSV (standard)
df = pd.read_csv('data.csv')

# CSV compressed
df = pd.read_csv('data.csv.gz', compression='gzip')

# TSV (tab-separated)
df = pd.read_csv('data.tsv', sep='\\t')

# JSON
df = pd.read_json('data.json')

# JSONL (JSON lines - one JSON per line)
df = pd.read_json('data.jsonl', lines=True)

# Gzipped JSONL
with gzip.open('data.jsonl.gz', 'rt') as f:
    data = [json.loads(line) for line in f]
df = pd.DataFrame(data)

# Parquet
df = pd.read_parquet('data.parquet')

# GraphML network
import networkx as nx
G = nx.read_graphml('graph.graphml')"""
        return f'<h4>1. Data Loading</h4><details class="code-block" open><summary>📝 Data Loading Code</summary><pre><code>{code}</code></pre></details>'
    
    def _pandas_section(self):
        code = """# ============================================================
# PANDAS ESSENTIAL OPERATIONS
# ============================================================
import pandas as pd
import numpy as np

# ----- GroupBy Aggregations -----
grouped = df.groupby('category')['value'].agg(['mean', 'std', 'count', 'sum'])

# Multiple columns
grouped = df.groupby('category').agg({
    'col1': ['mean', 'std'],
    'col2': ['sum', 'count']
})

# Named aggregations (cleaner)
grouped = df.groupby('category').agg(
    mean_val=('value', 'mean'),
    total=('value', 'sum'),
    n=('value', 'count')
)

# ----- Quintiles / Binning (Exam 2021) -----
# IMPORTANT: Use duplicates='drop' for ties
df['quintile'] = pd.qcut(df['score'], q=5, 
                         labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                         duplicates='drop')

# Equal-width bins (not quintiles)
df['bin'] = pd.cut(df['value'], bins=5, labels=['B1', 'B2', 'B3', 'B4', 'B5'])

# ----- Cross-tabulation -----
crosstab = pd.crosstab(df['category1'], df['category2'])
crosstab_norm = pd.crosstab(df['category1'], df['category2'], normalize='index')  # row-normalize

# ----- ID Parsing (Exam 2023 - Friends) -----
# Pattern: s01_e05_c03_u012 -> season, episode, conversation, utterance
def parse_id(id_str):
    parts = id_str.split('_')
    result = {}
    for part in parts:
        prefix = part[0]
        number = int(part[1:])
        if prefix == 's': result['season'] = number
        elif prefix == 'e': result['episode'] = number
        elif prefix == 'c': result['conversation'] = number
        elif prefix == 'u': result['utterance'] = number
    return result

parsed = df['id'].apply(parse_id).apply(pd.Series)
df = pd.concat([df, parsed], axis=1)

# ----- Merge DataFrames -----
merged = pd.merge(df1, df2, on='key_column', how='inner')  # inner, left, right, outer

# ----- Pivot Tables -----
pivot = df.pivot_table(values='value', index='row_cat', columns='col_cat', aggfunc='mean')

# ----- Value Counts -----
df['column'].value_counts()
df['column'].value_counts(normalize=True)  # proportions"""
        return f'<h4>2. Pandas Operations</h4><details class="code-block"><summary>📝 Pandas Code</summary><pre><code>{code}</code></pre></details>'
    
    def _classification_section(self):
        code = """# ============================================================
# CLASSIFICATION - All Methods
# ============================================================
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (confusion_matrix, accuracy_score, balanced_accuracy_score,
                            precision_score, recall_score, f1_score, roc_curve, auc)
from sklearn.preprocessing import StandardScaler

# ----- Train/Test Split -----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42  # Exam 2023: 0.3, Exam 2021: 0.4
)

# ----- Decision Tree (Exam 2023) -----
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
balanced_acc = balanced_accuracy_score(y_test, y_pred)
random_baseline = 1 / n_classes  # NOT most frequent class!

# CRITICAL: Confusion matrix normalized (ALL cells sum to 1)
cm = confusion_matrix(y_test, y_pred)
cm_normalized = cm / cm.sum()  # NOT cm / cm.sum(axis=1)!

# Feature importance
for feat, imp in sorted(zip(feature_names, dt.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat}: {imp:.4f}")

# ----- Logistic Regression (Exam 2021: C=10) -----
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=10, max_iter=2000, random_state=42)
lr.fit(X_train, y_train)
print(f"Train acc: {lr.score(X_train, y_train):.4f}")
print(f"Test acc: {lr.score(X_test, y_test):.4f}")

# ----- LogisticRegressionCV (Exam 2019) -----
from sklearn.linear_model import LogisticRegressionCV

Cs = [0.01, 0.1, 1, 10, 100]
lr_cv = LogisticRegressionCV(Cs=Cs, cv=3, random_state=42, max_iter=200)
lr_cv.fit(X_train, y_train)
print(f"Best C: {lr_cv.C_}")

# ----- k-NN Classifier (Lecture 4) -----
from sklearn.neighbors import KNeighborsClassifier

# IMPORTANT: Scale features for k-NN!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train_s, X_test_s, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)

# Find optimal k
for k in [1, 3, 5, 7, 9, 11]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_s, y_train)
    print(f"k={k}: Train={knn.score(X_train_s, y_train):.3f}, Test={knn.score(X_test_s, y_test):.3f}")

# ----- Random Forest (Lecture 5) -----
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42)
rf.fit(X_train, y_train)

# Cross-validation precision/recall
precision = cross_val_score(rf, X, y, cv=10, scoring='precision').mean()
recall = cross_val_score(rf, X, y, cv=10, scoring='recall').mean()

# ----- ROC/AUC (AppliedML exercise) -----
# For binary classification with probabilities
y_pred_proba = lr.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc_score = auc(fpr, tpr)
print(f"AUC: {auc_score:.4f}")

# Plot ROC curve
import matplotlib.pyplot as plt
plt.plot(fpr, tpr, label=f'ROC (AUC = {auc_score:.3f})')
plt.plot([0, 1], [0, 1], 'r--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()

# ----- SGDClassifier with Pipeline (Exam 2020) -----
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-4,
                         max_iter=5, tol=None, random_state=42,
                         class_weight='balanced'))  # Important for imbalanced!
])

parameters = {
    'vect__ngram_range': [(1, 1), (1, 2)],
    'clf__alpha': [1e-4, 1e-3]
}

gs_clf = GridSearchCV(text_clf, parameters, cv=5, scoring='balanced_accuracy')
gs_clf.fit(X_train_text, y_train)
print(f"Best params: {gs_clf.best_params_}")"""
        return f'<h4>3. Classification</h4><details class="code-block"><summary>📝 Classification Code</summary><pre><code>{code}</code></pre></details>'
    
    def _regression_section(self):
        code = """# ============================================================
# REGRESSION - OLS, Ridge, Treatment Encoding
# ============================================================
import statsmodels.formula.api as smf
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ----- OLS with Treatment Encoding (Exam 2023) -----
# Treatment encoding: one category is reference, others compared to it
formula = 'outcome ~ C(category, Treatment(reference="baseline"))'
model = smf.ols(formula=formula, data=df).fit()
print(model.summary())

# Interpretation:
# - Intercept = mean outcome when category = "baseline"
# - C(category)[T.X] = how much category X differs from baseline
# - p < 0.05 = statistically significant

# Simple categorical
formula = 'y ~ x1 + x2 + C(category)'
model = smf.ols(formula=formula, data=df).fit()

# Numeric predictors only
formula = 'y ~ x1 + x2 + x3'
model = smf.ols(formula=formula, data=df).fit()
print(f"R-squared: {model.rsquared:.4f}")
print(f"Coefficients:\\n{model.params}")

# ----- RidgeCV (Exam 2019) -----
from sklearn.linear_model import RidgeCV

alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
ridge = RidgeCV(alphas=alphas)
ridge.fit(X_train, y_train)

print(f"Best alpha: {ridge.alpha_}")
print(f"Train R²: {ridge.score(X_train, y_train):.4f}")
print(f"Test R²: {ridge.score(X_test, y_test):.4f}")

y_pred = ridge.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

# Coefficients
for feat, coef in zip(feature_names, ridge.coef_):
    print(f"  {feat}: {coef:.4f}")

# ----- Binary Target from Median (Exam 2019) -----
median_val = df['target_col'].median()
df['binary_target'] = (df['target_col'] > median_val).astype(int)
# Then use LogisticRegression on binary_target"""
        return f'<h4>4. Regression</h4><details class="code-block"><summary>📝 Regression Code</summary><pre><code>{code}</code></pre></details>'
    
    def _network_section(self):
        code = """# ============================================================
# NETWORK ANALYSIS - PageRank, Centrality, Triangles
# ============================================================
import networkx as nx
import numpy as np
from scipy import stats

# ----- Building Graphs -----
# From edge list (u, v columns)
G = nx.DiGraph()  # or nx.Graph() for undirected
for _, row in df.iterrows():
    G.add_edge(row['u'], row['v'])

# From pandas directly
G = nx.from_pandas_edgelist(df, source='u', target='v', 
                            edge_attr=True, create_using=nx.DiGraph())

# Signed graph (Exam 2022 - VOT attribute)
G = nx.Graph()
for _, row in df.iterrows():
    G.add_edge(row['SRC'], row['TGT'], VOT=row['VOT'])

# ----- PageRank (Exam 2021) -----
pagerank = nx.pagerank(G, alpha=0.85)
top_nodes = sorted(pagerank.items(), key=lambda x: -x[1])[:10]
for node, pr in top_nodes:
    print(f"{node}: {pr:.6f}")

# ----- Degree Centrality -----
degree_cent = nx.degree_centrality(G)
in_degree_cent = nx.in_degree_centrality(G)   # for DiGraph
out_degree_cent = nx.out_degree_centrality(G) # for DiGraph

# ----- Betweenness Centrality -----
betweenness = nx.betweenness_centrality(G)

# ----- Correlate PageRank with Score (Exam 2021) -----
scores = [G.nodes[n].get('score', 0) for n in G.nodes()]
pr_values = [pagerank[n] for n in G.nodes()]
corr, p_value = stats.spearmanr(scores, pr_values)
print(f"Spearman correlation: {corr:.4f}, p={p_value:.6f}")

# ----- Connectivity (Exam 2020) -----
# Directed graph
is_weakly = nx.is_weakly_connected(G)
is_strongly = nx.is_strongly_connected(G)
n_weak = nx.number_weakly_connected_components(G)
n_strong = nx.number_strongly_connected_components(G)

# Largest component
largest_cc = max(nx.weakly_connected_components(G), key=len)
G_largest = G.subgraph(largest_cc)

# ----- Structural Balance (Exam 2022) -----
# Find all triangles
triangles = []
for clique in nx.enumerate_all_cliques(G):  # G must be undirected!
    if len(clique) == 3:
        triangles.append(tuple(clique))
    elif len(clique) > 3:
        break

# Count configurations
configs = {(1,1,1): 0, (1,1,-1): 0, (1,-1,-1): 0, (-1,-1,-1): 0}

for tri in triangles:
    signs = []
    for i in range(3):
        u, v = tri[i], tri[(i+1) % 3]
        vot = G[u][v].get('VOT', 1)
        signs.append(1 if vot > 0 else -1)
    signs = tuple(sorted(signs, reverse=True))
    configs[signs] += 1

total = sum(configs.values())
strong_balance = (configs[(1,1,1)] + configs[(1,-1,-1)]) / total
weak_balance = (configs[(1,1,1)] + configs[(1,-1,-1)] + configs[(-1,-1,-1)]) / total

# Interpretation:
# (+,+,+): Friend of friend is friend - BALANCED
# (+,+,-): Two friends have common enemy - UNBALANCED
# (+,-,-): Enemy of enemy is friend - BALANCED
# (-,-,-): All enemies - BALANCED (weak)

# ----- Quintile Cross-Tab (Exam 2021) -----
# Assign nodes to quintiles based on score
node_df['quintile'] = pd.qcut(node_df['score'], q=5, 
                              labels=['Q1','Q2','Q3','Q4','Q5'],
                              duplicates='drop')
quintile_map = node_df.set_index('u')['quintile'].to_dict()

# Build matrix
matrix = np.zeros((5, 5))
q_idx = {'Q1':0, 'Q2':1, 'Q3':2, 'Q4':3, 'Q5':4}
for u, v in G.edges():
    if u in quintile_map and v in quintile_map:
        matrix[q_idx[quintile_map[u]], q_idx[quintile_map[v]]] += 1

# Row-normalize
matrix_norm = matrix / matrix.sum(axis=1, keepdims=True)"""
        return f'<h4>5. Network Analysis</h4><details class="code-block"><summary>📝 Network Code</summary><pre><code>{code}</code></pre></details>'
    
    def _statistics_section(self):
        code = """# ============================================================
# STATISTICAL TESTS & CONFIDENCE INTERVALS
# ============================================================
from scipy import stats
import numpy as np

# ----- T-Test (Lecture 11) -----
# Two independent samples
group1 = df[df['treatment'] == 1]['outcome']
group2 = df[df['treatment'] == 0]['outcome']

t_stat, p_value = stats.ttest_ind(group1, group2)
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.6f}")
if p_value < 0.05:
    print("Significant difference (reject H0)")

# Paired t-test (same subjects, two conditions)
t_stat, p_value = stats.ttest_rel(before, after)

# One-sample t-test (compare to known mean)
t_stat, p_value = stats.ttest_1samp(sample, popmean=0)

# ----- Chi-Squared Test (Lecture 5, 11) -----
contingency = pd.crosstab(df['var1'], df['var2'])
chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
print(f"Chi-squared: {chi2:.4f}")
print(f"p-value: {p_value:.6f}")
print(f"Degrees of freedom: {dof}")

# ----- Correlation with p-value -----
# Spearman (Exam 2021 - rank-based, monotonic)
corr, p_value = stats.spearmanr(df['x'], df['y'])

# Pearson (linear relationship)
corr, p_value = stats.pearsonr(df['x'], df['y'])

# ----- Bootstrap Confidence Interval (Describing_data) -----
def bootstrap_ci(data, n_iterations=1000, confidence=0.95):
    '''Bootstrap 95% confidence interval for the mean.'''
    means = np.zeros(n_iterations)
    n = len(data)
    
    for i in range(n_iterations):
        # Sample WITH REPLACEMENT (key!)
        sample = np.random.choice(data, size=n, replace=True)
        means[i] = np.mean(sample)
    
    alpha = (1 - confidence) / 2
    lower = np.percentile(means, alpha * 100)
    upper = np.percentile(means, (1 - alpha) * 100)
    return lower, upper

data = df['column'].dropna().values
ci_lower, ci_upper = bootstrap_ci(data)
print(f"Mean: {np.mean(data):.4f}")
print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

# For 10000 iterations:
# 95% CI uses indices 250 and 9750 (2.5th and 97.5th percentiles)

# ----- Effect Size (Cohen's d) -----
def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (group1.mean() - group2.mean()) / pooled_std

d = cohens_d(group1, group2)
# |d| < 0.2: small, 0.2-0.8: medium, > 0.8: large"""
        return f'<h4>6. Statistical Tests</h4><details class="code-block"><summary>📝 Statistics Code</summary><pre><code>{code}</code></pre></details>'
    
    def _text_section(self):
        code = """# ============================================================
# TEXT ANALYSIS - TF-IDF, Document Classification
# ============================================================
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# ----- TF-IDF Vectorization -----
vectorizer = TfidfVectorizer(
    max_features=150,      # Limit vocabulary size
    stop_words='english',  # Remove common words
    min_df=2,              # Minimum document frequency
    ngram_range=(1, 2)     # Unigrams and bigrams
)

X = vectorizer.fit_transform(documents).toarray()
feature_names = vectorizer.get_feature_names_out()

# Top terms by average TF-IDF
mean_tfidf = X.mean(axis=0)
top_indices = mean_tfidf.argsort()[-20:][::-1]
for idx in top_indices:
    print(f"{feature_names[idx]}: {mean_tfidf[idx]:.4f}")

# ----- Document Classification Pipeline (Exam 2020) -----
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(
        loss='log_loss',
        penalty='l2',
        alpha=1e-4,
        max_iter=5,
        tol=None,
        random_state=42,
        class_weight='balanced'  # Important for imbalanced data!
    ))
])

# Hyperparameter tuning
parameters = {
    'vect__ngram_range': [(1, 1), (1, 2)],
    'tfidf__use_idf': [True, False],
    'clf__alpha': [1e-4, 1e-3, 1e-2]
}

gs_clf = GridSearchCV(text_clf, parameters, cv=5, scoring='balanced_accuracy')
gs_clf.fit(X_train, y_train)

print(f"Best parameters: {gs_clf.best_params_}")
print(f"Best CV score: {gs_clf.best_score_:.4f}")
print(f"Test accuracy: {gs_clf.score(X_test, y_test):.4f}")

# ----- Word Frequency Analysis -----
from collections import Counter

# Simple word counts
all_words = ' '.join(df['text']).lower().split()
word_counts = Counter(all_words)
print(word_counts.most_common(20))"""
        return f'<h4>7. Text Analysis</h4><details class="code-block"><summary>📝 Text Code</summary><pre><code>{code}</code></pre></details>'
    
    def _clustering_section(self):
        code = """# ============================================================
# CLUSTERING & DIMENSIONALITY REDUCTION
# ============================================================
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ----- Feature Scaling (Always do this first!) -----
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----- K-Means Clustering -----
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

# Cluster centers
centers = kmeans.cluster_centers_

# ----- Finding Optimal k -----
inertias = []
silhouettes = []

for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))
    print(f"k={k}: Inertia={km.inertia_:.2f}, Silhouette={silhouettes[-1]:.4f}")

# Best k by silhouette score
best_k = silhouettes.index(max(silhouettes)) + 2
print(f"Best k: {best_k}")

# Elbow plot
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(range(2, 11), inertias, 'bo-')
ax1.set_xlabel('k')
ax1.set_ylabel('Inertia (SSE)')
ax1.set_title('Elbow Method')

ax2.plot(range(2, 11), silhouettes, 'go-')
ax2.set_xlabel('k')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Analysis')

# ----- PCA (Principal Component Analysis) -----
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"Explained variance: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")

# Plot PCA
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')

# ----- t-SNE -----
# Good for visualization, preserves local structure
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')

# KEY DIFFERENCES (EXAM):
# PCA: Linear, fast, preserves global structure, variance-based
# t-SNE: Non-linear, slow, preserves local structure, probability-based"""
        return f'<h4>8. Clustering & Dimensionality Reduction</h4><details class="code-block"><summary>📝 Clustering Code</summary><pre><code>{code}</code></pre></details>'
    
    def _visualization_section(self):
        code = """# ============================================================
# VISUALIZATION - All Common Plot Types
# ============================================================
import matplotlib.pyplot as plt
import seaborn as sns

# ----- ECDF Plot (Exam 2022) -----
# Method 1: Manual
data = df['column'].dropna().values
sorted_data = np.sort(data)
ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
plt.plot(sorted_data, ecdf)

# Method 2: Seaborn (preferred)
sns.ecdfplot(data=df, x='column')

# ----- Cumulative Sum Plot (Exam 2021) -----
df_sorted = df.sort_values('score', ascending=False)
cumsum = df_sorted['value'].cumsum()
cumsum_frac = cumsum / cumsum.iloc[-1]
x = np.arange(1, len(cumsum) + 1) / len(cumsum)

plt.plot(x, cumsum_frac, label='Actual')
plt.plot([0, 1], [0, 1], 'r--', label='Uniform')
plt.xlabel('Fraction of items')
plt.ylabel('Cumulative fraction')
plt.legend()

# ----- Confusion Matrix Heatmap -----
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm_norm = cm / cm.sum()  # ALL cells sum to 1

sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues',
           xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')

# ----- Correlation Heatmap -----
corr = df[numeric_cols].corr(method='spearman')
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', 
           center=0, vmin=-1, vmax=1)

# ----- Bar Plot with Error Bars -----
grouped = df.groupby('category')['value'].agg(['mean', 'std']).reset_index()
plt.bar(grouped['category'], grouped['mean'], yerr=grouped['std'], capsize=5)
plt.xticks(rotation=45)

# ----- Histogram with KDE -----
sns.histplot(df['column'], kde=True)
plt.axvline(df['column'].mean(), color='red', linestyle='--', label='Mean')
plt.axvline(df['column'].median(), color='green', linestyle='--', label='Median')
plt.legend()

# ----- Box Plot -----
sns.boxplot(data=df, x='category', y='value')

# ----- Violin Plot -----
sns.violinplot(data=df, x='category', y='value')

# ----- Scatter Plot with Regression Line -----
sns.regplot(data=df, x='x', y='y', scatter_kws={'alpha':0.5})

# ----- ROC Curve -----
from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc_score = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'ROC (AUC = {auc_score:.3f})')
plt.plot([0, 1], [0, 1], 'r--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()"""
        return f'<h4>9. Visualization</h4><details class="code-block"><summary>📝 Visualization Code</summary><pre><code>{code}</code></pre></details>'
    
    def _observational_section(self):
        code = """# ============================================================
# OBSERVATIONAL STUDIES - Propensity Score Matching (Lecture 6)
# ============================================================
import pandas as pd
import numpy as np
import networkx as nx
import statsmodels.formula.api as smf

# ----- Complete Propensity Score Matching Workflow -----

# Step 1: Standardize continuous covariates (optional but recommended)
covariates = ['age', 'education', 'income', 'experience']
for col in covariates:
    if df[col].dtype in ['int64', 'float64']:
        df[col + '_std'] = (df[col] - df[col].mean()) / df[col].std()

# Step 2: Estimate propensity scores using logistic regression
# P(treatment | covariates)
formula = 'treatment ~ age_std + education_std + income_std + C(gender) + C(region)'
propensity_model = smf.logit(formula=formula, data=df).fit()
df['propensity_score'] = propensity_model.predict()

print(propensity_model.summary())

# Step 3: Check propensity score overlap between groups
import matplotlib.pyplot as plt
treated = df[df['treatment'] == 1]['propensity_score']
control = df[df['treatment'] == 0]['propensity_score']

plt.hist(treated, bins=30, alpha=0.5, label='Treated', density=True)
plt.hist(control, bins=30, alpha=0.5, label='Control', density=True)
plt.xlabel('Propensity Score')
plt.ylabel('Density')
plt.legend()
plt.title('Propensity Score Distribution')

# Step 4: Match treated with control using max weight matching
def get_similarity(ps1, ps2):
    '''Similarity based on propensity score difference.'''
    return 1 - np.abs(ps1 - ps2)

treated_df = df[df['treatment'] == 1]
control_df = df[df['treatment'] == 0]

# Build bipartite graph
G = nx.Graph()

for t_idx, t_row in treated_df.iterrows():
    for c_idx, c_row in control_df.iterrows():
        similarity = get_similarity(t_row['propensity_score'], 
                                   c_row['propensity_score'])
        # Only add edge if similarity is high enough
        if similarity > 0.9:  # caliper
            G.add_weighted_edges_from([(t_idx, c_idx, similarity)])

# Find maximum weight matching (Hungarian algorithm)
matching = nx.max_weight_matching(G, maxcardinality=True)

# Step 5: Extract matched pairs
matched_treated = []
matched_control = []
for pair in matching:
    if pair[0] in treated_df.index:
        matched_treated.append(pair[0])
        matched_control.append(pair[1])
    else:
        matched_treated.append(pair[1])
        matched_control.append(pair[0])

matched_df = pd.concat([
    df.loc[matched_treated],
    df.loc[matched_control]
])

print(f"Matched {len(matching)} pairs")

# Step 6: Check covariate balance after matching
print("\\nCovariate Balance After Matching:")
for cov in covariates:
    t_mean = matched_df[matched_df['treatment'] == 1][cov].mean()
    c_mean = matched_df[matched_df['treatment'] == 0][cov].mean()
    std_diff = (t_mean - c_mean) / df[cov].std()
    print(f"{cov}: Treated={t_mean:.3f}, Control={c_mean:.3f}, StdDiff={std_diff:.3f}")

# Step 7: Estimate treatment effect
treated_outcome = matched_df[matched_df['treatment'] == 1]['outcome'].mean()
control_outcome = matched_df[matched_df['treatment'] == 0]['outcome'].mean()
ate = treated_outcome - control_outcome  # Average Treatment Effect

print(f"\\nTreated mean outcome: {treated_outcome:.4f}")
print(f"Control mean outcome: {control_outcome:.4f}")
print(f"Average Treatment Effect (ATE): {ate:.4f}")

# T-test for significance
from scipy import stats
t_stat, p_val = stats.ttest_ind(
    matched_df[matched_df['treatment'] == 1]['outcome'],
    matched_df[matched_df['treatment'] == 0]['outcome']
)
print(f"p-value: {p_val:.6f}")"""
        return f'<h4>10. Observational Studies</h4><details class="code-block"><summary>📝 Propensity Matching Code</summary><pre><code>{code}</code></pre></details>'


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_exam_blitz(data_dir, exam_notebook=None, output_file="exam_blitz_report.html"):
    """Main entry point."""
    print("=" * 70)
    print("🎯 EXAM BLITZ v5.0 (COMPLETE) - Full ADA Coverage")
    print("=" * 70)
    print("Coverage: ALL Exams 2019-2023 + Full Course Material")
    print("NEW: Propensity Matching, k-NN, Random Forest, ROC/AUC,")
    print("     Bootstrap CI, Hypothesis Testing, PCA/t-SNE, Silhouette")
    print("=" * 70)
    
    data_dir = Path(data_dir)
    if not data_dir.exists():
        print(f"❌ Directory not found: {data_dir}")
        sys.exit(1)
    
    report = ReportBuilder()
    
    # Parse exam notebook for context
    exam_context = None
    if exam_notebook:
        exam_context = ExamContextExtractor()
        exam_context.parse_notebook(exam_notebook)
        report.set_exam_context(exam_context)
    
    # Find and load data files
    files = DataLoader.find_data_files(data_dir)
    print(f"\n📁 Found {len(files)} data files")
    
    dataframes = {}
    for f in files:
        name, df = DataLoader.load_file(f)
        if df is not None:
            dataframes[name] = df
            print(f"   ✓ {name}: {df.shape[0]:,} rows × {df.shape[1]} cols")
    
    if not dataframes:
        print("❌ No data files found!")
        sys.exit(1)
    
    # Initialize ALL analyzers
    id_parser = IDParser(report)
    network_builder = NetworkBuilder(report)
    network_analyzer = NetworkAnalyzer(report)
    quintile_analyzer = QuintileAnalyzer(report)
    overview_analyzer = DataOverviewAnalyzer(report)
    distribution_analyzer = DistributionAnalyzer(report)  # NEW - ECDF, Cumsum
    groupby_analyzer = GroupbyAnalyzer(report)
    correlation_analyzer = CorrelationAnalyzer(report)
    regression_analyzer = RegressionAnalyzer(report)
    classification_analyzer = ClassificationAnalyzer(report)
    text_analyzer = TextAnalyzer(report)
    clustering_analyzer = ClusteringAnalyzer(report)
    # NEW in v5
    hypothesis_analyzer = HypothesisTestingAnalyzer(report)
    bootstrap_analyzer = BootstrapAnalyzer(report)
    propensity_analyzer = PropensityMatchingAnalyzer(report)
    
    # Process each dataframe
    for name, df in dataframes.items():
        print(f"\n{'='*60}")
        print(f"📂 Processing: {name}")
        
        df, _ = id_parser.detect_and_parse(df, name)
        dataframes[name] = df
        
        overview_analyzer.analyze(df, name)
        distribution_analyzer.analyze(df, name)  # NEW - ECDF, Cumsum
        groupby_analyzer.analyze(df, name)
        correlation_analyzer.analyze(df, name)
        hypothesis_analyzer.analyze(df, name)  # NEW
        bootstrap_analyzer.analyze(df, name)   # NEW
        regression_analyzer.analyze(df, name)
        classification_analyzer.analyze(df, name)
        text_analyzer.analyze(df, name)
        clustering_analyzer.analyze(df, name)
        propensity_analyzer.analyze(df, name)  # NEW
    
    # Generate Exam Cheat Sheet FIRST
    cheat_sheet = ExamCheatSheet(report)
    cheat_sheet.generate()
    
    # Build and analyze networks
    print(f"\n{'='*60}")
    print("🕸️ Building networks...")
    graphs, node_attrs = network_builder.detect_and_build(dataframes)
    
    for graph_name, G in graphs.items():
        node_df = node_attrs.get(graph_name)
        network_analyzer.analyze(G, graph_name, node_df)
        
        if node_df is not None:
            quintile_analyzer.analyze(G, node_df, graph_name)
    
    # Generate report
    print(f"\n{'='*60}")
    print("📝 Generating HTML report...")
    
    html = report.build(title="Exam Blitz v5.0 (Complete) - Full ADA Coverage")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\n✅ Report saved: {output_file}")
    print(f"📊 Total figures: {report.figure_counter}")
    print(f"📑 Total sections: {len(report.sections)}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exam Blitz v5.0 (Complete) - Full ADA Exam Coverage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
COMPLETE COVERAGE:
  Exams 2019-2023 + ALL Course Lectures and Exercises

NEW IN v5.0:
  - Propensity Score Matching (Lecture 6, Observational Studies)
  - k-NN Classifier (Lecture 4)
  - Random Forest (Lecture 5)
  - ROC/AUC Analysis (AppliedML)
  - Bootstrap Confidence Intervals (Describing_data)
  - Hypothesis Testing: t-test, chi-squared (Lecture 11)
  - PCA/t-SNE Visualization (Unsupervised_Learning)
  - Silhouette Score for Clustering

EXAMPLES:
  python exam_blitz_v5_complete.py ./exam_data/
  python exam_blitz_v5_complete.py ./data --exam exam_2024.ipynb -o report.html
        """
    )
    parser.add_argument("data_dir", help="Directory containing exam data files")
    parser.add_argument("--exam", "-e", help="Path to exam notebook for context extraction")
    parser.add_argument("-o", "--output", default="exam_blitz_report.html", help="Output HTML file")
    
    args = parser.parse_args()
    run_exam_blitz(args.data_dir, args.exam, args.output)