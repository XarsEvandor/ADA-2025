import React, { useState, useMemo } from 'react';
import { Search, ChevronDown, ChevronRight, Copy, Check, AlertTriangle, BookOpen, Network, BarChart3, FileText, Brain, Calculator, Moon, Sun, Zap, Target, TrendingUp } from 'lucide-react';

const questionTypes = [
  {
    id: 'data-exploration',
    title: 'Data Exploration',
    icon: BarChart3,
    color: 'blue',
    keywords: ['how many', 'count', 'distinct', 'unique', 'shape', 'number of rows', 'number of columns', 'describe'],
    blitzSearch: 'Data Overview, Value Counts, describe() table',
    templates: [
      {
        question: 'How many distinct [X] are in the dataframe?',
        answer: 'There are [NUMBER] distinct [X] in the dataframe.',
        note: 'Use df["column"].nunique() or len(df["column"].unique())'
      },
      {
        question: 'What is the shape of the dataframe?',
        answer: 'The dataframe has [ROWS] rows and [COLUMNS] columns.',
        note: 'Check df.shape in blitz output'
      }
    ]
  },
  {
    id: 'id-parsing',
    title: 'ID Parsing & Column Creation',
    icon: FileText,
    color: 'blue',
    keywords: ['extract', 'parse', 'create column', 'season', 'episode', 'format sAA_eBB'],
    blitzSearch: 'ID Parsing Analysis, String Operations',
    templates: [
      {
        question: 'Extract season/episode from ID like s01_e05_c03_u012',
        answer: 'The season is [s01] and the episode is [s01_e05]. The ID format is sAA_eBB_cCC_uDDD where AA=season, BB=episode, CC=scene, DDD=utterance.',
        note: 'Common pattern: id.split("_")[0] for season, "_".join(id.split("_")[:2]) for episode'
      }
    ]
  },
  {
    id: 'regression',
    title: 'Regression Interpretation',
    icon: TrendingUp,
    color: 'green',
    keywords: ['intercept', 'coefficient', 'C(variable', 'Treatment(reference=', 'significant', 'regression', 'ols', 'logit', 'p-value'],
    blitzSearch: 'Regression Analysis → find your specific formula',
    templates: [
      {
        question: 'What does the intercept represent?',
        answer: 'The intercept ([VALUE]) represents the mean [OUTCOME] when all categorical predictors are at their reference level. Specifically, it is the average [OUTCOME] for [REFERENCE CATEGORY, e.g., season 1].',
        note: 'From 2023 exam: "the intercept corresponds to the average utterance length of season 1 lines" when reference="s01"'
      },
      {
        question: 'What does coefficient C(X, Treatment(reference=Y))[T.Z] represent?',
        answer: 'This coefficient ([VALUE]) represents the difference in average [OUTCOME] between [Z] and the reference category [Y]. A [positive/negative] coefficient means [Z] has [higher/lower] [OUTCOME] by [VALUE] units compared to [Y].',
        note: 'Key insight: It represents BY HOW MUCH the average changes from reference to this category'
      },
      {
        question: 'Is the difference significant at 0.05?',
        answer: 'Looking at the p-value ([P-VALUE]), since p [< / ≥] 0.05, the difference [is / is not] statistically significant at the 0.05 level. We [can / cannot] reject the null hypothesis that there is no difference.',
        note: 'TRAP: p = 0.05 is NOT significant. Must be STRICTLY LESS than 0.05'
      },
      {
        question: 'Log-transformed outcome interpretation',
        answer: 'With a log-transformed dependent variable, the coefficient represents a multiplicative effect. A coefficient of [VALUE] means multiplying the outcome by e^[VALUE] ≈ [MULTIPLIER]. For example, -0.22 means multiplying by e^(-0.22) ≈ 0.80, an ~20% decrease.',
        note: 'From course: "whenever a patient has high blood pressure we multiply by e^(-0.22) ≈ 0.80"'
      }
    ]
  },
  {
    id: 'classification',
    title: 'Classification & Confusion Matrix',
    icon: Target,
    color: 'green',
    keywords: ['accuracy', 'confusion matrix', 'random baseline', 'decision tree', 'classifier', 'precision', 'recall', 'F1'],
    blitzSearch: 'Classification Analysis → target variable, normalized matrix',
    templates: [
      {
        question: 'Compare accuracy to random baseline',
        answer: 'The classifier achieves [X]% accuracy. The random baseline (predicting uniformly at random among [N] classes) would achieve 1/[N] = [Y]% accuracy. The classifier outperforms random by [X-Y] percentage points.',
        note: 'CRITICAL: Random baseline = 1/number_of_classes, NOT most frequent class!'
      },
      {
        question: 'Which character/class is most distinct? (from confusion matrix)',
        answer: 'Looking at the diagonal of the normalized confusion matrix, [CLASS] has the highest value ([VALUE]), meaning it is correctly classified most often and is therefore most distinct in its characteristics.',
        note: 'From 2023 exam solution: "Monica is the most recognisable by her speech, because she was the best predicted"'
      },
      {
        question: 'Which two classes are most similar?',
        answer: 'The highest off-diagonal value ([VALUE]) appears at position [actual=A, predicted=B], indicating [A] is most frequently misclassified as [B]. These classes are most similar.',
        note: 'Look for max value OFF the diagonal'
      },
      {
        question: 'Which two classes are least similar?',
        answer: 'The lowest off-diagonal values (approaching 0) appear between [CLASS A] and [CLASS B], indicating they are rarely confused and therefore least similar.',
        note: 'From 2023: "Monica and Phoebe have the least similar way of talking because Monica\'s lines were never interpreted as Phoebe\'s"'
      },
      {
        question: 'Reading a confusion matrix for metrics',
        answer: 'TP=[value], FP=[value], FN=[value], TN=[value]. Accuracy=(TP+TN)/total. Precision=TP/(TP+FP). Recall=TP/(TP+FN). F1=2×(P×R)/(P+R).',
        note: 'From Quiz 8: Given matrix with TP=30, FP=10, FN=10, TN=50: Accuracy=0.8, Precision=0.75'
      }
    ]
  },
  {
    id: 'network',
    title: 'Network & Centrality Analysis',
    icon: Network,
    color: 'green',
    keywords: ['PageRank', 'degree', 'centrality', 'graph', 'edges', 'nodes', 'in-degree', 'out-degree', 'networkx'],
    blitzSearch: 'Network Analysis → centrality tables',
    templates: [
      {
        question: 'What does PageRank measure?',
        answer: 'PageRank measures importance based on INCOMING connections from other important nodes. A high PageRank for [NODE] means they receive many responses/connections from other influential nodes.',
        note: 'PageRank depends ONLY on incoming edges!'
      },
      {
        question: 'What does out-degree mean?',
        answer: 'Out-degree counts outgoing edges. A high out-degree for [NODE] means they [reply to / connect to] many others. It does NOT indicate how many others interact with them.',
        note: 'TRAP: out_degree_centrality normalizes by (n-1), raw out_degree does not'
      },
      {
        question: 'If we remove X\'s outgoing edges, how does their PageRank change?',
        answer: 'PageRank depends ONLY on INCOMING edges. Removing [X]\'s outgoing edges does NOT affect [X]\'s own PageRank—it only affects the PageRank of nodes [X] was pointing to.',
        note: 'From 2023 exam: "True, since PageRank depends on incoming links"'
      },
      {
        question: 'If edges are inverted, does PageRank stay the same?',
        answer: 'FALSE. PageRank depends on incoming edges. If we invert every edge, the in-degree becomes the out-degree, completely changing PageRank values.',
        note: 'From 2023 solution: "False, because PageRank centrality depends on the number of incoming edges"'
      },
      {
        question: 'Does higher out-degree mean others spoke less?',
        answer: 'NO. Higher out-degree only means [NODE] made more outgoing connections. It implies nothing about the activity of others. Out-degree measures what [NODE] does, not what others do.',
        note: 'Common misconception tested in exams'
      },
      {
        question: 'New node with many outgoing edges but no incoming - highest PageRank?',
        answer: 'FALSE. A node with no incoming edges will have very LOW PageRank (only the dampening factor contribution), regardless of outgoing edges.',
        note: 'From 2023: "False, for the same reason [PageRank depends on incoming links]"'
      }
    ]
  },
  {
    id: 'significance',
    title: 'Statistical Significance',
    icon: Calculator,
    color: 'yellow',
    keywords: ['significant at 0.05', 'p-value', 'confidence interval', 'reject', 'null hypothesis', 'statistically'],
    blitzSearch: 'Regression output p-values, t-test results',
    templates: [
      {
        question: 'Is the difference statistically significant?',
        answer: 'With p = [VALUE], since p [< / ≥] 0.05, the difference [is / is not] statistically significant at the 0.05 level. We [can / cannot] reject the null hypothesis.',
        note: 'p < 0.05 → significant; p ≥ 0.05 → NOT significant'
      },
      {
        question: 'Two datasets with different p-values for same hypothesis',
        answer: 'The p-values only suggest how likely each dataset was under H0. More evidence is needed to determine if H0 is true or false. One dataset rejecting and one not rejecting H0 does not make either conclusion definitive.',
        note: 'From Quiz 4: "p-values only suggest how likely each dataset was under H0. More evidence is needed"'
      },
      {
        question: 'Interpreting confidence intervals',
        answer: 'If the 95% CI does not include 0, the coefficient is significant at the 0.05 level. If CIs of two groups overlap substantially, their difference may not be significant.',
        note: 'Visual inspection: non-overlapping error bars suggest significance'
      }
    ]
  },
  {
    id: 'tfidf',
    title: 'TF-IDF & Text Analysis',
    icon: FileText,
    color: 'purple',
    keywords: ['TF-IDF', 'tokens', 'word frequency', 'vocabulary', 'term frequency', 'inverse document'],
    blitzSearch: 'Text Analysis section',
    templates: [
      {
        question: 'What is TF-IDF?',
        answer: 'TF-IDF = TF × IDF where TF = count of word in document, IDF = log(total docs / docs containing word). High TF-IDF means a word is frequent in this document but rare across all documents.',
        note: 'IDF formula from exam: log(number of episodes / episodes containing word)'
      },
      {
        question: 'How do character name tokens help classification?',
        answer: 'Character names in utterances can predict the speaker. If "joey" appears and Ross typically replies to Joey, the utterance is likely Ross\'s. Names indicate conversational patterns.',
        note: 'From 2023 solution: names can indicate who typically speaks to whom'
      },
      {
        question: 'Why might TF-IDF classification have issues?',
        answer: 'Potential issues: (1) Context similarity - people in same conversation speak similarly, (2) Common words dominate, (3) Rare words may overfit, (4) Doesn\'t capture word order/meaning.',
        note: 'From 2023: "if people are in the same conversation, they might speak similarly simply because they are in the same social context"'
      }
    ]
  },
  {
    id: 'causation',
    title: 'Causation & Observational Studies',
    icon: Brain,
    color: 'purple',
    keywords: ['cause', 'causal', 'observational', 'confound', 'propensity', 'matching', 'RCT', 'treatment effect'],
    blitzSearch: 'Conceptual - not typically in blitz report',
    templates: [
      {
        question: 'Can we conclude X causes Y?',
        answer: '[If observational]: No. This is an observational study, so we can only identify correlation/association, not causation. There may be confounding variables. [If RCT]: Yes, properly randomized experiments support causal claims.',
        note: 'NEVER say "causes" for observational data'
      },
      {
        question: 'What steps for causal analysis with observational data?',
        answer: 'With unobserved confounders: Regression with treatment + covariates may NOT identify causal effect. Options: (1) Assume no unobserved confounders and run regression, (2) Use propensity score matching to balance groups.',
        note: 'From Quiz 6: "regression may not let you identify the causal effect" with unobserved confounders'
      },
      {
        question: 'Why might a treatment show negative effect in naive analysis?',
        answer: 'Confounding! Example: Treatment group may have worse baseline characteristics. The "negative effect" might be due to pre-existing differences, not the treatment itself. Propensity score matching can address this.',
        note: 'LaLonde study example: naive analysis showed negative treatment effect, matching showed positive'
      }
    ]
  },
  {
    id: 'methodology',
    title: 'Methodology & Conceptual',
    icon: Brain,
    color: 'purple',
    keywords: ['why', 'explain', 'what\'s wrong', 'improve', 'propose', 'data leakage', 'confounder'],
    blitzSearch: 'Usually conceptual - requires domain knowledge',
    templates: [
      {
        question: 'Why can\'t we use feature X for prediction?',
        answer: 'Feature X cannot be used because [it wouldn\'t be available at prediction time / it IS the target / it causes data leakage / it perfectly encodes the outcome].',
        note: 'Common: using future information, or variables that wouldn\'t exist for new data'
      },
      {
        question: 'How to address same-conversation similarity confound?',
        answer: 'Match utterances from the same conversation across speakers. Ensure each speaker\'s lines are paired with others from the same conversational context, neutralizing the confound.',
        note: 'From 2023: "one-to-one matching between all groups of lines whose conversation identifier is the same"'
      },
      {
        question: 'How to make confusion matrix more interpretable vs random?',
        answer: 'Divide all values by the expected random confusion matrix values (1/(n_classes²) if normalized). Values > 1 indicate better than random, < 1 worse than random.',
        note: 'From 2023: "divide all values by the values of the confusion matrix for the random classifier"'
      }
    ]
  },
  {
    id: 'truefalse',
    title: 'True/False Questions',
    icon: Check,
    color: 'yellow',
    keywords: ['true or false', 'justify', 'statement'],
    blitzSearch: 'Depends on the specific claim',
    templates: [
      {
        question: 'General approach to T/F with justification',
        answer: 'State TRUE or FALSE clearly. Then provide 1-2 sentences explaining WHY using specific data/logic from the problem. Reference exact numbers when available.',
        note: 'Always justify with evidence, even if the answer seems obvious'
      }
    ]
  }
];

const traps = [
  {
    trap: 'Confusion matrix normalization',
    wrong: 'Row-wise (rows sum to 1)',
    correct: 'ALL cells sum to 1 (when asked for fully normalized)',
    source: '2023 exam: "Normalize the confusion matrix such that all cells sum to 1"'
  },
  {
    trap: 'Random baseline for classification',
    wrong: 'Most frequent class accuracy',
    correct: '1 / number_of_classes',
    source: '2023 exam: "random baseline is around 1/6" for 6-class problem'
  },
  {
    trap: 'PageRank dependency',
    wrong: 'Depends on outgoing edges',
    correct: 'Depends ONLY on incoming edges',
    source: '2023 exam: multiple T/F questions on this'
  },
  {
    trap: 'out_degree vs out_degree_centrality',
    wrong: 'They\'re the same',
    correct: 'out_degree_centrality = out_degree / (n-1)',
    source: '2023 exam: "do not use nx.out_degree_centrality as it normalizes"'
  },
  {
    trap: 'Correlation vs causation',
    wrong: '"X causes Y"',
    correct: '"X is associated with Y" (unless RCT)',
    source: 'Observational studies lecture & exercises'
  },
  {
    trap: 'p-value threshold',
    wrong: 'p = 0.05 is significant',
    correct: 'p < 0.05 is significant (strictly less than)',
    source: 'Multiple quiz questions'
  },
  {
    trap: 'Coefficient interpretation',
    wrong: '"Increase of X"',
    correct: '"Difference compared to REFERENCE category"',
    source: '2023 exam regression questions'
  },
  {
    trap: 'Coefficient significance',
    wrong: 'Large coefficient = significant',
    correct: 'Check p-value, not magnitude',
    source: 'Regression analysis exercises'
  },
  {
    trap: 'Cross-validation K selection',
    wrong: 'Choose K with lowest training error',
    correct: 'Choose K with lowest VALIDATION error',
    source: 'Quiz 8: Choose K=1 (lowest validation error)'
  },
  {
    trap: 'Micro vs Macro average',
    wrong: 'Same for all metrics',
    correct: 'Micro for overall, Macro for per-class performance',
    source: 'Quiz 5: micro for overall, macro for per-class'
  }
];

const decisionTree = [
  { condition: 'Asks "how many" / "count" / "shape"', type: 'Data Exploration', color: 'blue' },
  { condition: 'Mentions "intercept" / "coefficient" / "C(variable"', type: 'Regression', color: 'green' },
  { condition: 'Mentions "accuracy" / "confusion matrix" / "classifier"', type: 'Classification', color: 'green' },
  { condition: 'Mentions "PageRank" / "degree" / "edges" / "nodes"', type: 'Network', color: 'green' },
  { condition: 'Asks about "p-value" / "significant at 0.05"', type: 'Significance', color: 'yellow' },
  { condition: 'Mentions "TF-IDF" / "tokens" / "vocabulary"', type: 'TF-IDF', color: 'purple' },
  { condition: 'Asks "can we conclude" / "causes" / "causal"', type: 'Causation', color: 'purple' },
  { condition: 'Says "True or false" / "Justify"', type: 'True/False', color: 'yellow' },
  { condition: 'Asks "why" / "what\'s wrong" / "improve"', type: 'Methodology', color: 'purple' },
];

// Color helper function that returns appropriate classes based on darkMode state
const getColorClasses = (color, darkMode) => {
  const colors = {
    blue: {
      bg: darkMode ? 'bg-blue-950' : 'bg-blue-50',
      border: darkMode ? 'border-blue-700' : 'border-blue-200',
      text: darkMode ? 'text-blue-100' : 'text-blue-800',
      accent: darkMode ? 'text-blue-200' : 'text-blue-600'
    },
    green: {
      bg: darkMode ? 'bg-emerald-950' : 'bg-emerald-50',
      border: darkMode ? 'border-emerald-700' : 'border-emerald-200',
      text: darkMode ? 'text-emerald-100' : 'text-emerald-800',
      accent: darkMode ? 'text-emerald-200' : 'text-emerald-600'
    },
    yellow: {
      bg: darkMode ? 'bg-amber-950' : 'bg-amber-50',
      border: darkMode ? 'border-amber-600' : 'border-amber-200',
      text: darkMode ? 'text-amber-100' : 'text-amber-800',
      accent: darkMode ? 'text-amber-200' : 'text-amber-600'
    },
    purple: {
      bg: darkMode ? 'bg-purple-950' : 'bg-purple-50',
      border: darkMode ? 'border-purple-700' : 'border-purple-200',
      text: darkMode ? 'text-purple-100' : 'text-purple-800',
      accent: darkMode ? 'text-purple-200' : 'text-purple-600'
    },
    red: {
      bg: darkMode ? 'bg-red-950' : 'bg-red-50',
      border: darkMode ? 'border-red-700' : 'border-red-200',
      text: darkMode ? 'text-red-100' : 'text-red-800',
      accent: darkMode ? 'text-red-200' : 'text-red-600'
    }
  };
  return colors[color] || colors.blue;
};

export default function ADAExamNavigator() {
  const [darkMode, setDarkMode] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [activeTab, setActiveTab] = useState('decision');
  const [expandedSections, setExpandedSections] = useState({});
  const [copiedId, setCopiedId] = useState(null);

  const toggleSection = (id) => {
    setExpandedSections(prev => ({ ...prev, [id]: !prev[id] }));
  };

  const copyToClipboard = (text, id) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const filteredTypes = useMemo(() => {
    if (!searchQuery) return questionTypes;
    const query = searchQuery.toLowerCase();
    return questionTypes.filter(type => 
      type.title.toLowerCase().includes(query) ||
      type.keywords.some(k => k.toLowerCase().includes(query)) ||
      type.templates.some(t => 
        t.question.toLowerCase().includes(query) ||
        t.answer.toLowerCase().includes(query)
      )
    );
  }, [searchQuery]);

  const filteredTraps = useMemo(() => {
    if (!searchQuery) return traps;
    const query = searchQuery.toLowerCase();
    return traps.filter(t =>
      t.trap.toLowerCase().includes(query) ||
      t.wrong.toLowerCase().includes(query) ||
      t.correct.toLowerCase().includes(query)
    );
  }, [searchQuery]);

  // Text color helpers - using explicit conditional colors
  const textPrimary = darkMode ? 'text-gray-50' : 'text-gray-900';
  const textSecondary = darkMode ? 'text-gray-200' : 'text-gray-600';
  const textMuted = darkMode ? 'text-gray-400' : 'text-gray-500';
  const bgPrimary = darkMode ? 'bg-gray-900' : 'bg-gray-50';
  const bgSecondary = darkMode ? 'bg-gray-800' : 'bg-white';
  const bgTertiary = darkMode ? 'bg-gray-700' : 'bg-gray-100';
  const borderColor = darkMode ? 'border-gray-600' : 'border-gray-200';

  return (
    <div className={`min-h-screen transition-colors duration-300 ${bgPrimary} ${textPrimary}`}>
      {/* Header */}
      <header className={`sticky top-0 z-50 border-b ${bgSecondary} ${borderColor}`}>
        <div className="max-w-6xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl">
                <BookOpen className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className={`text-xl font-bold ${textPrimary}`}>ADA Exam Navigator</h1>
                <p className={`text-sm ${textMuted}`}>EPFL Applied Data Analysis</p>
              </div>
            </div>
            <button
              onClick={() => setDarkMode(!darkMode)}
              className={`p-2 rounded-lg ${bgTertiary} hover:opacity-80 transition-opacity ${textPrimary}`}
            >
              {darkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
            </button>
          </div>
          
          {/* Search */}
          <div className="relative">
            <Search className={`absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 ${textMuted}`} />
            <input
              type="text"
              placeholder="Search question types, keywords, templates..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className={`w-full pl-10 pr-4 py-3 rounded-xl border ${bgTertiary} ${borderColor} ${textPrimary} placeholder-gray-400 outline-none focus:ring-2 focus:ring-blue-500 transition-all`}
            />
          </div>

          {/* Tabs */}
          <div className="flex gap-2 mt-4 overflow-x-auto pb-2">
            {[
              { id: 'decision', label: 'Decision Tree', icon: Zap },
              { id: 'types', label: 'Question Types', icon: Target },
              { id: 'traps', label: 'Common Traps', icon: AlertTriangle },
              { id: 'blitz', label: 'Blitz Guide', icon: FileText }
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium whitespace-nowrap transition-colors ${
                  activeTab === tab.id
                    ? 'bg-blue-600 text-white'
                    : `${bgTertiary} ${textSecondary} hover:opacity-80`
                }`}
              >
                <tab.icon className="w-4 h-4" />
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-6xl mx-auto px-4 py-6">
        {/* Decision Tree Tab */}
        {activeTab === 'decision' && (
          <div className="space-y-4">
            <div className={`p-4 rounded-xl ${bgSecondary} border ${borderColor}`}>
              <h2 className={`text-lg font-semibold mb-4 flex items-center gap-2 ${textPrimary}`}>
                <Zap className="w-5 h-5 text-yellow-500" />
                Quick Question Type Identifier
              </h2>
              <div className="space-y-2">
                {decisionTree.map((item, idx) => {
                  const colors = getColorClasses(item.color, darkMode);
                  return (
                    <div
                      key={idx}
                      className={`p-3 rounded-lg border ${colors.bg} ${colors.border}`}
                    >
                      <div className="flex items-center justify-between flex-wrap gap-2">
                        <span className={colors.text}>
                          If question {item.condition}
                        </span>
                        <span className={`font-semibold px-3 py-1 rounded-full text-sm ${colors.bg} ${colors.text} border ${colors.border}`}>
                          → {item.type}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            <div className={`p-4 rounded-xl ${bgSecondary} border ${borderColor}`}>
              <h2 className={`text-lg font-semibold mb-2 ${textPrimary}`}>Question Distribution (from past exams)</h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {[
                  { label: 'Data & Exploration', pct: '20%', color: 'blue' },
                  { label: 'Regression', pct: '15%', color: 'green' },
                  { label: 'Classification', pct: '15%', color: 'green' },
                  { label: 'Networks', pct: '15%', color: 'green' },
                  { label: 'Visualization', pct: '10%', color: 'blue' },
                  { label: 'Significance', pct: '10%', color: 'yellow' },
                  { label: 'Text/TF-IDF', pct: '10%', color: 'purple' },
                  { label: 'True/False', pct: '5%', color: 'yellow' }
                ].map((item, idx) => {
                  const colors = getColorClasses(item.color, darkMode);
                  return (
                    <div key={idx} className={`p-3 rounded-lg ${colors.bg} ${colors.border} border`}>
                      <div className={`text-2xl font-bold ${colors.text}`}>{item.pct}</div>
                      <div className={colors.accent}>{item.label}</div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}

        {/* Question Types Tab */}
        {activeTab === 'types' && (
          <div className="space-y-4">
            {filteredTypes.map(type => {
              const colors = getColorClasses(type.color, darkMode);
              const Icon = type.icon;
              const isExpanded = expandedSections[type.id];
              
              return (
                <div
                  key={type.id}
                  className={`rounded-xl border overflow-hidden ${bgSecondary} ${borderColor}`}
                >
                  <button
                    onClick={() => toggleSection(type.id)}
                    className={`w-full p-4 flex items-center justify-between hover:opacity-90 transition-opacity`}
                  >
                    <div className="flex items-center gap-3">
                      <div className={`p-2 rounded-lg ${colors.bg} ${colors.border} border`}>
                        <Icon className={`w-5 h-5 ${colors.text}`} />
                      </div>
                      <div className="text-left">
                        <h3 className={`font-semibold ${textPrimary}`}>{type.title}</h3>
                        <p className={`text-sm ${textMuted}`}>
                          Keywords: {type.keywords.slice(0, 4).join(', ')}...
                        </p>
                      </div>
                    </div>
                    {isExpanded ? <ChevronDown className={`w-5 h-5 ${textSecondary}`} /> : <ChevronRight className={`w-5 h-5 ${textSecondary}`} />}
                  </button>
                  
                  {isExpanded && (
                    <div className={`p-4 border-t ${borderColor}`}>
                      <div className={`mb-4 p-3 rounded-lg ${bgTertiary}`}>
                        <p className={`text-sm font-medium ${textPrimary}`}>🔍 Search in Blitz Report:</p>
                        <p className={`text-sm ${textSecondary}`}>{type.blitzSearch}</p>
                      </div>
                      
                      <div className="space-y-4">
                        {type.templates.map((template, idx) => {
                          const templateId = `${type.id}-${idx}`;
                          return (
                            <div key={idx} className={`p-4 rounded-lg ${colors.bg} ${colors.border} border`}>
                              <div className="flex justify-between items-start mb-2">
                                <p className={`font-medium text-sm ${colors.text}`}>Q: {template.question}</p>
                                <button
                                  onClick={() => copyToClipboard(template.answer, templateId)}
                                  className={`p-1.5 rounded-lg ${bgTertiary} hover:opacity-80 transition-opacity`}
                                  title="Copy answer template"
                                >
                                  {copiedId === templateId ? (
                                    <Check className="w-4 h-4 text-green-400" />
                                  ) : (
                                    <Copy className={`w-4 h-4 ${textMuted}`} />
                                  )}
                                </button>
                              </div>
                              <div className={`p-3 rounded-lg ${bgSecondary} mb-2 border ${borderColor}`}>
                                <p className={`text-sm ${textPrimary}`}>{template.answer}</p>
                              </div>
                              <p className={`text-xs ${colors.accent}`}>
                                💡 {template.note}
                              </p>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {/* Traps Tab */}
        {activeTab === 'traps' && (
          <div className={`rounded-xl border overflow-hidden ${bgSecondary} ${borderColor}`}>
            <div className={`p-4 border-b ${getColorClasses('red', darkMode).bg} ${getColorClasses('red', darkMode).border}`}>
              <h2 className={`text-lg font-semibold flex items-center gap-2 ${getColorClasses('red', darkMode).text}`}>
                <AlertTriangle className="w-5 h-5" />
                Critical Exam Traps to Avoid
              </h2>
              <p className={`text-sm mt-1 ${getColorClasses('red', darkMode).accent}`}>
                These are common mistakes that cost points on past exams
              </p>
            </div>
            
            <div className={`divide-y ${borderColor}`}>
              {filteredTraps.map((trap, idx) => {
                const redColors = getColorClasses('red', darkMode);
                const greenColors = getColorClasses('green', darkMode);
                return (
                  <div key={idx} className="p-4">
                    <div className={`font-medium mb-2 ${textPrimary}`}>{trap.trap}</div>
                    <div className="grid md:grid-cols-2 gap-3">
                      <div className={`p-3 rounded-lg ${redColors.bg} border ${redColors.border}`}>
                        <p className={`text-xs font-semibold mb-1 ${redColors.text}`}>❌ WRONG</p>
                        <p className={`text-sm ${redColors.accent}`}>{trap.wrong}</p>
                      </div>
                      <div className={`p-3 rounded-lg ${greenColors.bg} border ${greenColors.border}`}>
                        <p className={`text-xs font-semibold mb-1 ${greenColors.text}`}>✓ CORRECT</p>
                        <p className={`text-sm ${greenColors.accent}`}>{trap.correct}</p>
                      </div>
                    </div>
                    <p className={`text-xs mt-2 ${textMuted}`}>
                      📚 Source: {trap.source}
                    </p>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Blitz Guide Tab */}
        {activeTab === 'blitz' && (
          <div className="space-y-4">
            <div className={`p-4 rounded-xl ${bgSecondary} border ${borderColor}`}>
              <h2 className={`text-lg font-semibold mb-4 ${textPrimary}`}>🎯 Blitz Report Navigation Guide</h2>
              <p className={`mb-4 text-sm ${textSecondary}`}>
                Your exam_blitz.py script generates all analyses. Here's where to find each answer type:
              </p>
              
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className={bgTertiary}>
                    <tr>
                      <th className={`text-left p-3 rounded-tl-lg ${textPrimary}`}>Question About...</th>
                      <th className={`text-left p-3 rounded-tr-lg ${textPrimary}`}>Find in Blitz Report</th>
                    </tr>
                  </thead>
                  <tbody className={`divide-y ${borderColor}`}>
                    {[
                      ['Rows, columns, unique values', '"Data Overview" section → shape, nunique'],
                      ['Category distribution', '"Value Counts" → column name dropdown'],
                      ['Mean, median, std', '"Data Overview" → describe() table'],
                      ['Regression coefficients', '"Regression Analysis" → match your formula exactly'],
                      ['Classification accuracy', '"Classification Analysis" → target variable section'],
                      ['Confusion matrix', '"Classification Analysis" → normalized matrix heatmap'],
                      ['PageRank, degree centrality', '"Network Analysis" → centrality tables & rankings'],
                      ['Correlations', '"Correlation Analysis" → heatmap and correlation table'],
                      ['Group comparisons', '"Groupby Analysis" → select grouping column'],
                      ['Text tokens', '"Text Analysis" → TF-IDF values, word counts'],
                      ['Clustering', '"Clustering Analysis" → silhouette scores, cluster assignments']
                    ].map(([question, location], idx) => (
                      <tr key={idx} className={darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-50'}>
                        <td className={`p-3 ${textPrimary}`}>{question}</td>
                        <td className={`p-3 font-mono text-xs ${textSecondary}`}>{location}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className={`p-4 rounded-xl ${getColorClasses('blue', darkMode).bg} border ${getColorClasses('blue', darkMode).border}`}>
              <h3 className={`font-semibold mb-2 ${getColorClasses('blue', darkMode).text}`}>💡 Pro Tips</h3>
              <ul className={`text-sm space-y-2 ${getColorClasses('blue', darkMode).accent}`}>
                <li>• For regression: Match the EXACT formula from the exam question (including reference category)</li>
                <li>• For networks: Check if they want raw degree or normalized centrality</li>
                <li>• For classification: Always check random baseline = 1/n_classes</li>
                <li>• Use Ctrl+F to search for specific variable names in the blitz HTML</li>
              </ul>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className={`border-t py-4 mt-8 ${borderColor}`}>
        <div className="max-w-6xl mx-auto px-4 text-center">
          <p className={`text-sm ${textMuted}`}>
            Based on EPFL ADA exams 2019-2023 • Good luck! 🍀
          </p>
        </div>
      </footer>
    </div>
  );
}
