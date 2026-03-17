---
name: data-analysis
description: Analyze, explore, clean, and visualize datasets with statistical rigor. Use when user asks to analyze data, find patterns, compute statistics, create visualizations, clean messy data, or explore a dataset. Trigger when user says things like "analyze this data", "what trends do you see", "find patterns in", "create a chart", "clean this dataset", "run statistics on", "what does this data tell us", or provides CSV/Excel/JSON data for exploration. Also trigger for A/B test analysis, cohort analysis, and data quality assessments. Do NOT trigger for simple data format conversions, database query writing without analysis, or ETL pipeline design.
metadata:
  version: "2.0.0"
  author: Upsonic
  tags: [data, analysis, statistics, visualization]
---

# Data Analysis

Explore, clean, analyze, and communicate findings from data. The goal is always to answer a question — start with what the user wants to know and work backward to the analysis that answers it.

## Before You Analyze

### Understand the Question

Before touching the data, clarify:

1. **What question are we answering?** ("Is our conversion rate improving?" is an answerable question. "Analyze this data" is not — help the user sharpen it.)
2. **Who needs the answer?** (Engineer debugging an issue? Executive making a budget decision? Researcher testing a hypothesis?)
3. **What decisions will this inform?** (This determines how precise you need to be and what format the answer should take.)
4. **What's the timeline?** (A quick sanity check and a thorough statistical analysis require different approaches.)

If the user says "analyze this data" without a specific question, help them formulate one:
- "What would be most useful to know from this data?"
- "Are you looking for trends over time, comparisons between groups, or something else?"
- "Is there a specific business question this should answer?"

## Reference Materials and Scripts

- Execute `profile_data.py` with a data file path to get a quick profile of any CSV, Excel, or JSON dataset — it reports shape, types, missing values, stats, and value distributions. Run with `--help` for usage.
- Load `statistical-tests-guide.md` when choosing statistical tests — it has a decision matrix for test selection, effect size interpretation tables, and sample size guidelines.

### Understand the Data

Before analysis, get your bearings:

1. **Source and context**: Where did this data come from? How was it collected? What time period does it cover?
2. **Schema**: What are the columns/fields? What do they represent? What are the data types?
3. **Scale**: How many rows/records? What's the granularity? (Per user? Per day? Per transaction?)
4. **Known issues**: Is the data known to be incomplete, biased, or have quality problems?

```python
# First look at any dataset
import pandas as pd

df = pd.read_csv("data.csv")  # or read_excel, read_json, etc.
print(f"Shape: {df.shape}")
print(f"\nColumn types:\n{df.dtypes}")
print(f"\nFirst rows:\n{df.head()}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nBasic stats:\n{df.describe()}")
```

## Analysis Workflow

### Step 1: Clean and Validate

Data quality determines analysis quality. Don't skip this.

#### Handle Missing Values
- **Count them first**: What percentage of each column is missing?
- **Understand why**: Are they random? Systematic? (e.g., optional fields vs data collection failures)
- **Choose a strategy and document it**:
  - Drop rows: When missing data is rare and random (less than 5%)
  - Impute with median/mode: When missing data is moderate and the distribution is known
  - Flag as separate category: When missingness itself is informative
  - Leave as-is: When the analysis method handles nulls natively

```python
# Document your decisions
missing_pct = df.isnull().sum() / len(df) * 100
print("Missing data percentage per column:")
print(missing_pct[missing_pct > 0].sort_values(ascending=False))
```

#### Handle Outliers
- **Detect**: Use IQR method, z-scores, or domain knowledge
- **Investigate**: Are they errors or legitimate extreme values?
- **Document your decision**: Keep, cap, or remove — and explain why

```python
# IQR method for outlier detection
Q1 = df['value'].quantile(0.25)
Q3 = df['value'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['value'] < Q1 - 1.5 * IQR) | (df['value'] > Q3 + 1.5 * IQR)]
print(f"Found {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%)")
```

#### Validate Data Types and Ranges
- Dates should be dates, numbers should be numbers
- Check for impossible values (negative ages, future dates, percentages over 100)
- Verify categorical values are consistent (watch for "USA", "US", "United States")

### Step 2: Explore

Start broad, then focus on what's interesting.

#### Descriptive Statistics
Always start here — understand the basics before going deeper.

```python
# Numerical columns
print(df.describe())

# Categorical columns
for col in df.select_dtypes(include='object').columns:
    print(f"\n{col}: {df[col].nunique()} unique values")
    print(df[col].value_counts().head(10))
```

#### Distributions
Understanding shape matters for choosing the right tests.

```python
import matplotlib.pyplot as plt

# Distribution of key metrics
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, col in enumerate(['metric_a', 'metric_b', 'metric_c']):
    df[col].hist(ax=axes[i], bins=30)
    axes[i].set_title(col)
    axes[i].axvline(df[col].median(), color='red', linestyle='--', label='median')
    axes[i].legend()
plt.tight_layout()
plt.savefig("distributions.png")
```

#### Correlations and Relationships
Look for patterns between variables.

```python
# Correlation matrix for numerical columns
corr = df.select_dtypes(include='number').corr()
print("Strong correlations (|r| > 0.5):")
for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)):
        if abs(corr.iloc[i, j]) > 0.5:
            print(f"  {corr.columns[i]} vs {corr.columns[j]}: {corr.iloc[i,j]:.3f}")
```

#### Trends Over Time
If the data has a time dimension, always look at trends.

```python
# Time series analysis
df['date'] = pd.to_datetime(df['date'])
daily = df.groupby('date')['metric'].agg(['mean', 'count'])
daily['mean'].plot(figsize=(12, 4), title='Daily Average')
plt.savefig("trend.png")
```

### Step 3: Analyze

Choose the right method for the question.

#### Comparison Questions ("Is A different from B?")

Use the right statistical test:
- **Two groups, continuous outcome**: t-test (if normal) or Mann-Whitney U (if not)
- **Multiple groups**: ANOVA (if normal) or Kruskal-Wallis (if not)
- **Two groups, categorical outcome**: Chi-squared test
- **Before/after with same subjects**: Paired t-test or Wilcoxon signed-rank

Always report:
- Sample sizes for each group
- Effect size (not just p-value) — Cohen's d, odds ratio, or percentage difference
- Confidence intervals
- Practical significance, not just statistical significance

```python
from scipy import stats

# Example: comparing two groups
group_a = df[df['variant'] == 'A']['metric']
group_b = df[df['variant'] == 'B']['metric']

# Check normality first
_, p_normal_a = stats.shapiro(group_a.sample(min(5000, len(group_a))))
_, p_normal_b = stats.shapiro(group_b.sample(min(5000, len(group_b))))

if p_normal_a > 0.05 and p_normal_b > 0.05:
    stat, p_value = stats.ttest_ind(group_a, group_b)
    test_name = "t-test"
else:
    stat, p_value = stats.mannwhitneyu(group_a, group_b)
    test_name = "Mann-Whitney U"

# Effect size (Cohen's d)
pooled_std = ((group_a.std()**2 + group_b.std()**2) / 2) ** 0.5
cohens_d = (group_a.mean() - group_b.mean()) / pooled_std

print(f"Test: {test_name}")
print(f"Group A: mean={group_a.mean():.3f}, n={len(group_a)}")
print(f"Group B: mean={group_b.mean():.3f}, n={len(group_b)}")
print(f"Difference: {group_a.mean() - group_b.mean():.3f} ({(group_a.mean() - group_b.mean())/group_b.mean()*100:.1f}%)")
print(f"Cohen's d: {cohens_d:.3f}")
print(f"p-value: {p_value:.4f}")
```

#### Trend Questions ("Is this changing over time?")

- Use rolling averages to smooth noise
- Look for seasonality (day-of-week, monthly, quarterly patterns)
- Segment by key dimensions — overall trends can mask group-level differences
- Compare period-over-period (WoW, MoM, YoY)

#### Composition Questions ("What's the breakdown?")

- Use percentages and proportions
- Show both absolute numbers and percentages
- Look for the Pareto principle (80/20 rule)
- Segment into meaningful groups

#### A/B Test Analysis

For experiment analysis, follow this checklist:

1. **Sample size**: Was the test adequately powered?
2. **Duration**: Did it run long enough for weekly patterns?
3. **Randomization**: Were groups balanced on key dimensions?
4. **Metric definition**: Is the metric clearly defined and correctly computed?
5. **Statistical test**: Use the appropriate test for the metric type
6. **Multiple comparisons**: Correct for multiple testing if checking many metrics
7. **Practical significance**: Is the effect large enough to matter?

### Step 4: Communicate Findings

Structure matters as much as the analysis itself.

#### Lead With the Answer
Start with what the user asked. Then support it.

**Pattern:**
```
[Answer to the question in 1-2 sentences]

[Key supporting evidence: 2-4 bullet points with specific numbers]

[Methodology note: How you arrived at this, in 1-2 sentences]

[Caveats and limitations: What could affect this conclusion]

[Recommended next steps: What to do with this information]
```

#### Visualization Guidelines

Choose the right chart type:
- **Trend over time**: Line chart
- **Comparison between categories**: Bar chart (horizontal for many categories)
- **Distribution**: Histogram or box plot
- **Relationship between two variables**: Scatter plot
- **Part-of-whole**: Stacked bar or pie chart (only for 2-5 categories)
- **Multiple dimensions**: Heatmap or small multiples

For every chart:
- Title that states the finding, not just the data ("Conversion drops 40% on weekends" not "Conversion by day")
- Label axes with units
- Include sample size or time period
- Use consistent colors across related charts

#### What to Always Include
- **Sample sizes**: How much data backs each claim
- **Time periods**: When the data is from
- **Definitions**: How key metrics are calculated
- **Confidence**: How certain you are (confidence intervals, p-values where appropriate)
- **Limitations**: What the data can't tell you

## Common Pitfalls

- **Survivorship bias**: Only analyzing data from things that succeeded/persisted
- **Simpson's paradox**: A trend that appears in groups but reverses when combined
- **Confounding variables**: Correlation doesn't mean causation — look for third variables
- **Small sample sizes**: Don't draw strong conclusions from small N. State the limitation.
- **Cherry-picking timeframes**: Make sure the time period is representative
- **Precision theater**: Reporting "23.847%" when the data only supports "roughly 24%"
- **Missing baseline**: A number without context is meaningless ("10,000 errors" — is that a lot?)

## Handling Common Requests

### "Just give me a quick look"
Provide: shape, column summary, missing data count, top-level descriptive stats, one key insight. Keep it to one screen of output.

### "What's interesting in this data?"
Run the full exploration workflow (Step 2) and surface the 3-5 most notable patterns: unexpected distributions, strong correlations, outliers, trends.

### "Is this statistically significant?"
Clarify what comparison they mean, choose the right test, report effect size and p-value, and explain what it means in practical terms.

### "Create a dashboard / visualization"
Ask what decisions the dashboard supports. Design 3-5 charts that answer those questions. Use consistent styling and clear titles that state findings.
