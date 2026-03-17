# Statistical Tests Quick Reference

## Choosing the Right Test

### Comparing Two Groups

| Data Type | Normal Distribution? | Test |
|-----------|---------------------|------|
| Continuous | Yes | Independent t-test |
| Continuous | No | Mann-Whitney U test |
| Continuous (paired) | Yes | Paired t-test |
| Continuous (paired) | No | Wilcoxon signed-rank test |
| Categorical | N/A | Chi-squared test |
| Proportions | N/A | Z-test for proportions |

### Comparing Three or More Groups

| Data Type | Normal Distribution? | Test |
|-----------|---------------------|------|
| Continuous | Yes | One-way ANOVA |
| Continuous | No | Kruskal-Wallis H test |
| Continuous (repeated) | Yes | Repeated measures ANOVA |
| Continuous (repeated) | No | Friedman test |

### Relationships Between Variables

| Question | Test |
|----------|------|
| Linear relationship between two continuous variables? | Pearson correlation |
| Monotonic relationship (non-linear OK)? | Spearman correlation |
| Predict continuous outcome from predictors? | Linear regression |
| Predict binary outcome? | Logistic regression |
| Association between two categorical variables? | Chi-squared test of independence |

## Checking Normality

```python
from scipy import stats

# Shapiro-Wilk (best for n < 5000)
stat, p = stats.shapiro(data)
print(f"Normal: {'Yes' if p > 0.05 else 'No'} (p={p:.4f})")

# For larger samples, use visual inspection
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(data, bins=30)
axes[0].set_title("Histogram")
stats.probplot(data, plot=axes[1])
plt.tight_layout()
```

## Effect Size Interpretation

### Cohen's d (difference between means)
| d | Interpretation |
|---|---------------|
| 0.2 | Small effect |
| 0.5 | Medium effect |
| 0.8 | Large effect |

### Pearson r (correlation)
| r | Interpretation |
|---|---------------|
| 0.1 | Small/weak |
| 0.3 | Medium/moderate |
| 0.5 | Large/strong |

### Odds Ratio
| OR | Interpretation |
|----|---------------|
| 1.0 | No effect |
| 1.5 | Small effect |
| 2.0 | Medium effect |
| 3.0+ | Large effect |

## P-value Interpretation

- p < 0.001: Very strong evidence against null hypothesis
- p < 0.01: Strong evidence
- p < 0.05: Moderate evidence (conventional threshold)
- p < 0.10: Weak evidence
- p >= 0.10: Insufficient evidence

**Remember:** p-value tells you how surprising the data is IF the null hypothesis is true. It does NOT tell you:
- The probability that the null hypothesis is true
- The magnitude of the effect (use effect size for that)
- Whether the result is practically meaningful

## Multiple Comparisons

When testing multiple hypotheses, adjust for multiple comparisons:

- **Bonferroni**: Divide alpha by number of tests. Conservative. Use for few comparisons.
- **Holm-Bonferroni**: Step-down Bonferroni. Less conservative. Generally preferred over Bonferroni.
- **Benjamini-Hochberg (FDR)**: Controls false discovery rate. Use for many comparisons (e.g., gene expression).

```python
from statsmodels.stats.multitest import multipletests

p_values = [0.001, 0.013, 0.029, 0.04, 0.15]
reject, corrected_p, _, _ = multipletests(p_values, method='holm')
print("Holm-corrected p-values:", corrected_p)
print("Reject null:", reject)
```

## Sample Size Guidelines

| Analysis Type | Minimum Per Group | Recommended |
|--------------|-------------------|-------------|
| t-test | 20 | 30+ |
| ANOVA | 20 per group | 30+ per group |
| Chi-squared | 5 expected per cell | 10+ per cell |
| Correlation | 30 | 50+ |
| Regression | 10-20 per predictor | 50+ per predictor |

For A/B tests, calculate required sample size based on:
- Baseline conversion rate
- Minimum detectable effect (MDE)
- Desired power (typically 0.80)
- Significance level (typically 0.05)

```python
from statsmodels.stats.power import NormalIndPower

power = NormalIndPower()
n = power.solve_power(
    effect_size=0.1,    # Expected effect size
    alpha=0.05,         # Significance level
    power=0.8,          # Statistical power
    ratio=1.0,          # Equal group sizes
)
print(f"Required sample size per group: {int(n)}")
```
