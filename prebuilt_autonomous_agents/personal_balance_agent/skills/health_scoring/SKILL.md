# Health Scoring Skill

## Purpose
Compute a quantitative financial health score (0-100) from five weighted components. This is the single-number summary of the person's financial standing, backed by transparent component scores.

## When to Use
Phase 3 — after the balance sheet is computed.

## Input
| Parameter | Type | Description |
|-----------|------|-------------|
| analysis_path | path | `analyses/{analysis_name}/` |

## Actions

### 1. Calculate component scores

Each component scores 0-100. The formula scales linearly from 0 (worst) to 100 (at or beyond benchmark). Clamp all scores to [0, 100].

#### Savings Rate (Weight: 25%)
```
value = net_cashflow / total_income
benchmark = 0.20  (20%)
score = min(100, (value / benchmark) * 100)
```
- If `value < 0` (spending more than earning), score = 0
- Adjust benchmark by age: under 25 → 0.10, 25-35 → 0.20, 35-50 → 0.25, over 50 → 0.30

#### Debt Ratio (Weight: 20%)
```
value = total_monthly_debt_payments / total_income
benchmark = 0.36  (36% — standard DTI threshold)
score = max(0, (1 - value / benchmark) * 100)   // lower is better
```
- If `value = 0` (no debt), score = 100
- If `value > benchmark`, score decreases proportionally past 0

#### Emergency Fund (Weight: 25%)
```
value = liquid_assets / monthly_expenses
benchmark = 6  (months)
score = min(100, (value / benchmark) * 100)
```
- If `value = 0`, score = 0
- For freelancers/self-employed, benchmark = 9 months

#### Expense Stability (Weight: 15%)
```
value = std_deviation(monthly_expenses) / mean(monthly_expenses)
benchmark = 0.15  (15% coefficient of variation)
score = max(0, (1 - value / benchmark) * 100)   // lower variation is better
```
- If only one month of data, use `fixed_ratio` as proxy:
  ```
  score = min(100, fixed_ratio * 100 + 20)  // higher fixed ratio → more predictable
  ```
  Cap at 75 since single-month data has low confidence.

#### Investment Rate (Weight: 15%)
```
value = monthly_investment_contributions / total_income
benchmark = 0.15  (15%)
score = min(100, (value / benchmark) * 100)
```
- Count savings transfers to investment accounts
- If no investment data available but person is under 25, give partial credit (score = 30) with a note

### 2. Compute overall score

```
overall = (savings_rate_score * 0.25) +
          (debt_ratio_score * 0.20) +
          (emergency_fund_score * 0.25) +
          (expense_stability_score * 0.15) +
          (investment_rate_score * 0.15)
```

Round to nearest integer.

### 3. Assign letter grade

| Range | Grade |
|-------|-------|
| 95-100 | A+ |
| 90-94 | A |
| 85-89 | A- |
| 80-84 | B+ |
| 75-79 | B |
| 70-74 | B- |
| 65-69 | C+ |
| 60-64 | C |
| 55-59 | C- |
| 40-54 | D |
| 0-39 | F |

### 4. Assign status labels per component

| Score Range | Status |
|-------------|--------|
| ≥ 90 | `"excellent"` |
| ≥ 75 | `"healthy"` |
| ≥ 60 | `"on_track"` |
| ≥ 40 | `"needs_attention"` |
| < 40 | `"at_risk"` |

### 5. Determine trend

- If historical analyses exist in `analyses.json`, compare current overall score to the most recent one.
- `"improving"` if current > previous
- `"declining"` if current < previous
- `"stable"` if within ±2 points
- `"first_analysis"` if no history exists

### 6. Append Phase 3 entry to `log.json`

```json
{
  "name": "Phase 3: Health Scoring",
  "completed_at": "2026-04-25T10:15:00Z",
  "overall_score": 72,
  "grade": "B-",
  "components": {
    "savings_rate": {"score": 65, "value": 0.20, "benchmark": 0.20, "weight": 0.25, "status": "on_track"},
    "debt_ratio": {"score": 80, "value": 0.167, "benchmark": 0.36, "weight": 0.20, "status": "healthy"},
    "emergency_fund": {"score": 50, "value": 2.1, "unit": "months", "benchmark": 6, "weight": 0.25, "status": "needs_attention"},
    "expense_stability": {"score": 85, "value": 0.08, "benchmark": 0.15, "weight": 0.15, "status": "healthy"},
    "investment_rate": {"score": 60, "value": 0.05, "benchmark": 0.15, "weight": 0.15, "status": "on_track"}
  },
  "trend": "first_analysis",
  "confidence": "medium",
  "confidence_notes": ["Single-month data limits expense stability accuracy", "Investment data incomplete"]
}
```

## Output
- `{analysis_path}/log.json` — updated with Phase 3 health scoring entry
- No other files created or modified
