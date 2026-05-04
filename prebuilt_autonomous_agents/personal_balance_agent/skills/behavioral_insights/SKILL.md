# Behavioral Insights Skill

## Purpose
Analyze spending patterns, habits, and behavioral tendencies from transaction data. This goes beyond what is spent to understand *how* and *when* money is spent — revealing actionable behavioral patterns.

## When to Use
Phase 5 — after risk analysis is complete.

## Input
| Parameter | Type | Description |
|-----------|------|-------------|
| analysis_path | path | `analyses/{analysis_name}/` |

## Actions

### 1. Temporal pattern analysis

Analyze `normalized_data.json` transactions by time dimensions:

- **Day-of-week distribution**: which days see the most spending? Calculate per-day average.
- **Weekend vs weekday**: what percentage of discretionary spending occurs Fri-Sun?
- **Time-of-month patterns**: payday surge (spending spikes right after salary), end-of-month squeeze
- **First week vs last week**: do expenses front-load or back-load within the month?

Flag if:
- Weekend spending > 40% of weekly discretionary → `"Weekend Spending Spike"`
- First 5 days after salary > 30% of monthly discretionary → `"Payday Surge"`
- Last week spending drops sharply → `"End-of-Month Squeeze"`

### 2. Category pattern analysis

- **Subscription audit**: list all recurring charges, total monthly cost, flag any duplicates or unused services
- **Dining frequency**: meals out per week, average ticket size, dining-to-grocery ratio
- **Small purchase accumulation**: count transactions under a threshold (e.g., 50 TRY) — do many small purchases add up significantly?
- **Impulse indicators**: transactions in entertainment/shopping categories that are irregular and above-average amount
- **Category concentration**: is spending heavily concentrated in few categories or spread evenly?

### 3. Trend analysis (if multi-period data available)

If previous analyses exist in `analyses.json`:
- Month-over-month change in total spending
- Category-level trends (which categories are growing/shrinking?)
- Income growth vs expense growth comparison
- Savings rate trend

If only single-period data, note: `"trend_available": false` and skip.

### 4. Benchmark comparison

Compare key ratios against benchmarks for the person's income tier and location:

| Metric | This Person | Benchmark | Status |
|--------|-------------|-----------|--------|
| Housing as % of income | 33% | 25-30% | `"above_benchmark"` |
| Food (total) as % of income | 21% | 15-20% | `"above_benchmark"` |
| Transport as % of income | 7% | 10-15% | `"below_benchmark"` |
| Savings rate | 20% | 20% | `"at_benchmark"` |

Benchmarks should be adjusted based on `cost_of_living_tier` from user profile.

### 5. Construct insights

Each insight follows this structure:

```json
{
  "pattern": "Weekend Spending Spike",
  "type": "temporal",
  "description": "45% of discretionary spending occurs Friday through Sunday, averaging 1,800 TRY per weekend vs 900 TRY Mon-Thu combined.",
  "impact_monthly": 2500,
  "impact_annual": 30000,
  "severity": "medium",
  "suggestion": "Set a weekend cash budget of 1,200 TRY. Use a prepaid card or separate account for weekend spending to create a natural friction point.",
  "data_points": {
    "weekend_discretionary": 3600,
    "weekday_discretionary": 4400,
    "weekend_share": 0.45,
    "weekend_days": 8,
    "weekday_days": 22
  }
}
```

Insight types: `"temporal"`, `"category"`, `"behavioral"`, `"benchmark"`, `"trend"`.

**Important**: Be factual, not judgmental. "45% of discretionary spending occurs on weekends" is an observation. "You spend too much on weekends" is a judgment. Stick to observations and let the impact numbers speak.

### 6. Append Phase 5 entry to `log.json`

```json
{
  "name": "Phase 5: Behavioral Insights",
  "completed_at": "2026-04-25T10:25:00Z",
  "insights_count": 5,
  "insights": [
    {
      "pattern": "Weekend Spending Spike",
      "type": "temporal",
      "impact_monthly": 2500,
      "severity": "medium"
    },
    {
      "pattern": "High Dining-to-Grocery Ratio",
      "type": "category",
      "impact_monthly": 800,
      "severity": "low"
    },
    {
      "pattern": "Subscription Accumulation",
      "type": "category",
      "impact_monthly": 500,
      "severity": "low"
    }
  ],
  "benchmarks": {
    "housing_pct": {"value": 0.33, "benchmark": 0.28, "status": "above_benchmark"},
    "food_pct": {"value": 0.21, "benchmark": 0.18, "status": "above_benchmark"},
    "savings_rate": {"value": 0.20, "benchmark": 0.20, "status": "at_benchmark"}
  },
  "trend_available": false
}
```

## Output
- `{analysis_path}/log.json` — updated with Phase 5 behavioral insights entry
- No other files created or modified
