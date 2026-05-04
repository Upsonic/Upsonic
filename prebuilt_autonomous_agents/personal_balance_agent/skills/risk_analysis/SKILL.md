# Risk Analysis Skill

## Purpose
Identify, quantify, and prioritize financial risks specific to this person's situation. Each risk includes severity, concrete impact, and a mitigation strategy.

## When to Use
Phase 4 — after the health score is computed and all financial data is analyzed.

## Input
| Parameter | Type | Description |
|-----------|------|-------------|
| analysis_path | path | `analyses/{analysis_name}/` |

## Actions

### 1. Evaluate risk categories

Analyze each risk category using data from `normalized_data.json`, `user_profile.json`, and the Phase 2/3 log entries.

#### Income Risk
- **Single income source**: if only one salary source → severity `"high"` for single earners, `"medium"` if there's a working partner
- **Freelance/variable income**: if `employment_type` is `"freelance"` or `"mixed"`, measure income coefficient of variation
- **Industry concentration**: if profession is in a volatile industry, note the risk
- **No passive income**: 100% active income = higher risk

#### Expense Risk
- **High fixed cost ratio**: if `fixed_ratio > 0.70` → limited flexibility in crisis
- **Lifestyle inflation signals**: if discretionary spending > 20% of income
- **No expense buffer**: if `net_cashflow / total_income < 0.10`
- **Subscription creep**: total subscriptions as % of income

#### Debt Risk
- **High-interest debt**: any debt with rate > central bank rate + 10% → severity `"high"`
- **Debt-to-income ratio**: if DTI > 0.36 → `"high"`, 0.20-0.36 → `"medium"`, < 0.20 → `"low"`
- **Only minimum payments**: if debt payment barely covers interest
- **Multiple debt sources**: complexity risk

#### Emergency Fund Risk
- **Months of coverage**: < 1 month → `"critical"`, 1-3 → `"high"`, 3-6 → `"medium"`, 6+ → `"low"`
- **Dependents without fund**: if dependents > 0 and emergency fund < 6 months → escalate severity
- **No health insurance + low fund**: compound risk

#### Investment / Growth Risk
- **No investments**: 0% investment rate → missing long-term growth
- **Concentration risk**: all investments in one asset class
- **Currency risk**: if assets/income are in a high-inflation currency with no hedge
- **No retirement savings**: if age > 30 and no retirement contributions

#### Life Event Risk
- Run standard scenarios and assess impact:
  - **Job loss (3 months)**: can the person survive without income for 3 months?
  - **Medical emergency**: does insurance cover it? what's the out-of-pocket max?
  - **Major repair/expense**: unexpected 2x monthly income expense
  - **Interest rate change**: impact on variable-rate debts

### 2. Score and prioritize risks

Each risk gets:
```json
{
  "name": "Low Emergency Fund",
  "category": "emergency_fund",
  "severity": "high",
  "probability": "medium",
  "risk_score": 8,
  "description": "Only 2.1 months of expenses covered by liquid savings.",
  "impact": "A job loss or medical emergency would force taking on high-interest debt within 2 months.",
  "impact_amount": 35000,
  "mitigation": "Redirect 2000 TRY/month from discretionary spending to a dedicated emergency savings account. Reach 4-month coverage in 5 months.",
  "mitigation_effort": "medium",
  "mitigation_timeline": "5 months"
}
```

**Risk score** = severity_value × probability_value (each 1-5):
- Severity: critical=5, high=4, medium=3, low=2, minimal=1
- Probability: very_likely=5, likely=4, possible=3, unlikely=2, rare=1
- Risk score range: 1-25. Display as 1-10 by dividing by 2.5 and rounding.

### 3. Determine overall risk level

Based on the highest-severity active risk:
- Any `"critical"` risk → overall = `"critical"`
- Any `"high"` risk → overall = `"high"`
- Multiple `"medium"` risks → overall = `"moderate"`
- Only `"low"` risks → overall = `"low"`

### 4. Append Phase 4 entry to `log.json`

```json
{
  "name": "Phase 4: Risk Analysis",
  "completed_at": "2026-04-25T10:20:00Z",
  "overall_risk": "moderate",
  "risk_count": {"critical": 0, "high": 1, "medium": 2, "low": 1},
  "top_risk": "Low Emergency Fund",
  "risks": [
    {
      "name": "Low Emergency Fund",
      "category": "emergency_fund",
      "severity": "high",
      "probability": "medium",
      "risk_score": 6,
      "impact_amount": 35000,
      "mitigation_timeline": "5 months"
    },
    {
      "name": "Single Income Source",
      "category": "income",
      "severity": "medium",
      "probability": "possible",
      "risk_score": 5,
      "impact_amount": null,
      "mitigation_timeline": "6-12 months"
    }
  ]
}
```

## Output
- `{analysis_path}/log.json` — updated with Phase 4 risk analysis entry
- No other files created or modified
