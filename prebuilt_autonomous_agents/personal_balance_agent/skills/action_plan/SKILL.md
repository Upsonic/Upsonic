# Action Plan Skill

## Purpose
Create a prioritized, concrete, and measurable action plan based on the person's health score, risks, behavioral insights, and goals. Every action must be specific enough to execute this week and measurable enough to verify next month.

## When to Use
Phase 6 — after behavioral insights are complete.

## Input
| Parameter | Type | Description |
|-----------|------|-------------|
| analysis_path | path | `analyses/{analysis_name}/` |

## Actions

### 1. Gather inputs for prioritization

Read from `log.json`:
- Phase 3: health score components — which are weakest?
- Phase 4: risks — which are most severe?
- Phase 5: behavioral insights — which have the highest monthly impact?
- User goals from `user_profile.json`

### 2. Generate actions in four time horizons

#### Immediate (This Week)
Quick wins that require no financial sacrifice, just action:
```json
{
  "action": "Cancel unused Spotify Family plan (only 1 user active)",
  "category": "expense_reduction",
  "effort": "low",
  "priority": 1,
  "expected_impact": {"monthly": 50, "annual": 600},
  "addresses": ["Subscription Accumulation insight", "savings_rate component"],
  "verification": "Check next month's statement for absence of charge"
}
```

Examples: cancel unused subscriptions, set up automatic savings transfer, review and dispute incorrect charges, update insurance beneficiaries.

#### Short-term (1-3 months)
Structural changes that take some effort:
```json
{
  "action": "Set up automatic transfer of 2000 TRY to emergency savings on salary day",
  "category": "savings_building",
  "effort": "low",
  "priority": 1,
  "expected_impact": {"monthly": 2000, "annual": 24000},
  "addresses": ["Low Emergency Fund risk", "emergency_fund component"],
  "milestone": "Emergency fund reaches 3 months by July 2026",
  "verification": "Emergency savings balance >= 31000 TRY by end of July"
}
```

Examples: build emergency fund, refinance high-interest debt, create a weekend budget system, negotiate bills.

#### Medium-term (3-12 months)
Strategic financial moves:
```json
{
  "action": "Open a diversified investment account and start contributing 1500 TRY/month",
  "category": "investment",
  "effort": "medium",
  "priority": 2,
  "expected_impact": {"monthly": 1500, "annual": 18000},
  "addresses": ["investment_rate component", "No Investment Growth risk"],
  "milestone": "Investment portfolio reaches 18000 TRY by March 2027",
  "verification": "Monthly investment statement shows consistent contributions"
}
```

Examples: reach emergency fund target, start investing, pay off high-interest debt, build a side income stream, get life insurance.

#### Long-term (1-5 years)
Major financial milestones:
```json
{
  "action": "Accumulate 400,000 TRY for apartment down payment (20% of target)",
  "category": "goal_achievement",
  "effort": "high",
  "priority": 2,
  "expected_impact": {"monthly": 5000, "annual": 60000},
  "addresses": ["Buy apartment goal"],
  "milestone": "Down payment fund reaches 400,000 TRY by 2029",
  "verification": "Dedicated savings/investment account balance",
  "prerequisite": "Emergency fund must be at 6-month target first"
}
```

### 3. Prioritize actions

Priority scoring (1 = highest priority):
- **Impact ÷ Effort**: high impact + low effort = priority 1
- **Risk severity**: actions that mitigate `"high"` or `"critical"` risks get +1 priority
- **Goal alignment**: actions that directly serve stated user goals get +1 priority
- **Dependencies**: if action B requires action A, A gets higher priority

Sort within each time horizon by priority, then by expected monthly impact descending.

### 4. Calculate total potential impact

Sum up all expected impacts:
```json
{
  "total_potential_monthly_savings": 3550,
  "total_potential_annual_savings": 42600,
  "projected_savings_rate_after_plan": 0.37,
  "projected_health_score_after_plan": 85
}
```

### 5. Append Phase 6 entry to `log.json`

```json
{
  "name": "Phase 6: Action Plan",
  "completed_at": "2026-04-25T10:30:00Z",
  "actions_count": {"immediate": 3, "short_term": 4, "medium_term": 3, "long_term": 2},
  "total_actions": 12,
  "total_potential_monthly_savings": 3550,
  "total_potential_annual_savings": 42600,
  "projected_health_score_after_plan": 85,
  "top_3_actions": [
    "Set up 2000 TRY auto-transfer to emergency savings",
    "Cancel 3 unused subscriptions (150 TRY/month)",
    "Create weekend spending budget (cap at 1200 TRY)"
  ]
}
```

## Action Categories

Use these standard categories:
- `"expense_reduction"`: cutting or reducing an expense
- `"savings_building"`: increasing savings rate or building reserves
- `"debt_management"`: paying off, refinancing, or restructuring debt
- `"investment"`: starting or increasing investment contributions
- `"income_growth"`: actions to increase income
- `"insurance_protection"`: getting or improving insurance coverage
- `"goal_achievement"`: direct progress toward a stated goal
- `"financial_hygiene"`: organizational/administrative improvements

## Output
- `{analysis_path}/log.json` — updated with Phase 6 action plan entry
- No other files created or modified
