# Projection & What-If Skill

## Purpose
Model the person's financial future under different scenarios. Produce baseline, optimistic, and pessimistic projections plus custom what-if simulations. This is the final phase — it compiles everything into `result.json`.

## When to Use
Phase 7 — the final phase, after the action plan is complete.

## Input
| Parameter | Type | Description |
|-----------|------|-------------|
| analysis_path | path | `analyses/{analysis_name}/` |
| analysis_name | string | Name of this analysis |

## Actions

### 1. Build projection models

Use data from all previous phases (read `log.json`, `user_profile.json`, `normalized_data.json`).

#### Baseline Projection (current trajectory, no changes)
Project forward assuming:
- Income stays constant (or grows at stated/estimated rate)
- Expenses stay at current level (adjusted for inflation if in high-inflation currency)
- Debt payments continue as scheduled
- Savings rate remains constant
- No new debts or major expenses

```json
{
  "months_3": {
    "net_worth": 34000,
    "liquid_savings": 31000,
    "emergency_fund_months": 3.2,
    "total_debt": 19500,
    "cumulative_saved": 9000
  },
  "months_6": {
    "net_worth": 43000,
    "liquid_savings": 37000,
    "emergency_fund_months": 4.3,
    "total_debt": 14000,
    "cumulative_saved": 18000
  },
  "months_12": {
    "net_worth": 61000,
    "liquid_savings": 49000,
    "emergency_fund_months": 6.0,
    "total_debt": 3000,
    "cumulative_saved": 36000
  },
  "months_24": {
    "net_worth": 97000,
    "liquid_savings": 85000,
    "emergency_fund_months": 8.5,
    "total_debt": 0,
    "cumulative_saved": 72000
  }
}
```

#### Optimistic Projection (action plan fully followed)
Assumes:
- All immediate and short-term actions executed
- Savings rate increases to projected level from action plan
- Debt paid off faster via recommended strategy
- Investment contributions begin as planned

#### Pessimistic Projection (risks materialize)
Assumes:
- One major risk materializes (use the top risk from Phase 4)
- Income reduction of 20-30% for 3 months (or job loss)
- Emergency expenses equal to 1 month of income
- No behavioral improvements

### 2. Currency / inflation adjustment

For high-inflation currencies (TRY, ARS, etc.):
- Note the current annual inflation rate (from user context or general knowledge)
- Show projections in both nominal and real (inflation-adjusted) terms
- Flag that nominal projections may be misleading

```json
{
  "inflation_note": "Turkey's annual inflation is approximately 40%. Nominal projections are shown, but purchasing power may differ significantly.",
  "nominal": {...},
  "real_terms_estimate": {...}
}
```

### 3. Run what-if simulations

#### Standard simulations (always run these):

**Job loss for 3 months:**
```json
{
  "scenario": "Job loss for 3 months",
  "assumptions": ["Zero income for 3 months", "Fixed expenses continue", "Discretionary spending cut by 50%"],
  "result": {
    "months_until_depleted": 4.2,
    "lowest_balance": -8000,
    "debt_increase": 8000,
    "recovery_months": 8,
    "survivable": false,
    "critical_month": 3
  },
  "recommendation": "Current emergency fund is insufficient to survive 3-month job loss. Priority: build fund to 6 months."
}
```

**Income increase of 20%:**
```json
{
  "scenario": "Income increase of 20%",
  "assumptions": ["Net income rises to 18000 TRY", "Expenses remain constant", "All extra income goes to savings/investment"],
  "result": {
    "new_savings_rate": 0.33,
    "new_health_score": 82,
    "months_to_emergency_target": 4,
    "annual_additional_savings": 36000
  }
}
```

**Major unexpected expense (2x monthly income):**
```json
{
  "scenario": "Major unexpected expense of 30,000 TRY",
  "assumptions": ["One-time expense hits this month", "No insurance coverage"],
  "result": {
    "emergency_fund_after": -5000,
    "debt_required": 5000,
    "recovery_months": 5
  }
}
```

#### Custom simulations (from user goals):
If the user asked specific questions (e.g., "Can I afford a car loan?"), model that scenario:
```json
{
  "scenario": "Car loan: 200,000 TRY at 3.5% monthly for 36 months",
  "assumptions": ["Monthly payment of 7,200 TRY", "Insurance + fuel: 2,000 TRY/month"],
  "result": {
    "new_monthly_expenses": 21200,
    "new_savings_rate": -0.41,
    "new_debt_ratio": 0.61,
    "new_health_score": 28,
    "verdict": "NOT_AFFORDABLE",
    "explanation": "This loan would push the debt-to-income ratio to 61%, eliminate all savings capacity, and require spending more than income."
  },
  "alternative": "A car loan of 100,000 TRY over 48 months (payment ~3,100 TRY) would be more sustainable, keeping DTI at 37%."
}
```

### 4. Write `result.json`

Compile all phases into the final `result.json`. See the system prompt for the complete schema. This file must contain:
- `name`, `period`, `currency`
- `health_score` (from Phase 3)
- `balance_sheet` (from Phase 2)
- `risk_analysis` (from Phase 4)
- `behavioral_insights` (from Phase 5)
- `action_plan` (from Phase 6)
- `projections` (baseline, optimistic, pessimistic, what_if from this phase)
- `data_quality` (completeness, gaps, assumptions)

### 5. Update `analyses/analyses.json`

Set `status` to `"completed"`, fill in `health_score`, `grade`, `net_cashflow`, `risk_level`.

### 6. Append Phase 7 entry to `log.json`

```json
{
  "name": "Phase 7: Projections & What-If",
  "completed_at": "2026-04-25T10:35:00Z",
  "projections_generated": ["baseline", "optimistic", "pessimistic"],
  "what_if_scenarios": 4,
  "key_finding": "Current trajectory reaches 6-month emergency fund in 12 months. With action plan, this accelerates to 7 months.",
  "goal_feasibility": [
    {"goal": "6-month emergency fund by Dec 2026", "feasible": true, "confidence": "high"},
    {"goal": "Buy apartment by 2030", "feasible": "uncertain", "confidence": "low", "note": "Requires significant income growth or lower property target"}
  ],
  "files_written": ["result.json", "analyses.json"]
}
```

### 7. Update `progress.json`

Set `status: "COMPLETED"`, all phases `"done"`, `current_activity: "Done. See result.json."`.

## Output
- `{analysis_path}/result.json` — the complete financial analysis report
- `analyses/analyses.json` — updated with final status
- `{analysis_path}/log.json` — finalized with Phase 7 entry
- `{analysis_path}/progress.json` — set to COMPLETED
