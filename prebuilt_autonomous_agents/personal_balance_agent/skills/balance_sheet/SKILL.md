# Balance Sheet Skill

## Purpose
Build a structured personal balance sheet from normalized transaction data and user profile. This is the financial snapshot — income vs expenses, assets vs liabilities, net worth.

## When to Use
Phase 2 — after profile analysis is complete.

## Input
| Parameter | Type | Description |
|-----------|------|-------------|
| analysis_path | path | `analyses/{analysis_name}/` |

## Actions

### 1. Compute income breakdown

Read `normalized_data.json` and aggregate income by category:

```json
{
  "total": 15000,
  "breakdown": [
    {"category": "salary", "amount": 15000, "percentage": 100.0, "frequency": "monthly", "source": "TechCorp"},
  ],
  "regularity": "stable",
  "sources_count": 1
}
```

For multiple income sources, calculate each source's share and stability.

### 2. Compute expense breakdown

Aggregate expenses by category, sorted by amount descending:

```json
{
  "total": 12000,
  "breakdown": [
    {"category": "housing", "amount": 5000, "percentage": 41.7, "type": "fixed"},
    {"category": "food_groceries", "amount": 2000, "percentage": 16.7, "type": "variable"},
    {"category": "food_dining", "amount": 1200, "percentage": 10.0, "type": "discretionary"},
    {"category": "transport", "amount": 1000, "percentage": 8.3, "type": "semi_fixed"},
    {"category": "utilities", "amount": 800, "percentage": 6.7, "type": "fixed"},
    {"category": "subscriptions", "amount": 500, "percentage": 4.2, "type": "fixed"},
    {"category": "entertainment", "amount": 500, "percentage": 4.2, "type": "discretionary"},
    {"category": "debt_payment", "amount": 1500, "percentage": 12.5, "type": "fixed"},
    {"category": "miscellaneous", "amount": 500, "percentage": 4.2, "type": "variable"}
  ],
  "fixed_expenses": 7800,
  "variable_expenses": 2500,
  "discretionary_expenses": 1700,
  "fixed_ratio": 0.65
}
```

Classify each expense as:
- `"fixed"`: same amount every month (rent, loan payment, subscriptions)
- `"semi_fixed"`: roughly consistent but varies slightly (utilities, transport)
- `"variable"`: changes significantly (groceries, healthcare)
- `"discretionary"`: non-essential, fully flexible (dining, entertainment)

### 3. Calculate net cashflow

```
net_cashflow = total_income - total_expenses
savings_rate = net_cashflow / total_income
```

### 4. Map assets and liabilities

From `user_profile.json`, organize:

**Assets:**
```json
{
  "total": 50000,
  "liquid": 25000,
  "invested": 15000,
  "illiquid": 10000,
  "breakdown": [
    {"type": "savings_account", "value": 25000, "liquidity": "liquid", "currency": "TRY"},
    {"type": "stock_portfolio", "value": 15000, "liquidity": "semi_liquid", "currency": "USD"},
    {"type": "retirement_fund", "value": 10000, "liquidity": "illiquid", "currency": "TRY"}
  ]
}
```

**Liabilities:**
```json
{
  "total": 25000,
  "breakdown": [
    {"type": "credit_card", "balance": 5000, "interest_rate": 4.25, "monthly_payment": 1500, "remaining_months": 4},
    {"type": "student_loan", "balance": 20000, "interest_rate": 1.5, "monthly_payment": 1000, "remaining_months": 22}
  ],
  "total_monthly_payments": 2500,
  "highest_interest_first": ["credit_card", "student_loan"]
}
```

### 5. Calculate net worth

```
net_worth = total_assets - total_liabilities
```

### 6. Append Phase 2 entry to `log.json`

```json
{
  "name": "Phase 2: Balance Sheet",
  "completed_at": "2026-04-25T10:10:00Z",
  "income": {"total": 15000, "sources": 1},
  "expenses": {"total": 12000, "categories": 9, "fixed_ratio": 0.65},
  "net_cashflow": 3000,
  "savings_rate": 0.20,
  "assets": {"total": 50000, "liquid": 25000},
  "liabilities": {"total": 25000},
  "net_worth": 25000,
  "debt_to_income": 0.167,
  "notes": "High fixed expense ratio (65%) limits flexibility."
}
```

## Output
- `{analysis_path}/log.json` — updated with Phase 2 balance sheet entry
- Balance sheet data is stored in the log and will be compiled into `result.json` at Phase 7
