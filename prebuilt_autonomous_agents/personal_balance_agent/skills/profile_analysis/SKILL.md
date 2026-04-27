# Profile Analysis Skill

## Purpose
Build a complete user financial profile by combining the user's stated context with facts discovered in the transaction data. The user only needs to fill in what they know off the top of their head — the agent fills in the rest by reading the actual data.

**This phase writes back to the user context file.** After cross-referencing, the agent updates the original `user_context.md` with discovered and corrected values so the user can review, correct, and reuse it next month.

## When to Use
Phase 1 — after data ingestion is complete and `normalized_data.json` is available.

## Input
| Parameter | Type | Description |
|-----------|------|-------------|
| analysis_path | path | `analyses/{analysis_name}/` |
| user_context | ref | Path to the user context markdown file, or inline text/JSON |

## Actions

### 1. Parse user context

Read `user_context.md` (or whatever format was provided). Extract every field the user filled in. Fields left as `—`, `N/A`, blank, or placeholder text are treated as **unknown — to be discovered from data**.

### 2. Scan transaction data for discoverable facts

Read `normalized_data.json` and extract the following automatically:

#### Income discovery
- Detect salary deposits: recurring large credits on the same day each month → infer `net_monthly_income`, `payment_schedule`, `employer_name`
- Detect side income: irregular credits from different sources → infer `side_income`
- Detect investment income: dividends, interest payments

#### Debt discovery
- Detect loan payments: recurring fixed debits to banks/financial institutions → infer loan type, monthly payment amount, and lender
- Detect credit card payments: lump-sum debits matching credit card patterns → infer credit card payment amounts
- Detect installment payments: repeated payments with similar descriptions → infer installment debt
- Detect interest/fee charges: if visible in credit card statement → infer interest rate

#### Fixed commitment discovery
- Detect rent: large recurring monthly debit (same amount, same recipient) → infer rent amount
- Detect utilities: recurring debits to utility companies (electricity, water, gas, internet, phone) → infer each utility cost
- Detect subscriptions: small recurring debits to known services (Netflix, Spotify, YouTube, gym, etc.) → infer subscription list and costs
- Detect insurance premiums: recurring debits to insurance companies → infer insurance types and costs

#### Asset signals
- Detect savings transfers: recurring transfers to savings/investment accounts → signal existing savings habit
- Detect investment purchases: transactions with brokerages or crypto exchanges → signal investment accounts

### 3. Cross-reference stated vs discovered

For every field, compare what the user stated with what the data shows:

| Field | User Said | Data Shows | Action |
|-------|-----------|------------|--------|
| Net income | 45,000 TRY | Salary deposit: 44,800 TRY | Use data (close enough, note delta) |
| Net income | — (blank) | Salary deposit: 44,800 TRY | **Fill in from data** |
| Rent | 12,000 TRY | Monthly debit: 12,000 TRY | Confirmed ✓ |
| Rent | — (blank) | Monthly debit: 12,000 TRY | **Fill in from data** |
| Gym | — (blank) | Monthly debit: 1,500 TRY to "MACFIT" | **Fill in from data** |
| Credit card debt | 15,000 TRY | Statement balance: 18,200 TRY | Use data (more recent), flag discrepancy |
| Car loan | — (blank) | No loan payments detected | Leave as "None detected" |
| Subscriptions | Netflix 200 TRY | Netflix 200 TRY + Spotify 60 TRY + HBO Max 120 TRY | **Extend with discovered subscriptions** |

Source tracking rules:
- **User stated + data confirms** → keep user's value, mark `"source": "confirmed"`
- **User stated + data contradicts** → use data value, keep user's original as `"user_stated"`, mark `"source": "data_corrected"`
- **User left blank + data reveals** → fill in from data, mark `"source": "discovered"`
- **User stated + no data to verify** → keep user's value, mark `"source": "user_only"`
- **User left blank + no data found** → leave as null, mark `"source": "unknown"`

### 4. Write enriched `user_profile.json`

Save the complete, enriched profile to `{analysis_path}/user_profile.json`. Every field includes a `"source"` tag so downstream phases (and the user) know what was stated vs discovered vs corrected:

```json
{
  "profession": {"value": "Software Engineer", "source": "user_only"},
  "employment_type": {"value": "salaried", "source": "confirmed", "evidence": "Regular monthly salary deposit detected"},
  "net_monthly_income": {"value": 44800, "source": "data_corrected", "user_stated": 45000, "evidence": "Salary deposit of 44,800 TRY on 1st of month"},
  "currency": "TRY",
  "age_range": {"value": "25-30", "source": "user_only"},
  "dependents": {"value": 0, "source": "user_only"},
  "location": {"value": "Istanbul", "source": "user_only"},
  "cost_of_living_tier": "high",
  "existing_debts": [
    {
      "type": "credit_card",
      "balance": {"value": 18200, "source": "data_corrected", "user_stated": 15000},
      "monthly_payment": {"value": 3500, "source": "discovered", "evidence": "Credit card payment of 3,500 TRY on 15th"},
      "interest_rate": {"value": 4.25, "source": "user_only"},
      "lender": {"value": "Yapı Kredi", "source": "discovered"}
    }
  ],
  "fixed_commitments": [
    {"name": "Rent", "amount": {"value": 12000, "source": "confirmed"}, "frequency": "monthly"},
    {"name": "Internet", "amount": {"value": 450, "source": "discovered", "evidence": "Turk Telekom debit"}, "frequency": "monthly"},
    {"name": "Phone", "amount": {"value": 150, "source": "discovered", "evidence": "Vodafone debit"}, "frequency": "monthly"},
    {"name": "Gym", "amount": {"value": 1500, "source": "discovered", "evidence": "MACFIT monthly"}, "frequency": "monthly"},
    {"name": "Netflix", "amount": {"value": 200, "source": "confirmed"}, "frequency": "monthly"},
    {"name": "Spotify", "amount": {"value": 60, "source": "confirmed"}, "frequency": "monthly"},
    {"name": "YouTube Premium", "amount": {"value": 80, "source": "confirmed"}, "frequency": "monthly"},
    {"name": "HBO Max", "amount": {"value": 120, "source": "discovered"}, "frequency": "monthly"}
  ],
  "insurance": {
    "health": {"value": true, "source": "user_only"},
    "life": {"value": false, "source": "user_only"},
    "property": {"value": false, "source": "user_only"},
    "private_health": {"value": true, "source": "discovered", "evidence": "Monthly premium to Anadolu Sigorta: 800 TRY"}
  },
  "goals": [...],
  "risk_tolerance": "moderate",
  "income_stability": "stable",
  "data_confidence": {
    "stated_vs_observed_income_match": false,
    "income_delta": -200,
    "corrections": ["Credit card balance: 15,000 → 18,200 based on statement"],
    "discoveries": ["HBO Max subscription (120 TRY/month)", "Gym: MACFIT", "Insurance premium: 800 TRY/month", "Internet: 450 TRY", "Phone: 150 TRY"],
    "unverifiable": ["Age range", "Dependents", "Risk tolerance"]
  }
}
```

### 5. Write back updated `user_context.md`

**This is a critical step.** Update the original `user_context.md` file in its source folder with discovered and corrected values. The user gets a complete, accurate profile ready for next month.

#### Write-back rules:
- **Discovered values** (was blank/`—`): fill in the value, add `<!-- discovered from transactions -->` inline
- **Corrected values** (user was wrong): update the value, add `<!-- updated: was X, data shows Y -->` inline
- **New items** (not in template at all): add new rows to the relevant table/section, mark with `<!-- discovered -->`
- **Confirmed values**: leave exactly as-is, no annotation needed
- **Preserve the user's structure**: don't reorganize or reformat the markdown — keep their layout, just fill gaps and fix inaccuracies

#### Example of a written-back section:

Before (user filled):
```markdown
## Monthly Fixed Commitments

- **Rent:** 12,000 TRY
- **Internet + phone:** 600 TRY
- **Gym membership:** 1,500 TRY
- **Streaming subscriptions:** Netflix (200 TRY), Spotify (60 TRY), YouTube Premium (80 TRY)
```

After (agent enriched):
```markdown
## Monthly Fixed Commitments

- **Rent:** 12,000 TRY
- **Internet:** 450 TRY <!-- discovered from transactions: Turk Telekom -->
- **Phone:** 150 TRY <!-- discovered from transactions: Vodafone -->
- **Gym membership:** 1,500 TRY (MACFIT) <!-- discovered: MACFIT -->
- **Streaming subscriptions:** Netflix (200 TRY), Spotify (60 TRY), YouTube Premium (80 TRY), HBO Max (120 TRY) <!-- discovered: HBO Max -->
- **Insurance premium:** 800 TRY (Anadolu Sigorta) <!-- discovered from transactions -->
```

Before (user left blank):
```markdown
## Existing Debts

| Debt | Outstanding Balance | Monthly Payment | Interest Rate | Remaining Months |
|------|---------------------|-----------------|---------------|------------------|
| Credit card | 15,000 TRY | — | 4.25%/month | — |
| Personal loan | — | — | — | — |
```

After (agent enriched):
```markdown
## Existing Debts

| Debt | Outstanding Balance | Monthly Payment | Interest Rate | Remaining Months |
|------|---------------------|-----------------|---------------|------------------|
| Credit card (Yapı Kredi) | 18,200 TRY <!-- updated: was 15,000 --> | 3,500 TRY <!-- discovered --> | 4.25%/month | ~6 |
| Personal loan | None detected | — | — | — |
| Car loan | None detected | — | — | — |
| Mortgage | None detected | — | — | — |
```

### 6. Determine cost-of-living tier

Based on location, assign a tier:
- `"very_high"`: Zurich, San Francisco, New York, London, Singapore
- `"high"`: Istanbul, Berlin, Dubai, Toronto, Sydney
- `"moderate"`: Ankara, Warsaw, Lisbon, Bangkok
- `"low"`: smaller cities, rural areas

### 7. Append Phase 1 entry to `log.json`

```json
{
  "name": "Phase 1: Profile Analysis",
  "completed_at": "2026-04-25T10:05:00Z",
  "profession": "Software Engineer",
  "employment_type": "salaried",
  "net_monthly_income": 44800,
  "income_sources_detected": 1,
  "income_match": false,
  "income_delta": -200,
  "cost_of_living_tier": "high",
  "dependents": 0,
  "total_existing_debt": 18200,
  "total_existing_assets": 50000,
  "goals_count": 5,
  "profile_completeness": 0.92,
  "fields_discovered": 6,
  "fields_corrected": 2,
  "fields_confirmed": 8,
  "fields_unknown": 3,
  "user_context_updated": true,
  "missing_fields": ["gross_monthly_income"]
}
```

## Output
- `{analysis_path}/user_profile.json` — complete enriched profile with source tracking
- `user_context.md` — **updated in-place** with discovered/corrected values and inline annotations
- `{analysis_path}/log.json` — updated with Phase 1 entry
