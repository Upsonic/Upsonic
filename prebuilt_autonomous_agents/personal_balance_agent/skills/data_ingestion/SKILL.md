# Data Ingestion Skill

## Purpose
Set up the analysis workspace, copy raw data, parse and normalize financial data into a standard format. This is Phase 0 — it runs before any analysis begins. All bookkeeping files are JSON (never markdown).

## When to Use
- At the very start of a new analysis
- When updating `analyses.json` after an analysis completes

## Input
| Parameter | Type | Description |
|-----------|------|-------------|
| analysis_name | string | The analysis name **as given by the caller**. Use it verbatim. |
| financial_data | ref | Raw financial data. Can be: a file path (CSV, JSON, PDF, XLSX), pasted text, a structured JSON object, or a free-text summary. |
| user_context | ref | User's financial context (structured or free-text). |
| time_period | string | The period this data covers. |
| analyses_directory | path | The directory where analysis folders live. |
| goals | string | Optional specific questions or goals. |

## Actions

### 1. Create analysis directory
```
{analyses_directory}/{analysis_name}/
{analyses_directory}/{analysis_name}/raw_data/
```

### 2. Copy / capture raw data (NEVER modify originals)

Detect the format of `financial_data` and bring it into `raw_data/`:

- **File path on disk** (CSV, JSON, XLSX, PDF) → `cp` to `raw_data/`
- **Pasted text** → save verbatim to `raw_data/raw_input.txt`
- **Structured JSON object** → save to `raw_data/raw_input.json`
- **Multiple sources** → copy each into `raw_data/` preserving names

### 3. Parse and normalize financial data

#### Format detection and parsing

The agent accepts any file format. Detect the type by extension and content, then apply the appropriate parser:

**PDF (.pdf)**
1. Extract text with `pdftotext` or `pdfplumber`
2. If text extraction yields little/no text (scanned document) → use OCR via `pytesseract`
3. Detect document type from content: bank statement, credit card statement, invoice, payslip, etc.
4. Parse transaction tables — look for repeating row patterns with date, description, and amount columns
5. Extract summary info if present: period, opening/closing balance, totals

**Excel (.xlsx / .xls)**
1. Read with `openpyxl` or `pandas`
2. Scan all sheets — identify which contain transaction data
3. Detect header row (may not be row 1 — banks often add metadata rows above)
4. Handle merged cells, hidden columns, and formatted numbers
5. Map columns to: date, description, amount, and optionally category/type

**CSV / TSV (.csv / .tsv / .txt)**
1. Auto-detect delimiter (comma, tab, semicolon, pipe)
2. Auto-detect encoding (UTF-8, ISO-8859-9, Windows-1254, etc.)
3. Parse header row, map columns to standard fields
4. Handle quoted fields and multiline descriptions

**JSON (.json)**
1. Parse structure — could be an array of transactions, a nested object, or a summary
2. Map fields to standard format regardless of key naming conventions

**Pasted text / free-form**
1. Detect if structured (tabular) or unstructured (narrative)
2. If tabular → parse rows into transactions
3. If narrative → extract income/expense figures and categories

#### Locale-aware parsing

Detect number and date formats from the data or from `user_context`:

| Format | Number Example | Date Example |
|--------|---------------|--------------|
| US/UK | 1,234.56 | 03/25/2026 or 2026-03-25 |
| TR/EU | 1.234,56 | 25.03.2026 |
| FR | 1 234,56 | 25/03/2026 |

If ambiguous, use the currency/locale from `user_context` as the tiebreaker.

#### Multi-source deduplication

When multiple files are provided (e.g., bank statement + credit card statement, or bank export + manual spreadsheet):

1. Parse each file independently into its own transaction list
2. Merge into a single unified list
3. Apply deduplication rules:
   - **Credit card payments**: if a lump-sum credit card payment appears in the bank statement AND individual credit card transactions are also provided → remove the lump-sum from the bank data (the individual transactions are the real expenses)
   - **Internal transfers**: transfers between the person's own accounts → mark as `"internal_transfer"`, exclude from income/expense totals
   - **Duplicate entries**: same date + same amount + similar description across sources → keep one, flag the duplicate
   - **Income**: salary or other income appearing in multiple sources → count once

Parse whatever was provided and produce `normalized_data.json`:

```json
{
  "period": "March 2026",
  "currency": "TRY",
  "transactions": [
    {
      "date": "2026-03-01",
      "description": "Monthly Salary - TechCorp",
      "amount": 15000.00,
      "type": "income",
      "category": "salary",
      "subcategory": "primary_income",
      "original_description": "TECHCORP MAAS ODEMESI"
    }
  ],
  "summary": {
    "total_income": 15000.00,
    "total_expenses": 12000.00,
    "net_cashflow": 3000.00,
    "transaction_count": 87,
    "date_range": {"start": "2026-03-01", "end": "2026-03-31"},
    "income_sources": 1,
    "expense_categories": 12
  }
}
```

### 4. Auto-categorize transactions

Use these standard categories (extend if needed):

| Category | Examples |
|----------|----------|
| `salary` | Regular employment income |
| `freelance` | Contract/gig income |
| `investment_income` | Dividends, interest, capital gains |
| `other_income` | Gifts, refunds, misc income |
| `housing` | Rent, mortgage, property tax, maintenance |
| `utilities` | Electric, water, gas, internet, phone |
| `food_groceries` | Supermarket, market purchases |
| `food_dining` | Restaurants, cafes, delivery |
| `transport` | Fuel, public transit, parking, rideshare |
| `healthcare` | Insurance, medications, doctor visits |
| `education` | Courses, books, tuition |
| `entertainment` | Movies, games, hobbies, streaming |
| `subscriptions` | Recurring digital services |
| `clothing` | Apparel and accessories |
| `personal_care` | Gym, grooming, beauty |
| `debt_payment` | Loan payments, credit card payments |
| `savings_transfer` | Transfers to savings/investment accounts |
| `gifts_donations` | Gifts, charity |
| `insurance` | Life, auto, health insurance premiums |
| `miscellaneous` | Uncategorized |

For ambiguous transactions, assign best-guess category and flag `"confidence": "low"`.

### 5. Create `log.json`
```json
{
  "name": "{analysis_name}",
  "metadata": {
    "date": "YYYY-MM-DD",
    "period": "{time_period}",
    "currency": "TRY",
    "data_source_type": "csv | json | pdf | text | structured | mixed",
    "original_data": "{financial_data}",
    "user_goals": "{goals}"
  },
  "phases": []
}
```

### 6. Register in `analyses/analyses.json`
- If the file does not exist, create it with `{"analyses": []}`.
- Append a new entry with `status: "in_progress"`.

### 7. Create initial `progress.json`
(See `skills/progress/SKILL.md` for schema)
- `status: "RUNNING"`, all phases `pending`, Phase 0 `current`
- `started_at` and `updated_at` set to now (UTC ISO-8601)

### 8. Append Phase 0 entry to `log.json`
```json
{
  "name": "Phase 0: Data Ingestion",
  "completed_at": "2026-04-25T10:00:00Z",
  "data_source_type": "csv",
  "transactions_parsed": 87,
  "income_transactions": 2,
  "expense_transactions": 85,
  "date_range": {"start": "2026-03-01", "end": "2026-03-31"},
  "categories_detected": 12,
  "low_confidence_categorizations": 5,
  "data_quality": {
    "completeness": 0.95,
    "gaps": ["3 transactions missing descriptions"],
    "duplicates_removed": 0
  }
}
```

## Output
- `{analyses_directory}/{analysis_name}/` directory created
- `raw_data/` with copies of original data
- `normalized_data.json` with parsed, categorized transactions
- `log.json` initialized with Phase 0 entry
- `progress.json` initialized
- `analyses.json` updated
