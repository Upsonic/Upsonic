# Progress Skill

## Purpose
Maintain a **machine-readable** progress file so dashboards, CLIs, and notebooks can poll the analysis state at any time. The file is a JSON document — never markdown, never human-prose-first.

## When to Use
**Constantly.** This skill is not a phase — it runs alongside every phase. You must overwrite `progress.json` at these moments:

1. **Phase start** — when you begin a new phase
2. **Phase end** — when you complete a phase
3. **Before long operations** — before parsing a large CSV, computing projections, etc.
4. **On failure** — immediately when something goes wrong
5. **On completion** — when the full analysis finishes

## File Location
```
analyses/{analysis_name}/progress.json
```

## Format (CANONICAL — emit exactly)

The file is **overwritten** each time (not appended). It is always the full current snapshot. Use UTC ISO-8601 timestamps. Match this schema **byte-for-byte**.

```json
{
  "name": "{analysis_name}",
  "status": "RUNNING",
  "started_at": "2026-04-25T10:00:00Z",
  "updated_at": "2026-04-25T10:25:00Z",
  "phases": [
    {"index": 0, "name": "Data Ingestion",       "status": "done",    "summary": "Parsed 87 transactions from CSV."},
    {"index": 1, "name": "Profile Analysis",      "status": "done",    "summary": "Software engineer, 15K TRY net, Istanbul."},
    {"index": 2, "name": "Balance Sheet",          "status": "current", "summary": null},
    {"index": 3, "name": "Health Scoring",         "status": "pending", "summary": null},
    {"index": 4, "name": "Risk Analysis",          "status": "pending", "summary": null},
    {"index": 5, "name": "Behavioral Insights",    "status": "pending", "summary": null},
    {"index": 6, "name": "Action Plan",            "status": "pending", "summary": null},
    {"index": 7, "name": "Projections & What-If",  "status": "pending", "summary": null}
  ],
  "current_activity": "Computing expense breakdown by category.",
  "issues": []
}
```

### Field rules (strict)

- **`status`** is one of: `"RUNNING"`, `"COMPLETED"`, `"FAILED"`. Uppercase. Nothing else.
- **`phases`** is a **JSON array**, never an object. Exactly eight elements, in order: Data Ingestion, Profile Analysis, Balance Sheet, Health Scoring, Risk Analysis, Behavioral Insights, Action Plan, Projections & What-If. Use those exact `name` values.
- **`phases[].status`** is one of: `"done"`, `"current"`, `"pending"`, `"failed"`. Lowercase.
- **`phases[].index`** is a 0-based integer matching the position in the array.
- Exactly one phase may have `status == "current"` while the top-level `status == "RUNNING"`.
- **`phases[].summary`** is one short sentence, or `null` if the phase has not run yet.
- **`current_activity`** is one or two sentences describing what is happening **right now**.
- **`issues`** is an array of short strings; use `[]` when clean, never `null`.

## Rules

1. **Overwrite, don't append.** The file is a snapshot, not a log. `log.json` is the log.
2. **Valid JSON only.** Never write partial/invalid JSON.
3. **Update before, not after.** Update progress BEFORE starting a long operation.
4. **Be honest about failures.** On error, immediately set `status = "FAILED"`, mark the current phase `"failed"`, and append a message to `issues`.
5. **Always refresh `updated_at`.**

## Lifecycle

| Moment | Action |
|--------|--------|
| Phase 0 starts | Create `progress.json`, `status="RUNNING"`, all phases `pending`, Phase 0 → `current` |
| Phase N starts | Previous phase → `done` with summary; Phase N → `current`; refresh `current_activity` + `updated_at` |
| Long operation | Update `current_activity` + `updated_at` |
| Phase N ends | Mark Phase N → `done` with summary |
| Analysis completes | All phases `done`, `status="COMPLETED"`, `current_activity="Done. See result.json."` |
| Analysis fails | `status="FAILED"`, current phase → `"failed"`, `issues` populated |
