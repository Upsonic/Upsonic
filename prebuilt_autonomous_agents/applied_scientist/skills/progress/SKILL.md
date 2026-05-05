# Progress Skill

## Purpose
Maintain a **machine-readable** progress file so dashboards, CLIs, and notebooks can poll the experiment's state at any time. The file is a JSON document — never markdown, never human-prose-first.

## When to Use
**Constantly.** This skill is not a phase — it runs alongside every phase. You must overwrite `progress.json` at these moments:

1. **Phase start** — when you begin a new phase
2. **Phase end** — when you complete a phase
3. **Before long operations** — before training a model, installing dependencies, reading a large PDF
4. **On failure** — immediately when something goes wrong
5. **On completion** — when the full experiment finishes

## File Location
```
experiments/{research_name}/progress.json
```

## Format (CANONICAL — emit exactly)

The file is **overwritten** each time (not appended). It is always the full current snapshot. Use UTC ISO-8601 timestamps. Match this schema **byte-for-byte** — do not invent alternative field names, do not use a dict where a list is specified, do not translate status values to synonyms.

```json
{
  "name": "{research_name}",
  "status": "RUNNING",
  "started_at": "2026-04-17T10:00:00Z",
  "updated_at": "2026-04-17T10:25:00Z",
  "phases": [
    {"index": 0, "name": "Setup",           "status": "done",    "summary": "Copied notebook, data, paper."},
    {"index": 1, "name": "Analyze Current", "status": "done",    "summary": "Baseline is XGBoost, 85.3% accuracy."},
    {"index": 2, "name": "Research",        "status": "current", "summary": null},
    {"index": 3, "name": "Benchmark",       "status": "pending", "summary": null},
    {"index": 4, "name": "Implement",       "status": "pending", "summary": null},
    {"index": 5, "name": "Evaluate",        "status": "pending", "summary": null}
  ],
  "current_activity": "Reading research.pdf — extracting method summary and requirements.",
  "issues": []
}
```

### Field rules (strict)

- **`status`** is one of: `"RUNNING"`, `"COMPLETED"`, `"FAILED"`. Uppercase. Nothing else.
- **`phases`** is a **JSON array**, never an object. Exactly six elements, in order: Setup, Analyze Current, Research, Benchmark, Implement, Evaluate. Use those exact `name` values.
- **`phases[].status`** is one of: `"done"`, `"current"`, `"pending"`, `"failed"`. Lowercase. Do **not** use `"completed"`, `"in_progress"`, `"todo"`, or any other synonym.
- **`phases[].index`** is a 0-based integer matching the position in the array.
- Exactly one phase may have `status == "current"` while the top-level `status == "RUNNING"`. On `COMPLETED` / `FAILED`, no phase should be `"current"`.
- **`phases[].summary`** is one short sentence, or `null` if the phase has not run yet.
- **`current_activity`** is one or two sentences describing what is happening **right now**.
- **`issues`** is an array of short strings; use `[]` when clean, never `null`.
- Do **not** add extra top-level keys (e.g. `current_phase`), and do not use dict-of-phases shapes like `{"phase_0_setup": {...}}`.

## Rules

1. **Overwrite, don't append.** The file is a snapshot, not a log. `log.json` is the log.
2. **Valid JSON only.** Never write partial/invalid JSON. Write to a temp file and rename if needed.
3. **Update before, not after.** Update progress BEFORE starting a long operation. The user wants to know what's happening now, not what already happened.
4. **Be honest about failures.** On error, immediately set `status = "FAILED"`, mark the current phase `"failed"`, and append a message to `issues`.
5. **Always refresh `updated_at`** — a stale timestamp tells the user nothing is moving.

## Lifecycle

| Moment | Action |
|--------|--------|
| Phase 0 starts | Create `progress.json`, `status="RUNNING"`, all phases `pending`, Phase 0 → `current`, set `started_at` + `updated_at` |
| Phase N starts | Previous phase → `done` with one-line `summary`; Phase N → `current`; refresh `current_activity` + `updated_at` |
| Long operation starts | Update `current_activity` (e.g. `"Training model — this may take a few minutes"`) + `updated_at` |
| Phase N ends | Mark Phase N → `done` with one-line `summary` |
| Experiment completes | All phases `done`, `status="COMPLETED"`, `current_activity="Done. See result.json."` |
| Experiment fails | `status="FAILED"`, current phase → `"failed"`, `issues` populated, `current_activity` describes the error |
