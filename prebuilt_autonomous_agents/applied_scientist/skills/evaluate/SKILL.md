# Evaluate Skill

## Purpose
Compare baseline and new implementation results. Produce the machine-readable final report `result.json`, update `experiments.json`, and append a row to `comparison.json`.

## When to Use
Phase 5 — after the new implementation is complete and metrics are collected.

## Input
| Parameter | Type | Description |
|-----------|------|-------------|
| experiment_path | path | `experiments/{research_name}/` |
| research_name | string | Name of this experiment |

## Actions

1. **Collect all metrics** from `log.json` (Phase 3 baseline entry + Phase 4 new method entry).

2. **Determine verdict:**
   - `BETTER`: new method outperforms baseline on the majority of key metrics
   - `WORSE`: new method underperforms baseline on the majority of key metrics
   - `INCONCLUSIVE`: mixed results or differences within noise margin
   - `FAILED`: experiment could not produce comparable results (dependency failure, implementation crash, data incompatibility)

3. **Write `{experiment_path}/result.json`** in the exact schema below. Always valid JSON; never leave fields undefined — use `null` for unknown values.

   ```json
   {
     "name": "{research_name}",
     "verdict": "BETTER",
     "summary": "2-3 paragraphs explaining what the new method does, how it fundamentally differs from the baseline, and what trade-offs it makes.",
     "explanation": "2-3 sentences explaining WHY this verdict was reached. Reference specific metrics and their differences. Be concrete — mention numbers, not vague statements.",
     "comparison": {
       "metrics": [
         {
           "name": "accuracy",
           "current": 0.853,
           "new":     0.872,
           "diff":    0.019,
           "diff_display": "+0.019",
           "unit": null,
           "higher_is_better": true,
           "better": "new"
         },
         {
           "name": "training_time_seconds",
           "current": 2.0,
           "new":     45.0,
           "diff":    43.0,
           "diff_display": "+43.0",
           "unit": "seconds",
           "higher_is_better": false,
           "better": "current"
         }
       ]
     },
     "file_locations": {
       "current_notebook":   "experiments/{research_name}/current.ipynb",
       "current_data":       "experiments/{research_name}/current_data/",
       "new_notebook":       "experiments/{research_name}/new.ipynb",
       "research_paper":     "experiments/{research_name}/research.pdf",
       "experiment_log":     "experiments/{research_name}/log.json"
     }
   }
   ```

   ### Field rules
   - `verdict`: exactly one of `"BETTER"`, `"WORSE"`, `"INCONCLUSIVE"`, `"FAILED"`.
   - `summary` / `explanation`: plain text, no markdown headings. Short paragraphs only.
   - `comparison.metrics[]`:
     - `current` / `new` are numbers (or `null` if a side could not compute the metric).
     - `diff = new - current` (raw number). `diff_display` is the short string with sign (`"+0.019"`, `"-0.03"`).
     - `better`: `"new"` | `"current"` | `"tie"` | `null` — computed from `diff` and `higher_is_better`.
     - `unit` is a short unit string (`"seconds"`, `"%"`, etc.) or `null`.
   - `file_locations` uses paths relative to the experiments directory root.

4. **Update `experiments/experiments.json`:**
   - Set `status` to `"completed"` (or `"failed"` if the experiment failed).
   - Fill in `verdict`, `key_metric`, `baseline_model`, `new_method`.
   - `key_metric` is an object: `{"name": "...", "baseline": <num>, "new": <num>}`.

5. **Update `experiments/comparison.json`:**
   - If the file does not exist, create it with `{"experiments": []}`.
   - Append an entry:
     ```json
     {
       "name": "{research_name}",
       "date": "YYYY-MM-DD",
       "baseline": "{baseline_model}",
       "new_method": "{new_method}",
       "key_metric": {"name": "accuracy", "baseline": 0.853, "new": 0.872},
       "verdict": "BETTER"
     }
     ```

6. **Update `{experiment_path}/log.json`** — append a Phase 5 entry:
   ```json
   {
     "name": "Phase 5: Evaluate",
     "completed_at": "2026-04-17T11:40:00Z",
     "verdict": "BETTER",
     "key_change": "accuracy +0.019 (new > current)",
     "files_written": ["result.json", "experiments.json", "comparison.json"]
   }
   ```

## Output
- `{experiment_path}/result.json` — the final machine-readable report.
- `experiments/experiments.json` — updated with this experiment's final verdict.
- `experiments/comparison.json` — new row appended.
- `log.json` — finalized with Phase 5 entry.
