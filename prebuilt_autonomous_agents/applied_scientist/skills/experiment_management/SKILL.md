# Experiment Management Skill

## Purpose
Set up and manage the experiment folder structure. This is Phase 0 — it runs before any analysis begins. All bookkeeping files are JSON (never markdown).

## When to Use
- At the very start of a new experiment
- When updating `experiments.json` or `comparison.json` after an experiment completes

## Input
| Parameter | Type | Description |
|-----------|------|-------------|
| research_name | string | The experiment name **as given by the caller**. Use it verbatim — do not rename it, do not re-derive it from the paper title. |
| research_paper | path | Path to the PDF paper |
| current_notebook | path | Path to the current baseline .ipynb |
| current_data | path | Path to the current dataset (file or directory) |
| experiments_directory | path | The directory (inside the workspace) where experiment folders live (e.g. `./experiments`). |

## Actions

### Setup (start of experiment)

1. **Create experiment directory:**
   ```
   experiments/{research_name}/
   ```

2. **Copy files (NEVER move, NEVER modify originals):**
   ```bash
   cp {current_notebook} experiments/{research_name}/current.ipynb
   cp -r {current_data}  experiments/{research_name}/current_data/
   cp {research_paper}   experiments/{research_name}/research.pdf
   ```

   If `current_data` is a code-based download (e.g. `fetch_ucirepo(id=2)`), leave `current_data/` empty and record the download spec in `log.json`'s metadata.

3. **Create `log.json`** with the starting skeleton:
   ```json
   {
     "name": "{research_name}",
     "metadata": {
       "date": "YYYY-MM-DD",
       "original_notebook": "{current_notebook}",
       "original_data":     "{current_data}",
       "research_paper":    "{research_paper}"
     },
     "phases": []
   }
   ```
   Phases append entries here as they finish; never overwrite earlier entries.

4. **Register in `experiments/experiments.json`:**
   - If the file does not exist, create it with `{"experiments": []}`.
   - Append a new entry with `status: "in_progress"`:
     ```json
     {
       "name": "{research_name}",
       "date": "YYYY-MM-DD",
       "status": "in_progress",
       "paper": "{paper_title}",
       "baseline_model": null,
       "new_method": null,
       "verdict": null,
       "key_metric": null,
       "path": "experiments/{research_name}/"
     }
     ```

5. **Create initial `progress.json`** (see `skills/progress/SKILL.md` for schema) with:
   - `status: "RUNNING"`
   - all phases listed, all `pending`, Phase 0 marked `current`
   - `started_at` and `updated_at` set to now (UTC ISO-8601)

### Finalize (end of experiment)

1. Update the experiment entry in `experiments.json` with final `status`, `verdict`, and `key_metric` (see `skills/evaluate/SKILL.md`).
2. Append a row to `experiments/comparison.json`.
   - If the file does not exist, create it with `{"experiments": []}`.

## Output
- `experiments/{research_name}/` directory with copied files
- `experiments/{research_name}/log.json` initialized
- `experiments/{research_name}/progress.json` initialized
- `experiments/experiments.json` updated
