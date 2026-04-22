# Research Skill

## Purpose
Read the materialized research source and extract actionable information needed to implement the proposed method. Record the findings as a structured JSON entry.

## When to Use
Phase 2 — after the current implementation has been analyzed.

## Input
| Parameter | Type | Description |
|-----------|------|-------------|
| experiment_path | path | `experiments/{research_name}/` |

The research source was materialized into `{experiment_path}` during Phase 0. Its local path is recorded in `{experiment_path}/log.json` under `metadata.research_source`, and its kind (`pdf`, `file`, `git`, `kaggle_notebook`, `kaggle_dataset`, `web`, `other`) under `metadata.research_source_kind`. Expected layouts:

- `pdf` → `{experiment_path}/research.pdf`
- `file` → `{experiment_path}/research_source.{ext}` (Markdown, HTML, `.ipynb`, text, …)
- `git` → `{experiment_path}/research_source/` (cloned repository — read `README*`, `docs/`, the top-level code, and any papers/notebooks inside)
- `kaggle_notebook` → `{experiment_path}/research_source/` (pulled Kaggle kernel — read the `.ipynb` and any accompanying metadata; fall back to the saved HTML if the CLI was unavailable)
- `kaggle_dataset` → `{experiment_path}/research_source/` (downloaded Kaggle dataset — read the dataset description / README and skim files to understand the data; fall back to the saved HTML if the CLI was unavailable)
- `web` → `{experiment_path}/research_source.html` (and possibly `research_source.md` or `research.pdf`)
- `other` → whatever Phase 0 saved (check `metadata.research_source`)

## Actions

1. **Read the materialized research source** at `metadata.research_source` (falling back to `research.pdf` for legacy experiments) and extract:

   - **Method Summary:** 2-3 short paragraphs describing what the paper proposes, what problem it solves, and how it differs from traditional approaches.
   - **Pros:** each advantage the paper claims or demonstrates.
   - **Cons:** stated or inferred limitations, assumptions, or weaknesses.
   - **Implementation Requirements:**
     - Required libraries/packages (with versions if specified)
     - Required data format or preprocessing
     - Required compute resources (GPU, memory, etc.)
     - Key hyperparameters to set
   - **Compatibility Analysis:**
     - Can the method use the same data as the current baseline?
     - Does it need different preprocessing?
     - Does it output comparable predictions (same format)?
     - Can the same metrics be used for comparison?

2. **Append a Phase 2 entry to `{experiment_path}/log.json`** under `phases`:
   ```json
   {
     "name": "Phase 2: Research",
     "completed_at": "2026-04-17T10:30:00Z",
     "paper": {
       "title":   "CatBoost: Unbiased Boosting with Categorical Features",
       "authors": ["Prokhorenkova et al."],
       "method_summary": "CatBoost is a gradient-boosting framework that handles categorical features natively via ordered target statistics and uses oblivious decision trees to reduce overfitting."
     },
     "pros": [
       "Native categorical handling — no manual encoding needed",
       "Reduces target leakage with ordered boosting",
       "Strong out-of-the-box performance"
     ],
     "cons": [
       "Training slower than XGBoost for small data",
       "More memory intensive"
     ],
     "requirements": {
       "new_dependencies": ["catboost>=1.2"],
       "data_format": "pandas.DataFrame with categorical columns marked",
       "compute": "CPU is sufficient; GPU optional"
     },
     "compatibility": {
       "same_data":    true,
       "same_metrics": true,
       "preprocessing_notes": "CatBoost takes raw categorical columns; do NOT pre-encode them for the new notebook."
     }
   }
   ```

   Do not overwrite earlier entries; append to the `phases` array.

## Output
- `{experiment_path}/log.json` — updated with Phase 2 research entry
- No other files created or modified
