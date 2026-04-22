# Experiment Management Skill

## Purpose
Set up and manage the experiment folder structure. This is Phase 0 — it runs before any analysis begins. All bookkeeping files are JSON (never markdown).

## When to Use
- At the very start of a new experiment
- When updating `experiments.json` or `comparison.json` after an experiment completes

## Input
| Parameter | Type | Description |
|-----------|------|-------------|
| research_name | string | The experiment name **as given by the caller**. Use it verbatim — do not rename it, do not re-derive it from the source title. |
| research_source | ref | A reference describing the new method. Any of: local file path (PDF, Markdown, HTML, `.ipynb`, …), web URL (blog post, arXiv, docs), git repository URL, Kaggle notebook or dataset page, or another fetchable resource. |
| current_notebook | path | Path to the current baseline .ipynb |
| current_data | path | Path to the current dataset (file or directory) |
| experiments_directory | path | The directory (inside the workspace) where experiment folders live (e.g. `./experiments`). |

## Actions

### Setup (start of experiment)

1. **Create experiment directory:**
   ```
   experiments/{research_name}/
   ```

2. **Copy baseline files (NEVER move, NEVER modify originals):**
   ```bash
   cp {current_notebook} experiments/{research_name}/current.ipynb
   cp -r {current_data}  experiments/{research_name}/current_data/
   ```

   If `current_data` is a code-based download (e.g. `fetch_ucirepo(id=2)`), leave `current_data/` empty and record the download spec in `log.json`'s metadata.

3. **Materialize the research source.** Detect the kind of `{research_source}` and save a local copy inside the experiment folder:

   | Kind | Detection | Command / action | Local path |
   |---|---|---|---|
   | Local PDF | existing path, `.pdf` extension | `cp {research_source} experiments/{research_name}/research.pdf` | `research.pdf` |
   | Other local file | existing path, non-PDF (`.md`, `.html`, `.txt`, `.ipynb`, …) | `cp {research_source} experiments/{research_name}/research_source.{ext}` | `research_source.{ext}` |
   | Git repository | `git@…`, ends in `.git`, or known git host (`github.com`, `gitlab.com`, `bitbucket.org`, …) | `git clone --depth 1 {research_source} experiments/{research_name}/research_source/` | `research_source/` |
   | Kaggle notebook | `https://www.kaggle.com/code/<user>/<slug>` (or legacy `/kernels/…`) | if the Kaggle CLI is installed and authenticated: `kaggle kernels pull <user>/<slug> -p experiments/{research_name}/research_source/`. Otherwise fall back to fetching the page HTML. | `research_source/` |
   | Kaggle dataset | `https://www.kaggle.com/datasets/<user>/<slug>` | if the Kaggle CLI is installed and authenticated: `kaggle datasets download -d <user>/<slug> -p experiments/{research_name}/research_source/ --unzip`. Otherwise save the page HTML. | `research_source/` |
   | Web URL | `http(s)://` and not matched above | fetch the page (e.g. `curl -L -o research_source.html {research_source}`). If the URL clearly points at a PDF (e.g. arXiv `/pdf/...` link), save as `research.pdf`. Optionally also save a cleaned Markdown version as `research_source.md`. | `research_source.html` (or `research.pdf` / `research_source.md`) |
   | Other | fallback | fetch and save a readable local copy; document the choice in `log.json.metadata` | as chosen |

   Let `research_source_local` be whichever local path you produced. Use that path for Phase 2 onwards — never re-fetch.

4. **Create `log.json`** with the starting skeleton:
   ```json
   {
     "name": "{research_name}",
     "metadata": {
       "date": "YYYY-MM-DD",
       "original_notebook":      "{current_notebook}",
       "original_data":          "{current_data}",
       "research_source":        "{research_source_local}",
       "research_source_origin": "{research_source}",
       "research_source_kind":   "pdf | file | git | kaggle_notebook | kaggle_dataset | web | other"
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
