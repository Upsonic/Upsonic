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
| research_source | ref | A free-form reference describing the new method. The caller can pass anything that identifies the content — a local file path, any URL (blog post, arXiv, docs, Hugging Face page, …), a git repository, a Kaggle link, a paper ID, or **a plain text idea** describing the approach to try. Do not reject unusual values; investigate and fetch whatever was given, or, for pure text ideas, save the text verbatim. |
| current_notebook | path | Path to the current baseline .ipynb |
| current_data | ref \| placeholder | Path to the current dataset (file or directory), a short description of how the notebook loads data, **or** the literal placeholder `"(not provided — infer it from the current notebook's data-loading cells)"`. When you see that placeholder, read the current notebook yourself and infer the source from its data-loading cells; do not ask the user. |
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
   # Only when current_data is a real path on disk:
   cp -r {current_data}  experiments/{research_name}/current_data/
   ```

   Resolve `{current_data}` as follows before copying:

   - **Real local path** (file or directory that exists on disk) → `cp` / `cp -r` it into `current_data/`.
   - **Short description of a code-based download** (e.g. `"downloaded in notebook (ucimlrepo, id=2)"`) → leave `current_data/` empty and record the description verbatim in `log.json.metadata.original_data`.
   - **Placeholder `"(not provided — infer it from the current notebook's data-loading cells)"`** → open `current.ipynb` yourself, scan for data-loading cells (`pd.read_csv`, `fetch_openml`, `fetch_ucirepo`, `load_dataset`, `kaggle.api...`, `urllib`/`requests` downloads, `np.load`, local paths, …), write down the exact loader you found as `log.json.metadata.original_data`, and make sure Phase 4's `new.ipynb` uses the same loader. Do not ask the user for clarification — do the investigation yourself.

3. **Materialize the research source.** `{research_source}` can be anything — a local file, a URL of any kind, a git or Kaggle link, an arXiv / paper ID, a Hugging Face page, **or a plain text idea** describing the method to try. Your job is to bring its content into the experiment folder using whatever tool fits:

   - **Investigate first.** Check if the value is a path on disk (`ls` / `test -e`); if it looks like a URL, poke it with `curl -I`; look at the hostname; read any hint in the value itself. If the value does not look like a path or URL at all, treat it as a **text idea** (see below). Do not rely on a fixed detection list.
   - **Retrievable source** → fetch it with the most appropriate tool: `cp`, `git clone --depth 1`, `curl -L` / `wget`, `kaggle kernels pull …` / `kaggle datasets download …`, `huggingface-cli`, an arXiv PDF helper, Python downloaders, or anything else available. Install a missing CLI with `pip install` / `uv pip install` if it is the right tool for this source.
   - **Text idea** → do not fabricate a paper or URL. Save the description verbatim to `experiments/{research_name}/research_source.md` (optionally with a leading `# Idea` heading) and let Phase 2 turn it into a concrete implementation plan.
   - **Pick a sensible local name** based on what you actually produced:
     - A single PDF → `experiments/{research_name}/research.pdf`
     - Any other single file (including a text idea) → `experiments/{research_name}/research_source.{ext}` (`.md` for ideas; preserve the real extension for files you copied)
     - Multiple files or a cloned repo / dataset → `experiments/{research_name}/research_source/` (a directory)
     - A fetched HTML page → `research_source.html`, optionally with a cleaned `research_source.md` and/or a linked `research.pdf`
   - **Choose your own `research_source_kind` label** to describe what you did (e.g. `pdf`, `file`, `git`, `kaggle_notebook`, `kaggle_dataset`, `arxiv`, `huggingface_model`, `html`, `idea`, `other`). This label is just for observability — there is no closed enum.
   - **If fetching fails**, try an alternative (different CLI, raw HTML fallback, `curl` instead of a specialized tool). Only after genuine failure, log the attempts in `log.json`, mark the experiment `FAILED`, and explain what you tried in `result.json.explanation`. A text idea can never "fail to fetch" — it is always saved verbatim.

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
       "research_source_kind":   "a short label you pick, e.g. pdf, file, git, kaggle_notebook, kaggle_dataset, arxiv, huggingface_model, html, idea, other"
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
