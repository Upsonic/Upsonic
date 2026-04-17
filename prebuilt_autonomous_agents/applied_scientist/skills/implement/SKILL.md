# Implement Skill

## Purpose
Create a new Jupyter notebook implementing the method from the research paper, using the same data as the baseline. Record implementation details and measured metrics as a structured JSON entry.

## When to Use
Phase 4 — after benchmark metrics are defined and baseline values are extracted.

## Input
| Parameter | Type | Description |
|-----------|------|-------------|
| experiment_path | path | `experiments/{research_name}/` |

## Actions

1. **Install dependencies:**
   - Install any new packages identified in Phase 2.
   - Capture installed package names and versions for the log entry below.

2. **Write `{experiment_path}/new_requirements.txt`:**
   - List all packages the new notebook needs (one per line, `package==version`).
   - Include both existing dependencies and new ones from the paper.

3. **Create `{experiment_path}/new.ipynb`** with this structure:

   ```
   [Markdown] # {Research Name} - New Method Implementation
   [Markdown] ## 1. Setup & Imports
   [Code]     import statements + dependency checks

   [Markdown] ## 2. Data Loading
   [Code]     load from experiments/{research_name}/current_data/
              (use the SAME data loading logic as current.ipynb)

   [Markdown] ## 3. Data Preprocessing
   [Code]     preprocessing as required by the new method
              (note any differences from baseline preprocessing)

   [Markdown] ## 4. Model Implementation
   [Code]     implement the new method from the paper

   [Markdown] ## 5. Training
   [Code]     train the model
              (use same train/test split as baseline for fair comparison)

   [Markdown] ## 6. Evaluation
   [Code]     compute ALL comparison metrics defined in Phase 3

   [Markdown] ## 7. Results Summary
   [Code]     print all metrics in a structured format
   ```

4. **Implementation rules:**
   - Use the SAME train/test split (same random seed, same ratio) as the baseline.
   - Use the SAME data — load from `current_data/`, do not download new data.
   - Compute ALL metrics defined in Phase 3 (including any with `"needs_computation": true`).
   - Add timing measurements for training (`training_time_seconds`).
   - Handle errors gracefully — if the method fails, log why.
   - **Efficiency:** if data is large (100K+ rows), sample it to a manageable size (10K–30K rows). Both notebooks must use the exact same sample. Use paper's recommended hyperparameters — do not run exhaustive grid searches. If training takes more than 10 minutes, reduce data size or simplify config. The goal is a fair comparison, not a production model.

5. **Run the notebook** end-to-end and verify it executes without errors.

6. **Append a Phase 4 entry to `{experiment_path}/log.json`** under `phases`:
   ```json
   {
     "name": "Phase 4: Implement",
     "completed_at": "2026-04-17T11:30:00Z",
     "new_dependencies_installed": [
       {"name": "catboost", "version": "1.2.5"}
     ],
     "training": {
       "split": 0.2,
       "seed": 42,
       "stratified": true
     },
     "metrics": {
       "accuracy": 0.8721,
       "f1":       0.7310,
       "roc_auc":  0.9288,
       "training_time_seconds": 45.2
     },
     "notebook_executed": true,
     "errors":   [],
     "warnings": []
   }
   ```

   Do not overwrite earlier entries; append to the `phases` array.

## Output
- `{experiment_path}/new.ipynb` — complete, executed notebook
- `{experiment_path}/new_requirements.txt` — written
- `{experiment_path}/log.json` — updated with Phase 4 implementation entry
