# Analyze Current Skill

## Purpose
Read and understand the current baseline implementation. Extract all relevant information about the existing approach without modifying anything, and record the analysis as a structured JSON entry.

## When to Use
Phase 1 — after experiment setup is complete and files are copied to the experiment folder.

## Input
| Parameter | Type | Description |
|-----------|------|-------------|
| experiment_path | path | `experiments/{research_name}/` |

## Actions

1. **Read `{experiment_path}/current.ipynb`** and extract:
   - Model/algorithm used
   - Preprocessing steps (encoding, scaling, feature selection, etc.)
   - Training approach (train/test split ratio, cross-validation, etc.)
   - Hyperparameters
   - Metrics used and their values
   - Target variable and feature set

2. **Extract dependencies:**
   - Scan all import statements in the notebook.
   - Write `{experiment_path}/current_requirements.txt` with one package per line (`package==version` if determinable, otherwise just `package`).

3. **Read `{experiment_path}/current_data/`** (or, for code-based data, the download spec):
   - Identify data format (CSV, parquet, etc.)
   - Note number of rows, columns
   - Note data types and any special handling

4. **Append a Phase 1 entry to `{experiment_path}/log.json`** under `phases`:
   ```json
   {
     "name": "Phase 1: Analyze Current",
     "completed_at": "2026-04-17T10:15:00Z",
     "model": "XGBoost",
     "preprocessing": [
       "Drop rows with NaN",
       "LabelEncoder on target",
       "LabelEncoder on categorical features",
       "StandardScaler on numerical features"
     ],
     "training": {
       "split": 0.2,
       "seed": 42,
       "stratified": true
     },
     "hyperparameters": {
       "n_estimators": 200,
       "max_depth": 6,
       "learning_rate": 0.1
     },
     "metrics": {
       "accuracy": 0.8726,
       "f1":       0.7277,
       "roc_auc":  0.9274
     },
     "target": "income",
     "features_count": 14,
     "data": {
       "source": "ucimlrepo fetch_ucirepo(id=2)",
       "format": "pandas.DataFrame",
       "rows": 45222,
       "cols": 14
     },
     "notes": "Data downloaded programmatically; both notebooks must use the same source."
   }
   ```

   Do not overwrite earlier entries; append to the `phases` array.

## Output
- `{experiment_path}/log.json` — updated with complete Phase 1 analysis entry
- `{experiment_path}/current_requirements.txt` — written
- No other files created or modified
