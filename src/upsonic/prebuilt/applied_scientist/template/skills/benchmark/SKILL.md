# Benchmark Skill

## Purpose
Define the comparison metrics and extract baseline values from the current implementation. Record them as a structured JSON entry so downstream phases and final evaluation can read them directly.

## When to Use
Phase 3 — after both current analysis and research analysis are complete.

## Input
| Parameter | Type | Description |
|-----------|------|-------------|
| experiment_path | path | `experiments/{research_name}/` |

## Actions

1. **Define comparison metrics:**
   - Include ALL metrics already used in `current.ipynb`.
   - Add any additional metrics that are relevant for the new method.
   - For classification: accuracy, precision, recall, F1, AUC-ROC (as applicable).
   - For regression: MSE, RMSE, MAE, R² (as applicable).
   - Include training time if measurable.

2. **Extract baseline values:**
   - Read metric values from `current.ipynb` output cells.
   - If a metric is not computed in the notebook, record it as `null` and set `"needs_computation": true` — both notebooks must then compute it.

3. **Append a Phase 3 entry to `{experiment_path}/log.json`** under `phases`:
   ```json
   {
     "name": "Phase 3: Benchmark",
     "completed_at": "2026-04-17T10:45:00Z",
     "metrics": [
       {
         "name": "accuracy",
         "description": "Fraction of correctly classified samples.",
         "higher_is_better": true,
         "baseline": 0.8726,
         "needs_computation": false
       },
       {
         "name": "f1",
         "description": "F1 score (binary, positive class).",
         "higher_is_better": true,
         "baseline": 0.7277,
         "needs_computation": false
       },
       {
         "name": "roc_auc",
         "description": "Area under the ROC curve.",
         "higher_is_better": true,
         "baseline": 0.9274,
         "needs_computation": false
       },
       {
         "name": "training_time_seconds",
         "description": "Wall-clock training time.",
         "higher_is_better": false,
         "baseline": null,
         "needs_computation": true
       }
     ],
     "notes": "training_time_seconds must be added to both notebooks for a fair comparison."
   }
   ```

   Do not overwrite earlier entries; append to the `phases` array.

## Output
- `{experiment_path}/log.json` — updated with Phase 3 benchmark entry
- Clear list (in `metrics`) of what the new implementation must compute
