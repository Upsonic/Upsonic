New experiment.

**Experiment name:** {research_name}
**Research paper:** {research_paper}
**Current notebook:** {current_notebook}
**Current data:** {current_data}
**Experiments directory:** {experiments_directory}

Use `{research_name}` **exactly as given** for the experiment folder (`{experiments_directory}/{research_name}/`) and for the `"name"` field in every JSON file — do not rename it, do not add suffixes, do not derive a new one from the paper title.

Run the full experiment pipeline. Go from Phase 0 through Phase 5 without stopping. All bookkeeping is JSON — update `progress.json` continuously and, at the end, write `result.json` with the final verdict, summary, and comparison table. No markdown reports.

Start now.
