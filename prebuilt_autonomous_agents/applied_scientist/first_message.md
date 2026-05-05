New experiment.

**Experiment name:** {research_name}
**Research source:** {research_source}
**Current notebook:** {current_notebook}
**Current data:** {current_data}
**Experiments directory:** {experiments_directory}

The research source describes the new method to evaluate. It may be a local file (PDF, Markdown, HTML, notebook), a web URL (blog post, arXiv link, documentation page), a git repository URL, or any other reference you can fetch and read. Detect the type at Phase 0 and materialize it inside the experiment folder before reading it (see `skills/experiment_management/SKILL.md`).

Use `{research_name}` **exactly as given** for the experiment folder (`{experiments_directory}/{research_name}/`) and for the `"name"` field in every JSON file — do not rename it, do not add suffixes, do not derive a new one from the source title.

Run the full experiment pipeline. Go from Phase 0 through Phase 5 without stopping. All bookkeeping is JSON — update `progress.json` continuously and, at the end, write `result.json` with the final verdict, summary, and comparison table. No markdown reports.

Start now.
