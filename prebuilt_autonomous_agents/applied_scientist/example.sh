claude \
  --system-prompt-file "./system_prompt.md" \
  --dangerously-skip-permissions \
  --effort "medium" \
  "New experiment.
**Research paper:** example_1/tabpfn.pdf
**Current notebook:** example_1/Baseline XGBoost Adult.ipynb
**Current data:** downloaded in notebook (ucimlrepo, id=2)

Run the full experiment pipeline. Go from Phase 0 through Phase 5 without stopping. I want to see \`result.md\` at the end telling me whether this new method is better than what we have.

Start now."