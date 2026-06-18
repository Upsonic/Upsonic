claude \
  --system-prompt-file "./system_prompt.md" \
  --dangerously-skip-permissions \
  --effort "medium" \
  "New financial analysis.
**Analysis name:** march_2026_review
**Financial data:** example_1/credit_card_march2026.pdf, example_1/bank_account_march2026.pdf
**User context:** user_context.md
**Time period:** March 2026
**Analyses directory:** ./analyses
**Goals:** Can I afford a car loan of 500K TRY? What if I lose my job for 3 months? Should I pay off credit card first or build emergency fund first? What's a realistic timeline to save for a down payment?

Run the full analysis pipeline. Go from Phase 0 through Phase 7 without stopping. I want to see \`result.json\` at the end with my complete financial health assessment.

Start now."
