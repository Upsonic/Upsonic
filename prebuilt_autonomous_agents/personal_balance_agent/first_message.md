New financial analysis.

**Analysis name:** {analysis_name}
**Financial data:** {financial_data}
**User context:** {user_context}
**Time period:** {time_period}
**Analyses directory:** {analyses_directory}
**Goals:** {goals}

The financial data can be one or more file paths (PDF, Excel, CSV, JSON — any format) or pasted text. Detect the format of each file, parse it, and normalize into categorized transactions. When multiple files are provided, merge and deduplicate — e.g., a credit card payment in a bank statement is the same money as the individual credit card transactions, don't double count.

The user context can be a markdown file path, inline text, or structured JSON describing the person's financial situation. Parse and normalize it into a user profile.

Use `{analysis_name}` **exactly as given** for the analysis folder (`{analyses_directory}/{analysis_name}/`) and for the `"name"` field in every JSON file — do not rename it, do not add suffixes.

Run the full analysis pipeline. Go from Phase 0 through Phase 7 without stopping. All bookkeeping is JSON — update `progress.json` continuously and, at the end, write `result.json` with the complete financial health assessment including: balance sheet, health score, risk analysis, behavioral insights, action plan, and projections.

Start now.
