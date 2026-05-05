# Serena MCP — When and How to Use

Always-on. Consult before any non-trivial code work in this repo, including inside `/two` (Planner pre-flight, Critic claim-verification).

## When to Use

Always use Serena for:

- **Symbol lookups.** Finding a function, class, method, or variable by name.
- **Reference tracing.** *"What calls X? Where is Y used? What does Z extend?"*
- **Subsystem orientation.** Getting an overview of a module's symbols before reading file-by-file.

Tools:

- `mcp__serena__find_symbol` — locate a symbol by name, optionally scoped.
- `mcp__serena__find_referencing_symbols` — find everything that references a given symbol.
- `mcp__serena__get_symbols_overview` — list the top-level symbols in a file or directory.

## When NOT to Use

- Reading a single known file → use `Read`.
- Searching for a literal string (not a symbol) → use `Grep` / `Glob`.
- Any modification — Serena is **read-only in this repo** (saved feedback). For edits use the standard Edit / Write tools.

## Hard Rules

1. **Prefer Serena over `grep` for symbol queries.** A symbol query is more precise and won't false-match strings inside docs, comments, or unrelated identifiers.
2. **Read-only.** Serena does not write. Never attempt to mutate via Serena tools — saved feedback memory.
3. **Surface what you found.** When a Serena consultation produces something relevant, briefly note it in your reply, e.g. *"From Serena: existing X handler at src/upsonic/Y.py:42."* The consultation must be visible to the user.
4. **Trivial-work exception, but say so.** For single-line typo or comment edits, skipping Serena is fine — but state it explicitly: *"Skipping Serena lookup — single-line cosmetic edit."* Silent skipping is not allowed.

## Anti-Patterns

- Using `grep` for `find_symbol`'s job — slower and yields false positives across docs, strings, and unrelated identifiers.
- Attempting to mutate source code with Serena tools — read-only is a hard constraint, enforced by saved feedback.
- Quietly using Serena and not surfacing the finding — defeats the visibility intent and makes it look like the consultation never happened.

## Serena Memory (Optional, Currently Unused)

Beyond the code-lookup tools above, Serena MCP exposes a **memory system** distinct from Claude Code auto memory.

Tools: `mcp__serena__write_memory`, `read_memory`, `list_memories`, `edit_memory`, `rename_memory`, `delete_memory`. Files live in `.serena/memories/`.

### Current Status in This Repo

- `.serena/memories/` exists but is **empty** — Serena memory is not in active use.
- The whole `.serena/` directory is in the root `.gitignore`, so memories written today would be **machine-local** (the same scope as Claude Code memory), not team-shared.
- To make Serena memory team-shared, un-gitignore `.serena/memories/` (e.g., add `!.serena/memories/` after the `.serena` line in root `.gitignore`) and commit deliberately.

### When to Use vs Claude Code Memory

Claude Code memory is the default for personal learnings — it auto-loads and saves itself selectively (see `memory.md`). Reach for Serena memory only when you have a specific reason:

- You've un-gitignored `.serena/memories/` and want the note to ship with the repo.
- You want explicit, deliberate control over what gets persisted (no auto-save).
- The note belongs in the same tool surface as `find_symbol` calls (AI-readable codebase orientation notes that pair tightly with symbol lookups).

If none of those apply, **prefer Claude Code memory.**

### Hard Rules (when used)

1. **"Read-only" applies to source code, not Serena memory.** The codebase-mutating Serena tools (`replace_symbol_body`, `insert_after_symbol`, `replace_content`) are off-limits per saved user feedback. Memory tools write to `.serena/memories/`, not to `src/`, and are allowed when used deliberately.
2. **Surface findings like code lookups.** *"From Serena memory: <one-liner>."*
3. **Don't double-write.** A learning saved to Claude Code memory should not be duplicated to Serena memory; pick one.
4. **Read selectively.** Call `list_memories` first to see titles, then `read_memory` for the specific file. Don't blindly read everything.

### Anti-Patterns (Serena memory)

- Saving the same learning to both Claude Code memory and Serena memory — duplication, drift.
- Reading every memory file on every task — bloats context. Use `list_memories` first.
- Using Serena memory as a substitute for code documentation — code-level facts belong in docstrings or `documents/ai/explanation/`.
