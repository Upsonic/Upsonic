# Claude Auto Memory — When and How to Use

Always-on. Auto memory is enabled by default in Claude Code v2.1.59+ and lives at:

```
~/.claude/projects/-Users-dogankeskin-Desktop-Upsonic/memory/
```

The first 200 lines (or 25 KB) of `MEMORY.md` load into every Claude Code session automatically. Topic files referenced from `MEMORY.md` are lazy — read on demand. This is where saved user feedback, prior corrections, and project learnings live.

## When to Use

Always consult auto memory for:

- **Recurring user corrections.** Before doing work the user has steered before, check whether memory captured a rule. Saves a re-correction round trip.
- **Past mistakes when writing or reviewing tests.** Memory stores prior test-writing pitfalls (what was mocked when it shouldn't have been; tests that passed trivially; fixtures that hid bugs). Consult before drafting test scenarios.
- **Similar prior requests.** When a new task echoes something the user has done before, memory often holds the relevant decision or feedback.
- **Project conventions and feedback.** Anything the user once said *"don't do X again"* — memory remembers.
- **Continuity across sessions.** Recover state from memory rather than re-asking the user.

## When NOT to Use

- For things written in code, docs, or `CLAUDE.md`. Those are canonical — don't re-derive them from memory.
- For ephemeral conversation state (current task progress, in-flight todos). Use the conversation, not memory.
- For things that are obviously memory-irrelevant (a single typo fix, an unambiguous read).

## Hard Rules

1. **`MEMORY.md` is already loaded — don't re-read it unless context was compacted.** It arrived at session start. Topic files (`feedback_*.md`, `project_*.md`, etc.) ARE lazy — read them when their description matches the task.
2. **Surface what you found.** When a memory consultation produces something relevant, briefly note it in your reply, e.g. *"From memory: prior feedback don't mock the DB in integration tests."* The consultation must be visible to the user.
3. **Trivial-work exception, but say so.** For single-line typo or comment edits, skipping memory is fine — but say so explicitly: *"Skipping memory lookup — single-line cosmetic edit."* Silent skipping is not allowed.
4. **Stale memory loses to current code.** A memory written weeks ago can be wrong. If memory says symbol X exists but a quick check shows it doesn't, trust the code; update or remove the stale memory.
5. **Browse with `/memory`.** The slash command lists every loaded file (CLAUDE.md, rules, auto memory) and links to the auto memory folder. Open files in your editor at any time — they're plain markdown.

## When and What to Save (MUST)

Saving is an **active discipline**, not a passive auto-magical thing. Claude Code may auto-save in some flows, but when one of the triggers below fires, you **MUST** deliberately save — don't rely on the model to "notice."

**MUST save when:**

- **User corrects with a generalisable rule.** *"don't mock the DB in tests"*, *"never commit without explicit approval"*. Save the **rule** (not the specific instance), in one tight sentence with the **why** if known. The next session will hit the same situation; the rule is what generalises.
- **User confirms a non-obvious choice was right.** *"yes, the single bundled PR was correct here"*, *"the cap of 5 iterations is correct — don't reduce."* Save the *why*, not just the choice. Confirmations of non-obvious decisions are gold; without them, the next session re-debates the same point.
- **Discovered convention non-obvious from the code.** A pattern that requires reading 5+ files to notice but governs many touchpoints. Save the convention with one example file path so a future session can locate it.
- **End-of-workflow reflection (feature / refactor / bug-fix / testing flow).** Before handing off to `commit.md`, reflect: *"Did this work surface a rule, a convention, or a pitfall worth carrying to next session?"* If yes, save it. If no, say so explicitly in the workflow output: *"No memory-worthy learning from this task."* Silent skipping is not allowed — the act of asking is the discipline.

**Do NOT save:**

- Code patterns that can be re-derived from the codebase.
- Git history (`git blame` / `git log` is canonical).
- One-off task state or debug fixes already recorded in commits.
- Anything already in `CLAUDE.md` or `documents/ai/guides/`.
- Personal preferences a single session got right — wait for confirmation across at least two distinct turns before promoting to memory.

**How to save:** the entry should be one short paragraph at most — rule first, *Why* second, *How to apply* third. Keep memory files tight; bloated entries lose signal. See the existing `feedback_*.md` files in the memory folder for the format.

## API "Memory Tool" — Different Thing

The Anthropic API memory tool (`memory_20250818`) is an SDK-level primitive for apps that build their own memory backend (file system, DB, encrypted store, etc.). Unrelated to Claude Code's auto memory. Not used in this repo.

## Anti-Patterns

- Reading every topic file on every task — bloats context. Read only when the description matches.
- Ignoring memory and re-asking the user something they already taught — wastes a round trip.
- Saving every observation as memory — clutters the index and reduces signal.
- Trusting old memory blindly without verifying against current code.
- Quietly using memory and not surfacing the finding — defeats the visibility intent.
