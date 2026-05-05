# Subagent Dispatch — When and How to Use

Always-on. The goal is to keep the main agent's context window lean and let parallel investigations run without polluting the main thread. Applies repo-wide, including inside `/two` (Planner pre-flight, Critic claim-verification).

## When to Use

Always dispatch a subagent for:

- **Heavy reads.** Mapping a subsystem (10+ files), tracing a feature end-to-end across modules, or summarizing large diffs / long logs. Use the `Explore` subagent.
- **Separable investigations.** Two or more independent questions that don't share state — *"does X exist? what's Y's contract? is Z still used?"* Send them as multiple `Agent` tool calls in one message (parallel), or invoke `superpowers:dispatching-parallel-agents`.
- **Long-running tasks.** Comprehensive maps, exhaustive searches, or summaries of unfamiliar areas.
- **Planning and review passes.** Use the `Plan` agent for non-trivial implementation strategies and `code-reviewer` for review checkpoints after each major step.

## When NOT to Use

- A single `Read` of a known file path.
- A `grep` for a specific symbol when you already know the file or directory.
- A trivial one-step lookup the main agent can do without bloating context.

## Hard Rules

1. **Brief subagents like a colleague who walked into the room cold.** State the goal, what's already been ruled out, and what kind of answer you need. Don't write a flowchart — give the question and let them work.
2. **Cap response length when only a short answer is needed.** *"Report under 200 words."* Otherwise the result inflates the main context anyway.
3. **Trust but verify.** A subagent's summary describes what it intended to do, not necessarily what it did. When a subagent writes or edits code, inspect the actual changes before reporting work as done.
4. **Parallel for independent work.** When you launch multiple subagents on independent tasks, send them in a single message with multiple tool uses so they run concurrently.

## Anti-Patterns

- Loading a whole subsystem into the main context to ground a single claim → dispatch `Explore` instead.
- Sequential subagent calls when the work is independent → batch into one parallel message.
- Over-prescriptive prompts ("step 1 do X, step 2 do Y") that turn the subagent into a script runner — give them context and let them think.
- Trusting a subagent's summary without verifying actual file/code changes when the work was code-mutating.
