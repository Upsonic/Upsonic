---
title: "Refactor Workflow"
description: "Workflow for behaviour-preserving refactors in the Upsonic framework. Defines the four mandatory phases (Motivate & Scope, Characterize, Transform, Verify Behaviour Preserved), the per-phase hard gates, and the anti-patterns that turn a refactor into something else. Lighter peer to feature.md; cross-references its aspect appendix instead of duplicating it."
---

# Refactor Workflow

This document is the operational guide for **refactoring code** in this framework — changing internal structure *without* changing observable behaviour. It tells you *what to do, in what order, and what must be true at each step.* It is a peer to two other guides:

- [`feature.md`](./feature.md) — for adding features. Carves pure refactors out of scope and points here.
- [`coding-standards.md`](./coding-standards.md) — *how* code is written here (naming, types, async).
- [`commit.md`](./commit.md) — *how* commits are made here (format, approval gate).

Refactoring has a different shape than feature or bug-fix work — *motivate → characterize current behaviour → transform → verify nothing changed* — and a tighter rule set: the public API does not change, no new features, no behaviour drift. This guide is sized accordingly: same RFC 2119 vocabulary and gate structure as `feature.md`, but cross-references `feature.md §4` for shared aspects rather than restating them.

---

## 1. Audience and Scope

**Audience.** AI assistants restructuring code in `src/upsonic/`. Human contributors will also benefit, but the doc is written for AI consumption: prescriptive, gated, and cross-referenced to real files in the repo.

**In scope — use this guide when:**

- Renaming, extracting, inlining, or moving code without changing observable behaviour.
- Splitting an oversized module, class, or function.
- Removing dead code or unused exports.
- Replacing one internal helper with another that has the identical contract.
- Mechanical migrations (e.g., `dict[...]` → `TypedDict`, sync helper → standalone function).

**Out of scope — use a different guide for:**

- Anything that changes observable behaviour, even slightly → [`feature.md`](./feature.md).
- Any change motivated by a bug → [`bug-fix.md`](./bug-fix.md). Refactor afterwards if needed.
- Performance changes that alter latency characteristics observable to users → [`feature.md`](./feature.md).

**Boundary calls — common ambiguities:**

- *Refactor that incidentally adds a small helper class.* Still a refactor **IF** the helper is internal (not exported, no public callers) and no public API changes. Otherwise it is a feature.
- *Refactor that fixes a latent bug along the way.* **STOP.** Split the work: do the bug-fix first under [`bug-fix.md`](./bug-fix.md) (with a regression test), then the refactor on top. The two **MUST NOT** share a commit. A refactor that secretly fixes a bug is impossible to review and impossible to revert cleanly.
- *Refactor that requires renaming a public symbol.* Allowed, but only with a deprecation alias at the old location (so existing user imports keep working) — or with explicit user sign-off on a breaking change.

---

## 2. How to Use This Guide

The guide has three parts:

1. **§3 — The four phases** (Motivate & Scope → Characterize → Transform → Verify Behaviour Preserved). The spine. You **MUST** transition through all four in order. Each phase has an *entry gate*, *what to do*, *aspect references* into `feature.md §4`, and an *exit gate*.
2. **§4 — Hard Gates Summary.** Pre-flight checklist per phase boundary.
3. **§5 — Anti-patterns.** The most common AI mistakes specific to refactoring.

**RFC 2119 keywords.** **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, and **MAY** are used with their RFC 2119 meanings. Anything marked **MUST** is a hard gate; violating it requires explicit user approval.

**Aspect handling.** Every aspect referenced from a phase **MUST** be addressed. If an aspect does not apply, you **MUST** state so explicitly with a one-line reason. Silent skipping is forbidden.

---

## 3. The Four Phases

### Phase 1 — Motivate & Scope

**Entry gate**

- A reason to refactor exists.

**What to do**

1. **Mandatory pre-work consultation.** Per the **Default Pre-Work Consultation** in [`CLAUDE.md`](../../../CLAUDE.md), run a parallel pass before stating the target state:
   - **Claude Code memory** — see [`memory.md`](./memory.md). Pull recurring user corrections, similar prior refactors, project conventions.
   - **Serena code lookup** — see [`serena.md`](./serena.md). Find similar prior refactors, related symbols, all callers of the surface you plan to touch.
   - **Surface findings at the top of your response.** *"From memory: …. From Serena: …."* Refactors fail loudly when an unaudited caller breaks; Serena's reference tracing is non-optional.
2. State the motivation in **one sentence**.
   - **Valid motivations:** paying explicit debt, enabling a known upcoming feature, a file/class crossing a complexity threshold, removing duplication, splitting a module that has grown to do too many things.
   - **Invalid motivations:** *"the code feels off"*, *"I don't like the style"*, *"modernising for its own sake"*. Without a concrete reason, the refactor will drift.
3. State what is **OUT** of scope explicitly: public API does not change, no new features, no behaviour changes, no performance changes, no bug fixes. If any of these are wrong, you are not refactoring — pick the right guide.
4. Identify which subsystem(s) under `src/upsonic/` are affected. Cross-subsystem refactors are higher risk and require more care in Phase 4.
5. Define the target state: target file layout, target signatures, target naming. *"I will refactor X"* is not enough; you need the *post-refactor shape* described concretely.
6. **Plan refinement (MUST for non-trivial refactors).** Invoke `/write-plan` (superpowers) to produce the structured six-section plan, with Touchpoints pinning every file/symbol move. **The agent invokes this command** — mandatory output, not optional.

   Two delivery paths:
   - **Non-trivial refactors (multi-file move, public-symbol rename, splitting an oversized module):** Run plan-writing inside the `/two` skill (two-terminal debate). The user opens `/two planner` and `/two critic`. The Planner invokes `/write-plan` while drafting Turn 1; the Critic reviews; iterate to mutual close (cap: 5 iterations). Final output: `plan/final_plan.md`.
   - **Trivial mechanical refactors (single-helper rename of an internal symbol, no public surface, no caller-audit findings):** The agent invokes `/write-plan` directly in a single terminal — no `/two` debate. The user declares the path explicitly: *"Skipping /two — single-helper rename, /write-plan single-shot."*

   `/two` is the debate transport; `/write-plan` is the plan format. Both compose for non-trivial; only `/write-plan` for trivial.

**Aspect references**

- *Default Pre-Work Consultation* — see [`CLAUDE.md`](../../../CLAUDE.md), [`memory.md`](./memory.md), [`serena.md`](./serena.md). Caller-audit via Serena is non-negotiable for any move/rename of a public symbol.
- *Architecture & Subsystem Fit* — see [`feature.md §4.1`](./feature.md). Stay inside the existing subsystem layout. A refactor is not justification for a new top-level module.
- *Plan Refinement* — `/two` (preferred) or `/write-plan` (small) for non-trivial refactors. Trivial mechanical refactors may skip with explicit statement.

**Exit gate**

- Memory + Serena consulted; findings (or absence) surfaced.
- Motivation written down, in one sentence.
- Out-of-scope items listed explicitly.
- Target state described concretely.
- Affected subsystems identified.
- Locked plan exists (or trivial-refactor skip is stated explicitly).

---

### Phase 2 — Characterize

**Entry gate**

- Phase 1 exit cleared.

**What to do**

1. Identify all tests covering the surface to be refactored. *"Tests pass"* is meaningless if there are no tests covering the path you are touching.
2. Run those tests; confirm they pass before any change.
3. If coverage is insufficient (no tests, or tests don't exercise the path being touched), you **MUST** add **characterization tests** first. Follow [`testing.md`](./testing.md) Phases 1-3 (Derive Scenarios with user-first / Serena / memory; RED tests; manual review). These tests lock in current behaviour and ship as part of the refactor change — they are not throwaway.
4. Capture both sync and async paths in coverage. If the surface has `name()` and `aname()` siblings, both **MUST** be exercised before transformation.
5. If the surface has external I/O (Redis, Postgres, model API, filesystem), smoke tests apply — see [`feature.md §4.5`](./feature.md).
6. **Lock the test set before Phase 3.** Once the characterization tests are green, declare them locked: *"Tests locked. Transformation may not edit any test file beyond mechanical renames/moves."* This is the contract Phase 3 transforms against.

**Aspect references**

- *Tests* — see [`feature.md §4.5`](./feature.md). Test tier (`unit_tests` vs `smoke_tests`) by what the surface touches.
- *API Discipline* — see [`feature.md §4.2`](./feature.md). Sync + async parity verified by Phase 2 tests.

**Exit gate**

- Existing or newly-added tests cover the refactor surface.
- All those tests pass on the pre-refactor code.
- Sync + async paths both exercised.

---

### Phase 3 — Transform

**Entry gate**

- Phase 2 exit cleared.

**What to do**

1. Make the smallest set of changes to reach the target state. Ideally one mechanical move at a time (one rename, one extract, one move).
2. Run the Phase 2 tests after each meaningful step. If they fail, you have introduced a behaviour change — back off and narrow the scope of the step.
3. **Sync + async siblings move together.** Never refactor only the sync path; the drift between siblings is exactly what `feature.md §4.2` exists to prevent.
4. Update `__init__.py` exports to preserve canonical import paths. If a public symbol must move, add a **deprecation alias** at the old location that re-exports the symbol with a `DeprecationWarning`.
5. Lazy-import discipline preserved (no new heavy deps imported at top-level — see [`coding-standards.md §2.4`](./coding-standards.md)).
6. **NO** new features. **NO** bug fixes. **NO** performance changes. **NO** new public symbols that weren't part of the target state in Phase 1.
7. Standalone-functions discipline preserved — no new module-level mutable state, no hidden globals (see [`coding-standards.md §1`](./coding-standards.md) and [`§3.5`](./coding-standards.md)).

**Aspect references**

- *Architecture & Subsystem Fit* — see [`feature.md §4.1`](./feature.md). No cross-subsystem leakage introduced.
- *API Discipline* — see [`feature.md §4.2`](./feature.md). Sync + async parity, full type annotations preserved.
- *Integration & Distribution* — see [`feature.md §4.7`](./feature.md). `__init__.py` exports updated; deprecation aliases for moved public symbols.

**Exit gate**

- Code compiles.
- Phase 2 tests still pass — without modification beyond mechanical renames/moves.
- `mypy --strict` clean.
- Public API surface unchanged, OR deprecation aliases in place for moved/renamed symbols.

---

### Phase 4 — Verify Behaviour Preserved

**Entry gate**

- Phase 3 exit cleared.

**What to do**

1. Run the full unit test suite: `uv run --all-extras pytest tests/unit_tests -v`.
2. If I/O was touched, run `make smoke_tests`.
3. Run `pre-commit run --all-files`.
4. Run `mypy --strict` on `src/`.
5. **Diff the test files.** If existing tests changed in any way beyond renames/moves (assertions broadened, parametrize cases removed, fixtures restructured), you have likely introduced a behaviour change hiding in test code. **STOP** and split that out into a separate bug-fix or feature.
6. Verify public exports are unchanged. Smoke-import every public symbol from its canonical path; if anything fails, the refactor broke the public surface.
7. If any public symbol was renamed or moved, confirm the deprecation alias at the old location is in place and emits `DeprecationWarning`.
8. Confirm docstrings on touched public symbols still match the (preserved) behaviour. A refactor that desyncs docstrings from code is incomplete.
9. Confirm examples in `tests/doc_examples/` still execute. Any example that breaks is evidence of a public-surface drift you missed.
10. If the refactor was an enabler for a planned feature, link the future plan in the commit body.
11. **Memory hygiene (MUST, before commit).** Per [`memory.md`](./memory.md) "When and What to Save", reflect on whether this refactor surfaced anything worth carrying to the next session — typically a discovered convention, a non-obvious dependency direction, or a user correction about scope discipline. If yes, save a tight memory entry. If no, state explicitly: *"No memory-worthy learning from this refactor."* Silent skipping is not allowed.

**Aspect references**

- *Tests* — see [`feature.md §4.5`](./feature.md). Full test tiers green.
- *Integration & Distribution* — see [`feature.md §4.7`](./feature.md). Exports + deprecation aliases.
- *Docs & Examples* — see [`feature.md §4.6`](./feature.md). Docstrings match preserved behaviour; examples still run.

**Exit gate**

- Every item in the **Hard Gates Summary (§4)** passes.

> **Hard gate.** You **MUST NOT** call the work "refactored", "clean", or "done" before this gate clears. *"The tests still pass"* is necessary but not sufficient; *every item in §4 passes* is the gate.

---

## 4. Hard Gates Summary

```
─── Before transitioning OUT of MOTIVATE & SCOPE ──────────────────────
  [ ] Memory consulted (memory.md); findings surfaced or "no relevant entry"
  [ ] Serena consulted (serena.md); caller-audit done for any public symbol move/rename
  [ ] Motivation stated in one sentence
  [ ] Out-of-scope items listed (no new features, no behaviour change, no fixes)
  [ ] Target state described concretely
  [ ] Affected subsystems identified
  [ ] /write-plan invoked by the agent; six-section locked plan exists
  [ ] Plan delivered via /two debate (non-trivial) or /write-plan single-shot (trivial — explicitly declared)

─── Before transitioning OUT of CHARACTERIZE ──────────────────────────
  [ ] Tests covering the refactor surface exist (added if missing, via testing.md Phases 1-3)
  [ ] All those tests pass on pre-refactor code
  [ ] Sync + async paths both exercised
  [ ] Test set locked before Phase 3; declared explicitly

─── Before transitioning OUT of TRANSFORM ─────────────────────────────
  [ ] Phase 2 tests still pass without modification (renames/moves only)
  [ ] mypy --strict clean
  [ ] Public exports unchanged, or deprecation aliases added
  [ ] Sync + async siblings refactored together
  [ ] No new features; no bug fixes; no performance changes

─── Before claiming the refactor DONE ─────────────────────────────────
  [ ] Memory hygiene: end-of-workflow reflection complete; entry written or "no memory-worthy learning" stated
  [ ] uv run --all-extras pytest tests/unit_tests passes
  [ ] make smoke_tests passes (if I/O touched)
  [ ] pre-commit run --all-files clean
  [ ] mypy --strict clean
  [ ] Public API surface verified unchanged (smoke-imports succeed)
  [ ] Deprecation aliases in place for moved/renamed public symbols
  [ ] Docstrings still match the preserved behaviour
  [ ] Examples in tests/doc_examples/ still run
  [ ] Ready to hand off to commit workflow (commit.md)
```

If any box cannot be ticked, the refactor is not done. Naming an item N/A is allowed, but **MUST** be accompanied by a one-line reason.

---

## 5. Anti-Patterns

The most common AI mistakes when refactoring in this framework.

1. **Refactoring without tests.** *"The tests pass"* when there are no tests is a meaningless gate. Add characterization tests first under Phase 2, or you have no behaviour-preserving guarantee. *(Aspect: §4.5)*

2. **Bundling a feature into the refactor.** *"While I was extracting the helper I added a new option to the constructor."* Now this isn't a refactor; it is a feature. Split the work: refactor first, feature on top under [`feature.md`](./feature.md). *(Aspect: §4.2)*

3. **Bundling a bug-fix into the refactor.** *"I noticed a latent bug and fixed it as part of moving the function."* The bug-fix needs a regression test and a separate change under [`bug-fix.md`](./bug-fix.md). Bundling makes the diff impossible to review and the bug impossible to revert independently. *(Aspect: §4.3)*

4. **Renaming a public symbol without a deprecation alias.** Existing user imports break silently on next release. Public API is a contract; renames need a deprecation alias at the old location, or explicit user sign-off on a breaking change. *(Aspect: §4.2)*

5. **Refactoring only the sync path.** Sync and async siblings are two implementations of one API. Refactoring one without the other introduces drift that bugs will exploit. *(Aspect: §4.2)*

6. **"Improving" tests during the refactor.** Updating tests, broadening assertions, removing parametrize cases, *"modernising"* — these are behaviour-affecting changes hiding in test code. Phase 2 tests change only mechanically (file moves, renames). *(Aspect: §4.5)*

7. **Top-level subsystem creation framed as a "refactor".** Splitting `tools/` into `tools/` and `tool_processors/` is an architectural change requiring justification under [`feature.md §4.1`](./feature.md), not a quiet refactor. *(Aspect: §4.1)*

8. **Inverting a dependency direction without auditing callers.** *"I moved this from X to Y."* Now `from upsonic.x import Foo` breaks for every existing caller. Deprecation alias at the old location plus a caller audit, or it isn't a refactor. *(Aspect: §4.7)*

9. **Skipping the pre-work consultation (memory + Serena).** A refactor that moves or renames a public symbol without a Serena caller-audit is a guaranteed broken-import release. Both memory and Serena run before stating the target state. *(Aspect: Default Pre-Work Consultation in CLAUDE.md, Phase 1)*

10. **Producing a refactor without `/write-plan`.** Multi-file moves, public-symbol renames, splitting an oversized module — these need `/write-plan` output before any `src/` edit. Free-form refactors drift; structured plans don't. Non-trivial refactors deliver via `/two` debate; trivial mechanical refactors may go single-shot — the path MUST be declared explicitly. *(Aspect: Phase 1)*

11. **Editing the characterization tests during transformation.** The whole point of Phase 2 is locking behaviour. If a Phase 2 test seems wrong while transforming, the refactor's *scope* is wrong — split the change, don't edit the test silently. *(Aspect: testing.md Phase 4, this guide Phase 3)*

12. **Skipping memory hygiene at the end of the refactor.** Refactors often surface conventions and scope-discipline corrections that should travel forward. End-of-workflow memory reflection is mandatory; either save or state "no memory-worthy learning." *(Aspect: memory.md, Phase 4)*

---

## 6. Quick Map

```
Phase 1: Motivate & Scope  — memory.md + serena.md pre-work pass (MUST)
                             + caller-audit for public symbol moves
                             + /write-plan invocation → six-section locked plan (MUST for non-trivial)
                             + delivered via /two debate (non-trivial) or single-shot (trivial)
                             + why, target state, what stays the same
Phase 2: Characterize      — tests cover the surface (testing.md Phases 1-3)
                             + lock test set before Phase 3 (MUST)
Phase 3: Transform         — small mechanical steps, sync+async together
                             + locked tests stay locked (no semantic edits)
Phase 4: Verify            — tests pass UNCHANGED, public API unchanged, no drift
                             + memory hygiene reflection (MUST)
Then: commit.md            — propose message, wait for approval, commit
```

When unsure at any point, re-read the relevant phase section and the cross-referenced aspect rules in [`feature.md §4`](./feature.md). The hard-gates summary in §4 is the single most important page in this document.
