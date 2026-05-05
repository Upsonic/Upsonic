---
title: "Test Discipline"
description: "Workflow for deriving, writing, reviewing, and locking tests. Complements feature.md §4.5 (test placement and types) — that guide says where tests live and what types; this one says how scenarios are derived, vetted, and locked, and how production code iterates against locked tests."
---

# Test Discipline

Use this guide when deriving, writing, or reviewing tests for any feature, bug-fix, or refactor in this repo.

This document is **process** — the order in which tests are conceived, written, vetted, and locked. It complements `feature.md` §4.5 ([feature.md](./feature.md)), which defines *where* tests live and *what types* they are (unit / smoke / integration / doc-examples / sync vs async). Read both: §4.5 governs placement; this guide governs derivation and lifecycle.

## When to Use

- Deriving test scenarios from a plan or specification.
- Writing test code for new behaviour.
- Reviewing AI-generated test code before locking.
- Modifying production code while tests are locked.

## When NOT to Use

- Documentation typo fixes — no test impact.
- Pure dependency bumps with no code change.
- Generated artifacts (changelogs, lockfiles).

---

## The Four Phases

### Phase 1 — Derive Scenarios

**Entry gate.** A plan or specification exists (e.g. `plan/final_plan.md`, a feature ticket, a bug repro).

**What to do:**

1. **User provides their scenarios first.** Before any AI lookup, the user states the cases they care about — golden path plus the edge cases they have in mind. These are the seed.
2. **Pull similar prior tests via Serena.** Use `mcp__serena__find_symbol` and `mcp__serena__find_referencing_symbols` to locate analogous tests already under `tests/`. Reuse fixtures and patterns where they fit; don't reinvent. See `serena.md`.
3. **Pull past test-writing mistakes via auto memory.** Read `MEMORY.md` (already loaded) plus any topic files matching test-writing or testing-feedback. The point is to avoid mistakes the user has already corrected once. See `memory.md`.
4. **Surface findings.** Note in the reply what was found, e.g. *"From Serena: similar fixture pattern in `tests/agent/test_agent_accumulation.py:34`. From memory: prior feedback don't mock the DB in integration tests."*
5. **Lock the scenario list.** Once scenarios are written down (in chat or in the plan file), state explicitly: *"Scenario list locked. No additions / removals without approval."* The scenario list is the contract that follows.

**Aspect cross-references**
- Test placement (unit / smoke / integration) — `feature.md` §4.5.
- Sync / async parity — `feature.md` §4.5 and `coding-standards.md` §3.

**Exit gate.** A locked scenario list exists, with each case named and its expected behaviour described.

---

### Phase 2 — Generate Tests (RED First)

**Entry gate.** Phase 1 exit cleared.

**What to do:**

1. **Write tests against the locked scenarios — no production code yet.** Each test must fail when run against the current codebase. This is the RED of TDD.
2. **Tests must actually exercise the path.** A test that passes against unimplemented code is broken — it's testing nothing or testing the test setup.
3. **One scenario, one test.** No combining cases into one mega-test; that hides which case actually failed when red.
4. **Cross-reference `feature.md` §4.5 for placement:**
   - Pure logic → `tests/unit_tests/<subsystem>/`.
   - External services → `tests/smoke_tests/<subsystem>/` (Docker-backed via `make smoke_tests`).
   - Cross-subsystem → `tests/integration_tests/`.
   - New public API surface → also add a `tests/doc_examples/` exercise.
   - Sync and async paths each get their own test.
5. **Run the tests.** Confirm every test fails. Note the failure mode for each — *"AttributeError on `agent.latency`"* is good; *"test passed"* means broken-in-place.

**Exit gate.** Every locked scenario has a corresponding test, and every test currently fails when run.

---

### Phase 3 — Manual Review (Hard Gate)

**Entry gate.** Phase 2 exit cleared.

The user manually reads every generated test before locking. This phase is **non-skippable**. Tests that look right but don't actually test the change are the worst possible outcome — green CI lying about correctness.

**Reject and fix when:**

- **Test would pass without the production change.** Setup-only assertions, tautologies (`assertEqual(x, x)`), or assertions on `Mock.return_value`.
- **Test doesn't cover the edge it claims.** A test named `test_handles_empty_input` that only feeds non-empty input.
- **Test depends on order or shared state.** Each test must run in isolation.
- **Test mocks the framework boundary being added.** Mock third-party clients at the edge; let your own code run (per `feature.md` §4.5).
- **Test asserts on implementation, not contract.** Asserting on a private attribute or on the literal call shape locks the implementation, not the behaviour.

**Exit gate.** Every test, when read, *clearly exercises its scenario* and would fail if the production behaviour were absent.

---

### Phase 4 — Lock and Iterate via Code

**Entry gate.** Phase 3 exit cleared. All tests RED.

**What to do:**

1. **Lock the test files.** State explicitly: *"Tests locked. From here on, no edits to anything under `tests/...`. Behaviour changes go through `src/`, never the test."*
2. **The user pins the touchpoints.** Class names, service names, and the file paths where edits land are decided by the user, not unilaterally invented by the agent. The plan's **Touchpoints** section is the contract; the agent fills in bodies.
3. **Implement production code** to turn each test green, one scenario at a time. Locked tests are the contract.
4. **If a test looks wrong while implementing**, the test is not necessarily wrong — the *scope* may have been wrong. Stop implementation, surface the conflict, get an explicit scenario-list change approved (which means re-entering Phase 1 for that scenario), then re-lock. Never silently edit a test to make it pass.

**Exit gate.** All locked tests pass against the implementation, with no test edits since the lock. Done.

---

### Memory Hygiene at the End

Before declaring the test work complete and handing off to the parent workflow (`feature.md` / `refactor.md` / `bug-fix.md`), reflect per [`memory.md`](./memory.md) "When and What to Save":

- Did this scenario list surface a **test-writing pitfall** worth carrying to the next session? (e.g., *"trivially-passing tests slip through when assertions target `Mock.return_value`"*).
- Did the user correct a fixture / mocking / parametrize approach with a **generalisable rule**?
- Did manual review reject a kind of test you were about to write again next time?

If yes, save a tight memory entry — the rule, the *why*, and one example of how to apply it. If no, state explicitly: *"No memory-worthy learning from this test phase."* Silent skipping is not allowed.

---

## Hard Rules

1. **Locked tests are immutable.** Behaviour changes happen in `src/`. The temptation to "tweak the test to match the code" is the failure mode this entire discipline exists to prevent.
2. **No green test without RED first.** A test that passes immediately after writing it is testing nothing — there is no proof the assertion target is what made it pass. Re-derive RED before declaring done.
3. **User defines class/service names and change locations.** The agent does not unilaterally invent module structure; the plan's Touchpoints are the contract.
4. **One scenario, one test.** Combining cases obscures which assertion failed. Split.
5. **Surface the consultation chain.** Phase 1 must show what came from Serena and what came from memory. Phase 3 must show that every test was actually read.
6. **Trivial-work exception, but say so.** For a doc-only change with no test impact, skip the entire workflow but state it: *"Skipping test discipline — docs-only change."*

## Anti-Patterns

- AI generates tests, runs them, they pass, "done" — without manual Phase 3 review.
- Editing a test to make it pass when implementation diverges from spec — the test was the contract; the implementation broke it.
- Bundling new test cases into an existing test method ("just one more assertion") — locks unrelated behaviour by accident and makes future test edits ambiguous.
- Skipping the user-scenarios-first step and letting the AI propose the entire scenario list — the user's own edge-case intuitions never get encoded.
- Using `grep` to find similar tests when Serena would be precise — slower and yields false positives in docs / strings.
- Mocking the framework boundary you're adding — invalidates the test against real wiring (per `feature.md` §4.5).
- Combining sync and async into the same test — passing one does not exercise the other.
