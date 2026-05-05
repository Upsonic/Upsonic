---
title: "Bug-Fix Workflow"
description: "Workflow for fixing reported bugs in the Upsonic framework. Defines the four mandatory phases (Reproduce, Diagnose, Fix, Verify), the per-phase hard gates, and the anti-patterns that produce broken or partial fixes. Lighter peer to feature.md; cross-references its aspect appendix instead of duplicating it."
---

# Bug-Fix Workflow

This document is the operational guide for **fixing bugs** in this framework. It tells you *what to do, in what order, and what must be true at each step.* It is a peer to two other guides:

- [`feature.md`](./feature.md) — for adding features. Carves bug-fixes out of scope and points here.
- [`coding-standards.md`](./coding-standards.md) — *how* code is written here (naming, types, async).
- [`commit.md`](./commit.md) — *how* commits are made here (format, approval gate).

Bug-fixing has a different shape than feature work — *reproduce → root cause → minimal fix → regression test* — and a smaller surface area. This guide is sized accordingly: same RFC 2119 vocabulary and gate structure as `feature.md`, but cross-references `feature.md §4` for shared aspects rather than restating them.

---

## 1. Audience and Scope

**Audience.** AI assistants fixing reported bugs in `src/upsonic/`. Human contributors will also benefit, but the doc is written for AI consumption: prescriptive, gated, and cross-referenced to real files in the repo.

**In scope — use this guide when:**

- A reported bug, failing test, traceback, regression, or unexpected behaviour exists.
- Behaviour contradicts a docstring, type annotation, or stated contract.
- Crash, hang, deadlock, or race condition.
- Silent wrong output (a computed value is incorrect, not just an error).

**Out of scope — use a different guide for:**

- *"I want to add a feature so this works differently"* → [`feature.md`](./feature.md).
- *"Code is fine, I just want to clean it up"* → [`refactor.md`](./refactor.md).
- Changing documented behaviour deliberately → [`feature.md`](./feature.md) (it is a breaking change).

**Boundary calls — common ambiguities:**

- *Bug-fix that needs a new typed exception in `exceptions.py`.* Still a bug-fix. Adding the exception is part of the minimal fix and ships in the same change.
- *Bug-fix that reveals adjacent dead code or messy structure.* **STOP.** The bug-fix stays narrow. Note the cleanup for a separate refactor pass under [`refactor.md`](./refactor.md) — do not bundle it.
- *Bug that does not reproduce.* Do not apply a speculative fix. Stop and ask the user for more context.

---

## 2. How to Use This Guide

The guide has three parts:

1. **§3 — The four phases** (Reproduce → Diagnose → Fix → Verify). The spine. You **MUST** transition through all four in order. Each phase has an *entry gate*, *what to do*, *aspect references* into `feature.md §4`, and an *exit gate*.
2. **§4 — Hard Gates Summary.** Pre-flight checklist per phase boundary.
3. **§5 — Anti-patterns.** The most common AI mistakes specific to bug-fixing.

**RFC 2119 keywords.** **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, and **MAY** are used with their RFC 2119 meanings. Anything marked **MUST** is a hard gate; violating it requires explicit user approval.

**Aspect handling.** Every aspect referenced from a phase **MUST** be addressed. If an aspect does not apply, you **MUST** state so explicitly with a one-line reason. Silent skipping is forbidden.

---

## 3. The Four Phases

### Phase 1 — Reproduce

**Entry gate**

- A bug report, failing test, traceback, or observed contradiction-of-contract exists.

**What to do**

1. Read the report twice. Distinguish the symptom (what the user saw) from the underlying behaviour.
2. Build a deterministic reproduction. Capture the exact command, input, and environment.
3. The reproduction **MUST** be a runnable test or script. *"I see it sometimes"* is not a reproduction.
4. If you cannot reproduce, you **MUST** stop and ask the user for more context. Do not apply a speculative fix.

**Aspect references**

- *Tests* — see [`feature.md §4.5`](./feature.md). The reproduction will become the regression test in Phase 3.

**Exit gate**

- A runnable command or test exists that fails the same way the user describes.
- The failure is deterministic across runs.

---

### Phase 2 — Diagnose

**Entry gate**

- Phase 1 exit cleared.

**What to do**

1. Trace from symptom to cause by reading the actual code path. Do not pattern-match from memory.
2. State the root cause in **one sentence**. If you cannot, you have not finished diagnosing.
3. **Audit related code paths for the same root cause.** Same module? Other modules with similar shape? Same `except Exception:` swallow elsewhere? Same off-by-one pattern? A bug-fix that closes only one site of a multi-site bug is a partial fix.
4. If the bug spans subsystems (e.g., `agent/` + `tools/`), **MUST** identify each affected subsystem before proposing a fix.
5. For non-obvious bugs, use the `superpowers:systematic-debugging` skill.

**Aspect references**

- *Architecture & Subsystem Fit* — see [`feature.md §4.1`](./feature.md). Identify which subsystem owns the cause.
- *Reliability & Safety* — see [`feature.md §4.3`](./feature.md). Typed exceptions, error semantics, no bare `except`.

**Exit gate**

- Root cause stated in one sentence.
- List of related code paths checked — either clean, or with their own fixes scoped.

---

### Phase 3 — Fix

**Entry gate**

- Phase 2 exit cleared.

**What to do**

1. Write the **minimal** fix targeting the root cause. The diff should be small.
2. Convert the Phase 1 reproduction into a regression test, in the **same change**.
3. Test placement: `tests/unit_tests/<subsystem>/` for pure-logic bugs; `tests/smoke_tests/<subsystem>/` for I/O / external-service bugs.
4. If the fix needs a new typed exception, add it to `src/upsonic/exceptions.py` in the same change.
5. **Sync + async parity.** If the bug exists in both `name()` and `aname()`, fix both in the same change. See [`feature.md §4.2`](./feature.md).
6. **No scope creep.** Spotted dead code, weird naming, an outdated comment? Note it for a later refactor under [`refactor.md`](./refactor.md) — do **NOT** include in this change.
7. New keyword arguments **MUST** default to behaviour-preserving values. Public surface is a contract.

**Aspect references**

- *API Discipline* — see [`feature.md §4.2`](./feature.md). Sync + async parity, no breaking-default kwargs.
- *Reliability & Safety* — see [`feature.md §4.3`](./feature.md). Typed exceptions, no `except: pass`.
- *Tests* — see [`feature.md §4.5`](./feature.md). Regression test placement and tier.

**Exit gate**

- Regression test fails on pre-fix code.
- Regression test passes on post-fix code.
- `mypy --strict` clean on touched code.
- No unrelated changes in the diff.

---

### Phase 4 — Verify

**Entry gate**

- Phase 3 exit cleared.

**What to do**

1. Run the full unit test suite: `uv run --all-extras pytest tests/unit_tests -v`.
2. If I/O was touched, run `make smoke_tests`.
3. Run `pre-commit run --all-files`.
4. Run `mypy --strict` on `src/`.
5. Manually trace the original failing path through the new code — confirm the bug is gone.
6. Manually trace one related code path audited in Phase 2 — confirm it still works.
7. Bug-fixes are user-visible. Flag CHANGELOG / version-bump impact for the user.
8. Confirm `UPSONIC_TELEMETRY=False` is still respected if any telemetry path was touched.

**Aspect references**

- *Tests* — see [`feature.md §4.5`](./feature.md).
- *Observability & Cost* — see [`feature.md §4.4`](./feature.md). `UPSONIC_TELEMETRY=False` respected; no PII in error messages.

**Exit gate**

- Every item in the **Hard Gates Summary (§4)** passes.

> **Hard gate.** You **MUST NOT** call the work "fixed", "done", or "ready" before this gate clears. *"The original test passes"* is not enough; *every item in §4 passes* is the gate.

---

## 4. Hard Gates Summary

```
─── Before transitioning OUT of REPRODUCE ─────────────────────────────
  [ ] Deterministic reproduction (runnable command or test) exists
  [ ] Reproduction failure mode matches the report

─── Before transitioning OUT of DIAGNOSE ──────────────────────────────
  [ ] Root cause stated in one sentence
  [ ] Related code paths audited; either clean or scoped for fixes
  [ ] No speculative "I think it's because…" — actual evidence

─── Before transitioning OUT of FIX ───────────────────────────────────
  [ ] Diff is minimal — no unrelated changes
  [ ] Regression test added (sync AND async if applicable)
  [ ] Typed exceptions used; no bare except
  [ ] mypy --strict clean on touched code
  [ ] No new keyword args without behaviour-preserving defaults

─── Before claiming the bug-fix DONE ──────────────────────────────────
  [ ] Regression test fails on pre-fix code, passes after
  [ ] uv run --all-extras pytest tests/unit_tests passes
  [ ] make smoke_tests passes (if I/O touched)
  [ ] pre-commit run --all-files clean
  [ ] mypy --strict clean
  [ ] Manual trace of original failure shows fix works
  [ ] Manual trace of one related path shows no regression
  [ ] CHANGELOG / version-bump impact flagged
  [ ] Ready to hand off to commit workflow (commit.md)
```

If any box cannot be ticked, the bug-fix is not done. Naming an item N/A is allowed, but **MUST** be accompanied by a one-line reason.

---

## 5. Anti-Patterns

The most common AI mistakes when fixing bugs in this framework.

1. **Fixing the symptom, not the cause.** Wrapping in `try/except` or adding a `None` check without finding *why* `None` showed up. The `except` swallow you just added IS the next bug. *(Aspect: §4.3)*

2. **No reproduction, only a guess.** Pattern-matching the report to a known bug shape and shipping a fix without running the failing case. Most "fixes" without repro do not fix the bug. *(Aspect: §4.5)*

3. **Bundling refactor into the fix.** *"While I was here I cleaned up the surrounding code."* Now the diff is too big to review and the fix is hidden inside it. Keep the bug-fix narrow; do the refactor under [`refactor.md`](./refactor.md) afterwards. *(Aspect: §4.2)*

4. **Test that doesn't fail without the fix.** A regression test that passes whether or not the fix is applied is not a regression test. Run it against the unfixed code first to confirm it fails. *(Aspect: §4.5)*

5. **Fixing only one site of a multi-site bug.** Same bare-except in three files; fixing one ships a partial fix and pretends the issue is closed. Audit related paths in Phase 2. *(Aspect: §4.1)*

6. **Async sibling forgotten.** The bug exists in both `read` and `aread`; the fix lands only in `read`. Sync + async parity applies to fixes too. *(Aspect: §4.2)*

7. **Catching exception to swallow.** Fix is *"catch and log"*. This loses information and creates the next bug. Catch typed, re-raise typed. *(Aspect: §4.3)*

8. **Bug-fix that silently changes the public API.** Fix introduces a new keyword argument with a non-preserving default; existing callers break. Public surface is a contract; defaults preserve old behaviour. *(Aspect: §4.2)*

---

## 6. Quick Map

```
Phase 1: Reproduce  — deterministic failing case, no fix without one
Phase 2: Diagnose   — root cause in one sentence; audit related paths
Phase 3: Fix        — minimal change + regression test + sync/async parity
Phase 4: Verify     — tests, types, manual trace of failure AND related paths
Then: commit.md     — propose message, wait for approval, commit
```

When unsure at any point, re-read the relevant phase section and the cross-referenced aspect rules in [`feature.md §4`](./feature.md). The hard-gates summary in §4 is the single most important page in this document.
