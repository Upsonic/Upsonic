---
title: "Feature Workflow"
description: "Workflow + standards for adding features and non-trivial enhancements to the Upsonic framework. Defines the four mandatory phases (Understand, Design, Implement, Verify), the eight cross-cutting aspects checked at each phase, the hard gates that block progression, and the anti-patterns to avoid. For AI assistants planning or implementing feature work."
---

# Feature Workflow

This document is the operational guide for building **features and non-trivial enhancements** in this framework. It tells you *what to do, in what order, and what must be true at each step*. It is the connective tissue between two existing standards documents:

- [`coding-standards.md`](./coding-standards.md) — *how* code is written here (naming, types, async, structure).
- [`commit.md`](./commit.md) — *how* commits are made here (format, approval gate).

Where those documents define rules, this document defines a process and embeds the framework-specific musts/shoulds/should-nots that an AI without prior context would otherwise miss.

---

## 1. Audience and Scope

**Audience.** AI assistants planning or implementing feature work in `src/upsonic/`. Human contributors will also benefit, but the doc is written for AI consumption: prescriptive, gated, and cross-referenced to real files in the repo.

**In scope — use this guide when adding:**
- a new public API (function, class, method) that users will import
- a new subsystem under `src/upsonic/` (rare)
- a new provider in an existing subsystem (e.g. a new VectorDB / Model / Storage / Tool / Embedding / OCR / Loader / Text-splitter provider)
- a new agent type or new prebuilt agent
- a new policy in `safety_engine/`
- a new RAG component (retriever, reranker, chunker)
- a new public method or public config knob on an existing class (`Direct`, `Agent`, `Task`, `Team`, `KnowledgeBase`, etc.)
- a new error type intended to be caught by users

**Out of scope — do not run this workflow for:**
- bug fixes (different shape: reproduce → root cause → minimal fix → regression test)
- pure refactors with no externally observable behaviour change
- typo / single-line documentation fixes
- version bumps and dependency-only bumps
- internal-only helpers used by exactly one caller
- generated code (changelogs, lockfiles)

When in doubt about whether something is in scope, **default to running the workflow** — the cost of unneeded ceremony is small; the cost of skipping a hard gate is a broken release.

---

## 2. How to Use This Guide

The guide has three parts:

1. **§3 — The four phases** (Understand → Design → Implement → Verify). The spine. You **MUST** transition through all four in order. Each phase has an *entry gate*, *what to do*, an *aspect checklist for that phase*, and an *exit gate*.
2. **§4 — Aspect reference (Appendix A).** The eight cross-cutting concerns (Architecture, API Discipline, Reliability & Safety, Observability & Cost, Tests, Docs & Examples, Integration & Distribution, Performance) listed in full with MUST / SHOULD / SHOULD NOT rules. Phases reference these by name; this is where the rules live.
3. **§5–6 — Hard-gates summary (Appendix B) and Anti-patterns (Appendix C).** Pre-flight checklists and the most common AI mistakes specific to this codebase.

**RFC 2119 keywords.** **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, and **MAY** are used with their RFC 2119 meanings throughout. Anything marked **MUST** is a hard gate; violating it requires explicit user approval.

**Skipping aspects.** Every aspect listed in a phase checklist **MUST** be addressed. If an aspect does not apply, you **MUST** state so explicitly with a one-line reason — e.g. *"Observability & Cost: N/A, this feature does not call any model and adds no I/O."* Silent skipping is forbidden.

---

## 3. The Four Phases

### Phase 1 — Understand

**Entry gate**
- A feature request exists (user instruction, ticket, follow-up).

**What to do**
1. Read the feature request twice. The second read is for the things you missed the first time.
2. **Mandatory pre-work consultation.** Per the **Default Pre-Work Consultation** in [`CLAUDE.md`](../../../CLAUDE.md), run a parallel pass before any sketch:
   - **Claude Code memory** — see [`memory.md`](./memory.md). Pull recurring user corrections, similar prior requests, project conventions.
   - **Serena code lookup** — see [`serena.md`](./serena.md). Find similar prior implementations, related symbols, referencing patterns.
   - **Surface findings at the top of your response.** *"From memory: …. From Serena: …."* If neither produced anything relevant, say so explicitly.
3. **Flag a fuzzy brief.** If the user's request leaves bottlenecks, success criteria, or constraints unstated, **MUST** ask the user to sharpen it before any plan refinement. *"What does success look like? Which subsystem boundaries are off-limits?"* Sharpening is the user's call — the agent's job is to flag the gap, not to invoke brainstorming on the user's behalf.
4. Identify which subsystem(s) under `src/upsonic/` will be touched. Read the `base.py` of each affected subsystem first (every subsystem with extension points has one).
5. Find the closest existing implementation to the feature being requested and read it end to end. Examples:
   - new VectorDB provider → read `src/upsonic/vectordb/providers/chroma.py`
   - new Model provider → read `src/upsonic/models/openai.py`
   - new Storage backend → read `src/upsonic/storage/postgres/`
   - new Tool → read `src/upsonic/tools/common_tools/duckduckgo.py`
6. Check whether an equivalent feature already exists or is partially implemented. Do not duplicate.
7. If the request has more than one reasonable interpretation, **MUST** ask a clarifying question before proceeding.

**Aspect checklist (this phase)**
- *Default Pre-Work Consultation* — see [`CLAUDE.md`](../../../CLAUDE.md), [`memory.md`](./memory.md), [`serena.md`](./serena.md).
  - **MUST** consult both memory and Serena before sketching a solution.
  - **MUST** surface findings at the top of the response.
- *Architecture & Subsystem Fit* — see §4.1.
  - **MUST** identify which existing subsystem owns this feature.
  - **MUST NOT** propose a new top-level module under `src/upsonic/` without justifying why no existing subsystem fits.

**Exit gate**
You can answer in writing:
- which subsystem(s) are affected,
- what the user-visible API will look like (rough shape, not yet typed),
- what the simplest plausible solution looks like,
- what existing code is the reference pattern,
- what memory and Serena returned (or that nothing relevant was found).

---

### Phase 2 — Design

**Entry gate**
- Phase 1 exit cleared.

**What to do**
1. Write the **public interface first**, with full type annotations. Sketch the function/class signatures before any implementation thought.
2. Decide **sync vs async parity from day one**. Every public method that does I/O or wraps an awaitable **MUST** ship as both `name()` and `aname()`. Do not defer the second one.
3. Decide whether a new `base.py` extension point is needed. Default answer is **no** — extending an existing one is almost always correct.
4. Propose **at least two alternative approaches** for any non-trivial design decision and pick one with a stated reason. Single-option designs hide trade-offs.
5. Sketch error and edge cases: what does the API do on bad input, on network failure, on timeout, on partial result, on cancellation?
6. If it is a new provider, locate up front:
   - the registry it must register with (`src/upsonic/models/model_registry.py` for models, etc.)
   - the `__init__.py` exports that must be updated
   - the optional-dependency group in `pyproject.toml` that the new third-party libraries belong to
7. List the tests you plan to write before writing them.
8. **Plan refinement (MUST).** Invoke `/write-plan` (superpowers) to produce the structured six-section plan: Goal & Success Criteria, Scope, Touchpoints (concrete file paths, class names, service names — pinned by the user), Test Strategy, Risks & Mitigations, Step-by-Step Execution Plan. **The agent invokes this command** — it is mandatory output, not optional.

   Two delivery paths, depending on the work's size:
   - **Non-trivial features (default):** Run plan-writing inside the `/two` skill (two-terminal debate). The user opens `/two planner` and `/two critic`. The Planner invokes `/write-plan` while drafting Turn 1 in `plan/conversation.md`; the Critic reviews; iterate to mutual close (cap: 5 iterations). Final output: `plan/final_plan.md`.
   - **Trivial features (single-file, no public API, no tests beyond a doc-example):** The agent invokes `/write-plan` directly in a single terminal — no `/two` debate. The user declares the path explicitly: *"Skipping /two — trivial feature, /write-plan single-shot."*

   `/two` is the **two-terminal debate transport** (defines roles + dialogue protocol). `/write-plan` is the **plan format and content** (mandatory in either path). They compose: `/two` for the debate, `/write-plan` for the actual writing.

   The plan is the contract for Phase 3 and Phase 4. **No production code or tests without it.**

**Aspect checklist (this phase)**
- *Architecture & Subsystem Fit* (§4.1) — extension point chosen, no new subsystem unless justified.
- *API Discipline* (§4.2) — signatures fully typed, sync + async parity planned.
- *Reliability & Safety* (§4.3) — does this feature affect model output quality, user content, or external trust boundaries? If yes, plan integration with `reliability_layer/` and/or `safety_engine/`.
- *Plan Refinement* — `/two` (preferred) or `/write-plan` (small features). Six-section locked plan exists; Touchpoints pinned.
- *Integration & Distribution* (§4.7) — registry entries, exports, optional-dep group identified.
- *Performance* (§4.8) — any blocking I/O? Any hot path that needs caching?

**Exit gate**
A locked plan exists (`plan/final_plan.md` from `/two`, or the equivalent six-section plan from `/write-plan`) that lists:
- the public API with types,
- the alternatives considered and the choice with reason,
- the subsystem touchpoints and integration points (pinned file paths, class names, service names),
- the tests planned.

> **Hard gate.** You **MUST NOT** write production code or tests before this exit gate is cleared and the user has had a chance to review the locked plan. "Production code" here means anything outside a quick experimental sketch — and even sketches **SHOULD** be deleted once the plan is locked.

---

### Phase 3 — Implement

**Entry gate**
- Phase 2 design approved.

**What to do**

> **Order of operations is mandatory.** Tests come **first**, `src/` comes **second**. See [`testing.md`](./testing.md) for the full RED-first / lock-first discipline.

1. **Test discipline (MUST, ahead of any production code).** Run [`testing.md`](./testing.md) end-to-end:
   - **Phase 1 — Derive Scenarios.** User provides their scenarios first; you augment with Serena ([`serena.md`](./serena.md)) for similar existing tests and Claude Code memory ([`memory.md`](./memory.md)) for past test-writing mistakes. Lock the scenario list.
   - **Phase 2 — Generate RED Tests.** Each scenario gets one failing test. Run them; confirm every test fails before any `src/` change.
   - **Phase 3 — Manual Review (Hard Gate).** User reads each test; trivially-passing tests, missed edges, and order-coupled tests are rejected and fixed.
   - **Phase 4 — Lock.** State explicitly: *"Tests locked. From here on, no edits to anything under `tests/...`. Behaviour changes go through `src/`, never the test."*
2. Implement the smallest **vertical slice** first: one provider, one method, one path through the new code, end to end. Do not build a horizontal layer (all signatures, no bodies) and then come back to fill it in.
3. Add **sync and async in pairs**, not in waves. The async sibling is not a follow-up commit; it ships in the same change.
4. Add types as you write. No `Any` unless an upstream third-party API forces it; when that happens, isolate the `Any` at the boundary and convert to a typed model immediately. See `coding-standards.md` §4 (Type System) for full type rules.
5. Keep functions **standalone** — every dependency is a parameter, never a global, never a hidden module-level singleton. See `coding-standards.md` §1 (Philosophy) and §3.5 (Standalone, Self-Contained Functions).
6. Update `__init__.py` exports **as you go**, not at the end. If your code is not importable from its parent package by the time you stop for the day, you have not finished.
7. If the feature pulls in a new third-party library, add it to the appropriate optional-dependency group in `pyproject.toml` *in the same change*. Lazy-import the dependency inside the function or class that uses it (per `coding-standards.md` §2.4). The minimal install **MUST** stay minimal.
8. **If a locked test seems wrong while implementing**, the test isn't necessarily wrong — the *scope* may have been wrong. Stop, surface the conflict, get the scenario list re-opened explicitly (re-enter [`testing.md`](./testing.md) Phase 1 for that scenario), then re-lock. **Never silently edit a test to make it pass.**

**Aspect checklist (this phase)**
- *Test Discipline* — see [`testing.md`](./testing.md). Tests RED first, manual review cleared, locked before any `src/` change. No silent test edits after lock.
- *API Discipline* (§4.2) — sync + async parity present for every public method, types enforced, standalone functions.
- *Reliability & Safety* (§4.3) — typed exceptions raised from `src/upsonic/exceptions.py` (or a new typed exception added there), no bare `except`, no exception swallowing.
- *Observability & Cost* (§4.4) — telemetry hooks added where the feature emits events; cost tracking via `genai-prices` for any direct model call; `UPSONIC_TELEMETRY=False` respected.
- *Tests* (§4.5) — placement and types: `tests/unit_tests/` for logic; `tests/smoke_tests/` for I/O / external services / provider behaviour; both sync and async paths exercised. Lifecycle: see *Test Discipline* above.
- *Integration & Distribution* (§4.7) — exports updated, registry registered, optional-dep group added.
- *Performance* (§4.8) — no blocking I/O inside `async def` functions (use `httpx`/`aiohttp`, not `requests`).

**Exit gate**
- Code compiles.
- `mypy --strict` passes for the new code.
- Locked tests pass against the implementation. **No semantic test edits since the lock** (verify with `git diff` on `tests/`).
- Sync **and** async are both implemented and both tested.
- All new public symbols are exported from the relevant `__init__.py`.

---

### Phase 4 — Verify

**Entry gate**
- Phase 3 exit cleared.

**What to do**

> **Verify the test-lock invariant first.** Skim `git diff` on `tests/` since the Phase 3 lock; any non-mechanical edit (changed assertion, broadened parametrize, removed case) means the lock was broken — stop, identify the scope drift, and either re-justify in the plan or split out the change.

1. Run the full unit test suite: `uv run --all-extras pytest tests/unit_tests -v`.
2. If the feature touches Docker-backed services (Redis, Postgres, MongoDB, Mongo, Qdrant, etc.), run smoke tests: `make smoke_tests`.
3. Run pre-commit: `pre-commit run --all-files`. CI will run the same hooks; fix any failures locally.
4. Run `mypy --strict` on `src/`.
5. **Manually trace one happy-path call and one error-path call** through the new code. Walk the bytes from public entry point to underlying I/O and back. This catches integration mistakes that unit tests miss.
6. Confirm no telemetry is emitted when `UPSONIC_TELEMETRY=False` is set in the environment.
7. Verify no public API was changed in a breaking way without a deprecation note. If you removed or renamed anything users could import, you **MUST** flag it for the user.
8. Add or update Google-style docstrings on every new public symbol.
9. Add an example to `tests/doc_examples/` (or `examples/` for larger demos) for any new public API surface.
10. Consider version-bump and CHANGELOG impact. The framework uses `chore: New Version X.Y.Z` commits — check whether the change merits one.
11. **Memory hygiene (MUST, before commit).** Per [`memory.md`](./memory.md) "When and What to Save", reflect on whether this work surfaced any of the following worth carrying to the next session:
    - A user correction with a generalisable rule (saved as the rule, not the instance).
    - A confirmed non-obvious decision (saved with the *why*).
    - A discovered codebase convention non-obvious from a single file (saved with one example path).

    If yes, write a tight memory entry under the auto-memory folder before handing off to `commit.md`. If no, state explicitly: *"No memory-worthy learning from this task."* Silent skipping is not allowed.

**Aspect checklist (this phase)**
- *Memory Hygiene* — see [`memory.md`](./memory.md). End-of-workflow reflection complete; either a memory entry written or "no memory-worthy learning" stated.
- *Tests* (§4.5) — all relevant tiers green.
- *Docs & Examples* (§4.6) — public API has docstrings, an example exists, `CONTRIBUTING.md` has been updated if a new extension point was added.
- *Reliability & Safety* (§4.3) — error path manually traced.
- *Integration & Distribution* (§4.7) — new optional-dep groups installable in a fresh env, exports importable from the top-level package.

**Exit gate**
- Every item in the **Hard Gates Summary (§5)** passes.
- Ready to hand off to the commit workflow in [`commit.md`](./commit.md).

> **Hard gate.** You **MUST NOT** call the work "done", "ready", or "complete" before this gate clears. *"Tests pass on my machine"* is not the gate; *every item in §5 passes* is the gate.

---

## 4. Aspect Reference (Appendix A)

The eight aspects, in order. Each aspect is a small rule list. Generic Python rules (naming, formatting, async syntax, type-checker config) live in [`coding-standards.md`](./coding-standards.md); only Upsonic-specific calls-out are duplicated here.

### 4.1 Architecture & Subsystem Fit

- **MUST** locate the feature inside an existing subsystem under `src/upsonic/` whenever one fits. New top-level modules are rare and **SHOULD** require explicit user sign-off.
- **MUST** follow the `base.py` extension-point pattern when the subsystem has one. Do not add public-facing concrete classes that bypass the base interface.
- **MUST NOT** introduce cross-subsystem leakage — `tools/` does not import from `agent/`, `storage/` does not import from `team/`, etc. If a shared concept is needed, it lives in `utils/` or its own small module.
- **SHOULD** mirror the directory layout of the closest existing example. Symmetry across subsystems is a feature.
- *Reference patterns:* `src/upsonic/vectordb/providers/chroma.py`, `src/upsonic/models/openai.py`, `src/upsonic/storage/postgres/`, `src/upsonic/tools/common_tools/duckduckgo.py`.

### 4.2 API Discipline

- **MUST** ship sync + async parity for every public function/method that performs I/O, awaits a coroutine, or wraps an async client. Sync method has the bare name (`run`, `read`, `invoke`); async sibling is `a`-prefixed (`arun`, `aread`, `ainvoke`). See `coding-standards.md` §2.3.
- **MUST** fully type-annotate every public signature. `Any` is permitted only at third-party boundaries and **MUST** be converted to a typed model immediately inside the function.
- **MUST** keep functions standalone — dependencies as parameters, no module-level mutable state, no implicit global clients. See `coding-standards.md` §1 principle 4 (*"Determinism is a seam"*) and the Standalone Functions section.
- **MUST** model structured input and output as Pydantic models. Plain dicts are not part of the public API.
- **SHOULD** preserve the boundary between `Direct`, `Agent`, `Task`, and `Team`. Adding a method to `Team` that duplicates `Agent` behaviour is a smell; reuse via composition.
- **SHOULD NOT** add a new keyword argument to a public method without a default that preserves prior behaviour. Public surface is a contract.

### 4.3 Reliability & Safety

- **MUST** raise typed exceptions from `src/upsonic/exceptions.py`. If no existing exception fits, add one there (in the same change) rather than raising a bare `Exception` or `RuntimeError`.
- **MUST NOT** use bare `except:` or `except Exception:` without re-raising. Catching to swallow is forbidden; catching to translate to a typed framework exception is fine.
- **MUST** integrate with `src/upsonic/reliability_layer/` for any feature that affects model output quality (verifier passes, editor passes, retry-on-bad-output flows).
- **MUST** integrate with `src/upsonic/safety_engine/` for any feature that ingests user content or emits user-visible model output where a policy could apply.
- **SHOULD** define explicit timeout and retry behaviour for any new I/O call. The default `httpx`/`aiohttp` timeout is rarely the right one.
- **SHOULD NOT** silently coerce or normalize user input — fail fast with a typed exception, let the caller decide.

### 4.4 Observability & Cost

- **MUST** respect the `UPSONIC_TELEMETRY=False` environment variable. New telemetry hooks **MUST** check the flag before emitting.
- **MUST** route any direct model call through the framework's cost-tracking path so `genai-prices` records token usage. Hardcoding model names or bypassing the existing `Usage` accumulation breaks cost reporting.
- **MUST** emit Sentry-friendly error context — typed exception, no PII in messages, no token contents in logs. The Sentry SDK is already wired (`sentry-sdk[opentelemetry]`); use it.
- **SHOULD** prefer `rich` for any user-visible structured output (tables, panels, progress). Print-debugging artifacts **MUST NOT** ship.
- **SHOULD NOT** add a new telemetry sink without an opt-out. If `UPSONIC_TELEMETRY=False` is the only switch, that is the switch your code uses.

### 4.5 Tests

- **MUST** add `tests/unit_tests/<subsystem>/test_*.py` coverage for any new pure-logic public surface. Pure-logic means it can be exercised without network, disk, or a running service.
- **MUST** add `tests/smoke_tests/<subsystem>/` coverage for any feature that touches an external service, a database, an API, or a filesystem in non-trivial ways. Smoke tests use real Docker-backed services per the `make smoke_tests` workflow.
- **MUST** test sync and async paths separately — passing the sync test does not exercise the async sibling.
- **MUST** add or update an example in `tests/doc_examples/` for new public API surface so the public usage stays exercised.
- **SHOULD** prefer `tests/integration_tests/` for cross-subsystem behaviour (e.g. agent + storage + tools wired together).
- **SHOULD NOT** mock the framework boundary you are adding. Mock the third-party client at the edge; let your own code run.

### 4.6 Docs & Examples

- **MUST** add a Google-style docstring to every new public class and public function. Single-paragraph intent + Args/Returns/Raises. See `coding-standards.md` §2.7.
- **MUST** add a recipe to `CONTRIBUTING.md` whenever a new extension point is introduced (a new `base.py` interface that third parties may implement). Mirror the existing "Adding a VectorDB Provider" / "Adding a Storage Provider" recipes.
- **MUST** add at least one usage example for any new public API. Smaller examples live in `tests/doc_examples/`; larger ones (multi-file demos) live in `examples/`.
- **SHOULD** add a README or short note inside the subsystem directory if the feature changes how the subsystem is used overall.
- **SHOULD NOT** write tutorial-length prose inside docstrings — keep them tight, link out to `documents/ai/explanation/<subsystem>/` for depth.

### 4.7 Integration & Distribution

- **MUST** update the relevant `__init__.py` exports so the new public symbols are importable from the package's canonical path.
- **MUST** register new providers in their subsystem's registry (e.g. `src/upsonic/models/model_registry.py` for new model providers).
- **MUST** add new third-party dependencies under the appropriate optional-dependency group in `pyproject.toml`. Never add to the base `dependencies` list unless the package would be unusable without the new library.
- **MUST** lazy-import any optional heavy dependency inside the function or class that uses it (per `coding-standards.md` §2.4) — the minimal install stays minimal.
- **SHOULD** update the umbrella optional groups (`vectordb`, `storage`, `models`, etc.) when adding a new provider to a category that has one.
- **SHOULD NOT** introduce a singleton or process-wide registration as a side effect of import. Registration is explicit and called by the user / framework.

### 4.8 Performance

- **MUST NOT** call blocking I/O (`requests.get`, `time.sleep`, `open(...).read()` on a large file, blocking DB clients) inside an `async def`. Use `httpx`, `aiohttp`, `asyncio.sleep`, `aiofiles`, or the async-native client of your dependency.
- **MUST** support cancellation in long-running async paths. Catching `asyncio.CancelledError` to swallow it is a bug.
- **SHOULD** consider a cache when the call is on a hot path — embeddings, repeated identical model calls, retrieved documents. The framework has a `cache/` subsystem; use it instead of inventing a private cache.
- **SHOULD NOT** issue N+1 model calls when one batched call would do. Loops over user data that each invoke an LLM are a code smell.

---

## 5. Hard Gates Summary (Appendix B)

A pre-flight checklist. Use it at every phase boundary; the items below are non-negotiable without explicit user approval to skip.

```
─── Before transitioning OUT of UNDERSTAND ────────────────────────────
  [ ] Memory consulted (memory.md); findings surfaced or "no relevant entry"
  [ ] Serena consulted (serena.md); findings surfaced or "no similar prior work"
  [ ] Fuzzy brief flagged for the user to sharpen, if applicable

─── Before transitioning OUT of DESIGN ────────────────────────────────
  [ ] Public API signatures written down with full type annotations
  [ ] Sync + async parity decided (which methods need both, named correctly)
  [ ] Affected subsystem identified; no unjustified new top-level module
  [ ] At least two alternatives considered; chosen one justified in writing
  [ ] Tests planned (which tier, what they cover)
  [ ] /write-plan invoked by the agent; six-section locked plan exists
  [ ] Plan delivered via /two debate (non-trivial) or /write-plan single-shot (trivial — explicitly declared)
  [ ] Touchpoints pinned (concrete file paths, class names, service names)
  [ ] User had a chance to review the locked plan

─── Before transitioning OUT of IMPLEMENT ─────────────────────────────
  [ ] testing.md Phase 1-3 complete: scenarios locked, RED tests written, manual review cleared
  [ ] Tests locked before any src/ change; declared explicitly
  [ ] Locked tests now pass against the implementation
  [ ] No semantic test edits since the lock (verified via git diff)
  [ ] Sync + async both implemented for every public method that needs both
  [ ] mypy --strict passes on the new code
  [ ] __init__.py exports updated for every new public symbol
  [ ] Registry entries added (model_registry, etc.) where applicable
  [ ] New third-party deps added to pyproject.toml under the right extras group
  [ ] Heavy optional deps lazy-imported inside the function/class that uses them
  [ ] No blocking I/O inside any async function
  [ ] No bare `except:` or `except Exception:` swallowing errors
  [ ] No `Any` outside justified third-party boundaries

─── Before claiming the feature DONE ──────────────────────────────────
  [ ] Memory hygiene: end-of-workflow reflection complete; entry written or "no memory-worthy learning" stated
  [ ] Test-lock invariant verified — no semantic test edits since Phase 3 lock
  [ ] uv run --all-extras pytest tests/unit_tests -v passes
  [ ] make smoke_tests passes (if I/O / external services were touched)
  [ ] pre-commit run --all-files clean
  [ ] mypy --strict clean
  [ ] Public API has Google-style docstrings (Args / Returns / Raises)
  [ ] tests/doc_examples updated (or examples/) for new public API
  [ ] CONTRIBUTING.md updated if a new extension point was added
  [ ] UPSONIC_TELEMETRY=False respected by any new telemetry hook
  [ ] Manual happy-path AND error-path trace performed
  [ ] No silent breaking change to public surface (or breaking change flagged)
  [ ] Ready to hand off to commit workflow (commit.md)
```

If any box cannot be ticked, the work is not done. Naming an item N/A is allowed, but **MUST** be accompanied by a one-line reason.

---

## 6. Anti-Patterns (Appendix C)

The most common AI mistakes when adding to this framework. Each one ships a bug; recognise the shape so you do not write it.

1. **Calling `requests.get(...)` inside an `async def`.** Blocks the event loop; defeats async. Use `httpx.AsyncClient` or `aiohttp`. *(Aspect: §4.8)*

2. **Implementing only the async sibling and "leaving sync for later."** Sync + async parity is required from day one — a follow-up commit "to add the sync version" is forbidden. *(Aspect: §4.2)*

3. **Importing a heavy optional dep at module top-level.** `import chromadb` at the top of a provider file forces every user to install ChromaDB. Move it inside the function or class. *(Aspect: §4.7, §4.8)*

4. **`except Exception: pass` (or re-raising as a generic `Exception`).** Loses type information, hides bugs, breaks Sentry context. Catch typed exceptions and re-raise typed exceptions. *(Aspect: §4.3)*

5. **Adding a new top-level subsystem when the feature fits inside an existing one.** Cross-subsystem entropy is permanent; resist it. If the feature is "a new kind of retriever", it goes in `rag/` or `vectordb/`, not in a new sibling directory. *(Aspect: §4.1)*

6. **Marking the work "done" without running smoke tests after touching I/O.** Unit tests are necessary but not sufficient when external services are involved. Run `make smoke_tests` before claiming completion. *(Aspect: §4.5)*

7. **Forgetting to update `__init__.py`.** The feature ships, the test passes inside the file, but users cannot import it because nothing was exported. The exit gate of Phase 3 explicitly requires this. *(Aspect: §4.7)*

8. **Hardcoding a model name in a cost-tracked path.** Bypasses `genai-prices` registration and produces silently wrong cost reports. Route every model call through the existing usage-accumulation path. *(Aspect: §4.4)*

9. **Adding a public method without a default value for a new keyword argument.** Existing user code breaks. New params **MUST** default to behaviour-preserving values. *(Aspect: §4.2)*

10. **Mocking the framework boundary you just added.** Tests pass without exercising the real code. Mock the third-party client at the edge; let your own new code run end-to-end in tests. *(Aspect: §4.5)*

11. **Skipping the pre-work consultation (memory + Serena).** Walking into a feature without checking memory means re-discovering what the user already taught; without Serena means duplicating an existing helper. Both run before any sketch. *(Aspect: Default Pre-Work Consultation in CLAUDE.md, Phase 1)*

12. **Producing a final plan without `/write-plan`.** A plan written off-the-cuff in chat — without the six-section format produced by `/write-plan`, without Touchpoints pinned, without critique iteration where applicable — is too underspecified to drive Phase 3. `/write-plan` is mandatory; the only choice is the delivery path (`/two` debate for non-trivial, `/write-plan` single-shot for trivial). *(Aspect: Phase 2)*

13. **Writing `src/` before tests are locked.** The locked tests are the contract. Producing implementation against unlocked tests means the tests will get bent to fit the code. `testing.md` is mandatory, not optional. *(Aspect: testing.md, Phase 3)*

14. **Editing tests after the lock.** Even a "small fix" to an assertion edges the contract. If a test is wrong, scope was wrong — re-enter `testing.md` Phase 1 explicitly. Never silently edit a locked test. *(Aspect: testing.md Phase 4, this guide Phase 4)*

15. **Skipping memory hygiene at the end of the workflow.** A user correction or a confirmed non-obvious choice that doesn't get saved means next session re-debates it. End-of-workflow reflection is mandatory; either save or state "no memory-worthy learning." *(Aspect: memory.md, Phase 4)*

---

## 7. Quick Map

```
Phase 1: Understand   — memory.md + serena.md pre-work pass (MUST)
                        + flag fuzzy brief for user to sharpen
                        + read code, find pattern, identify subsystem
Phase 2: Design       — public API + types + alternatives + tests planned
                        + /write-plan invocation → six-section locked plan (MUST)
                        + delivered via /two debate (non-trivial) or single-shot (trivial)
Phase 3: Implement    — testing.md (RED → manual review → LOCK) FIRST (MUST)
                        then vertical slice in src/, sync+async pairs
Phase 4: Verify       — test-lock invariant, tests green, types, docs, manual trace
                        + memory hygiene reflection (MUST)
Then: commit.md       — propose message, wait for explicit approval, commit
```

When unsure at any point, re-read the relevant phase section and the corresponding aspect rules in §4. The hard-gates summary in §5 is the single most important page in this document.
