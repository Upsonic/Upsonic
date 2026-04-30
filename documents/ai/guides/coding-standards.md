---
title: "Coding Standards"
description: "Language and engineering standards for any Python codebase in this repository: SOLID, DRY, reusability, unification, typing, async, naming, structure, testing, and review process."
---

# Coding Standards

This document is a **pure coding standard**. It says nothing about *what* this codebase builds — no rules here describe how any particular concept must be implemented, no required flow for any specific component, no opinions about any domain object.

It only says **how code is written** in this repository: how it is named, typed, structured, composed, tested, documented, and reviewed. The same standard applies whether the file in front of you is a parser, a CLI, a database adapter, a math library, or a framework.

The thread running through every rule below: **write a thing once, type it precisely, expose it through one obvious path, and don't change the shape on users without saying so.**

---

## 1. Philosophy

Five principles take precedence over every concrete rule. When a rule below conflicts with a principle, the principle wins.

1. **Public surface is forever.** Internal code can churn freely. Anything a user can import is a contract; changing its shape is a versioned, deliberate event.
2. **One obvious path.** Each concept has exactly one home. There is one canonical way to import it, one canonical name for it, and one canonical implementation behind it. Aliases, parallel APIs, and "convenience copies" rot.
3. **Code once, don't repeat.** If two functions look the same with different names, one of them must go. If two modules solve the same problem, they must be unified. Duplication is a debt that always comes due.
4. **Determinism is a seam.** Anything non-deterministic (network, clock, randomness, third-party I/O) lives behind a clean interface so the rest of the code is unit-testable in isolation.
5. **Progressive disclosure.** The simplest call must be one line. The advanced call must be possible without subclassing, monkey-patching, or reading internal source.

---

## 2. Python Language Standards

### 2.1 Version and Tooling

- **Minimum Python**: 3.10. We rely on `match` statements, PEP 604 union syntax (`X | Y`), and `ParamSpec`.
- **Package manager**: `uv`. `pip` is not used in development. Lockfile is `uv.lock` and is committed.
- **Formatter**: `ruff format` (Black-compatible).
- **Linter**: `ruff check` with `E`, `F`, `W`, `I`, `B`, `UP`, `SIM`, `PL` enabled.
- **Type checker**: `mypy --strict` on all first-party code; specific, narrowly-scoped `# type: ignore[code]` only where a third-party import is genuinely untyped. Bare `# type: ignore` is forbidden.
- **Pre-commit**: `ruff`, `ruff format --check`, `mypy`, end-of-file fixer, trailing whitespace. CI re-runs the same hooks.

### 2.2 Formatting

- Line length: **120**. Long enough to fit a typed signature; short enough to diff in a side-by-side review.
- Indent: 4 spaces. No tabs. No mixed.
- Trailing commas in multi-line function calls and collection literals.
- One statement per line. No `if x: do_y()` one-liners except in trivial guards.

### 2.3 Naming

| Kind | Convention | Example |
| --- | --- | --- |
| Module | `snake_case` | `pipeline.py` |
| Package | `snake_case` | `text_splitter` |
| Class | `PascalCase` | `JsonStore`, `RetryPolicy` |
| Function / method | `snake_case` | `build_payload`, `compute_hash` |
| Constant | `SCREAMING_SNAKE_CASE` | `DEFAULT_TIMEOUT_S` |
| Type variable | `PascalCase`, single word | `OutputT`, `InputT` |
| Private | leading underscore | `_internal_helper` |
| Protected | single underscore prefix | `_client` |
| Boolean | `is_`, `has_`, `should_` prefix | `is_open`, `has_changes` |
| Async method (public) | `a`-prefixed | `aread`, `arun`, `aclose`, `ainvoke` |
| Sync method (public) | bare name, no suffix | `read`, `run`, `close`, `invoke` |
| Async-only iterator (no sync sibling) | `a`-prefix kept for consistency | `astream` (see §5.2 streaming exception) |
| Internal async helper | bare name, no `a`-prefix needed | `_dispatch`, `_load` |

The `a`-prefix is reserved for **public** async methods that have a sync sibling (see §5.2). Internal/private code is async-by-default and does not need the prefix — adding `a` to a function with no sync twin is noise.

Avoid abbreviations that aren't industry-standard. `cfg` and `ctx` are fine; `mngr`, `usr_msg`, `tkn_cnt` are not.

### 2.4 Imports

- Group: stdlib → third-party → first-party → local (`from .x import y`). Blank line between groups.
- `ruff isort` enforces grouping and ordering. Don't argue with it.
- **No wildcard imports** (`from x import *`) anywhere.
- **No relative imports across packages.** Within a package, relative imports (`from .types import Thing`) are fine and preferred. Between packages, always absolute.
- **Lazy-import optional heavy dependencies inside the function or class that uses them.** A minimal install must stay minimal. This applies to every optional third-party module across the codebase.

```python
def _open_redis():
    import redis  # lazy
    return redis.Redis.from_url(...)
```

### 2.5 Strings

- F-strings for interpolation. `%`-formatting and `str.format` are forbidden in new code.
- Triple-quoted strings for multi-line text. Use `textwrap.dedent` for embedded text inside indented code.
- Never build SQL, HTML, or shell commands by f-string concatenation. Use the proper escaping API.

### 2.6 Comprehensions and Generators

- Comprehensions for transforming/filtering. Loops for side-effects.
- Generator expressions when the result is iterated once and not stored.
- A comprehension that needs more than one `for` and one `if` is a `for` loop in disguise — write it as one.

### 2.7 Comments and Docstrings

- **Default to no comments.** Names and types should already say what the code does.
- Add a comment **only** when the *why* is non-obvious: a hidden invariant, a workaround for a specific bug, behaviour that would surprise a reader. Never narrate the code (`# increment counter`, `# return result`).
- Public classes and public functions: one-paragraph docstring describing intent, parameters, return type, and a short example. We use Google-style docstrings.
- No `# TODO` without a tracking issue ID and an owner.
- Never reference the current task, ticket, or PR in a code comment — that information rots.

### 2.8 Errors

- Never `except:` or `except Exception:` without re-raising or logging with the exception object.
- Never swallow an exception to "keep things going". Convert it to a typed event the caller can handle.
- `raise NewError(...) from original` to preserve the chain.
- Custom exceptions inherit from a single project-wide root so users can catch all of the codebase's errors with one class.

---

## 3. Project Structure

This section governs **how files are organised**, not what lives in them.

### 3.1 Layout Principles

- We use the `src/` layout. Everything shipped on PyPI lives under `src/<package_name>/`.
- Top-level repository scratch (`examples/`, `notebooks/`, `benchmarks/`, `scripts/`, ad-hoc `*_project/` directories, stray test files at the repo root) is **not** packaged and **must not** be imported from inside `src/`.
- Tests mirror `src/` directory-for-directory **per concept**, then split horizontally by category (e.g. `unit_tests/`, `integration_tests/`, `smoke_tests/`, `doc_examples/`). New code adds tests in the matching subfolder of each relevant category.
- Files prefixed with `_` (e.g. `_internal.py`) are private. They are not re-exported from the package's public surface; users importing them take a private dependency at their own risk.
- `py.typed` is committed. We ship type information.

### 3.2 Public vs Private API

- Anything re-exported from the top-level `__init__.py` is **public**: changing its shape is a deliberate, breaking event and must be flagged in the changelog.
- Anything else is internal. Users who import from internal paths take a dependency at their own risk.
- A leading underscore on a module name (`_internal.py`) marks it private even within its package.
- No public symbol depends on a private one across package boundaries.

### 3.3 One Obvious Path Per Symbol

- **Every component is accessible through exactly one path: its own innermost sub-module.** Re-exports up the tree are explicit, and there is no second way to import the same thing.
- Convenience aliases (`from x import Foo as Bar`) on the public API are forbidden — they create two names for one symbol and force readers to learn both.
- If the same concept appears in two places, one of them is wrong. Pick the right home and delete the other.

### 3.4 Module Cohesion

- One concept per module. A file named for one thing must not become a junk drawer for unrelated helpers.
- Files larger than ~600 lines are a smell. Split by sub-concept.
- A helper module called `utils.py` or `helpers.py` is a smell. Name it for what it does (`text_normalization.py`, `path_resolution.py`).
- Circular imports are forbidden. If you hit one, you have a layering bug — don't paper over it with deferred imports.

### 3.5 Standalone, Self-Contained Functions

- Every function and method must be **runnable as standalone as possible**. It takes everything it needs as explicit input and returns everything the caller needs as explicit output.
- No reaching into module-level mutable state. No implicit reads from globals, environment variables, or singletons inside the function body — those reads happen at the call site and the resolved values are passed in.
- A function whose behaviour depends on hidden context is a function that cannot be tested or reused.

---

## 4. Type System

### 4.1 Type Hints Are Mandatory

- Every public function, method, and attribute has full type hints on inputs, output, and class attributes.
- `Any` in a public signature requires a comment justifying it. `Any` in a private signature is permitted only when an exact type is genuinely impossible.
- Use the precise type that the value actually is. `dict` is not a type — `dict[str, int]` is.

```python
async def aload(
    self,
    key: str,
    *,
    timeout: float | None = None,
) -> bytes | None: ...

def load(
    self,
    key: str,
    *,
    timeout: float | None = None,
) -> bytes | None: ...
```

### 4.2 Pydantic at Boundaries

- Public data structures that cross a boundary or are serialised are Pydantic models with `model_config = ConfigDict(frozen=True, extra="forbid")` unless mutation is genuinely required.
- Internal data structures can be dataclasses or plain classes; Pydantic is for data that crosses a public boundary.
- Use `Field(...)` for descriptions when the field is user-facing — they show up in generated schemas.

### 4.3 Generics

- Generic classes and functions use `TypeVar` with `bound=` or `default=` (PEP 696) where it tightens the contract.
- `ParamSpec` for decorators that wrap callables.
- Single-letter `T` is acceptable only when there is exactly one type parameter and its meaning is obvious. Otherwise use `OutputT`, `InputT`, etc.

### 4.4 Discriminated Unions for Variant Types

When a value can be one of several shapes, model it as a discriminated union of typed Pydantic models, not as a generic dict or a string tag:

```python
from typing import Annotated, Literal
from pydantic import BaseModel, Field

class Created(BaseModel):
    type: Literal["created"] = "created"
    id: str

class Updated(BaseModel):
    type: Literal["updated"] = "updated"
    id: str
    revision: int

class Deleted(BaseModel):
    type: Literal["deleted"] = "deleted"
    id: str

ChangeEvent = Annotated[
    Created | Updated | Deleted,
    Field(discriminator="type"),
]
```

This gives callers `match event:` exhaustiveness and stops generic dicts from leaking through the public surface.

### 4.5 `None`, `Optional`, Sentinels

- `X | None` is the canonical "absent" type. Don't use `Optional[X]` in new code.
- For "unset vs explicitly None" semantics, use a private sentinel (`_UNSET = object()`) and document it. Never use a magic string.

### 4.6 `typing.Protocol` over Inheritance

Prefer structural typing for plugin points. It avoids forcing users to inherit from a framework class to implement an interface.

```python
class KeyValueBackend(Protocol):
    async def aget(self, key: str) -> bytes | None: ...
    async def aput(self, key: str, value: bytes) -> None: ...
```

---

## 5. Async Programming

### 5.1 Async-First, Always

This repository is **async-first**. The canonical implementation of any I/O-bound method is `async`. Sync versions exist as **thin wrappers** around the async core, never the other way around.

- All I/O is `async` underneath. HTTP, file, DB — all of it.
- New methods are written as `async` first. The sync sibling is written afterwards and delegates to the async version.
- No `time.sleep` in library code; use `asyncio.sleep`.
- No `requests`; use `httpx` or the relevant async client.
- No blocking calls inside an async function. If you must call blocking code, wrap it with `asyncio.to_thread`.

### 5.2 Dual Public API: Sync **and** Async

**Every public method that performs work has both a sync and an async version.** This is non-negotiable for the public API surface across the entire codebase.

Naming convention (see §2.3):

- **Async** version → `a`-prefixed: `aread`, `arun`, `astream`, `aclose`, `ainvoke`.
- **Sync** version → bare name: `read`, `run`, `stream`, `close`, `invoke`.

The async version is the source of truth. The sync version delegates:

```python
class Worker(Generic[OutputT]):
    async def arun(
        self,
        payload: Payload,
        *,
        timeout: float | None = None,
    ) -> Result[OutputT]:
        """Canonical async implementation."""
        ...

    def run(
        self,
        payload: Payload,
        *,
        timeout: float | None = None,
    ) -> Result[OutputT]:
        """Sync wrapper over :meth:`arun`."""
        return _run_sync(self.arun(payload, timeout=timeout))
```

Rules for the sync wrapper:

- Use a single shared helper (e.g. `_run_sync(coro)`) that handles "no running loop" and "already inside a loop" correctly. The wrapper is the **only** place in the codebase that calls `asyncio.run` or equivalent.
- Inside an already-running event loop (e.g. Jupyter), the sync wrapper either runs on a worker thread with its own loop, or raises a clear error directing the user to the `a`-prefixed method. Never silently call `asyncio.run` in a running loop — that is a `RuntimeError`.
- The sync wrapper must not duplicate logic, retry policy, hooks, or error mapping. If you find yourself copying anything beyond the `await`, you are doing it wrong.
- Type signatures, parameter names, defaults, docstrings, and exception classes are **identical** between the sync and async versions. The only difference is `async def` and the `a`-prefix on the name.

What this rule applies to:

- Every public class method that performs work.
- Anything imported from the package's top-level `__init__`.
- Lifecycle hooks the **user calls**: `close`/`aclose`, `connect`/`aconnect`.

What this rule does **not** apply to:

- Internal helpers and private methods — those are async-only and do not need an `a`-prefix.
- Callback protocols the **user implements**. Those remain async-only because they are awaited inside the loop; offering a sync version would invite blocking the loop.
- Streaming iterators where a sync version would silently buffer the whole stream. If a sync `stream` cannot avoid degenerating into "fetch all then yield", document the limitation and consider raising on misuse.

### 5.3 Don't Mix Colors

- Async functions stay async all the way down. Don't bury `asyncio.run` inside library code — it breaks nested loops, notebooks, and Jupyter. The sync wrapper helper from §5.2 is the only sanctioned exception.
- Never call `asyncio.get_event_loop()` — it is deprecated and broken in modern Python. Use `asyncio.get_running_loop()` only inside an async context.
- Internal code calls the async version. The sync wrapper is for users, not for us — never call the sync sibling from within the codebase when the async version is available.

### 5.4 Cancellation and Timeouts

- Every external call has a timeout. No exceptions.
- Use `asyncio.timeout()` (Python 3.11+) for cancellation; it propagates cleanly.
- A cancelled task must actually stop. If your code catches `asyncio.CancelledError`, you must re-raise it after cleanup.

```python
async def acall(self, request: Request) -> Response:
    try:
        async with asyncio.timeout(self._timeout_s):
            return await self._client.send(request)
    except asyncio.CancelledError:
        await self._cleanup(request)
        raise
    except asyncio.TimeoutError as e:
        raise OperationTimeoutError(f"call to {request.target} timed out") from e
```

### 5.5 Concurrency Primitives

- Use `asyncio.TaskGroup` (Python 3.11+) over bare `asyncio.gather` — it propagates exceptions deterministically.
- For producer/consumer streams, use `asyncio.Queue` with a `maxsize` to give backpressure.
- For shared state across coroutines, prefer immutability over locks. If you need a lock, document the invariant it guards.

---

## 6. SOLID Principles

SOLID is enforced as a coding standard across every module. None of the rules below depend on what the module does.

### 6.1 Single Responsibility

A class or function does **one** thing. If you can describe it only by saying "and", split it.

- A function name with `and` in it is a refactor waiting to happen.
- A class with two unrelated state machines is two classes pretending to be one.

### 6.2 Open / Closed

Code is **open for extension, closed for modification**.

- New behaviour is added by introducing a new implementation of an existing interface, not by editing the existing implementation with a new branch.
- If adding a feature requires you to grow an `if`/`elif` ladder, you are missing an abstraction.

### 6.3 Liskov Substitution

Any implementation of an interface is substitutable for any other implementation of that interface, with no loss of guarantees.

- A subclass that strengthens preconditions or weakens postconditions is broken.
- "It works as long as you don't pass X" is not substitutable.

### 6.4 Interface Segregation

Prefer many small, focused interfaces over one large interface that everything must implement.

- Don't define a single `IComponent` interface that forces every implementer to implement methods it doesn't use.
- Split by capability: a thing that only reads should not be forced to implement `write`.

### 6.5 Dependency Inversion

High-level code depends on abstractions, not on concrete implementations.

- Constructors take interfaces (or `Protocol`s), not concrete classes.
- Concrete wiring happens at the edge of the program (entry points, factories), never deep inside business logic.
- This is what makes everything in §13 testable without networking.

---

## 7. DRY, Reusability, Unification

### 7.1 Write It Once

- If the same logic appears in two places, extract it. There is one canonical implementation, in one canonical home, called from both sites.
- Copy-pasting "for now" is permanent unless you write the cleanup PR before merging the duplicate.

### 7.2 Unify Related Things

- Two functions that do the same job with different names are one function with two callers using the wrong name.
- Two modules that solve the same problem with different APIs are a missing abstraction. Pick one shape, migrate, and delete the other.
- Constants, enums, and configuration knobs that describe the same idea live in **one** place.

### 7.3 Stable Surface, Churning Internals

- The contract a user calls is written once and changed deliberately.
- The implementation behind that contract is free to be rewritten at any time as long as the contract holds.
- Refactoring inside the boundary is encouraged. Refactoring across the boundary is a versioned event.

### 7.4 No Speculative Code

- Build only what is used. Methods, parameters, attributes, hooks, and configuration knobs that no caller in this repository or its documented users will use must not exist.
- Speculative generality is dead weight. It complicates types, drags on tests, and confuses readers about which path is real.
- "We might need this later" is not a reason to ship code today.

---

## 8. State Management

### 8.1 Immutability by Default

- Public data structures are frozen Pydantic models. Updates produce new objects (`obj.model_copy(update={...})`).
- An immutable object can be safely shared across coroutines without locking.
- Mutability is permitted internally where it pays for itself in performance, and only behind an interface that hides it.

### 8.2 No Module-Level Mutable State

- No global registries you can't override per-instance.
- Resources (clients, connection pools, caches) live on an instance, not at module scope.
- Singletons are a smell. If you genuinely need one, hide it behind a function that returns it and make that function injectable for tests.

### 8.3 Explicit Lifetimes

- Anything that owns a resource exposes both `close` and `aclose` (§5.2) and supports both `with` and `async with`.
- A resource that is never closed is a resource leak waiting for production traffic.

---

## 9. Error Handling

### 9.1 One Root Exception

- The codebase has a single root exception. Every custom exception inherits from it. Users can catch all of our errors with one class.
- Custom exceptions form a typed hierarchy, not a wall of unrelated classes.

### 9.2 Retry Is Policy

- Retry, backoff, and jitter are policies expressed as data, not control flow scattered across the call stack.
- Retry only on the exception classes whose retry semantics you have explicitly modelled. Never catch `Exception` and retry — you will retry the wrong things and burn time.
- A retry is a logged event, not a silent recovery.

### 9.3 Construction vs Operation

- Construction errors (bad config, missing required argument) raise immediately at construction time. Users should be able to instantiate an object and assert on it without performing I/O.
- Operation errors raise during the operation, not before.

### 9.4 Error Messages

- The message says **what** failed and **what to do next**, not just **what was wrong**.
- Bad: `"Invalid value"`.
- Good: `"Unknown backend 'sqlitex'. Use one of: sqlite, postgres, redis."`

---

## 10. Configuration

### 10.1 Sources, In Priority Order

1. Explicit constructor arguments.
2. Environment variables.
3. Hard-coded defaults.

No config files are read implicitly. If a user wants `.env`, they call `dotenv` themselves.

### 10.2 Defaults

- Every parameter has a sensible default. The 80% case must work without configuration.
- Defaults are documented in the parameter docstring **and** echoed in `repr()`.
- Defaults that drift over time (anything tied to fast-moving external systems) are marked with a clear comment and are reviewed each release.

### 10.3 Optional Extras

Heavy or rarely-used dependencies live behind PEP 621 optional extras. A user installing the package with no extras must end up with a minimal, working install.

---

## 11. Logging

### 11.1 One Logger, Stdlib Only

- We use the stdlib `logging` module via a single named logger per package: `logging.getLogger("<package_name>")`.
- Default level is `WARNING`. The library never reconfigures the root logger.
- Levels:
  - `DEBUG`: full payloads, full request/response bodies. Off by default. May contain PII.
  - `INFO`: lifecycle events (started, completed, retried). No content.
  - `WARNING`: degraded paths, deprecated API used, truncated output.
  - `ERROR`: a operation failed terminally.

### 11.2 Redaction

- API keys, tokens, and secrets are redacted before logging. The HTTP client redacts `Authorization`, `x-api-key`, and any header containing `key`/`token`/`secret`.
- Tests assert that secrets never appear in any captured log line.

### 11.3 Hooks Don't Block

- Where the codebase exposes user-implemented async callbacks, they may not raise and may not block the loop. A callback that raises is logged and dropped — it does not break the calling code.

---

## 12. Configuration of Dependencies

### 12.1 Lazy Imports for Optional Deps

- Optional third-party modules are imported inside the function or class that needs them, never at module top-level. A user who never touches that feature must never pay its import cost.
- This applies regardless of which package the optional dep belongs to.

### 12.2 Connection Pooling

- Reusable clients (HTTP, DB) are created once per owning instance and shared. Don't create a client per request.
- Clients are closed on `close`/`aclose`.

---

## 13. Testing

### 13.1 Test Layout

The `tests/` directory contains exactly **four** top-level test folders. Every test in the repository belongs to one of them:

```
tests/
├── unit_tests/          # pure logic, no I/O, no network
├── integration_tests/   # code under test wired to recorded fakes for its external deps
├── smoke_tests/         # broader behaviour checks; some require real infra (Docker, real services)
└── doc_examples/        # the runnable snippets shown in docs/, executed as tests
```

There are no other test folders. Anything that doesn't fit one of the four belongs in one of the four anyway — pick the closest match and put it there. New folders next to these four are not added.

Within each top-level folder, the layout **mirrors `src/` directory-for-directory per concept**. A new feature's tests usually land in the matching subfolder under each relevant top-level folder.

### 13.2 The Four Folders

- **`unit_tests/`** — Pure functions and pure logic. No network, no disk I/O beyond temporary files, no external services, no clocks. The default home for any test that can be written without I/O. Runs on every commit.
- **`integration_tests/`** — The code under test wired to recorded fakes for every external dependency it touches. Verifies that components compose correctly without hitting the network. Recordings live alongside the tests. Runs on every commit.
- **`smoke_tests/`** — Broad behaviour checks across larger surfaces. Some tests in here legitimately require real infrastructure (Docker-managed databases, real third-party services, real credentials). The folder ships its own `docker-compose.yml` and `DOCKER_SETUP.md` for the dependencies it needs. Smoke tests that need credentials are gated on the relevant env vars and skipped when unavailable.
- **`doc_examples/`** — Tests that execute the snippets shown in `docs/`. A snippet that ships in user-facing documentation must execute as a test. If a doc page changes, the matching `doc_examples/` test changes with it.

### 13.3 Mock at the Boundary, Not Inside

- Tests mock at the **outermost adapter** boundary. They do not patch random functions inside the implementation.
- Recorded fakes (`pytest-recording` / `vcr.py` style) live next to the integration test that uses them.
- A new test that needs a recording records once against a real dependency, then commits the recording. The recording is the contract.

### 13.4 Test Discipline

- No `time.sleep` in a test. Use fake clocks (`freezegun`) or an injected clock.
- No real network in `unit_tests/` or `integration_tests/`. Real network is allowed only in `smoke_tests/` and only when the test is explicitly gated on credentials.
- No `print` in a test. Use `caplog` for log assertions.
- A flaky test is disabled and tracked, not retried until green.

### 13.5 Coverage

- Public API: 90%+ line coverage, 80%+ branch.
- Internal modules: best-effort. Coverage is not a goal; behaviour is.
- A bug fix lands with a regression test pinned to a recorded scenario in `unit_tests/` or `integration_tests/`.

---

## 14. Security

### 14.1 Secrets

- Credentials come from env vars or constructor args. Never from disk by default.
- Logs and exception messages are redacted. Tests assert that secret values never appear in any captured log.
- `repr()` of a client never includes the secret.

### 14.2 PII

- Default log level (`INFO`) emits no message content.
- `DEBUG` logs are explicit about possibly containing PII. The level is documented in the README and in this standard.
- The codebase does not call analytics or telemetry endpoints by default. Any opt-in mode must be explicit.

### 14.3 SSRF / URL Fetching

- Any code that fetches a URL on behalf of an untrusted input has an SSRF guard: no `localhost`, no link-local, no private ranges, by default. Override with an explicit flag.

---

## 15. Performance

### 15.1 The Event Loop Is Sacred

- No blocking call in async code. CI runs a watchdog that fails on a 100ms+ block.
- CPU-heavy work goes to a process pool, not the loop.

### 15.2 Cache Reference Data Once

- Reference data (capability tables, pricing tables, language tables, anything loaded from disk at import) is loaded once and cached at the instance level, not recomputed per call.

### 15.3 Stream When You Can

- For long results, prefer streaming over buffering the entire response. Buffering the full response before showing anything is jarring and wastes memory.

---

## 16. Documentation

### 16.1 Public Docstrings

Every public class and function: one-paragraph intent, parameters with units, return type, one short example. Google style.

```python
async def aload(
    self,
    key: str,
    *,
    timeout: float | None = None,
) -> bytes | None:
    """Load a value from the store (async, canonical implementation).

    Args:
        key: Lookup key.
        timeout: Maximum wall-clock time in seconds. ``None`` means no
            timeout from this layer.

    Returns:
        The stored bytes, or ``None`` if the key is not present.

    Raises:
        OperationTimeoutError: If ``timeout`` is exceeded.

    Example:
        >>> data = await store.aload("user:42")
    """

def load(
    self,
    key: str,
    *,
    timeout: float | None = None,
) -> bytes | None:
    """Load a value from the store (sync wrapper over :meth:`aload`).

    Identical semantics, parameters, and exceptions as :meth:`aload`. Prefer
    :meth:`aload` from async code; use this only from synchronous call sites.
    """
    return _run_sync(self.aload(key, timeout=timeout))
```

Both docstrings are required. The sync docstring may defer to the async version, but it must exist — users (and tooling) read whichever they call.

### 16.2 Conceptual Docs

- Live in `docs/`. Format is MDX.
- Every conceptual page has a runnable example. We test that the examples still execute on each release.
- A new public feature is not "done" until its docs page is merged.

### 16.3 Changelog

- `CHANGELOG.md`, kept by hand, grouped under "Added / Changed / Deprecated / Removed / Fixed / Security".
- Every user-visible change has an entry. Internal refactors do not.
- Each entry links the PR.

---

## 17. Dependencies

### 17.1 Adding a Dependency

A new runtime dependency requires:

- A specific need that can't be solved with stdlib in `< 100` lines.
- Active maintenance (commit in last 12 months).
- Compatible license (MIT, Apache 2.0, BSD).
- A pinned version range.
- An entry in the relevant optional extra if it's heavy.

### 17.2 Dev Dependencies

`uv` dev-dependencies for `pytest`, `pytest-asyncio`, `pytest-recording`, `hypothesis`, `ruff`, `mypy`, `pre-commit`. These are not in any user-facing extra.

### 17.3 Lockfile

- `uv.lock` is committed.
- CI runs `uv sync --locked` to verify the lockfile is up to date with `pyproject.toml`.

---

## 18. Bounded Concurrency

These rules apply whenever new code fans out, loops, or recurses.

### 18.1 Fan-out

- Use `asyncio.TaskGroup` for any concurrent fan-out. Bare `asyncio.gather` is forbidden in new code.
- A failure in one task cancels the group. Partial successes are surfaced as typed events on the result, never silently dropped.

### 18.2 Loops Bounded by External Output

- Any loop whose termination depends on data from an external source declares a hard step cap. No exceptions.
- Hitting the cap raises a typed exception carrying the partial result. The caller, not the inner loop, decides whether to resume.

### 18.3 Recursion

- Recursion depth is bounded by an explicit limit, never by Python's `RecursionError`.
- Recursive entry into a public function goes through the same caps as the outer call.

---

## 19. Code Review Checklist

A reviewer signs off only when they can answer **yes** to every relevant item:

**Types and Public Surface**
- [ ] No `Any` introduced in a public signature without justification.
- [ ] New public types are Pydantic models with `extra="forbid"` and `frozen=True` where appropriate.
- [ ] Re-exports updated if a new public symbol was added.
- [ ] The new symbol is reachable through exactly one import path.

**Async**
- [ ] No `time.sleep`, `requests`, or `asyncio.run` in library code (except the single sync-wrapper helper).
- [ ] Every external call has a timeout.
- [ ] `CancelledError` is re-raised after cleanup.
- [ ] Async method is the canonical implementation; sync sibling delegates to it.

**Dual API (Sync + Async)**
- [ ] Every new public method has both an `a`-prefixed async version **and** a bare-name sync version.
- [ ] Async name uses the `a`-prefix (`arun`, `aread`, `astream`, `aclose`); sync name is the bare form (`run`, `read`, `stream`, `close`).
- [ ] Sync version is a thin wrapper — no logic, retry policy, or error mapping is duplicated.
- [ ] Signatures, defaults, exceptions, and docstrings match between the two versions.
- [ ] Internal callers always `await` the async version, never the sync wrapper.
- [ ] Behaviour inside an already-running event loop is documented and does not silently call `asyncio.run`.

**Architecture**
- [ ] No new module-level mutable state.
- [ ] No new global singleton without an injection seam.
- [ ] No new `utils.py` / `helpers.py` junk drawer.
- [ ] No duplicated logic that could collapse into one implementation.
- [ ] Optional third-party imports are lazy.

**Errors**
- [ ] New exceptions inherit from the project root exception.
- [ ] No bare `except:` or `except Exception:` swallowing.
- [ ] Error messages name the next action, not just the failure.

**Tests**
- [ ] Unit tests for new pure logic.
- [ ] Integration test with a recorded fake for new code paths that touch external systems.
- [ ] Regression test for any bug fix.
- [ ] No new `time.sleep` in tests.

**Docs**
- [ ] Public docstrings updated.
- [ ] CHANGELOG entry under the right section.
- [ ] Conceptual MDX page updated if user-visible behaviour changed.

---

## 20. Anti-Patterns (Hall of Shame)

Patterns we have done before and will not do again. If you see one, fix it.

1. **Two import paths for one symbol.** One canonical home; the rest are deleted.
2. **Convenience aliases on the public API** (`from x import Foo as Bar`). Pick one name and keep it.
3. **A helper module called `utils.py` or `helpers.py`.** Name it for what it does.
4. **A new global singleton.** Make it injectable.
5. **`# TODO: fix this`** without an issue link.
6. **A flag that is silently ignored on some implementations.** Either honour it or raise.
7. **A subclass added to inject one extra parameter.** Compose, don't inherit.
8. **Catching `Exception` to "make it more robust".** It hides bugs.
9. **`asyncio.run` deep in library code.** It breaks Jupyter and nested loops.
10. **Stringly-typed variant data.** Use a discriminated union.
11. **A test that passes only when the network is up.** Record the transcript.
12. **Logging full payloads at INFO level.** That's PII. Move it to DEBUG.
13. **A migration path that requires users to read a blog post.** It belongs in the deprecation warning.
14. **A public async method shipped without a sync sibling.** Both versions are mandatory — see §5.2.
15. **Naming an async method without the `a`-prefix when a sync sibling exists.** Use `arun`/`run`, never `run_async`/`run` or `run`/`run_sync`.
16. **The sync version reimplements the async logic instead of awaiting it.** Code drift is guaranteed; bugs land in only one half.
17. **Internal code calls the sync wrapper.** Internal callers always `await` the async version.
18. **An `a`-prefix on a private/internal helper that has no sync twin.** The prefix is reserved for public methods with a sync sibling.
19. **A function that reaches into module globals or env vars instead of taking them as input.** Pass them in; resolve at the call site.
20. **Speculative parameters / methods / hooks no caller uses.** Delete them. Build only what is used.

---

## 21. Living Document

This standard is wrong somewhere — every standard is. When you find the place it's wrong, propose a change with a PR. The diff against this file is the change. Don't ship a parallel "RFC" doc that nobody reads.

The standard becomes binding the day a CI rule enforces it. Until then, it's a guideline. Move rules from "guideline" to "enforced" as quickly as the tooling allows.
