---
name: persistent-text-canvas
description: Use when working with Upsonic's file-backed, LLM-edited text canvas subsystem under `src/upsonic/canvas/`. Use when a user asks to give an agent a persistent scratchpad, draft long-form documents iteratively, expose `get_current_state_of_canvas` / `change_in_canvas` as agent tools, configure the canvas editor model, debug canvas filename sanitisation, or wire a `Canvas` into `Agent` or `AutonomousAgent`. Trigger when the user mentions Canvas, canvas_name, change_in_canvas, get_current_state_of_canvas, Empty Canvas, Canvas Editor, persistent text canvas, agent scratchpad, deep agent canvas, file-backed document, _clean_canvas, _save_canvas, or `upsonic.canvas`.
---

# `src/upsonic/canvas/` — Persistent, LLM-Edited Text Canvas

## 1. What this folder is

The `canvas` package implements a tiny, focused subsystem inside Upsonic: a **persistent text canvas** that an `Agent` (or any other Upsonic component) can read from and write to as if it were a shared scratchpad. Unlike a simple in-memory string, the canvas is materialized on disk as a `.txt` file named after the canvas, and modifications go through an LLM "editor" pass so callers can describe edits in *natural language* ("replace the section about pricing", "append a conclusion") rather than computing string diffs themselves.

In Upsonic's architecture, the canvas is a *first-class agent capability*. When an `Agent` is instantiated with a `canvas=Canvas(...)` argument, the agent's `_register_agent_tools` method automatically harvests the canvas's exposed functions and registers them with the agent's `ToolManager`. The agent can therefore call `get_current_state_of_canvas()` and `change_in_canvas(...)` mid-run via normal tool-calling, exactly as it would call any other tool. This makes the canvas useful for long-running tasks (drafting documents, iterative writing, plan refinement, "deep agent" workflows) where the LLM needs a stable, mutable workspace that survives across turns.

The folder is intentionally small — just two Python files (`__init__.py` and `canvas.py`) — and its public surface is a single class, `Canvas`. The class composes with the rest of Upsonic by referring to `upsonic.models.infer_model` for model resolution and by lazily importing `upsonic.Task` and `upsonic.agent.agent.Agent` at edit-time (rather than at module-import time) to avoid circular dependencies.

The canvas pattern is one of Upsonic's "agent-state primitives" — comparable in spirit to memory and knowledge bases, but scoped to *free-form authored text* that the agent itself owns and rewrites. It is not a database, not a vector store, and not message history; it is a single mutable document keyed by name.

## 2. Folder layout

```
src/upsonic/canvas/
├── __init__.py     # Lazy re-export of the Canvas class
└── canvas.py       # The Canvas class (file-backed, LLM-edited text canvas)
```

## 3. Top-level files

### 3.1 `__init__.py`

A pure re-export module that uses **PEP 562 module-level `__getattr__`** for lazy loading. Importing `Canvas` directly from `upsonic.canvas` does not pay the cost of importing the implementation file (which transitively touches `upsonic.models`) until the symbol is actually accessed.

| Symbol | Kind | Purpose |
| --- | --- | --- |
| `TYPE_CHECKING` block | import | Imports `Canvas` only for static type checkers, so IDEs and `mypy` see the symbol without runtime import cost. |
| `__getattr__(name: str) -> Any` | module function | Implements lazy attribute access. When `name == "Canvas"`, it imports `from .canvas import Canvas` and returns the class. Any other attribute name raises `AttributeError` with a hint to import from a sub-module. |
| `__all__` | module attribute | Set to `['Canvas']`, controlling `from upsonic.canvas import *` semantics and signalling the public API. |

Net effect: `from upsonic.canvas import Canvas` works as expected, but module load is deferred to first use.

### 3.2 `canvas.py`

The full implementation of the canvas primitive. It depends on:

- `re` — used for sanitising the canvas name and stripping fenced code blocks from the canvas text.
- `typing.Optional`, `typing.Any` — type hints.
- `upsonic.models.infer_model`, `upsonic.models.Model` — for resolving the LLM that will perform edits.
- (Lazy, function-local) `upsonic.Task` and `upsonic.agent.agent.Agent` — used inside `change_in_canvas` to drive the edit.

#### Class: `Canvas`

A file-backed text canvas that delegates non-trivial edits to an LLM "editor" agent.

| Attribute | Type | Meaning |
| --- | --- | --- |
| `canvas_name` | `str` | Logical name of the canvas. Used to derive the on-disk filename (`<sanitised_name>.txt`). |
| `model` | `Model` | Resolved model instance used for edit operations. Defaults to `openai/gpt-4o-mini` when none is supplied. |

##### `__init__(self, canvas_name: str, model: Optional[Model | str] = None)`

Constructs the canvas.

1. Stores `canvas_name`.
2. Resolves the model:
   - If `model is None`, calls `infer_model("openai/gpt-4o-mini")`.
   - Else if `model` is a string (e.g. `"anthropic/claude-3-5-sonnet"`), calls `infer_model(model)`.
   - Otherwise assumes `model` is already a `Model` instance and uses it as-is.
3. Calls `_clean_canvas()` to immediately scrub any fenced code blocks present in the existing on-disk file. This is a defensive cleanup: if the previous run wrote a canvas that contained markdown fences (which the LLM is told *not* to emit), they are removed on startup.

##### `_clean_canvas(self) -> None`

Reads the current canvas, removes any triple-backtick fenced blocks (` ```...``` `, multiline) using `re.sub(r'```.*?```', '', canvas_text, flags=re.DOTALL)`, and writes the result back via `_save_canvas`. This is invoked on every `Canvas` construction and ensures canvas text remains plain prose.

##### `_save_canvas(self, canvas_text: str) -> None`

Persists `canvas_text` to disk.

- Filename is derived by `re.sub(r'[^\w\s-]', '', self.canvas_name).strip().replace(' ', '_')` then suffixed with `.txt`. This strips anything that is not a word char, whitespace, or hyphen, then converts internal whitespace to underscores. So `Canvas("My Project!")` writes to `My_Project.txt`.
- Opens the file in `'w'` mode with `encoding='utf-8'` and writes the entire content. The previous content is overwritten.

##### `_load_canvas(self) -> str`

Reads the on-disk canvas using the same filename derivation.

- On `FileNotFoundError` (canvas has never been saved), returns the empty string `""`.
- Otherwise returns the full file content.

##### `get_current_state_of_canvas(self) -> str`

The first of the two **public, agent-callable** functions. Wraps `_load_canvas()` and substitutes the literal string `"Empty Canvas"` whenever the file does not exist or is empty. This is intentional: when the LLM editor sees `"Empty Canvas"` it knows to write fresh content rather than try to *modify* nothing. The string also makes the value safe to drop directly into a prompt without ambiguity.

##### `async change_in_canvas(self, new_text_of_part: str, part_definition: str) -> str`

The second public, agent-callable function and the heart of the canvas. Instead of computing diffs in code, it asks an LLM to perform the edit.

Algorithm:

1. Lazily imports `upsonic.Task` and `upsonic.agent.agent.Agent` to avoid a circular import at module load time (`agent.py` itself imports `Canvas` for typing).
2. Instantiates a one-off `Agent(model=self.model, name="Canvas Editor")`.
3. Reads the current canvas via `get_current_state_of_canvas()`.
4. **Fast path — empty canvas:** if the current state is `"Empty Canvas"` or `""`, it writes `new_text_of_part` straight to disk without calling the LLM and returns it. This avoids paying for an LLM call on the first write.
5. **Edit path:** builds a structured prompt asking the editor to either:
   - Replace the line/section that contains `part_definition` with exactly `new_text_of_part`, or
   - If no such section exists, append `new_text_of_part` as a new section at the end.
   - Crucially, the prompt forbids explanations, code blocks, or formatting and demands *only* the complete updated document.
6. Wraps the prompt in a `Task`, awaits `agent.do_async(task)`, reads `task.response`, saves it via `_save_canvas`, and returns the new canvas text.

This means `change_in_canvas` is the single mutation entry point for callers and uses the LLM as a fuzzy, semantic find-and-replace engine. Because the prompt is anchored on a `part_definition` string (a human-supplied marker), the LLM can find the right region even when the section header text is paraphrased.

Note: the function is `async`. When the canvas is exposed as an agent tool, Upsonic's tool dispatch supports both sync and async callables, so the agent will simply `await` it.

##### `functions(self) -> list[Callable]`

Returns `[self.get_current_state_of_canvas, self.change_in_canvas]`. This is the integration hook that `Agent._register_agent_tools` uses to add the canvas to the agent's tool registry. Each element is a *bound method* of the specific `Canvas` instance, so the agent's tool calls go through the right canvas without any extra wiring.

## 4. Subfolders

There are no subfolders. The canvas package is intentionally flat — a deliberate design choice given that its public API is a single class.

## 5. Cross-file relationships

Within the package, the relationship is trivial: `__init__.py` re-exports `Canvas` from `canvas.py`. There is no internal registry, no plugin system, no abstract base class — just one class and one re-export shim.

The interesting relationships are *outward*:

| Direction | Module | Nature of dependency |
| --- | --- | --- |
| **Outbound** | `upsonic.models.infer_model`, `upsonic.models.Model` | Top-level import in `canvas.py`. Resolves a model spec string (e.g. `"openai/gpt-4o-mini"`) to a `Model` instance and provides the `Model` type. |
| **Outbound (lazy)** | `upsonic.Task` | Imported inside `change_in_canvas` so that constructing the editor task does not couple module-load order. |
| **Outbound (lazy)** | `upsonic.agent.agent.Agent` | Imported inside `change_in_canvas` to spin up an ephemeral editor agent per edit. |
| **Inbound** | `upsonic.agent.agent.Agent` (`src/upsonic/agent/agent.py`) | Accepts `canvas: Optional["Canvas"] = None` in its constructor (line 259), stores it as `self.canvas` (line 496), and harvests `self.canvas.functions()` inside `_register_agent_tools` (lines 1878-1883) so the canvas's two methods become regular tools. |
| **Inbound** | `upsonic.agent.autonomous_agent.autonomous_agent.AutonomousAgent` (`src/upsonic/agent/autonomous_agent/autonomous_agent.py`) | Accepts a `canvas` kwarg (line 126) and forwards it down to the underlying `Agent` (line 316), so prebuilt autonomous agents can also opt into a canvas. |

Both inbound users import `Canvas` only under `TYPE_CHECKING`, which complements the lazy-loading `__init__.py` and avoids any import cycles.

## 6. Public API

What you import from this folder:

```python
from upsonic.canvas import Canvas
# or, equivalently:
from upsonic.canvas.canvas import Canvas
```

### Public methods on `Canvas`

| Method | Async? | Returns | Used by |
| --- | --- | --- | --- |
| `Canvas(canvas_name, model=None)` | n/a | `Canvas` | User code |
| `get_current_state_of_canvas()` | sync | `str` (current canvas text or `"Empty Canvas"`) | User code, **Agent (auto-registered tool)** |
| `change_in_canvas(new_text_of_part, part_definition)` | **async** | `str` (new canvas text after edit) | User code, **Agent (auto-registered tool)** |
| `functions()` | sync | `list[Callable]` of the two public methods above | `Agent._register_agent_tools` |

### Internal (underscore) helpers

These are not part of the public API but are documented here for completeness:

| Method | Purpose |
| --- | --- |
| `_clean_canvas()` | Strip fenced code blocks from current canvas content. |
| `_save_canvas(canvas_text)` | Write text to `<sanitised_name>.txt`. |
| `_load_canvas()` | Read `<sanitised_name>.txt`, return `""` if missing. |

## 7. Integration with the rest of Upsonic

```
                +-------------------------------+
                |   user code                   |
                |   Canvas("draft", model="...")|
                +---------------+---------------+
                                |
                                v
   +-------------------------------------------------+
   |  upsonic.canvas.Canvas                          |
   |  - get_current_state_of_canvas()                |
   |  - async change_in_canvas(new, part_def)        |
   |  - functions()  -> [get..., change_in_...]      |
   +-----------+--------------------+----------------+
               |                    |
               | uses               | tools harvested by
               v                    v
   +----------------------+   +-------------------------------+
   | upsonic.models       |   | upsonic.agent.agent.Agent     |
   |  - infer_model(spec) |   |  - canvas: Optional[Canvas]   |
   |  - Model             |   |  - _register_agent_tools()    |
   +----------------------+   |    appends canvas.functions() |
                              |    to ToolManager registry    |
                              +---------------+---------------+
                                              |
                                              v
                              +-------------------------------+
                              | autonomous_agent /            |
                              | deepagent / ralph workflows   |
                              | pass `canvas=` through to     |
                              | Agent and gain canvas tools.  |
                              +-------------------------------+
```

### Specific touch-points

- **`src/upsonic/agent/agent.py`**
  - Line 81: `from upsonic.canvas.canvas import Canvas` (under `TYPE_CHECKING`).
  - Line 259: `canvas: Optional["Canvas"] = None` parameter in `Agent.__init__`.
  - Line 496: `self.canvas = canvas`.
  - Lines 1878–1883: in `_register_agent_tools`, when `self.canvas` is set, each callable from `self.canvas.functions()` is appended to `final_tools` (deduplicating against existing tools), so the canvas methods become normal LLM-callable tools.

- **`src/upsonic/agent/autonomous_agent/autonomous_agent.py`**
  - Line 13: `from upsonic.canvas.canvas import Canvas` (under `TYPE_CHECKING`).
  - Line 126: accepts `canvas: Optional["Canvas"] = None`.
  - Line 316: forwards `canvas=canvas` to the inner `Agent` constructor.

This is the entire integration surface. Tests, prebuilts, RAG, safety, storage, and team modules do not currently reference the canvas — it is opt-in per agent.

## 8. End-to-end flow

A typical interaction in which an agent uses the canvas to draft a document looks like this:

1. **User code constructs a canvas and an agent:**
   ```python
   from upsonic import Agent, Task
   from upsonic.canvas import Canvas

   canvas = Canvas("Project Brief", model="openai/gpt-4o-mini")
   agent = Agent(
       model="openai/gpt-4o",
       name="Writer",
       canvas=canvas,            # opt in
   )
   ```
2. **`Canvas.__init__`** resolves the model via `infer_model`, then immediately runs `_clean_canvas()`. If `Project_Brief.txt` already exists, fenced code blocks are stripped and the file is rewritten; otherwise `_load_canvas()` returns `""`, `_clean_canvas` writes an empty file (technically, it writes `""` back).
3. **`Agent.__init__`** stores the canvas and, inside `_register_agent_tools`, calls `canvas.functions()`, getting back `[get_current_state_of_canvas, change_in_canvas]`. These are merged into `final_tools` and forwarded into the `ToolManager`. The LLM now sees them as tools.
4. **User dispatches a task:**
   ```python
   await agent.do_async(Task("Draft a one-page brief for Project X."))
   ```
5. **The LLM, via tool calls, drives the canvas:**
   - It calls `get_current_state_of_canvas()` and receives `"Empty Canvas"`.
   - It calls `change_in_canvas(new_text_of_part="# Project X\n...", part_definition="title")`.
   - Inside `change_in_canvas`, since the current canvas is `"Empty Canvas"`, the **fast path** triggers: the new text is written to `Project_Brief.txt` immediately, no LLM editor call, and the new content is returned.
6. **Subsequent edit:**
   - The agent later calls `change_in_canvas(new_text_of_part="## Risks\n...", part_definition="risks section")`.
   - Now the canvas is non-empty, so a one-off `Agent(name="Canvas Editor")` is created with the canvas's `model`. A `Task` is built whose prompt embeds the *current canvas*, the *target part definition*, and the *replacement text*, with explicit instructions to either replace-or-append and to return only the document.
   - The editor agent runs `agent.do_async(task)`. Its `task.response` becomes the new canvas content, which is saved via `_save_canvas` and returned.
7. **Persistence:** every successful edit ends with a fresh `Project_Brief.txt` on disk. The next process start can pick up where the previous one left off simply by constructing `Canvas("Project Brief")` again.

### Behavioural notes and edge cases

- **Filename derivation is lossy.** `Canvas("My Project!")` and `Canvas("My Project")` both map to `My_Project.txt`. Consumers must avoid name collisions across canvases that should be distinct.
- **No locking.** Two `Canvas` instances pointing at the same name will race on the file. The canvas does not coordinate concurrent edits.
- **No size or rate limits.** The whole canvas is sent into the editor prompt on every non-empty edit, so very large canvases will incur correspondingly large prompt costs and may run into context limits of the chosen `model`.
- **Output discipline relies on the prompt.** Because the editor LLM is asked to return *only* the document, any chattiness from the model would leak into the canvas. The `_clean_canvas` pass on next construction removes fenced code blocks but not other extraneous text.
- **Default editor model is small (`openai/gpt-4o-mini`).** This is a deliberate cost choice; pass `model=` explicitly for higher-quality edits.
- **Async only for edits.** `get_current_state_of_canvas` is synchronous (a plain file read), while `change_in_canvas` must be awaited.

### Summary

`upsonic.canvas` is a deliberately minimal subsystem: a single class, two public methods, two private I/O helpers, and a lazy-loading `__init__`. Its value comes from the convention that any `Agent` constructed with `canvas=Canvas(...)` automatically gains LLM-callable read/write tools over a persistent, file-backed text document — turning the canvas into a long-lived, shared workspace that survives process restarts and integrates seamlessly with Upsonic's tool-calling machinery.
