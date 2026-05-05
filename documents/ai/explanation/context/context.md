---
name: context-source-primitives
description: Use when working on how Upsonic feeds contextual information into LLM prompts via `Task.context`, including serializing tasks/agents into prompt-injectable JSON and wiring graph predecessor outputs into successor prompts. Use when a user asks to add a new context source type, debug `<Context>`/`<YourCharacter>`/`<PreviousTaskNodeOutput>` blocks, customize the default fallback system prompt, or trace how `Graph` auto-injects predecessor outputs. Trigger when the user mentions `ContextSource`, `TaskOutputSource`, `turn_task_to_string`, `turn_agent_to_string`, `DefaultPrompt`, `default_prompt`, `src/upsonic/context/`, `task.context`, `ContextManager`, `SystemPromptManager`, `<PreviousTaskNodeOutput>`, `<YourCharacter>`, graph passthrough, retrieval_mode, or prompt assembly.
---

# `src/upsonic/context/` — Context Source Primitives

## 1. What this folder is

`upsonic/context/` is a tiny, foundational support module that provides the **typed primitives and serialization helpers Upsonic uses to feed contextual information into an LLM call**. It is intentionally lightweight: it does not run any logic on its own and it does not directly invoke models. Its job is to define *what kinds of things can appear inside a `Task.context` list* and to provide the shims that turn rich domain objects (a `Task`, an `Agent`) into compact JSON payloads suitable for stuffing into a prompt.

In Upsonic, every call to an `Agent` or `Direct` ultimately runs against a `Task`. A `Task` carries a free-form `context: list[Any]` field that the framework's runtime must transform into a single string that gets prepended to the user/system message stack. The module under `src/upsonic/context/` defines the building blocks for this pipeline:

- A discriminator type, `ContextSource`, that any "abstract context object" can subclass so it can be detected at runtime.
- A first concrete subclass, `TaskOutputSource`, used by the `Graph` engine to forward a previous task's output into a downstream task's prompt.
- Two stringification helpers — `turn_task_to_string` and `turn_agent_to_string` — that produce JSON snapshots of a `Task` and an `Agent` respectively.
- A canonical `DefaultPrompt` model with a fallback system prompt used when an agent has no character configured.

These pieces are then consumed by `agent/context_managers/context_manager.py` (which builds the per-call `<Context>` block), `agent/context_managers/system_prompt_manager.py` (which builds the `<YourCharacter>` block and falls back to the default prompt) and by `direct.py` and `graph/graph.py` (which auto-inject `TaskOutputSource` items between graph nodes). In short, `context/` is the **data-shape contract** between the `Task` API surface and the prompt-assembly machinery.

The folder is deliberately import-light. Each file imports only `pydantic.BaseModel` plus typing primitives, and the heavier callers (`Task`, `Agent`) are referenced via `TYPE_CHECKING` to avoid circular imports at module load time. This is a recurring pattern in Upsonic for "leaf" support modules.

## 2. Folder layout

```
src/upsonic/context/
├── __init__.py            # empty re-export point (zero-byte)
├── agent.py               # turn_agent_to_string()
├── default_prompt.py      # DefaultPrompt + default_prompt()
├── sources.py             # ContextSource + TaskOutputSource
└── task.py                # turn_task_to_string()
```

| File | Lines | Public symbols | Depends on |
|------|------:|----------------|------------|
| `__init__.py` | 0 | — (no re-exports) | — |
| `agent.py` | 21 | `turn_agent_to_string` | `json`, `upsonic.agent.agent.Agent` (TYPE_CHECKING) |
| `default_prompt.py` | 10 | `DefaultPrompt`, `default_prompt` | `pydantic.BaseModel` |
| `sources.py` | 27 | `ContextSource`, `TaskOutputSource` | `pydantic.BaseModel`, `typing.Optional` |
| `task.py` | 16 | `turn_task_to_string` | `json`, `pydantic.BaseModel`, `upsonic.tasks.tasks.Task` (TYPE_CHECKING) |

## 3. Top-level files — file-by-file walkthrough

### 3.1 `__init__.py`

The package init file is intentionally empty (zero bytes). The folder does **not** re-export its symbols at the package level — every consumer imports the leaf module explicitly:

```python
from upsonic.context.task import turn_task_to_string
from upsonic.context.sources import TaskOutputSource
from upsonic.context.agent import turn_agent_to_string
from upsonic.context.default_prompt import default_prompt
```

This keeps `import upsonic.context` cheap and avoids dragging the `Task`/`Agent` heavy graph into the import chain.

### 3.2 `task.py` — `turn_task_to_string`

```python
def turn_task_to_string(task: "Task"):
    the_dict = {}
    the_dict["id"] = task.task_id
    the_dict["description"] = task.description
    the_dict["attachments"] = task.attachments
    the_dict["response"] = str(task.response)
    string_of_dict = json.dumps(the_dict)
    return string_of_dict
```

| Symbol | Kind | Notes |
|--------|------|-------|
| `turn_task_to_string(task)` | function | Serializes a `Task` to a JSON string with four fields: `id`, `description`, `attachments`, `response`. |

Key behaviors:

- **Forward reference** — `Task` is imported only under `TYPE_CHECKING`, so this function never triggers a circular import at module load time. The argument is annotated as the string `"Task"`.
- **`response` is forced to `str`** — even if the task already produced a structured `BaseModel` response, this helper coerces it via `str(...)` before JSON-encoding. That means structured responses will be embedded as their `repr`/`str` representation, not as nested JSON. (`ContextManager._format_task_output` in the agent layer takes a more sophisticated path for `TaskOutputSource` outputs, preferring `model_dump_json` when available.)
- **No PII redaction or truncation** — the helper is a pure dump; trimming, summarization, or filtering are the caller's responsibility.
- **`from pydantic import BaseModel`** is imported at the top of the module but unused inside the function; it is kept so callers that want a stable typed handle can reference the `Task`'s pydantic provenance.

The helper is consumed by `ContextManager._build_context_prompt` (`agent/context_managers/context_manager.py`) when a `Task` instance appears inside another task's `context` list:

```python
if isinstance(item, Task):
    task_parts.append(f"Task ID ({item.get_task_id()}): " + turn_task_to_string(item))
```

The resulting string ends up in the `<Tasks>` … `</Tasks>` block of the assembled prompt.

### 3.3 `default_prompt.py` — `DefaultPrompt` and `default_prompt()`

```python
class DefaultPrompt(BaseModel):
    prompt: str

def default_prompt():
    return DefaultPrompt(prompt="""
You are a helpful agent that can complete tasks. Try to complete the task as best as you can. If you need any external information, check for tools. If not found, inform the user or ask for help. You MUST respect the cultural guidelines provided to you IF PROVIDED.
""")
```

| Symbol | Kind | Notes |
|--------|------|-------|
| `DefaultPrompt` | `pydantic.BaseModel` subclass | Single field `prompt: str`. Acts as a typed envelope around the canonical fallback string. |
| `default_prompt()` | function | Returns a freshly constructed `DefaultPrompt` instance every call. |

The default prompt is used by `SystemPromptManager._build_system_prompt` whenever **all** of the following are true:

1. `self.agent.system_prompt is None`
2. The agent has no company metadata (`name`, `company_url`, `company_objective`, `company_description`)
3. Thinking is not enabled (no reflective Operation Deliberate Thought briefing has already been emitted)
4. No "culture" is attached (cultures get a more restrictive replacement prompt)

```python
if self.agent.system_prompt is None and not has_any_info and not is_thinking_enabled:
    if has_culture:
        base_prompt = "You are an agent with specific cultural guidelines ..."
    else:
        base_prompt = default_prompt().prompt
```

Because the function instantiates a fresh model each call, callers cannot mutate a shared singleton — small but important for thread-safety and deterministic prompts.

### 3.4 `sources.py` — `ContextSource` and `TaskOutputSource`

```python
class ContextSource(BaseModel):
    enabled: bool = True
    source_id: Optional[str] = None

class TaskOutputSource(ContextSource):
    task_description_or_id: str
    retrieval_mode: str = "full"  # Options: "full", "summary".
```

| Symbol | Kind | Fields | Notes |
|--------|------|--------|-------|
| `ContextSource` | abstract base `BaseModel` | `enabled: bool = True`, `source_id: Optional[str] = None` | Marker base class. The codebase does not check `isinstance(item, ContextSource)` directly today — runtime uses `isinstance(item, TaskOutputSource)` against the concrete subclass — but the base exists so future source types can subclass and share `enabled` / `source_id`. |
| `TaskOutputSource` | `ContextSource` subclass | `task_description_or_id: str`, `retrieval_mode: str = "full"` | Identifies a *previous* task whose result should be retrieved from the graph state and inlined into the current task's prompt. |

The `from __future__ import annotations` at the top makes all annotations lazy strings, so even though the file imports `Optional` from `typing`, the annotated types are never resolved at import time.

**Lifecycle of a `TaskOutputSource`:**

1. The object is created either manually by user code or **auto-injected** by the `Graph` engine in `graph/graph.py` between predecessor and successor nodes (see §5).
2. It lands in `task.context` alongside other allowed types (`Task`, `KnowledgeBase`, `str`, ...).
3. At prompt-assembly time `ContextManager._process_task_output_source` (or `Direct._build_messages_from_task` in the lighter `Direct` path) reads `state.get_task_output(item.task_description_or_id)`, formats it, and emits a `<PreviousTaskNodeOutput id='...'>` block.
4. `retrieval_mode` is currently treated as advisory — only `"full"` is honored; `"summary"` is reserved for future summarization paths (the comment "Might be useful" in the source flags this as a future hook).
5. `enabled` is similarly a forward-looking switch and is not yet branched on by the runtime.

### 3.5 `agent.py` — `turn_agent_to_string`

```python
def turn_agent_to_string(agent: Agent):
    the_dict = {}
    the_dict["id"] = agent.agent_id
    the_dict["name"] = agent.name
    the_dict["company_url"] = agent.company_url
    the_dict["company_objective"] = agent.company_objective
    the_dict["company_description"] = agent.company_description
    the_dict["company_name"] = agent.company_name
    the_dict["system_prompt"] = agent.system_prompt
    string_of_dict = json.dumps(the_dict)
    return string_of_dict
```

| Symbol | Kind | Notes |
|--------|------|-------|
| `turn_agent_to_string(agent)` | function | Serializes the *identity* of an `Agent` (id, name, company metadata, system prompt) to a JSON string. |

Notable points:

- **Identity-only serialization** — runtime state (`memory`, `knowledge`, `tools`, model bindings) is intentionally excluded. The helper exists so that *another* agent can be told who it is talking to or impersonating, not so it can clone another agent's state.
- **Forward reference** — same `TYPE_CHECKING` trick as `task.py` to avoid the import cycle between `upsonic.agent.agent` and `upsonic.context.agent`.
- **`from __future__ import annotations`** enables string-based annotations everywhere in the module, which is why the `Agent` annotation can resolve later.

This helper is invoked from `SystemPromptManager._build_system_prompt` whenever a *foreign* `Agent` instance is dropped into another task's `context`:

```python
agent_context_str = "<YourCharacter>"
found_agent_context = False
if self.task.context:
    for item in self.task.context:
        if isinstance(item, type(self.agent)):
            agent_context_str += f"\nAgent ID ({item.get_agent_id()}): {turn_agent_to_string(item)}"
            found_agent_context = True
if found_agent_context:
    agent_context_str += "\n</YourCharacter>"
    prompt_parts.append(agent_context_str)
```

The wrapping XML-ish tag `<YourCharacter>` signals to the LLM that the embedded JSON describes the persona/character it should adopt.

## 4. Subfolders

There are **no subfolders** under `src/upsonic/context/`. The package is intentionally flat — each context primitive sits directly under the package root, in its own focused module.

## 5. Cross-file relationships

Within the package itself the four leaf modules are independent — none of them imports another. The "graph" of relationships lives between this package and the rest of Upsonic. The diagram below summarizes those edges:

```
                           ┌──────────────────────────┐
                           │ upsonic/tasks/tasks.py   │
                           │   class Task             │
                           │     context: list[Any]   │◀──┐
                           └──────────────────────────┘   │
                                                          │ holds list of …
                                                          │
   ┌─────────────────────────────────────────────────┐    │
   │ upsonic/context/                                 │   │
   │  ├── sources.py                                  │   │
   │  │     class ContextSource(BaseModel)            │   │
   │  │     class TaskOutputSource(ContextSource) ────┼───┘
   │  ├── task.py    turn_task_to_string(task)        │
   │  ├── agent.py   turn_agent_to_string(agent)      │
   │  └── default_prompt.py  DefaultPrompt / default_prompt()
   └─────────────────────────────────────────────────┘
                       ▲           ▲           ▲
                       │           │           │
   ┌───────────────────┘           │           └────────────────────────┐
   │                               │                                    │
┌──┴──────────────────────────┐ ┌──┴──────────────────────────┐ ┌──────┴────────────────────┐
│ agent/context_managers/      │ │ agent/context_managers/      │ │ direct.py                  │
│  context_manager.py          │ │  system_prompt_manager.py    │ │  Direct._build_messages_   │
│   ContextManager             │ │   SystemPromptManager        │ │     from_task()            │
│   - reads Task / KB /        │ │   - injects default_prompt   │ │   - reads TaskOutputSource │
│     str / TaskOutputSource   │ │   - injects                  │ │     for graph passthrough  │
│   - calls turn_task_to_      │ │     turn_agent_to_string     │ └────────────────────────────┘
│     string                   │ └──────────────────────────────┘
└──────────────────────────────┘                                    ┌─────────────────────────┐
                                                                    │ graph/graph.py          │
                                                                    │  - auto-injects         │
                                                                    │    TaskOutputSource     │
                                                                    │    between predecessors │
                                                                    │    and successors       │
                                                                    └─────────────────────────┘
```

| Edge | Where it appears | Purpose |
|------|------------------|---------|
| `ContextManager → turn_task_to_string` | `agent/context_managers/context_manager.py:72` | Render a sibling `Task` into the `<Tasks>` block. |
| `ContextManager → TaskOutputSource` | `agent/context_managers/context_manager.py:84,176,329,373` | Detect graph passthrough items and pull from `state`. |
| `SystemPromptManager → default_prompt` | `agent/context_managers/system_prompt_manager.py:276` | Fall back when no character/system prompt is set. |
| `SystemPromptManager → turn_agent_to_string` | `agent/context_managers/system_prompt_manager.py:291` | Inline another agent's identity into `<YourCharacter>`. |
| `Direct._build_messages_from_task → TaskOutputSource` | `direct.py:183-210` | Same passthrough behavior in the lighter `Direct` execution path. |
| `Graph._execute_node → TaskOutputSource` | `graph/graph.py:31, 784, 792, 798` | Auto-injects `TaskOutputSource` entries derived from predecessor task IDs. |

## 6. Public API

The de-facto public API of `upsonic.context` is the union of the four leaf-module symbols. Although `__init__.py` does not re-export them, every internal caller imports them directly, so any external consumer should follow the same pattern.

```python
# Marker / discriminator base
from upsonic.context.sources import ContextSource

# Concrete: pull a previous graph node's output into a downstream task
from upsonic.context.sources import TaskOutputSource

# Serialization helpers
from upsonic.context.task import turn_task_to_string
from upsonic.context.agent import turn_agent_to_string

# Default fallback prompt
from upsonic.context.default_prompt import DefaultPrompt, default_prompt
```

| Symbol | Stability | Typical usage |
|--------|-----------|---------------|
| `ContextSource` | Stable, intended for subclassing | Future custom context source types should extend this and inherit `enabled` / `source_id`. |
| `TaskOutputSource` | Stable | Pass `TaskOutputSource(task_description_or_id="prior_task_id")` inside `Task(context=[...])`. |
| `turn_task_to_string` | Stable utility | Serialize a `Task` to a compact JSON snapshot for prompt injection. |
| `turn_agent_to_string` | Stable utility | Serialize an `Agent`'s identity for `<YourCharacter>` blocks. |
| `DefaultPrompt` | Stable model | Typed wrapper around the fallback prompt string. |
| `default_prompt()` | Stable factory | Returns a fresh `DefaultPrompt` containing the canonical fallback. |

## 7. Integration with the rest of Upsonic

### 7.1 With `upsonic.tasks.tasks.Task`

`Task.context` is annotated to accept a heterogeneous list. The runtime contract is *"each item is one of: another `Task`, a `KnowledgeBase`, a `str`, or a `ContextSource` (currently `TaskOutputSource`)"*. The `context/` package owns the last category. Any future kinds of source — for example a `MemoryOutputSource` or a `ToolOutputSource` — would slot in the same place, simply by subclassing `ContextSource` and adding handling in `ContextManager._build_context_prompt`.

### 7.2 With `upsonic.agent.context_managers.context_manager.ContextManager`

`ContextManager` is the primary consumer. Its `_build_context_prompt` walks `task.context` and dispatches by type:

| Item type | Bucket | Helper used | Output tag |
|-----------|--------|-------------|------------|
| `Task` | `task_parts` | `turn_task_to_string` | `<Tasks>...</Tasks>` |
| `KnowledgeBase` | `knowledge_base_parts` | (RAG search; not in this package) | `<Knowledge Base>...</Knowledge Base>` |
| `str` | `additional_parts` | — | `<Additional Context>...</Additional Context>` |
| `TaskOutputSource` | `previous_task_output_parts` | `_process_task_output_source` | `<PreviousTaskNodeOutput id='...'>...</PreviousTaskNodeOutput>` |

All buckets are concatenated under a single `<Context>...</Context>` envelope which becomes part of the system message.

### 7.3 With `upsonic.agent.context_managers.system_prompt_manager.SystemPromptManager`

`SystemPromptManager` uses two of the four primitives:

1. `default_prompt()` — only when the agent has no system prompt and no company metadata and no thinking briefing and no culture.
2. `turn_agent_to_string` — when `task.context` contains another `Agent` instance.

Together with the agent's own `system_prompt`, company metadata, culture (if any), tool instructions, and skills section, these contributions form the final system prompt assembled inside `_build_system_prompt`.

### 7.4 With `upsonic.direct.Direct`

`Direct._build_messages_from_task` is a slimmer pipeline that does **not** instantiate a `ContextManager`. It still respects `TaskOutputSource` (and plain `str`) so that graph passthrough works in `Direct` mode. The same `<PreviousTaskNodeOutput id='...'>` framing is produced.

### 7.5 With `upsonic.graph.graph.Graph`

`Graph._execute_node` *generates* `TaskOutputSource` instances on the fly. For every `TaskNode` whose predecessors have already executed, the graph engine constructs:

```python
source = TaskOutputSource(task_description_or_id=pred.id)
node.task.context.append(source)
```

For predecessors that are themselves `DecisionFunc` / `DecisionLLM` nodes, the engine traces through the decision and grabs the upstream `TaskNode` IDs instead. Existing entries are deduplicated against `existing_source_ids`. This is the mechanism by which graph state implicitly flows downstream without the user having to wire connections manually.

## 8. End-to-end flow

The diagram below traces a single task call where the user has supplied a mixed `context` list, plus an example with a `Graph` predecessor.

### 8.1 Standalone agent call

```
User builds:
  task = Task(
      description="...",
      context=[
          previous_task,                # Task
          knowledge_base,               # KnowledgeBase
          "extra clarification text",   # str
          another_agent,                # Agent  (consumed by SystemPromptManager only)
      ],
  )

agent.do(task)
   │
   ▼
Agent runtime spawns SystemPromptManager and ContextManager.
   │
   ├── SystemPromptManager._build_system_prompt
   │      │
   │      ├── if no system_prompt / company / culture / thinking →
   │      │       base_prompt = default_prompt().prompt           ← default_prompt.py
   │      │
   │      └── for item in task.context:
   │             if isinstance(item, Agent):
   │                 agent_context_str += turn_agent_to_string(item)  ← agent.py
   │      → injects <YourCharacter>...</YourCharacter>
   │
   └── ContextManager._build_context_prompt
          │
          ├── for item in task.context:
          │     ├── Task              → turn_task_to_string(item)        ← task.py
          │     ├── KnowledgeBase     → RAG search (out of scope here)
          │     ├── str               → appended to <Additional Context>
          │     └── TaskOutputSource  → _process_task_output_source       ← sources.py
          │
          └── concatenates all parts under <Context>...</Context>

The two strings (system prompt + context block) are then handed to the
model client as the SystemPromptPart of the request.
```

### 8.2 Graph node execution

```
Graph._execute_node(node)
   │
   ├── If node is a TaskNode:
   │     node.task.context = []        ← cleared each step
   │     predecessors = self._get_predecessors(node)
   │     for pred in predecessors:
   │         if pred is a Decision*:
   │              trace through to underlying TaskNodes
   │              for task_pred:
   │                  context.append(TaskOutputSource(task_description_or_id=task_pred.id))
   │         else:
   │              context.append(TaskOutputSource(task_description_or_id=pred.id))
   │
   └── _execute_task(node, state)
          │
          ▼
         ContextManager (or Direct) reads each TaskOutputSource:
            output = state.get_task_output(item.task_description_or_id)
            renders <PreviousTaskNodeOutput id='...'>...</PreviousTaskNodeOutput>
```

### 8.3 Cheat-sheet for adding a new context source

Because the package is so small, extending it is straightforward:

1. **Define the type** in `sources.py`:

   ```python
   class MyMemoryRecallSource(ContextSource):
       memory_query: str
       top_k: int = 3
   ```

2. **Handle it** in `ContextManager._build_context_prompt` by adding a new `elif isinstance(item, MyMemoryRecallSource)` branch and emitting an appropriate XML-tagged section into `final_context_parts`.

3. **(Optional) Mirror** the same handling in `Direct._build_messages_from_task` if you want it to flow in `Direct` mode too.

4. **(Optional) Auto-inject** in `Graph._execute_node` if it is something the graph engine should produce automatically (analogous to `TaskOutputSource`).

The `enabled` and `source_id` fields are inherited for free, giving you a uniform place to gate or identify a source.

## Summary

`upsonic/context/` is the **type and serialization layer for prompt context**. It is small (four files, ~75 lines of code total) but load-bearing: every Upsonic agent call funnels its `Task.context` list through helpers defined here, and the `Graph` engine relies on `TaskOutputSource` to wire predecessor outputs into successor prompts. The deliberate flatness, the lazy `TYPE_CHECKING` imports, and the empty `__init__.py` keep this module a cheap-to-import primitive that the heavier `agent/context_managers/`, `direct.py`, and `graph/graph.py` modules can compose on top of.
